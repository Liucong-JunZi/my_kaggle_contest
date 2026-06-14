# ROGII Hill Climb 推进路线图

**Created**: 2026-06-15
**Owner**: round_010
**Goal**: OOF per-well RMSE 9.18 → 7.x（对应 LB 11.38 → 8.x）

---

## 核心原则

1. **OOF 是货币，LB 是裁判**：所有路径决策由 OOF 引导，但每 N 步用 LB 做一次校准（防 OOF-LB gap 自欺）
2. **候选池只增不减**：失效候选的权重会被 hill climb 自动归零，不需要主动剔除
3. **Hill climb 廉价（5min/run）**：每加 3-5 个候选就重跑一次，不需要等
4. **新候选准入门槛**：单模 perwell < 13 即可入池（更差的拖慢搜索但偶尔提供 diversity）

---

## 阶段 0 — 当前态（now）

```
候选池: 12 (legacy 8 + c01-c04)
Hill climb best: 9.182 (legacy 8 only, 用 R10 fold 评估)
Best single: c03_lgb_huber 9.582 (R10 fold)

In-flight:
  - c05-c08 batch（约 1-2h 完成）
  - r9 v2 LB PENDING（约 30min 出分）
  - rogii-harvester（持续注入新候选材料）
```

---

## 阶段 1 — 即时收紧（今晚 1-2h）

### 1.1 c05-c08 完成后 → Hill Climb v2
- 触发条件：`results/candidates/` 多 4 个 parquet
- 自动操作：`python orchestrator/hillclimb.py --label run_v2_after_c0X`
- 期望 perwell：**9.05 - 9.15**（c03 huber 9.582 是新强 single，加进去应该 -0.05 ~ -0.15）

### 1.2 r9 v2 LB 出分 → 路线分支决策

| r9 v2 LB | 含义 | 行动 |
|---|---|---|
| **≤ 8.5** | 纯 PF 128-seed 强大，LB 校准证实 | 启动 r9 OOF 本地生成（5-7h），优先级最高 |
| **8.5 - 10** | 纯 PF 中等，作为 ensemble 一员有用 | r9 OOF 生成排到中等优先级 |
| **> 10** | r9 单跑不够强，LB 8 那个传言可能假 | 跳过 r9 OOF，重心放别处 |

### 1.3 LB 第一次提交（v13 + r9 v2 + 当前 hill climb 结果三选一或 blend）

提交策略**只用 1 个配额**：
- 如果 hill climb v2 OOF 比 v13 OOF (9.28) 改善 ≥ 0.1 → 提交 hill climb 的 ensemble submission
- 否则 → 提交 v13 直接（v13 已经在 Kaggle 上跑完了 submission.csv，我们没真正提交过）

**关键**：拿到 1 个新 LB 数字，建立 OOF→LB 映射函数 `f(OOF) = LB`。当前只有 v8 一个数据点（9.91→11.383）。多 1 个点能开始拟合。

---

## 阶段 2 — 多样化扩池（明天 1 天）

### 2.1 自动接入新候选（关键基础设施）

写 `orchestrator/auto_hillclimb.sh`：
```bash
last_count=$(ls results/candidates/*.parquet | wc -l)
while :; do
    sleep 600
    new_count=$(ls results/candidates/*.parquet | wc -l)
    if [ $new_count -gt $((last_count + 2)) ]; then
        python orchestrator/hillclimb.py --label "auto_$(date +%H%M)"
        last_count=$new_count
    fi
done
```

让它后台跑，每出 3 个新候选自动重跑 hill climb，输出到 `results/hillclimb_runs/auto_*.json`，**不需要手动调度**。

### 2.2 第二批候选（来源：rogii-harvester staging）

harvester 把 `experiments/public_harvest/` 填好后，从中挑选：
- **优先 1**：feature_engineering 里的新 PF 变体（multi-PN、multi-seed、不同 init_spread）
- **优先 2**：model_params 里的 LGB/CAT 强配置（按公开 kernel LB 排）
- **优先 3**：preprocessing 里的 GR smoothing 变体

每个 stub 改成 round_010 的 candidate 接口（10 行模板），train_one 跑出 OOF。每天 3-5 个新候选。

### 2.3 自家加 5-8 个候选（不依赖 harvester）

- **c09**: LGB seed bagging（seed=7）
- **c10**: LGB seed bagging（seed=123）
- **c11**: CAT seed bagging（seed=7）
- **c12**: LGB on subsampled features（colsample=0.5，强多样性）
- **c13**: HistGradientBoosting（sklearn，第三类 GBDT lib）
- **c14**: ExtraTrees on dense features（树平均，diverse from boosting）
- **c15**: Ridge on PF/beam offsets only（线性 fallback）

每个 6-15 min 训练，半天能加完。

### 2.4 Hill climb v3 触发条件
- 候选总数 ≥ 25 → 跑全量 hill climb
- 期望 perwell：**8.7 - 8.9**

---

## 阶段 3 — LB 校准 & 关键信号注入（后天 1 天）

### 3.1 LB 校准点拟合

到这阶段我们应该有 3-5 个 LB 数据点（v8、v13、hill climb v2、可能 r9 v2、可能 hill climb v3）。

拟合简单线性：`LB = a × OOF_perwell + b`

每次出 OOF 数字，预报 LB。如果 OOF 8.7 预报 LB 10.1 但实际 9.6 → b 在变 → 模型间 gap 在缩。

### 3.2 r9 OOF 注入（关键信号 #1）

**前提**：r9 v2 LB ≤ 9.5。

写 `experiments/round_010/candidates/c20_r9_pf128_full.py`：
- 内部调用 r9 v2 的 PF 128-seed 函数（从 `experiments/round_009/r9_pf_only_submit_v2.py` import）
- 在 R10 fold 上跑 OOF（每 fold 跑 5 train/val 的 PF）
- 5-7h 一次性投资，永久候选

加进池后，hill climb 期望 perwell：**8.0 - 8.4**。

### 3.3 multi-PN PF 注入（关键信号 #2）

写 c21-c24，每个用不同 PN（0.005/0.01/0.03/0.08）跑 16-seed ensemble：
- 利用 c20 已有的 PF infra，只改 PN 常数
- 每个 1-2h
- 加进池后期望 perwell：**7.7 - 8.1**

### 3.4 LB 提交（每次 hill climb v4/v5 都提一次）

每天 3-5 配额，按重要性顺序：
1. 阶段 3 后第一个 hill climb（验证 OOF-LB 校准）
2. 加 r9 OOF 后的 hill climb（验证 r9 注入有效）
3. 加 multi-PN 后的 hill climb（验证组合效应）

每个提交后更新 LB 校准函数。

---

## 阶段 4 — 公开资源融合（harvester 输出 → 候选池）

### 4.1 Harvester 完成后的 triage

harvester 产出 `experiments/public_harvest/` 后，**人工 review** 每个 stub（不是 30 个全跑）：

按公开 kernel 的 LB 分数排序：
- LB ≤ 8.5 的 kernel 的所有技术 → **必试**
- LB 8.5-10 的 kernel 的独特技术 → **选试**（看跟我们重叠度）
- LB > 10 的 kernel → **跳过**（除非有特别新的 idea）

### 4.2 三类融合方式

| 类型 | 融合到哪 | 频次 |
|---|---|---|
| 新 feature | `shared/data_loader.py` 加 column → 重训现有候选（c09+ 用扩充 feature set） | 每 3-5 feat batch 一次 |
| 新模型 | `candidates/cXX.py` 新建 | 每 1-2 个 |
| 新 blending recipe | `experiments/round_010/recipes/<name>.json` 当 hill climb 初始化预设 | 单独评估 |

### 4.3 物理捷径（visible-overlap）

按 LB-7.776 kernel 的 `tvt_from_contacts` 模式：
- 写 `submit_ensemble.py` 时，对 visible-overlap wells 直接用物理模型替换 ML 输出
- **不进 OOF**（OOF 评估时 visible-overlap = 0，意义不大），但**进 submission**
- 期望 LB：-0.5 ~ -1.0（取决于 visible-overlap 比例）

---

## 决策节点 & Switch 条件

### Switch A: r9 v2 LB 出分（阶段 1.2）
```
if LB(r9_v2) <= 8.5:
    祭出 r9 OOF 路线  → 阶段 3.2 优先级 = 1
elif LB(r9_v2) <= 10:
    r9 OOF 路线 优先级 = 2
else:
    跳过 r9 OOF，重心转向 GBDT 多样化（阶段 2.3）
```

### Switch B: Hill climb v2 改善幅度（阶段 1.1 后）
```
delta = 9.182 - perwell(v2)
if delta >= 0.10:
    走原路线（增加多样化）
elif delta < 0.05:
    停下来诊断 — 候选池可能高度相关，需要更激进多样化（阶段 2.3 立刻执行）
```

### Switch C: 阶段 2 后 perwell（阶段 2.4）
```
if perwell(v3) <= 8.8:
    走阶段 3（PF 路线）有信心继续突破
elif perwell(v3) <= 9.0:
    走阶段 3，但加 LB 校准提交先确认方向
else:
    GBDT 路线撞墙，必须激进上 PF infra（跳到 3.2）
```

### Switch D: r9 OOF 注入后效果（阶段 3.2）
```
if perwell(v4) <= 8.3:
    走 multi-PN 路线（3.3）
elif perwell(v4) <= 8.6:
    走 multi-PN，但同时加 selector 路线（kernel signal）
else:
    PF 注入边际有限，转向 visible-overlap 物理捷径（4.3）
```

### Switch E: 7.x 不可达（保险阀）
```
如果阶段 3 末 perwell > 8.5 且 LB > 9.5:
    7.x 这一目标在当前 feature/model 信息量下不可达
    → 只能等 harvester 给出 visible-overlap leak 模式或新 image-track 信号
    → 或者放弃 7.x 改打稳 8.x 提交
```

---

## 7.x 目标的可达路径（最佳情况）

| 阶段 | 累计 OOF perwell | 累计候选 | 主要新增 | 工作量 |
|---|---|---|---|---|
| 0 (now) | 9.18 | 12 | — | — |
| 1 (c05-c08 + hill climb v2) | 9.05 | 16 | GBDT 变种 | 2h |
| 2 (扩池到 25) | 8.85 | 25 | seed bagging + 算法多样化 | 1 day |
| 3.2 (r9 OOF 加进) | 8.20 | 26 | 128-seed PF | 7h |
| 3.3 (multi-PN PF) | 7.85 | 30 | 4-PN PF | 4h |
| 4.3 (visible-overlap) | 7.50 (LB 估计 8.5) | 30 | 物理捷径 | 1 day |
| 上限 | **7.50** | 30+ | — | — |

**结论**：7.5 是最佳情况，**真正破 7.0 需要 image track 或全新信号源**，本周不太现实。**目标定在 OOF 7.5，对应 LB 8.5**，跟当前 LB 7.776 kernel 持平/稍逊。

---

## 风险登记

| 风险 | 概率 | 影响 | 缓解 |
|---|---|---|---|
| r9 v2 LB > 10（纯 PF 路线破灭）| 中 | 高 | 阶段 2 多样化能保 -0.3，路线不死 |
| Hill climb 在 8.8 撞墙 | 高 | 中 | Switch C 自动转 PF 路线 |
| r9 OOF 跑 7h 失败 | 低 | 中 | 容错：先跑 1 fold sanity check |
| harvester 抓的全是 LB > 10 的劣质 kernel | 中 | 低 | 按 LB 排序，劣质跳过 |
| visible-overlap 实际比例小 | 高（未知）| 高 | 阶段 1 提交后从 LB 反推；fallback 仍有 8.5 |
| OOF-LB gap 大（提交后回退）| 中 | 高 | 校准函数 + 多次 LB 提交 |

---

## 操作 checklist（按时间线执行）

- [ ] c05-c08 batch 完成（自动）
- [ ] 第一次自动 hill climb v2（c05-c08 完成后立即触发）
- [ ] r9 v2 LB 出分 → 决策 Switch A
- [ ] 提交 hill climb v2 或 v13 到 LB（用第 1 个新配额）
- [ ] 写 c09-c15 7 个候选（自家多样化）
- [ ] 写 auto_hillclimb.sh 后台调度器
- [ ] harvester 出第一批 staging → 挑 3-5 个加进池
- [ ] hill climb v3（候选 ≥ 25）
- [ ] r9 OOF 生成（前提 Switch A 通过）
- [ ] hill climb v4 + LB 校准提交
- [ ] multi-PN PF 候选 c21-c24
- [ ] hill climb v5 + LB 提交
- [ ] visible-overlap 物理捷径 submit_ensemble 改造
- [ ] 最终 hill climb v6 + 终极 LB 提交

---

## 附录 — Hill climb 内部循环约束

- 当前 STEP_GRID = `[0.05, 0.10, 0.20, 0.30]`，N_ITER = 50
- 候选 ≥ 30 时考虑加 0.02 步长（更细搜索），N_ITER 提到 80
- 候选 ≥ 50 时考虑 random restarts（5 次随机起点取最佳），防止贪心局部最优
- TOL = 1e-4 不动（够细）
