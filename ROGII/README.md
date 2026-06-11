# ROGII Wellbore Geology Prediction — CV Direction

预测水平井的 True Vertical Thickness (TVT)，基于井轨迹和 typewell 参考 GR 数据。

## 项目结构

```
ROGII/
├── README.md                       # 本文件
├── environment.yml                 # conda 环境定义
│
├── src/                            # 核心代码
│   ├── gen_images.py               # 2D 匹配网格图像生成 (HDF5)
│   ├── gen_features.py             # 表格特征工程 (NPZ)
│   ├── train.py                    # SegFormer 训练 + raw/anchored 双指标评估
│   ├── decode.py                   # SDF→TVT 解码 + 已知段 partial anchor (R4-A)
│   └── eval.py                     # 预训练 checkpoint 评估
│
├── data/
│   ├── cache/                      # 预处理后的数据集 (HDF5/NPZ)
│   │   ├── cfg-img-medium/         # ⭐ 当前最佳: 50井, raw 15.84 / anc 13.67
│   │   ├── feat-lgb-base/          # 表格特征基线
│   │   └── feat-lgb-domain/        # +领域特征
│
├── docs/
│   ├── competition-description.md  # 比赛说明
│   ├── competition-insights.md     # 论坛精华整理
│   └── hengck23-reference/         # hengck23 参考代码 + 预训练权重
│
├── experiments/                    # 每轮设计文档 + 一次性脚本
│   ├── round_000-pf-discrete/      # 早期 PF/Viterbi 探索（notes + scripts）
│   ├── round_003/notes.md          # backbone/channel ablations
│   ├── round_004/                  # decoding + fair data scaling
│   └── round_005/                  # soft-argmin TVT loss (DEAD), MTP planning
│
├── results/                        # 各轮最终指标 (JSON + 日志)
│   ├── summary.json                # 全部 config 总览
│   ├── round_002/  round_003/  round_004/
│
├── notebooks/                      # Jupyter (待用)
├── rogii-wellbore-geology-prediction/   # 原始竞赛数据 (只读)
└── .claude/skills/                 # 多 Agent 协作 skills
```

## 当前最佳结果

| Config | raw RMSE | **anchored RMSE** | 训练 | 备注 |
|--------|----------|-------------------|------|------|
| **cfg-img-medium (R4 重训)** | 15.84 | **13.67** ⭐ | 77s, 50井×20ep mit-b0 | R4-A 流水线 + anchored-best ckpt |
| cfg-img-medium-mitb1 | 15.93 | — | 114s | R3 - b1 持平 b0，容量饱和 |
| cfg-img-medium-4ch (+gr_diff) | 25.57 | — | 85s | R3 - gr_diff 通道反向 |
| cfg-img-medium-100 (fair) | 25.35 | — | 204s, 100井×20ep | R4-B - 100井反而退化 |
| cfg-img-medium-200-fair | 25.29 | 34.17 ❌ | 391s | R4-B - 200井 anc>raw，新井已知段偏移 |
| feat-lgb-domain | 121.32 | — | 3s | LightGBM |
| feat-lgb-base | 211.73 | — | 3s | LightGBM |

**死路确认（R3-R5）**：
- mit-b1 backbone (R3) — 50 井容量饱和
- gr_diff 通道 (R3) — 派生信号目标泄漏
- 数据扩规模 (R2 + R4-B ×2) — 额外井分布与 val 不一致
- subpixel / smooth 解码 (R4-A) — 量化误差不是瓶颈
- **Soft-argmin + Huber TVT loss (R5-A)** — 与 SDF MSE 梯度冲突

**Round 4 增益**：R4-A partial anchor (α=0.75) + anchored-best ckpt = **15.89 → 13.67 ft (-2.22)** 零训练成本。

**Round 5 候选**：
1. ~~Soft-argmin + Huber TVT loss~~ — R5-A 已死路
2. **MTP head（dip / uncertainty / layer mask）** — aux loss 走独立 head，与 SDF gradient 解耦 ⭐
3. Beam search / DP decode — 低优 (R4-A 已证空间平滑增益小)
4. 数据聚类采样再扩 — R4-B 改进（先按 GR/TVT 分布筛井）

**参考**：hengck23 早期预训练 (00004053.pth) 在他 val 上 14.82 mean / 11.68 median。我们 anchored 13.67 已超过其平均。

## 核心方法

**SDF (Signed Distance Function) + SegFormer**
- 输入: (C=3, T=192, H=576) 2D 图像 — typewell GR / horizontal GR / history path
- 输出: SDF 热力图 = (h_tvt - t_tvt) / 40, clip [-3, 3]
- 解码: argmin(|sdf|) per column → t_tvt[idx] → partial anchor (R4-A)

参考 hengck23 [discussion #699853](https://www.kaggle.com/competitions/rogii-wellbore-geology-prediction/discussion/699853)。

## 使用方法

```bash
# 1. 环境
pip install torch transformers h5py lightgbm scipy opencv-python-headless

# 2. 生成数据（gen_images.py 内部用 CONFIGS 字典；fair scaling 看 experiments/round_004/）
python src/gen_images.py
python src/gen_features.py

# 3. 训练（统一入口，自动报告 raw + anchored RMSE）
python src/train.py --dataset data/cache/cfg-img-medium
python src/train.py --dataset data/cache/cfg-img-medium --backbone nvidia/mit-b1 --epochs 20

# 4. 评估 hengck23 预训练
python src/eval.py --ckpt docs/hengck23-reference/00004053.pth
```

## Multi-Agent 实验流程

使用 [claude-agent-sdk](https://github.com/anthropics/claude-agent-sdk-python)：

```
cv-orchestrator (我)
  ├─ image-searcher  → 生成图像数据集
  ├─ feature-searcher → 生成特征矩阵
  ├─ train-validator → 训练 + 评估
  └─ experiment-tracker → 记录实验
```

详见 `.claude/skills/cv-orchestrator/SKILL.md` + `WORK_PROBLEMS.md`。
