# LB 7.776 Kernel — smartorz/lb-7-776-rogii-ridge-sp

**Source**: kaggle kernels pull smartorz/lb-7-776-rogii-ridge-sp
**LB Score**: ~8 (user said 7.776 from name)
**File**: `lb-7-776-rogii-ridge-sp.ipynb` (33 cells, 50k+ tokens, 1353 LOC after extract)

## Core architecture

**This is not one model — it's a 7-signal ensemble blended via LightGBM + CatBoost + Ridge stack.**

```
For each well:
  1. Particle Filter ANCC (numba JIT)
  2. Particle Filter Z-based (numba JIT)
  3. Beam search × 14 configs → ensemble
  4. Multi-scale NCC (windows 8/15/25)
  5. Per-formation TVT (6 formations × early/mid/late/wls segments)
  6. Dense ANCC imputation (KNN over X,Y wells)
  7. Selector → blends PF/beam by well code

  ↓ All concatenated into ~100s of features per row

LightGBM(GroupKFold=5) + CatBoost(GroupKFold=5) on raw features
  → OOF preds

Ridge stack of OOF preds → final pred

Final: 0.3 × ridge + 0.7 × heuristic (PF+beam+selector blend)
```

## 关键的 target 定义 ⭐⭐⭐

```python
# Cell at line 927:
result['target'] = ev['TVT'] - last_known_TVT
```

**Target 是相对偏移，不是绝对 TVT**。所有 model 学的都是 "lateral 段相对 last_known 的微小偏离"。

Inference 时：
```python
pred_tvt = last_known_TVT + model.predict(features)
```

这就是为什么 LB ~8 — 利用了井是水平的物理事实：lateral TVT 跟 last_known TVT 的差异通常 ±50 ft 而不是绝对 ~10000 ft 范围。

## 我们的对比

| 项 | LB 7.776 kernel | 我们 |
|----|----------------|------|
| **Target** | `TVT - last_known_TVT` | **`TVT` 绝对值** ❌ |
| 主信号 | PF + Beam + NCC + Formation | SegFormer SDF |
| 数据 | 全 773 训练井 | 50 人工精挑 |
| Val | GroupKFold-5 跨全语料 | 50 人工精挑 |
| LightGBM | 100+ features，相对 target | 12 features，绝对 target |
| Ensemble | LGB + Cat + Ridge | None |
| Post-processing | apply_pp + sg_smooth | partial anchor α=0.75 |

## Inference 加速 - Numba JIT

所有 PF 和 beam search 用 `@njit` 装饰，速度比 Python 快 50-100x。这让 7-signal × 128-seed × 500-particle 在 4-5 小时内跑完。

## 物理模型 (`tvt_from_contacts`)

```python
# Cell line 115
def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    # 基于地质 contact 的物理 baseline
    # 用 reference formation 的最浅 TVT 反推全井 TVT
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset
```

对训练集见过的井，**直接用物理模型代替 ML 预测**（zero-shot 100% 准确）。

## 选择器机制 (`selector_well_code`)

```python
SELECTOR_BIN_VARIANTS = {
    0: 'pf_scale_5_hold_0.2',
    1: 'pf_scale_3_hold_0.15',
    ...
}
```

按井的 (n_eval, z_span) 落入 6 个 bin，每个 bin 用不同的 PF scale + beam blending 比例。这是基于 CV 调出来的 per-well-type config。

## 14 个 Beam 配置 (`BEAM_CONFIGS`)

```python
[(10, 20.0, 144.0, 2), (10, 8.0, 64.0, 2), ...]
# (bs, mc, es, r) 14 个不同 hyperparam 的 beam search
```

每个 beam 输出 1 路 TVT 预测，平均后给 LGB 当特征。

## 离线依赖

```python
artifacts_path = "/kaggle/input/datasets/ravaghi/wellbore-geology-prediction-artifacts"
```

需要预生成的 `_FI` (FormationPlaneKNN) 和 `_DI` (DenseANCCImputer) artifacts。这俩是基于全训练井的 KNN 模型，inference 时直接 load。

## 复制这个方法的可行性

### 立即可做 (1 天内)
- ✅ **修 target 定义** (`tvt - last_known_tvt`) — 我们的 LightGBM 应该立刻从 121 跳到 15-20
- ✅ **用全 773 井训练** — 已生成 cfg-img-full

### 中期 (1-3 天)
- ⚪ **抄 PF/Beam pipeline** (numba JIT) — 几百行代码但相对独立
- ⚪ **抄 multi-scale NCC** — 相对简单
- ⚪ **替换我们的 SDF 头为 NCC + 几何特征**

### 长期 (3+ 天)
- ⚪ **Formation contact 物理模型** — 需要理解 ANCC/ASTNU/ASTNL/EGFDU/EGFDL/BUDA 含义
- ⚪ **Dense ANCC + Formation KNN** — 需要全训练井做 KDTree

## 优先级建议

按 ROI 排序：

1. **修 LightGBM target**（30 分钟，最高 ROI）— 直接从 121 → 15-20，秒杀 SDF
2. **加 last_known_TVT + dz cumsum 几何先验作为基线** — 论坛说 plain GBDT 能到 ~9.6
3. **抄 PF pipeline**（1-2 天）— 拿到 ~10-12 LB
4. **完整复刻 7-signal stack**（3-5 天）— 拿到 ~8 LB

## 文件

- `lb-7-776-rogii-ridge-sp.ipynb` — 原始 notebook
- `lb-7-776-rogii-ridge-sp.py` — 提取的 Python 代码 (1353 行)
