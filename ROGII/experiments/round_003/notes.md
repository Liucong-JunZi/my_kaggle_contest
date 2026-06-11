# Round 3 — Backbone & Channel Ablations

**Goal**: 排除两个直觉性方向 — 更大 backbone / 增加输入通道。

## Setup

- 数据：cfg-img-medium 同源（50 train / 50 val 井，T=192 H=576 comp=24）
- 模型：GeoSteerNet (SegFormer + FPN fusion + history head)
- 训练：15 epochs, Adam lr=1e-3, batch=2, MPS

## Experiments

| Config | Backbone | Channels | best RMSE | Δ vs baseline 15.89 |
|--------|----------|----------|-----------|---------------------|
| cfg-img-medium (baseline) | mit-b0 | 3 [t_gr,h_gr,history] | **15.89** | — |
| cfg-img-medium-mitb1 | mit-b1 | 3 | 15.93 | +0.04 (tied) |
| cfg-img-medium-4ch | mit-b0 | 4 [+gr_diff] | 25.57 | **+9.68 (hurt)** |

## Findings

### 1. Backbone scaling: dead end at 50 wells
- mit-b1 (~4× params of b0) gave +0.04 ft — pure noise.
- train_loss 在 ep2 后就稳定在 0.20–0.25，从未继续下降 → 模型在该数据规模下已饱和。
- 推论：b2/b3 在 50 井数据上预期同样无增益。**更大 backbone 之前先增数据。**

### 2. gr_diff channel: actively harmful
- 在公平几何下加 gr_diff (= h_gr - t_gr lookup)，val RMSE 退化 +9.68 ft。
- val 曲线在 ep4 就卡 25.7，loss 仍降但 val 不动 → 模型抓到 gr_diff 锚定的捷径，无法突破。
- 可能原因：
  1. `gr_diff` 计算时已隐含某个 TVT 假设 → 目标泄漏到输入的"伪信号"，对未知段失效。
  2. 第 4 通道 patch_embedding 是新初始化，破坏了 mit-b0 预训练对齐。

## 副实验（已废弃）

- **cfg-img-grdiff (旧 comp=16)**: RMSE 60.99。事后发现 11/50 val 井 t_tvt 退化 — 是几何参数 bug，与 gr_diff 通道无关。已用 cfg-img-medium-4ch 重做公平对比。

## 工程修复

- 发现 `src/gen_images.py` 之前回退丢了 `/t_tvt` 字段保存逻辑 → 已修复。
- 新增 SKILL 经验：comp ≤ 24，新配置必须先 std 检查 t_tvt 不退化。

## Round 4 候选

| 路径 | 优先 | 理由 |
|------|------|------|
| 公平 200 井 (同 val) | ⭐⭐⭐ | 验证是否能解锁 b1 容量 |
| SDF scale 细搜索 (20/40/80) | ⭐⭐ | 低成本 |
| MTP head | ⭐⭐⭐⭐ | hengck23 真正提分关键 |
| Beam search 后处理 | ⭐⭐⭐ | 几何约束利好 |

死路（不再尝试）：
- mit-b2 / mit-b3 单独升级（先加数据）
- 当前形式的 gr_diff 通道
