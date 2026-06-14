# Kernel: hengck23/cnn-mtp-example (reference)

**Author**: hengck23 (Kaggle Grandmaster, original reference for the SDF approach)
**Last run**: ~2026-04
**Files**: cnn-mtp-example.ipynb

## Architecture (one-paragraph)
A small CNN trained on **GR-difference matching heatmaps** (2-channel: heatmap + history mask) producing a **multi-trajectory prediction (MTP)** output: K=10 candidate trajectories, each of length L=24, plus K logits selecting which trajectory is best. The CNN backbone is 4 ConvBlocks with 3 AvgPool stages (output 96 channels), flattened, then head MLP 2304→512→1024→4096, then two linear heads: `path_head: 4096→K*L=240` and `logit_head: 4096→K`. The MTP loss = winner-takes-all regression on the closest trajectory + cross-entropy classification of that winner.

## Key Techniques

### Feature Engineering
- 2D matching heatmap input: H×W = T×H or similar (computed externally, not in this kernel).
- History mask channel marks where TVT is known.

### Model & Hyperparams
- 4× ConvBlock with GELU, BatchNorm
- 3× AvgPool 2x2 stages (so input is downsampled 8x)
- Head: 2304 (3*8*96 flat dim) → 512 → 1024 → 4096
- K = 10 candidate trajectories, L = 24 path length
- Output: `paths (B,K,L)` + `logits (B,K)`

### MTP Loss
```python
def do_mtp_loss(pred, logit, target, alpha=1.0):
    error = ((pred - target[:, None]) ** 2).mean(dim=-1)  # [B, K]
    best_k = error.argmin(dim=1)                          # [B]
    reg_loss = error[torch.arange(B), best_k].mean()      # only the best trajectory
    cls_loss = F.cross_entropy(logit, best_k)             # classify the best
    return reg_loss + alpha * cls_loss
```

### Particle Filter / Beam Search
- None (pure CNN).

### Ensemble / Blending
- None at the kernel level.

### CV Methodology
- Not specified in this minimal example (just architecture demo).

## Anything novel vs LB-7.776 kernel?
**YES, fundamentally different formulation.** Where SDF predicts a continuous distance field (anchor-then-decode), MTP predicts K discrete candidate trajectories with a soft selector. Avoids the SDF→TVT decoding problem entirely. Two interesting characteristics:
1. The K=10 trajectories can capture **multimodality** (e.g., dip up vs. dip down, two plausible formations) — single SDF outputs cannot.
2. Winner-takes-all training avoids gradient conflict between modes.

This formulation is closer to the "Multipath" trajectory prediction literature than to ROGII's typical PF/Ridge setup. Worth experimenting with for our R5-A failure (TVT loss + SDF MSE conflict): MTP could be the right framing.

## Score-relevant constants
| name | value |
|------|-------|
| K (trajectories) | 10 |
| L (path length) | 24 |
| MTP cls weight α | 1.0 |
| Conv channels | 8, 16, 32, 96 |
| Head dims | 2304→512→1024→4096 |
| Activation | GELU |

## Cross-refs
- preprocessing/se_attention_block.md
- docs/hengck23-reference/ — full SDF reference implementation