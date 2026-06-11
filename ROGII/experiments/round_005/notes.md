# Round 5 — TVT-aware loss & MTP head

**Date**: 2026-06-12  (in progress)
**Branch**: `round-5-tvt-aware-loss`
**Baseline to beat**: cfg-img-medium R4 = **raw 15.84 / anchored 13.67 ft**

## Motivation (carry-over from R4)

R4 closed three death axes (R3 backbone, R3 channel, R4-B data scaling)
and locked R4-A's free 2.22-ft post-processing win. Two characterizations
of the remaining bottleneck:

1. **Train loss plateaus at ~0.20 from epoch 2 while val RMSE stops at 13.67.**
   Train objective (`masked_MSE(sdf_pred, sdf_target)`) is satisfied without
   the actual TVT metric improving — train and eval objectives are misaligned.

2. **R4-A bias analysis**: per-well bias has mean=-5.37 std=11.23. Anchor with
   α=0.75 (partial shrinkage) was best; α=1.0 hurt some wells. Confidence
   proxies were useless. ⇒ the model has **no calibration signal** to know
   when its prediction is off-shift.

Both point to needing TVT-space supervision in training.

## R5-A — Soft-argmin + Huber TVT loss (FAILED)

**Status**: dead end. Two independent sweeps (τ=0.10/β=0.05 with 5-ep warmup,
and τ=0.30/β=0.05 without warmup) both show the same pathology — TVT loss
disrupts SDF convergence to the point that the anchored RMSE is worse than the
R4 baseline.

### Loss-coupling pathology

Pilot (τ=0.10, β linearly 0→0.05 over 5 ep, then constant):

| ep | β | sdf_loss | tvt_loss | anc | note |
|----|---|---------:|---------:|-----:|------|
| 1 | 0.000 | 1.226 | 146.94 | 16.54 | warmup ep1, sdf comparable to R4 baseline 1.50 |
| 2 | 0.010 | 1.010 | 184.92 | **16.04** ⭐ | first sign of trouble — sdf already lower |
| 3 | 0.020 | 1.305 | 117.13 | 33.27 | sdf bounces up, anc explodes |
| 4-7 | 0.03-0.05 | 0.65-1.30 | 80-110 | 21-33 | sdf can't converge below ~0.6 |
| 8 | 0.050 | 0.761 | 86.16 | 14.63 | one lucky anchored hit but unstable |
| 9-14 | 0.050 | 0.76-2.4 | 86-137 | 17-45 | catastrophic anc oscillation |

R4 baseline for reference: sdf_loss reaches 0.20 by ep5 and stays; anc settles
at 13.67-14.3.

### Diagnosis

soft-argmin via softmax distributes gradient over **all T positions**, not just
the zero-crossing. With β > 0.01 this pressure rewrites the global SDF shape to
optimize TVT regression, not the true SDF surface. Result: argmin location
becomes mis-aligned because the SDF distribution flattens away from a clean
zero-crossing. R4 baseline's anchored RMSE explicitly assumes a clean SDF
surface for the parabolic subpixel + per-well bias estimate, so R5-A's
mis-shaping breaks both decoding stages.

**The two losses fight gradient direction at the SDF feature map level. Cannot
be solved by warmup or beta annealing alone.**

### Operational failures during R5-A

- Initial 4-way concurrent sweep created 7 phantom retry processes that
  overwrote each other's caches/checkpoints. Lessons:
  - Always confirm `ps aux | grep <script> | wc -l == 1` before claiming
    a single run is in flight.
  - Use TaskStop on agent IDs, not just pkill on PIDs — failed agents
    auto-retry.
- Updated WORK_PROBLEMS.md with "concurrent train sweep on MPS produces
  ckpt clobber + process zombification" entry.

### Verdict

- R5-A buried. Both pilots crashed in identical ways.
- Soft-argmin is structurally wrong here — too coupled to the dense SDF.
- R5-B (MTP head) is the next move: extra heads supervised by aux targets
  give TVT-related signal *without* a gradient that rewrites the SDF surface.

## R5-B (next) — MTP head

Architecture:
- Keep SDF head unchanged; add 2-3 auxiliary heads sharing the same backbone:
  - **`dip_head`**: predict local geological dip angle per column → forces
    backbone to encode geometry, not just GR matching
  - **`uncertainty_head`**: predict per-column σ of TVT estimate → gives R4-A
    anchor a real confidence signal (which we showed was missing)
  - **`layer_head`** (optional): predict layer boundary mask per row →
    typewell-side structural supervision

Loss:
```
L = MSE(sdf) + λ_d · MSE(dip) + λ_u · NLL(tvt | sigma) + λ_l · BCE(layer)
```
None of the aux losses touch the SDF gradient direction the way soft-argmin did
— they go to **separate output heads** sharing only the backbone features.

Setup cost: a few hours of architecture work + label generation (dip from
trajectory CSV is straightforward; layer mask harder).

## R5-C — Beam search / DP decode (low priority)

Deferred. R4-A's savgol smoothing showed spatial smoothness gains are < 0.05 ft.

## Files

- `experiments/round_005/notes.md` — this document
- `experiments/round_005/train_r5a.py` — R5-A training entry
- `results/round_005/r5a-*.json` — per-sweep metrics (when done)
