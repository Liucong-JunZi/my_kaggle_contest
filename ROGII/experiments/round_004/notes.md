# Round 4 — Decoding Post-processing & Fair Data Scaling

**Date**: 2026-06-11 (in progress; R4-B running)

## Motivation

R3 showed neither bigger backbone (mit-b1 tied) nor extra channel (gr_diff +9.68 ft hurt) move the needle. The diagnosis:
- train_loss plateaus at 0.20 from epoch 2 → not a fitting problem
- val RMSE stuck at 15.89 → **train objective (SDF MSE) misaligned with eval (TVT RMSE)**

Two cheap angles to disambiguate before going to MTP head / loss changes:
- **R4-A** — Improve decoding (no retrain). If most of the 15.89 is decoding noise, fix it for free.
- **R4-B** — Fair data scaling. The 200-well existing run was on a DIFFERENT val set so its 21.77 is misleading; redo with the same 50-well val.

## R4-A — Decoding Ablation (no retraining)

Loaded existing `cfg-img-medium/best_model.pth`, ran one inference pass, then decoded the same SDF four ways.

### Findings

| Strategy | mean RMSE | median | Δ vs baseline |
|----------|-----------|--------|---------------|
| 1. argmin int (baseline)               | 15.89 | 11.02 | — |
| 2. + subpixel parabolic                | 15.88 | 11.01 | -0.01 |
| 3. + savgol smoothing (w∈[7..101])     | 15.88 | 10.99 | -0.01 |
| 4. + known-segment anchor α=1.0        | 14.40 | 10.35 | -1.49 |
| **5. + partial anchor α=0.75 (best)**  | **14.28** | 10.88 | **-1.61** ⭐ |

### What we learned

1. **Subpixel precision is NOT the bottleneck**. Sweep-window savgol also useless. Quantization at compression=24 ft/row contributes essentially 0 ft of error.
2. **Per-well bias IS the bottleneck**. Bias distribution across 50 val wells: mean=-5.37, std=11.23, |max|=37.39. The model can localize relative geology fine but drifts in absolute TVT.
3. **Partial > full anchor**. α=1.0 fixes badly-biased wells (25050f63: -34 ft) but **hurts** already-well-calibrated ones (91b301ce: +24 ft). α=0.75 balances both.
4. **Threshold-gated and confidence-weighted variants were inferior** to a flat shrinkage. The bias is a continuous nuisance, not a binary "needs fixing" signal.

### Operational change

- New module `src/decode.py`: `decode_sdf_to_tvt()` + `anchor_known_segment(alpha=0.75)`.
- `src/train.py` updated: report both `raw` and `anchored` RMSE per epoch, select best checkpoint by **anchored** RMSE.

### Effective new baseline

| Config | raw RMSE | anchored RMSE |
|--------|---------|---------------|
| cfg-img-medium | 15.89 | **14.28** |

## R4-B — Fair Data Scaling (50 → 100 → 200 wells)

Setup:
- Identical 50-well val set (same well IDs as cfg-img-medium baseline).
- Train pools: 50 (original) / 100 / 200 wells, drawn from the 773-well training corpus, none overlapping val.
- Same geometry (T=192, H=576, comp=24, 3 channels), mit-b0, 20 epochs.
- Generator: `experiments/round_004/gen_fair_scaling.py` (calls `process_wells` + `save_h5` from `src/gen_images.py`).

### Status

| Config | n_train | best RMSE (raw) | epochs run | status |
|--------|---------|-----------------|-----------|--------|
| cfg-img-medium (baseline) | 50 | 15.89 (anchored 14.28) | 15 | reference |
| cfg-img-medium-100 | 100 | 25.49 @ ep4 | 14+ | converged at ~25 — **worse than 50-well** |
| cfg-img-medium-200-fair | 200 | running | — | pending |

### Preliminary signal (pending 200-well finish)

100-well training plateaus at ~25.5 ft — **worse than 50-well 15.89**. This is the second time data scaling hurt (R2 cfg-img-medium-200 was 21.77). Hypotheses:
- Pool wells beyond the original 50 may have systematically different characteristics (geology, length, GR distribution).
- Larger train set may need more epochs / different LR schedule to converge.
- The original 50 train wells were possibly hand-curated to match the val distribution.

If 200-well also lands ≥ 20 ft, the scaling axis is dead and we should pivot to **R5: TVT-aware loss (soft-argmin + Huber)** or **R6: MTP head**.

## Files

- `experiments/round_004/notes.md` — this document
- `experiments/round_004/gen_fair_scaling.py` — fair-scaling dataset generator (R4-B)
- `results/round_004/decoding_ablation_v1.json` — R4-A1 first sweep
- `results/round_004/decoding_ablation_v2.json` — R4-A2 alpha + threshold + confidence sweep
- `results/round_004/cfg-img-medium-100.json` — R4-B 100-well metrics (when done)
- `results/round_004/cfg-img-medium-200-fair.json` — R4-B 200-well metrics (when done)
