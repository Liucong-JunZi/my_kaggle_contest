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

### Final results

| n_train | raw RMSE | anchored RMSE | best epoch | wall time |
|---------|---------:|--------------:|-----------:|----------:|
| **50 (cfg-img-medium re-run)** | **15.84** | **13.67** ⭐ | 10 | 77 s |
| 100 | 25.35 | (anchor breaks) | 18 | 204 s |
| 200 (fair) | 25.29 | 34.17 ❌ | 18 | 391 s |

### Findings

1. **Data scaling is dead** — adding wells made it ~10 ft worse, twice. Combined with R2's `cfg-img-medium-200` (different-val 21.77), this is the THIRD failure of the scaling axis.
2. **The extra wells are distributionally off from val**. The original `TRAIN_IDS` in `src/gen_images.py` was hand-curated to match val geology; sorted-alphabetical extras don't.
3. **Anchored 34.17 > raw 25.29 on 200-fair** is the smoking gun: the anchor logic computes a per-well bias from the known H_H=48 horizontal columns, which should always help. When it dramatically hurts, those known columns themselves carry mis-calibrated predictions for the new wells, which means the model's *known-segment* error is what's broken — not just future-segment extrapolation.
4. **Anchor-best ckpt selection added 0.6 ft on 50-well** — R4-A re-run with the new train.py (selecting best by anchored RMSE) reaches 13.67, vs the earlier 14.28 (which used the raw-best ckpt and then anchored post-hoc). Selecting on what you care about evaluating actually matters.

### Verdict

- Stop trying to add wells naively. To revisit the data axis later: cluster wells by GR / TVT / trajectory features and pick extras that match val distribution, or stratified random sampling with multiple seeds.
- Cache for `cfg-img-medium-100/` and `cfg-img-medium-200-fair/` discarded (regenerable in ~3 s if needed); metrics + logs archived under `results/round_004/`.
- **Effective baseline after R4**: cfg-img-medium with R4-A pipeline = **raw 15.84 / anchored 13.67** (re-trained ckpt).

## Round 4 → Round 5 next step

Three axes confirmed dead through R3-R4:
- Bigger backbone (R3 mit-b1)
- Extra input channel as derived signal (R3 gr_diff)
- More training data (R2 / R4-B ×2)

Remaining promising axes:
1. **R5-A**: TVT-aware loss (soft-argmin + Huber on TVT) — directly attacks the train/eval objective gap (the loss plateau at 0.20 while val RMSE doesn't drop)
2. **R5-B**: MTP head (dip / uncertainty / layer boundary multi-task) — gives model the calibration signal R4-A's per-well bias analysis showed is missing
3. **R5-C**: Beam search / DP path decoding — exploits spatial smoothness across H columns

R5-A is cheapest (loss-only change); R5-B is best long-term ROI (hengck23's true source of LB gains).

## Files

- `experiments/round_004/notes.md` — this document
- `experiments/round_004/gen_fair_scaling.py` — fair-scaling dataset generator (R4-B)
- `experiments/round_004/gen_fair_scaling.log` — dataset generation log
- `results/round_004/decoding_ablation_v1.json` — R4-A1 first sweep
- `results/round_004/decoding_ablation_v2.json` — R4-A2 alpha + threshold + confidence sweep
- `results/round_004/cfg-img-medium-100.{json,log}` — R4-B 100-well metrics & training log
- `results/round_004/cfg-img-medium-200-fair.{json,log}` — R4-B 200-well metrics & training log
- `results/round_004/cfg-img-medium-r4repro.log` — re-train of 50-well baseline under current pipeline (13.67 ⭐)
- `data/cache/cfg-img-medium/{best_model.pth, metrics.json}` — UPDATED to the 13.67-ckpt
