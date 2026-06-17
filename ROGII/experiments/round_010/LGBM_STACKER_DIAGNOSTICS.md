# Round 010 — Two-stage LGBM stacker diagnostics

**Request**: try a second-stage LGBM stacker on top of candidate OOF predictions.

## Protocol

Implemented `orchestrator/stacking_lgbm.py` with an honest 5-fold meta protocol:

- First-stage candidates are loaded from `results/candidates/*.parquet`.
- Meta fold `f`: train LGBM only on rows where `fold != f`, using first-stage OOF predictions as features.
- Predict rows where `fold == f`.
- Report honest OOF per-well RMSE.

This avoids direct row leakage in the second stage.

## Runs

| label | setup | OOF perwell | Notes |
|---|---:|---:|---|
| `lgbm_stack_regularized_w` | 43 cands + derived pred features, inverse-per-well row weights | **7.3609** | Bad. Per-well weights distort training objective for LGBM. |
| `lgbm_stack_linearish_w` | shallow/regularized, inverse-per-well row weights | **7.3369** | Still bad. |
| `lgbm_stack_linearish_flat` | shallow, no per-well weights, all 43 cands + derived features | **7.2501** | Better but still much worse than hill climb 7.0127. |
| `lgbm_stack_top8_linearish_flat_noextra` | top 8 candidates only, no derived features, no weights | **7.2110** | Best LGBM stacker tested, still worse by +0.1983 RMSE. |

Baselines:

| method | OOF perwell |
|---|---:|
| Hill climb v4/v5 | **7.0127** |
| NNLS unnormalized | **6.9792** (LB-risky: sum_w≈1.18 amplification) |
| NNLS normalized | **7.1278** |

## Conclusion

**Do not use LGBM stacker for submission.** Even the conservative top-8/no-extra/no-weight version is 7.2110, far worse than hill climb (7.0127) and NNLS normalized (7.1278).

Likely reason: the second-stage tree meta-model learns fold/well-mix-specific nonlinear corrections from OOF predictions, but those corrections do not generalize across held-out wells. The candidate-prediction feature space is small and highly correlated; linear constrained blends are more robust than tree splits here.

The only strong result remains NNLS without sum normalization (6.9792), but its gain is mostly global amplitude expansion (`sum_w≈1.18`) caused by pool shrinkage bias, not a robust nonlinear interaction.

## Files

- `orchestrator/stacking_lgbm.py` — implemented stacker script
- `results/hillclimb_runs/lgbm_stack_regularized_w.{json,parquet}`
- `results/hillclimb_runs/lgbm_stack_linearish_w.{json,parquet}`
- `results/hillclimb_runs/lgbm_stack_linearish_flat.{json,parquet}`
- `results/hillclimb_runs/lgbm_stack_top8_linearish_flat_noextra.{json,parquet}`

## Next useful direction

Do not spend more time tuning tree meta-models on prediction-only features.

The more promising local next step is **calibrated linear blending**:

1. Fit per-candidate, per-fold scalar calibration on fold-train rows.
2. Apply to fold-val OOF predictions.
3. Run hill climb/NNLS with sum_w=1 on calibrated predictions.

The T4 MLPs are also not useful in their current full-feature form; if revisited, train disjoint-feature MLPs (GR-only / PF-only / geometry-only) to create lower-correlation residual signals.
