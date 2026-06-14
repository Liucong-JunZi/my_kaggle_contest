# Feature: Pairwise estimator divergence (v4 features)

**Source kernel**: mitchgansemer/gr-features-outlier-detection-rogii-wellbore (LB ~10, 144 votes)

## What it does
Encodes the **disagreement among per-row TVT estimators** as a feature family. Six pairwise differences and three aggregates over five estimators:

```python
form_drift   = tvt_form_full - last_anchor_tvt    # plane-KNN formation TVT
ncc_drift    = GR_ncc_delta                        # multi-scale NCC TVT
beam_drift   = beam_med2_delta                     # beam search median TVT
pf_drift     = pf_ancc_delta                       # PF-ANCC TVT
extrap_drift = tvt_extrap - last_anchor_tvt        # linear extrapolation

# Pairwise (6 features)
form_vs_ncc, form_vs_pf, form_vs_beam, ncc_vs_pf, ncc_vs_beam, beam_vs_pf

# Aggregates over all 5 (3 features)
estimator_drift_range = drifts.max - drifts.min
estimator_drift_max
estimator_drift_min

# Extrapolation comparisons (2 features)
extrap_k50_vs_extrap200, form_vs_extrap
```

## Why it matters
- When all estimators agree, the well is "easy" — model can trust its prediction.
- When estimators disagree (large `estimator_drift_range`), the well is in a hard interval — the model can learn to widen its prediction (output the mean) or rely on a particular estimator family.
- The author reports these rank in the **top 20 SHAP features** despite each being a pure "metadata" comparison.

## Why this works analytically
Each estimator has different failure modes:
- Formation KNN fails when the well is far from neighbors (distance) — the `spatial_knn_dist` already encodes this.
- NCC fails on monotonic GR (no template features to match).
- Beam fails on long laterals (drift accumulates).
- PF fails when the GR is too noisy to give likelihood signal.

A high `estimator_drift_range` is a robust indicator of "this is one of the hard outlier wells" — the LGB can learn to use that to switch behavior.

## Cross-refs
- feature_engineering/anchored_gr_offsets.md (same families enter via offset features)
- kernels/mitchgansemer_gr-features-outlier-detection-rogii-wellbore.md