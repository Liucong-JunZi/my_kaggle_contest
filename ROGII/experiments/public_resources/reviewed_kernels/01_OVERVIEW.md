# Reviewed Kernels Overview — 2026-06-15

## Files Reviewed

| File | Author | LB | Votes | Type |
|------|--------|----|-------|------|
| lightningv08_lb-7-776-rogii-ridge-sp.md | lightningv08 (→ aidensong123 → ravaghi) | **7.776** | 86 | Ridge-stacked LGB+CB + heuristic blend |
| cdeotte_xgb-starter-cv-15.md | cdeotte | ~15 | 162 | Single XGB, pure tabular |
| medali1992_rogii-tcn-train-with-ddp-layernorm-se-atten.md | medali1992 | unknown (deep) | 78 | TCN + cross-attention transformer |
| nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based.md | nihilisticneuralnet | **9.251** | 593 | Ridge-stacked LGB+CB + DTW features |

## Priority Ranking (actionable value for us)

1. **lightningv08 (LB 7.776)** — THE kernel to learn from. Our biggest gap.
2. **nihilisticneuralnet (LB 9.251)** — DTW feature family we should integrate.
3. **medali1992 (TCN)** — Deep learning alternative track; typewell cross-attention is novel.
4. **cdeotte (XGB starter)** — Clean minimal baseline; we already exceed this.

## What We Already Have (Checklist)

| Component | Status in our R10 | Notes |
|-----------|-------------------|-------|
| Relative target (TVT - last_known_TVT) | ✅ | In features_full since R8 |
| PF-ANCC (600 particles) | ✅ | Single seed only |
| PF-Z (600 particles) | ✅ | Single seed only |
| 14-config Beam Search | ✅ | In beam_features |
| 16-seed PF likelihood ensemble | ✅ | 4 scales: 3/5/8/12 |
| GroupKFold-5 CV | ✅ | |
| Geometry offsets (z_rel, x_rel, etc.) | ✅ | |
| GR rolling mean+std (4 windows) | ✅ | |
| Formation residuals (z - formation) | ✅ | |
| Geometric tangents | ✅ | |

## What We're MISSING (Gap Analysis)

### CRITICAL — Easiest to add, big expected impact:

1. **Per-formation TVT segment biases (b_well)** — Stub exists in `public_harvest/` but NOT in data_loader
2. **Plane-KNN per-formation imputation** — Stub exists but NOT integrated
3. **Post-processing (exponential ramp + SG-smooth)** — Zero lines of code

### HIGH VALUE — Medium effort:

4. **Anchored GR offsets** (4 anchors × 11 offsets = 44 features) — replaces our single `gr_diff_from_last`
5. **Multi-scale NCC features** (windows 8/15/25) — typewell correlation signal
6. **Dense ANCC imputation via kNN** — spatial smoothing
7. **GR shift lags + diffs + env + nrg** — richer GR signal
8. **DTW Sakoe-Chiba features** (from nihilisticneuralnet) — 4 radii + stochastic

### TRANSFORMATIVE — Harder but bridges largest gap:

9. **128-seed PF likelihood-weighted ensemble** (we only do 16)
10. **Selector branch** (binned by n_eval_rows & z_span)
11. **Ridge meta-stack on 3×LGB + 2×CAT OOFs** (we use simple LGB+CAT blend)
12. **Heuristic blend 0.3 Ridge + 0.7 Selector** (the biggest single contributor to 7.776)

### NICHE:

13. **Typewell cross-attention (TCN kernel)** — deep learning avenue
14. **Physical model tvt_from_contacts** — for visible-overlap wells
15. **Optuna tuning of post-process params** (nihilisticneuralnet)
