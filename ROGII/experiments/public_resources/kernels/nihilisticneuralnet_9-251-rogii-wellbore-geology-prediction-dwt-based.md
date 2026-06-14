# Kernel: nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based (LB 9.251)

**Author**: nihilisticneuralnet
**Last run**: ~2026-05
**Total votes**: 593 (highest-voted public kernel)
**Files**: 9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb

## Architecture (one-paragraph)
Same backbone as LB7.776 (PF-ANCC, PF-Z, 7 beams, multi-scale NCC, plane-KNN per-formation, dense ANCC, GR rolls + lags + diffs, anchored GR offsets, slope baselines), but **adds a DTW alignment family**: 4-radius constrained Sakoe-Chiba DTW (radii 20/50/100/200) producing cost-weighted ensemble TVT + per-radius warp-path slopes, plus 12-realization stochastic DTW with Gumbel-noise traceback giving mean/std/cv per row. Adds an extra anchor offset family (`DTW_OFFS=[-20,-10,-5,-2,0,2,5,10,20]`) for the DTW-anchored GR-difference offsets. Trains the same ridge-stacked LGB+CB ensemble on the relative target and uses an Optuna-tuned postprocess + SG-smooth.

## Key Techniques

### Feature Engineering
- All LB7.776 features: PF-ANCC + PF-Z deltas + std + relative
- 7 beam-search variants → cons/loose/vcons/sm5/vloose/mid/stiff (same configs as LB7.776)
- Multi-scale NCC (hws 8/15/25) + score-weighted ensemble (softmax T=3)
- Plane-KNN per-formation (k=10) + dense ANCC kNN (60 spw, k=20)
- Segment b_well (full/early/mid/late/wls) — 5 segment variants × 6 formations
- GR rolls {5,21,51,101} mean+std + lag/lead {1,5,15,30} + diff/diff-diff + env + nrg
- Anchored GR offsets: 4 LB7.776 anchors + **NEW DTW anchor**

### NEW: DTW family (see feature_engineering/dtw_sakoe_chiba.md)
- `dtw_ens_d`: cost-weighted ensemble TVT − last_known_tvt
- `dtw_mean_d, dtw_std, dtw_cv`: stochastic realization aggregates
- per-radius warp-path slopes (4 radii)
- `dtw_cost_min, dtw_cost_range`
- `tdtw{o}` GR offsets (9 features)

### Model & Hyperparams
- Same as LB7.776 stack (3× LGB + 2× CB + Ridge meta + PP + SG)
- Optuna tunes: alpha, tau, w_pf, sg_w, sg_p (multivariate TPE, 1000 trials, 100 warmup)

### Particle Filter / Beam Search
- Identical to LB7.776 (PF-ANCC, PF-Z, 7 beams). No 14-beam selector branch — pure ridge stack output.

### Ensemble / Blending
- 3× LGB + 2× CB → Ridge stack (positive, alpha tuned by Optuna)
- PP: blend with `pf_ancc`, exp ramp, alpha scale
- SG smoothing per well (sg_w/sg_p tuned)

### CV Methodology
- GroupKFold(n_splits=5) on well_id
- Target: `target = TVT - last_known_TVT`

## Anything novel vs LB-7.776 kernel?
**YES** — adds DTW alignment family. Two key novelties:
1. Multi-scale Sakoe-Chiba constrained DTW (4 radii) → cost-weighted ensemble TVT + warp-path local slope features
2. Stochastic DTW with Gumbel-noise cost perturbation → per-row uncertainty quantification (std/cv)
3. Optuna-tuned PP including sg_w/sg_p (more search than fixed 17/3 in LB7.776)

The full kernel achieves **LB 9.251** — significantly worse than LB7.776 (which had the heuristic 0.3/0.7 blend). This suggests the ridge-only branch lands ~9-10 LB and the heuristic branch is what pushes to 7.776.

## Score-relevant constants extracted
| name | value | source |
|------|-------|--------|
| DTW_RADII | (20, 50, 100, 200) | DTW_RADII |
| DTW_STOCH_K | 12 | DTW_STOCH_K |
| DTW_STOCH_TEMP | 3.0 | DTW_STOCH_TEMP |
| DTW_OFFS | [-20,-10,-5,-2,0,2,5,10,20] | DTW_OFFS |

## Cross-refs
- feature_engineering/dtw_sakoe_chiba.md (NEW)
- ensemble_weights/ridge_pp_smooth.md (Optuna-tuned variant)
- model_params/lightgbm_lb7776.json (same params)