# Kernel: mitchgansemer/gr-features-outlier-detection-rogii-wellbore (LB ~10, 144 votes)

**Author**: mitchgansemer
**Last run**: ~2026-05
**Total votes**: 144
**Files**: gr-features-outlier-detection-rogii-wellbore.ipynb

## Architecture (one-paragraph)
Tree-ensemble (XGBoost + CatBoost + HistGradientBoosting) on the relative target with novel **estimator-divergence features** (pairwise differences among formation, NCC, beam, PF, and extrapolation drifts) and **wavelet (db4) GR decomposition** (5-level DWT approximation, level-3 detail energy, residual). The author frames this explicitly as an outlier-detection task — most wells are easy, but a small tail of wells with large structural dip dominates the aggregate RMSE; the divergence features encode "this is a hard well" as a signal.

## Key Techniques

### Feature Engineering
- All LB7.776 base features (PF-ANCC, beam search, formation KNN, multi-scale NCC, anchored GR offsets, slopes)
- **NEW**: 5 estimator drifts (form, ncc, beam, pf, extrap)
- **NEW**: 6 pairwise divergences (form_vs_ncc, ..., beam_vs_pf)
- **NEW**: 3 aggregate drifts (range, max, min)
- **NEW**: 2 extrap comparisons (k50_vs_200, form_vs_extrap)
- **NEW**: DWT GR features (`gr_dwt_approx5`, `gr_dwt_detail_energy`, `gr_dwt_residual`)
- Total: 21 v4 features added on top of base 100+ features

### Model & Hyperparams
- XGBoost + CatBoost + HistGradientBoosting (sklearn) ensemble
- (specific hyperparams require deeper inspection of the kernel)

### Particle Filter / Beam Search
- Same PF-ANCC + beam ensemble as LB7.776 baseline.

### Ensemble / Blending
- 3-model average (XGB + CB + HGB) — author notes HGB pulls outlier wells toward truth.

### CV Methodology
- GroupKFold on well_id with **leave-one-out at the well-builder level** (`loo=True` in `build_features`) — for each train well, when imputing formation surfaces, exclude that well from KNN. Mirrors the test-time imputer behavior.

## Anything novel vs LB-7.776 kernel?
**YES — three unique additions**:
1. **Estimator divergence pairwise features** (top-20 SHAP). 11 cheap features that capture "estimators disagree → hard well".
2. **Wavelet (db4, periodization) GR decomposition** with 5-level approximation + level-3 detail energy.
3. Explicit outlier-aware framing: the author reports per-well OOF distribution and confirms a small tail drives most error.

## Score-relevant constants
| name | value |
|------|-------|
| DWT wavelet | db4 |
| DWT mode | periodization |
| DWT n_levels | 5 |
| Detail level used | 3 |
| Rolling-RMS window | 16 |

## Cross-refs
- feature_engineering/estimator_divergence.md (NEW)
- feature_engineering/wavelet_dwt_gr.md (NEW)