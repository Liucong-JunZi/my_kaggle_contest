# Kernel: cdeotte/xgb-starter-cv-15 (LB ~15)

**Author**: cdeotte (Grandmaster, frequent Kaggle gold)
**Last run**: ~2026-04
**Total votes**: 162
**Files**: xgb-starter-cv-15.ipynb

## Architecture (one-paragraph)
Compact XGBoost regressor predicting **residual = target_TVT - last_known_TVT** (the canonical relative target). Per-well features mix slope baselines (`baseline_tvt_all_slope`, `baseline_tvt_recent_slope`), trajectory deltas from the prediction-start anchor (`md_from_ps`, `xy_dist_from_ps`, `xyz_dist_from_ps`), known-segment statistics (last/min/max/mean/std/range of TVT and GR), GR rolling mean+std at windows {11,51,151}, GR diff(1) and diff(10), and typewell GR lookups at `last_known_TVT` and at `baseline_tvt_*`. Train rows are only `TVT_input.isna() & TVT.notna()` (the eval segment of train wells). 5-fold GroupKFold on well_id.

## Key Techniques

### Feature Engineering
- `baseline_tvt = last_known_tvt` is the residual target — flat baseline scores ~15.9 RMSE.
- Slope baselines (`recent_slope` from last 200 known rows; `slope_z_recent` is dTVT/dZ)
- Distance features from prediction-start: `md_from_ps`, `x_from_ps`, `y_from_ps`, `z_from_ps`, `xy_dist_from_ps`, `xyz_dist_from_ps`
- `row_frac = row_index / (n_rows-1)` (positional fraction along well)
- GR rolling at {11, 51, 151}, diff(1), diff(10), `gr_minus_last_known`
- Typewell features: `tw_gr_at_last_known_tvt`, `tw_gr_at_baseline_tvt` (linear interp), `gr_minus_tw_baseline`

### Model & Hyperparams
- `XGBRegressor`: n_estimators=450, lr=0.035, max_depth=5, min_child_weight=20, subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, reg_alpha=0.05, tree_method=hist, max_bin=256, device=cuda

### Particle Filter / Beam Search
- None (pure tabular).

### Ensemble / Blending
- None (single model averaged across 5 folds for test).

### CV Methodology
- GroupKFold(n_splits=5) on well_id — strict; matches our R8 protocol.
- Target: `target_residual = target_tvt - baseline_tvt = target_tvt - last_known_tvt`
- Inference: `tvt = baseline_tvt + xgb.predict(X)`

## Anything novel vs LB-7.776 kernel?
Not novel — strict subset. But it’s the cleanest minimal residual-XGB baseline; useful as a reference point and a sanity benchmark for our LightGBM stack.

## Score-relevant constants extracted
| name | value | source line |
|------|-------|-------------|
| n_estimators | 450 | XGB_PARAMS |
| learning_rate | 0.035 | XGB_PARAMS |
| max_depth | 5 | XGB_PARAMS |
| min_child_weight | 20 | XGB_PARAMS |
| reg_lambda | 4.0 | XGB_PARAMS |
| reg_alpha | 0.05 | XGB_PARAMS |
| recent_window | 200 | recent = known.tail(min(200, len(known))) |
| GR rolling windows | 11, 51, 151 | for window in [11, 51, 151]: |
| GR diff lags | 1, 10 | gr_diff_1, gr_diff_10 |

Cross-refs:
- model_params/xgb_cdeotte.json
- feature_engineering/baseline_residual_target.md
- cv_methodology/groupkfold_well.md
