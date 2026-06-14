# Kernel: lightningv08/lb-7-776-rogii-ridge-sp (LB 7.776 — public top reference)

**Author**: lightningv08 (fork chain through aidensong123/rogii-sel15-rerun → ravaghi artifacts)
**Last run**: ~2026-05
**Total votes**: 86
**Files**: lb-7-776-rogii-ridge-sp.ipynb

## Architecture (one-paragraph)
A two-branch system blended at the very end. **Branch 1 (Ridge stack)** computes a rich per-row feature matrix from 8 base signals — PF-ANCC, PF-Z, 7 beam-search variants, 3 multi-scale NCC + score-weighted NCC ensemble, plane-KNN per-formation TVT, dense ANCC kNN imputation, slope baselines, GR rolls — then trains 3× LightGBM (large + 2× small-deep) and 2× CatBoost on the relative target (target = TVT − last_known_TVT), stacks via Ridge with positive weights, and applies a postprocess: blend with raw PF-ANCC, exponential ramp `(1 − exp(−md_since/τ))` with τ=85 ft, and α=1.0 scale, then Savitzky-Golay smoothing per well. **Branch 2 (Heuristic selector)** runs 128-seed likelihood-weighted PF ensembles at 4 scales {3,5,8,12} + 14-config beam ensemble, then a binned selector picks one of 6 PF/beam blend variants based on `n_eval_rows` and `z_span`. Final submission = `0.3 * Ridge stack + 0.7 * Heuristic selector`.

## Key Techniques

### Feature Engineering
- **Per-formation TVT** with **segment b_well** (early/mid/late thirds + WLS tail-upweighted): for each formation `F` in {ANCC,ASTNU,ASTNL,EGFDU,EGFDL,BUDA}, compute `b_full = median(ktvt + kz - F_kn)`, `b_late` (last 50), `b_early`, `b_mid`, `b_wls` (exp(0.02*i) weights, tail-heavy). 6 formations × 5 b-variants → 30 segment-bias features.
- **Plane-KNN per-formation imputer**: kNN (k=10) over per-well median (X,Y,F) values, fits a local plane via 3×3 weighted normal equations, predicts each formation surface at any (X,Y). Distance features `spatial_knn_dist`.
- **Dense ANCC kNN imputer**: subsamples 60 (X,Y,ANCC) points per well, builds cKDTree over all train wells, predicts dense ANCC + variance + nearest-neighbor distance.
- **GR rolling stats**: windows {5,21,51,101} mean+std; GR shift lags {1,5,15,30} forward+backward; gr_diff_1, gr_diff_2, gr_env (rolling max), gr_nrg (sqrt of rolling mean square).
- **Anchored GR offsets**: `tda{o}` = `hgr − tw_gr(last_known_tvt + o)` for o ∈ {-80,-40,-20,-10,-5,0,5,10,20,40,80}; same for offsets relative to beam_ref (`tdbc`), sc_ens (`tdsc`), and pf_use (`tdpf`). Each is 9-11 features.
- **Geometric tangents along trajectory**: dxdmd, dydmd, dzdmd from MD-finite-differences.
- **Signal aggregates**: across 11 base signals (PF + 7 beams + 3 NCC + sc_ens + ANCC formation + dense), compute `sig_std`, `sig_mean_d`.

### Model & Hyperparams
- **LightGBM #1 (big)**: num_leaves=255, lr=0.030, n_estimators=5000, reg_lambda=3.0, reg_alpha=0.05, subsample=0.8 (freq=1), colsample_bytree=0.8, min_child_samples=15, max_bin=255, device=gpu
- **LightGBM #2,#3 (small-deep, seed 0/29)**: num_leaves=64, lr=0.00934, n_estimators=10000, reg_lambda=95.75, reg_alpha=10.79, subsample=0.474, colsample_bytree=0.393, min_child_samples=40, min_child_weight=0.241
- **CatBoost ×2**: depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254, iterations=8000, lr=0.020 (and 0.030), od_wait=300 (early-stop), seeds 7/123, task_type=GPU
- **Ridge meta**: alpha=1.66, tol=5e-4, positive=True, fit_intercept=True

### Particle Filter / Beam Search
- **PF-ANCC**: 600 particles, ANCC_ALPHA=0.998, ANCC_RN=0.002, ANCC_PN=0.005, ANCC_IS=0.3 (init spread), ANCC_RP=0.1 (resample pos jitter), ANCC_RR=0.001 (resample rate jitter), RESAMP=0.5*N effective threshold. Likelihood: gaussian on (gr − tw_gr_grid_lookup)/gr_sigma.
- **PF-Z**: 600 particles, MOM=0.993, VN=0.005, PN=0.01, GR_WT=0.3 mix between raw GR likelihood and rolling-5 smoothed GR likelihood; uses regression of dTVT/dMD on dZ/dMD (slope β, intercept icpt) as motion prior.
- **Beam Search (Numba JIT, ±2 step)**: 7 configs `(BS, mc, es, r, tag)` = `[(10,20,144,2,cons), (10,8,64,2,loose), (8,35,220,1,vcons), (10,14,90,5,sm5), (20,4,36,3,vloose), (12,12,100,3,mid), (15,25,180,2,stiff)]`. Cost = squared GR diff / es + step penalty mc*|d|.
- **Branch-2 PF ensemble**: 128 seeds × 500 particles, likelihood-weighted at 4 scales `{3,5,8,12}` (softmax of log-likelihood / scale).
- **Branch-2 beam ensemble**: 14 configs (different mix from feature beams).
- **Selector** (binned by `n_eval > 4840` and `z_span` thresholds {136.73, 185.51}): one of {pf_scale_5_hold_0.2, pf_scale_3_hold_0.15, pf_scale_12_beam_0.2_hold_0.15, pf_scale_5_hold_0.15, pf_scale_5_beam_0.05_hold_0.05, pf_scale_12_beam_0.2_hold_0.05}; default `pf_scale_8_hold_0.2`. `hold` is the weight on `last_known_tvt`, `beam` is the weight on the beam ensemble.

### Ensemble / Blending
- **Ridge stack** of 3×LGB + 2×CB OOF preds (positive Ridge, alpha=1.66).
- **Postprocess on Ridge OOF**: `delta = (md*(1-w_pf) + pf_ancc*w_pf) * (1 - exp(-md_since/τ)) * α` with grid-searched α ∈ {0.98..1.02}, τ ∈ {35,50,65,85,105,130,170,220}, w_pf ∈ {0.03..0.16}; baseline τ=85, w_pf=0.09, α=1.0. Then SG-filter (window=17, polyorder=3) per well.
- **Final blend**: `0.3 * sub_1 (Ridge stack) + 0.7 * sub_2 (Selector)` — the heuristic dominates.

### CV Methodology
- GroupKFold(n_splits=5) on well_id, parallel via `koolbox.Trainer` wrapper.
- Target: relative `target = TVT - last_known_TVT`.
- Per-fold early-stopping (250 rounds for both LGB and CB).

## Anything novel vs LB-7.776 kernel?
This IS the LB-7.776 kernel — already covered in `docs/lb-references/ANALYSIS.md`. Re-anchored here for reference.

## Score-relevant constants extracted
| name | value | source line |
|------|-------|-------------|
| Branch blend | 0.3 ridge + 0.7 heuristic | sub.assign |
| Ridge alpha | 1.6602834637650032 | ridge_params |
| PP τ | 85 | pp_params |
| PP w_pf | 0.09 | pp_params |
| PP α | 1.0 | pp_params |
| SG window/order | 17, 3 | sg_smooth |
| PF n_seeds | 128 | run_pf_lik_ensemble |
| PF n_particles | 500 | run_pf_lik_ensemble |
| Selector z_span thresholds | 136.73, 185.51 | SELECTOR_Z_SPAN_THRESHOLDS |
| Selector n_eval threshold | 4840 | SELECTOR_N_EVAL_THRESHOLD |
| Beam ±step | 2 | _beam_jit MOVES |
| Anchored offsets (GR) | -80,-40,-20,-10,-5,0,5,10,20,40,80 | ANCH_OFFS |

Cross-refs:
- feature_engineering/seg_b_well.md
- feature_engineering/plane_knn_formation.md
- feature_engineering/dense_ancc_imputer.md
- feature_engineering/anchored_gr_offsets.md
- feature_engineering/multi_scale_ncc.md
- model_params/lightgbm_lb7776.json
- model_params/catboost_lb7776.json
- ensemble_weights/ridge_pp_smooth.md
- ensemble_weights/selector_binned.md
- preprocessing/savgol_smooth.md
- cv_methodology/groupkfold_well.md
