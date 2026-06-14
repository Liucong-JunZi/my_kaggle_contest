# Ensemble: Ridge stack + post-processing + Savitzky-Golay smoothing

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp (Ridge branch, the 0.3-weighted half)

## Pipeline (in order)

### 1. Ridge stack of 5 OOF predictions
Models: 3× LightGBM + 2× CatBoost (all trained on the relative target `target = TVT - last_known_TVT`).
```python
ridge = Ridge(alpha=1.6602834637650032, tol=5e-4, positive=True, fit_intercept=True)
ridge.fit(stack_oof_predictions, y_residual)
ridge_pred = ridge.predict(stack_test_predictions)
```
- `positive=True` → all stack weights non-negative (interpretable, no model "subtracts" another).
- alpha and tol tuned by Optuna in some forks.

### 2. PF blend + exponential ramp + α scale
```python
def apply_pp(df, md, pd_, alpha, tau, w_pf):
    d = md * (1 - w_pf) + pd_ * w_pf            # blend Ridge model output with raw PF-ANCC residual
    if tau:
        d *= (1. - np.exp(-np.maximum(df['md_since'].values, 0.) / tau))   # ramp from 0 at MD=last to ~1 far out
    return d * alpha
```
- `w_pf = 0.09` (small but nonzero — PF as regularizer)
- `tau = 85` ft (the ramp half-life)
- `alpha = 1.0` (no global scaling in baseline; Optuna grids 0.98..1.02)

The ramp `(1 − exp(−md_since/τ))` is **physical**: at the first eval row (md_since=0), `d=0`, so `tvt_pred = last_known_tvt` exactly. Far from the anchor it asymptotes to the full Ridge+PF estimate. Small τ → quick ramp; large τ → most rows get a strongly damped delta.

### 3. Per-well Savitzky-Golay smoothing
```python
def sg_smooth(df, col, sg_w=17, sg_p=3):
    for well, g in df.groupby('well'):
        v = g[col].values
        wl = min(sg_w, len(v))
        if wl % 2 == 0: wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df
```
- Window 17 rows, polynomial order 3.
- Damps high-frequency noise without over-flattening structure.

### 4. PP grid search (selects best (α, τ, w_pf))
```python
pp_grid = [{α, τ, w_pf} for α in [0.98..1.02], τ in [35,50,65,85,105,130,170,220], w_pf in [0.03..0.16]]
```
Optuna-tuned variant in DWT kernel includes `sg_w, sg_p` as searched dimensions.

## Score-relevant constants
| name | value | source |
|------|-------|--------|
| Ridge alpha | 1.66028 | ridge_params |
| Ridge positive | True | ridge_params |
| PP alpha | 1.0 | pp_params |
| PP tau | 85 | pp_params |
| PP w_pf | 0.09 | pp_params |
| SG window | 17 | sg_smooth default |
| SG polyorder | 3 | sg_smooth default |
| Final 2-branch blend | 0.3 ridge + 0.7 selector | sub.assign |

## Cross-refs
- model_params/lightgbm_lb7776.json
- model_params/catboost_lb7776.json
- ensemble_weights/selector_binned.md (the other 0.7-weight branch)
- preprocessing/savgol_smooth.md