# Kernel: cdeotte/xgb-starter-cv-15 (LB ~15)

**Author**: cdeotte (Kaggle Grandmaster) | **Votes**: 162
**File**: xgb-starter-cv-15.ipynb

## Architecture

Clean minimal XGBoost baseline. Pure tabular — no PF, no beam search, no NCC.

```
Features (17):
  - Slope baselines: baseline_tvt_all_slope, baseline_tvt_recent_slope, slope_z_recent
  - Distance from prediction-start: md_from_ps, x_from_ps, y_from_ps, z_from_ps,
    xy_dist_from_ps, xyz_dist_from_ps
  - Known segment stats: last/min/max/mean/std/range of TVT and GR
  - Row position: row_frac = row_index / (n_rows-1)
  - GR rolling: mean+std at {11, 51, 151}, diff(1), diff(10), gr_minus_last_known
  - Typewell lookups: tw_gr_at_last_known_tvt, tw_gr_at_baseline_tvt, gr_minus_tw_baseline
```

## Hyperparams
```python
XGBRegressor(
    n_estimators=450, lr=0.035, max_depth=5, min_child_weight=20,
    subsample=0.85, colsample_bytree=0.85, reg_lambda=4.0, reg_alpha=0.05,
    tree_method='hist', max_bin=256, device='cuda'
)
```

## What's Interesting

### 1. Slope baselines
`baseline_tvt_recent_slope` — linear fit to last 200 known rows of TVT vs MD.
`slope_z_recent` — same but for TVT vs Z.
These are simpler but more interpretable versions of our geometry offsets.

### 2. Distance from prediction-start
Instead of our `md_offset`, they use `md_from_ps`, `xy_dist_from_ps`, `xyz_dist_from_ps` (Euclidean distance in 2D/3D). The 3D distance might capture well curvature that linear MD offset misses.

### 3. Row fraction
`row_frac = row_index / (n_rows-1)` — simple positional encoding.

### 4. Typewell lookups at baseline
`tw_gr_at_baseline_tvt` — interpolate typewell GR at the slope-baseline TVT position. This is similar to our anchored offsets but uses a learned slope baseline instead of last_known_TVT.

## What It's Missing (vs our R10)

- No PF/beam features (we have these)
- No formation residuals (we have these)
- No multi-scale NCC (we don't have this either)
- Single model, no ensemble (we have hill climb)
- No post-processing

## Takeaways

This kernel confirms:
- **Slope baselines** provide a useful signal that's different from pure last_known_TVT offsets
- **Typewell lookups** are a cheap way to add geological context
- Even a simple XGB hits ~15 LB with minimal features — our 9.182 OOF (~11 LB) is already a major improvement

**Potential quick addition**: Add typewell GR lookup features to our data_loader. We already have the typewell data, just need to interpolate.
