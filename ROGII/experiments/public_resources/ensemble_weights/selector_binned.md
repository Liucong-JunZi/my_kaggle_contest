# Ensemble: Selector — per-well binned PF/beam selector

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp (heuristic branch)

## Concept
For each test well, classify into 1 of 6 "buckets" based on simple structural properties, then apply that bucket's fixed PF+beam blend.

## Bucket coordinates
- `n_eval = number of rows where TVT_input is NaN` (i.e., the eval segment length)
- `z_span = max(Z_eval) - min(Z_eval)` (how steep the lateral)

```
n_bin = int(n_eval > 4840)      # 0/1
z_bin = searchsorted([136.73, 185.51], z_span, 'right')  # 0/1/2
code  = n_bin + 2 * z_bin       # 0..5
```

## Bucket → variant mapping
```
0: 'pf_scale_5_hold_0.2'
1: 'pf_scale_3_hold_0.15'
2: 'pf_scale_12_beam_0.2_hold_0.15'
3: 'pf_scale_5_hold_0.15'
4: 'pf_scale_5_beam_0.05_hold_0.05'
5: 'pf_scale_12_beam_0.2_hold_0.05'
```

`SELECTOR_GLOBAL_VARIANT = 'pf_scale_8_hold_0.2'` is the default fallback.

## Variant grammar
A variant name `pf_scale_S` (or `pf_scale_S_beam_B_hold_H`) corresponds to:
```
base = pf_by_scale[f'pf_scale_{S}']                         # likelihood-weighted PF at scale S
pred = (1 - B) * base + B * tvt_beam_ensemble               # mix in beam search
pred = (1 - H) * pred + H * last_known_tvt                  # shrink toward last_known anchor
```

The `hold` parameter is the shrinkage factor toward the constant `last_known_tvt`. Larger hold → more conservative (closer to flat baseline).

## Available scales
PF likelihood softmax temperatures: `(3.0, 5.0, 8.0, 12.0)`. Smaller scale → sharper softmax (one PF realization dominates); larger → uniform average.

## Why it matters
This **is** the heuristic branch that gets the 0.7 weight in LB7.776's final blend. The bucketing implicitly models which physics regime applies (low n_eval = short forecast horizon, large z_span = steep lateral with strong dip).

## Score-relevant constants
| name | value |
|------|-------|
| n_eval threshold | 4840 |
| z_span thresholds | 136.73, 185.51 |
| PF scales | 3, 5, 8, 12 |
| PF n_seeds | 128 |
| PF n_particles | 500 |
| Beam configs | 14 (different from feature-pipeline's 7) |
| Final blend weight (w/ Ridge) | 0.7 (heuristic) + 0.3 (Ridge) |

## Cross-refs
- feature_engineering/anchored_gr_offsets.md (the Ridge pipeline)
- ensemble_weights/ridge_pp_smooth.md (the other branch)