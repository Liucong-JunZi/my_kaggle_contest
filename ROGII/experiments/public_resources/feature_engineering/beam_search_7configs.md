# Feature: 7-config beam search ensemble

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp

## What it does
Numba-JIT beam search over the typewell TVT axis. At each horizontal-well row, the beam tracks BS best (typewell-index, cost) candidates and explores ±2 step transitions. Cost = `(GR_diff)² / es + mc * |step|`.

7 hand-picked configs (tag, beam_size, motion_cost, edge_smoothing, smooth_radius):
| tag | bs | mc | es | r |
|-----|----|----|-----|---|
| cons | 10 | 20 | 144 | 2 |
| loose | 10 | 8 | 64 | 2 |
| vcons | 8 | 35 | 220 | 1 |
| sm5 | 10 | 14 | 90 | 5 |
| vloose | 20 | 4 | 36 | 3 |
| mid | 12 | 12 | 100 | 3 |
| stiff | 15 | 25 | 180 | 2 |

The smoothing radius `r` controls Savitzky-Golay-like presmoothing of the horizontal GR before beam tracking (window `2r+1`).

## Outputs as features
- 7× `beam_{tag}_d` (delta from last_known_tvt)
- `beam_mean_d`, `beam_std_d`, `beam_med_d` (aggregates)
- `beam_ref` = (beam_cons + beam_sm5) / 2 used as anchor for offset family `tdbc*`

## Variants seen
- LB7.776 heuristic branch uses **14 configs** (different list from feature pipeline).
- Numba JIT acceleration is mandatory; pure-Python is too slow (~7 sec/well × 7 configs × 723 wells).

## Cross-refs
- feature_engineering/anchored_gr_offsets.md (uses beam_ref as anchor)
- ensemble_weights/selector_binned.md (uses 14-config beam ensemble)