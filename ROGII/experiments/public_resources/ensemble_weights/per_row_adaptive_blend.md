# Ensemble: Per-row adaptive blend (h-blend)

**Source kernel**: nina2025/rogii-h-blend-v1 (LB ~164 votes)

## Concept
Instead of a fixed global weight per submission (like Ridge stacking), h-blend computes a **per-row dynamic weight**:

1. For each row, collect all submission predictions.
2. Sort them by value (descending or ascending, or even random).
3. Each submission gets a **base weight** (fixed) + a **rank-correction weight** (depends on its rank at that row).
4. The final blend at row `i` = Σ over submissions `j`: `weight[j] + rank_correction[rank_j]`.

## Segment-based adaptive weights
The row space can be segmented by `mx-m = max_subm - min_subm` (spread among submissions):
- If spread in [seg_low, seg_high], use weight block A
- If spread in [seg_mid_low, seg_mid_high], use weight block B
- Else use weight block C

This lets the blender behave differently on consensus rows (tight spread) vs. diverging rows (wide spread).

## Why it matters
- The per-rank correction captures the "reliability gradient" — when a submission disagrees with the majority, it gets downweighted.
- Segmenting by spread captures the uncertainty regime.
- This is more flexible than fixed Ridge blending and potentially captures the LB7.776 0.3/0.7 duality in a continuous way.

## Cross-refs
- ensemble_weights/ridge_pp_smooth.md (fixed Ridge blend)
- ensemble_weights/selector_binned.md (discrete per-bucket selector)