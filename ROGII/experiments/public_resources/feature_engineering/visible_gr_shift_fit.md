# Feature: Visible GR shift fit (per-well)

**Source kernel**: sanidhyavijay24/9-946-rogii-geostat-softmax-ncc-hybrid

## What it does
Brute-force find the **best TVT shift** that aligns the known-prefix GR with the typewell GR:

```python
for shift in np.arange(-30.0, 30.1, 2.0):     # 31 candidates, 2 ft step
    candidate_tw_gr = interp(tw_tvt, tw_gr, known_tvt + shift)
    corr = corrcoef(known_gr, candidate_tw_gr)[0,1]
    track best corr
return {visible_gr_shift_ft, visible_gr_shift_corr, visible_gr_bias}
```

## Why it matters
Provides a **per-well global TVT bias** estimate from the known prefix alone — orthogonal to the per-row PF and beam signals. Three per-well constants:

- `visible_gr_shift_ft`: the best shift (−30..30 ft, step 2)
- `visible_gr_shift_corr`: how confident we are in the shift (Pearson corr)
- `visible_gr_bias`: mean(known_gr − tw_gr(known_tvt+shift)) at the best shift — captures GR-amplitude bias

This is essentially a sub-pixel detector of structural offset between the horizontal and the typewell.

## Score-relevant constants
| name | value |
|------|-------|
| Shift range | -30 to +30 ft |
| Shift step | 2 ft |
| Min known points | 50 |
| Min valid candidate points | 30 |

## Cross-refs
- feature_engineering/anchored_gr_offsets.md (uses fixed offsets per row, no fit)