# Preprocessing: GR interpolation before PF (and any GR-using model)

**Source kernel**: sunnywu27/rogii-wellbore-tvt-physical-model
**Also forum**: tid 707702 (msg from organizer/data team)

## What it does
Interpolate the horizontal well GR sequence before passing it to PF, beam search, NCC, DTW, or any other GR-driven matcher:

```python
gr_interp = hw['GR'].interpolate(limit_direction='both').fillna(tw_gr_mean)
```

## Why it matters
- The lateral GR is **~32% NaN** on average (per organizer in tid 707702). Specific wells can be much worse — `000d7d20` has 47% NaN GR in the prediction segment.
- A particle filter with NaN observations cannot update its weights → drift accumulates linearly.
- Per sunnywu27, "Local avg: **4.71 ft (vs 5.95 ft without GR interpolation)**" — a 1.24 ft RMSE drop just from this preprocessing change.

## Best practices
- Use `limit_direction='both'` to fill at both ends.
- Fall back to `tw_gr.mean()` (or `tw_gr.median()`) to avoid leaving any NaN.
- Apply to the FULL well GR (known + eval), not just the eval segment, to give rolling/diff features more support.
- Keep a `gr_missing` boolean mask as a separate feature so the model knows which rows had imputation.

## Score-relevant constants
| name | value |
|------|-------|
| Interpolation method | linear (default for `pd.Series.interpolate`) |
| Fallback fill value | `tw_gr.mean()` |
| Direction | both |

## Cross-refs
- feature_engineering/pf_ancc_pf_z.md (the immediate consumer)
- feature_engineering/anchored_gr_offsets.md
- feature_engineering/dtw_sakoe_chiba.md