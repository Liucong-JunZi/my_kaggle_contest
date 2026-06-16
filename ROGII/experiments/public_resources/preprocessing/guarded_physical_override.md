# Trick: Guarded physical override (verified at runtime)

**Source kernel**: pixiux/rogii-dual-pipeline-blend (LB ~9, 201 votes)

## Background
Some test wells have well_ids that match training wells. Their TVT can be reconstructed near-exactly using `tvt_from_contacts(hw_train, tw_train)`. **Naive application** of this trick caused some submitters silent regressions because:
- At submission-rerun time, hidden test copies may not be row-aligned or same-version as the train copies.
- A blind row-index lookup can inject large errors.

## The fix: runtime verification before override
Per-well, the override **must earn the right to fire**:

1. Reconstruct TVT from the **train** copy's formation contacts.
2. Compare against the **test** copy's known prefix (`TVT_input`), interpolated **by MD, not by row index**.
3. Override **only if**:
   - Prefix RMSE < 1.0 ft
   - At least 50 comparable rows
   - And only override rows whose MD lies inside the train copy's MD range.

## Pseudocode
```python
def guarded_override(test_well, train_well_phys, test_known_prefix, threshold_rmse=1.0, min_rows=50):
    # Interpolate the train physical TVT prediction onto test MD
    train_md = train_well_phys.MD.values
    train_tvt = train_well_phys.TVT.values
    test_md = test_well.MD.values
    train_tvt_at_test_md = np.interp(test_md, train_md, train_tvt,
                                      left=np.nan, right=np.nan)
    # Compare against test known prefix
    valid = test_known_prefix.notna() & np.isfinite(train_tvt_at_test_md[:len(test_known_prefix)])
    if valid.sum() < min_rows:
        return None  # Not enough overlap — skip override
    rmse = np.sqrt(np.mean((test_known_prefix[valid] - train_tvt_at_test_md[:len(test_known_prefix)][valid])**2))
    if rmse > threshold_rmse:
        return None  # Doesn't agree on known prefix — skip override
    # Override only rows in train MD range
    in_range = (test_md >= train_md.min()) & (test_md <= train_md.max())
    return train_tvt_at_test_md, in_range
```

## Why this matters
- The LB7.776 kernel applies physical override blindly with `0.3 * tabular + 0.7 * physics` — at risk of regression on misaligned wells.
- pixiux's guarded version: by construction, the final submission is **never worse** than the plain blend on misaligned wells; gets the full benefit on truly overlapping wells.

## Score-relevant constants
| name | value |
|------|-------|
| Prefix RMSE threshold | 1.0 ft |
| Min comparable rows | 50 |
| MD interpolation | linear, nan outside range |

## Cross-refs
- preprocessing/tvt_from_contacts_physical.md (the underlying physical model)
- preprocessing/exact_coord_train_overlap_blend.md (per-row coord variant, also at risk)