"""
Candidate stub: Guarded physical override for overlap wells

Source: pixiux/rogii-dual-pipeline-blend
Stage: STUB ONLY — parent agent should review before integrating.

A safe variant of the public LB7.776 0.3/0.7 physical override. Verifies
the train-derived TVT against the test well's known prefix BEFORE applying.

Apply as the LAST step of the pipeline, just before submission write.
"""
import numpy as np
import pandas as pd


def tvt_from_contacts(hw, tw, ref_col='EGFDU'):
    """Reconstruct TVT directly from formation contacts. Returns pd.Series of length len(hw)."""
    tw_g = tw.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw['TVT'] - (ref_tvt - (hw['Z'] - hw[ref_col]))).mean()
    return ref_tvt - (hw['Z'] - hw[ref_col]) + offset


def guarded_override_for_well(test_hw, train_hw, train_tw,
                              prefix_rmse_threshold=1.0,
                              min_comparable_rows=50,
                              ref_col='EGFDU'):
    """For one (test_well, train_well) pair, decide whether to apply the override.

    Returns: (override_tvt: np.ndarray or None, override_mask: np.ndarray or None)
        If verification passes, returns the train-derived TVT interpolated onto
        the test_hw MD axis, plus a boolean mask for which test rows fall in the
        train MD range.
        Otherwise returns (None, None).
    """
    try:
        train_phys_tvt = tvt_from_contacts(train_hw, train_tw, ref_col=ref_col)
    except Exception:
        return None, None

    train_md = train_hw['MD'].values
    test_md = test_hw['MD'].values

    # Interpolate train physical TVT onto test MD axis
    train_tvt_at_test_md = np.interp(test_md, train_md, train_phys_tvt.values,
                                      left=np.nan, right=np.nan)

    # Verification: prefix RMSE on known rows
    test_known = test_hw['TVT_input'].values
    valid = (~np.isnan(test_known)) & (~np.isnan(train_tvt_at_test_md))
    if valid.sum() < min_comparable_rows:
        return None, None

    rmse = np.sqrt(np.mean((test_known[valid] - train_tvt_at_test_md[valid]) ** 2))
    if rmse > prefix_rmse_threshold:
        return None, None

    # Override mask: only rows whose MD lies inside the train MD range
    in_range = (test_md >= train_md.min()) & (test_md <= train_md.max())
    return train_tvt_at_test_md, in_range


def apply_guarded_overrides(test_wells_data, sub_df, prefix_rmse_threshold=1.0,
                            min_comparable_rows=50, blend_weight=1.0):
    """Apply guarded physical override across all test wells.

    Args:
        test_wells_data: dict[wid] = {'test_hw': df, 'train_hw': df_or_None, 'train_tw': df_or_None}
        sub_df: submission DataFrame with 'id' and 'tvt' columns
        blend_weight: how much to apply the override; 1.0 = full override, <1.0 = blend

    Returns:
        Updated sub_df.
    """
    out = sub_df.copy()
    n_overrides = 0
    for wid, data in test_wells_data.items():
        if data['train_hw'] is None or data['train_tw'] is None:
            continue
        override_tvt, in_range = guarded_override_for_well(
            data['test_hw'], data['train_hw'], data['train_tw'],
            prefix_rmse_threshold=prefix_rmse_threshold,
            min_comparable_rows=min_comparable_rows,
        )
        if override_tvt is None:
            continue
        # Apply per-row override only on rows we're predicting (TVT_input is NaN) AND in range
        eval_mask = data['test_hw']['TVT_input'].isna().values
        ovr_mask = eval_mask & in_range & (~np.isnan(override_tvt))
        if not ovr_mask.any():
            continue
        for row_idx in np.where(ovr_mask)[0]:
            row_id = f'{wid}_{row_idx}'
            sel = out['id'] == row_id
            if sel.any():
                out.loc[sel, 'tvt'] = (
                    (1.0 - blend_weight) * out.loc[sel, 'tvt'].values
                    + blend_weight * override_tvt[row_idx]
                )
                n_overrides += 1
    print(f'Guarded override applied to {n_overrides} rows.')
    return out
