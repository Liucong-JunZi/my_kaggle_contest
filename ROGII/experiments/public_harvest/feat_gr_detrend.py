"""
Candidate stub: GR linear-detrend residual

Source: romantamrazov/rogii-super-solution-lb-top-3
Stage: STUB ONLY — parent agent should review before integrating.

Removes the monotonic GR-vs-MD drift so local geological features dominate.
"""
import numpy as np


def gr_detrend_resid(gr_arr, md_arr):
    """Return GR - linear_fit(MD)."""
    gr_arr = np.asarray(gr_arr, dtype=float)
    md_arr = np.asarray(md_arr, dtype=float)
    m = np.isfinite(gr_arr) & np.isfinite(md_arr)
    if m.sum() < 5 or np.nanstd(md_arr[m]) < 1e-6:
        return np.zeros_like(gr_arr, dtype=np.float32)
    a, b = np.polyfit(md_arr[m], gr_arr[m], 1)
    return (gr_arr - (a * md_arr + b)).astype(np.float32)


def add_gr_detrend_features(hw_df, eval_idx):
    """Per-row detrended GR (well-global trend removed)."""
    gr = hw_df['GR'].values
    md = hw_df['MD'].values
    resid = gr_detrend_resid(gr, md)
    return {
        'gr_detrend':       resid[eval_idx].astype(np.float32),
        'gr_detrend_d1':    np.gradient(resid)[eval_idx].astype(np.float32),
    }
