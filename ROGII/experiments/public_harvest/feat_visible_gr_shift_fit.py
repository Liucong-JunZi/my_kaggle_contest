"""
Candidate stub: Visible GR shift fit feature

Source: sanidhyavijay24/9-946-rogii-geostat-softmax-ncc-hybrid
Stage: STUB ONLY — parent agent should review before integrating.

Returns 3 per-well scalar features broadcasted to eval-row count.
"""
import numpy as np


def visible_gr_shift_fit(known_tvt, known_gr, tw_tvt, tw_gr,
                         shift_range=(-30.0, 30.1, 2.0)):
    """Find the best TVT shift to align known GR with typewell GR.

    Args:
        known_tvt: (N_known,) TVT_input values for the known prefix
        known_gr:  (N_known,) GR values for the known prefix (NaNs OK)
        tw_tvt:    (M,) typewell TVT (sorted)
        tw_gr:     (M,) typewell GR

    Returns:
        dict with 'visible_gr_shift_ft', 'visible_gr_shift_corr', 'visible_gr_bias'
    """
    out = {'visible_gr_shift_ft': 0.0,
           'visible_gr_shift_corr': 0.0,
           'visible_gr_bias': 0.0}

    known_tvt = np.asarray(known_tvt, dtype=float)
    known_gr  = np.asarray(known_gr, dtype=float)
    if len(known_tvt) < 50 or len(tw_tvt) < 10:
        return out

    best_corr = -np.inf
    best_shift = 0.0
    for shift in np.arange(*shift_range):
        cand = np.interp(known_tvt + shift, tw_tvt, tw_gr,
                         left=np.nan, right=np.nan)
        m = np.isfinite(cand) & np.isfinite(known_gr)
        if m.sum() < 30 or np.nanstd(cand[m]) <= 1e-6 or np.nanstd(known_gr[m]) <= 1e-6:
            continue
        corr = float(np.corrcoef(known_gr[m], cand[m])[0, 1])
        if np.isfinite(corr) and corr > best_corr:
            best_corr = corr
            best_shift = float(shift)

    if not np.isfinite(best_corr):
        return out

    cand = np.interp(known_tvt + best_shift, tw_tvt, tw_gr,
                     left=np.nan, right=np.nan)
    m = np.isfinite(cand) & np.isfinite(known_gr)
    out['visible_gr_shift_ft'] = best_shift
    out['visible_gr_shift_corr'] = best_corr
    out['visible_gr_bias'] = float(np.nanmean(known_gr[m] - cand[m])) if m.any() else 0.0
    return out


def add_visible_gr_shift_features(hw_df, tw_df, n_eval_rows):
    """Top-level entry: compute and broadcast to per-row features.

    Returns dict {feature_name: np.ndarray (n_eval_rows,)}
    """
    kn = hw_df[hw_df['TVT_input'].notna()]
    if len(kn) < 50:
        return {
            'visible_gr_shift_ft':   np.zeros(n_eval_rows, np.float32),
            'visible_gr_shift_corr': np.zeros(n_eval_rows, np.float32),
            'visible_gr_bias':       np.zeros(n_eval_rows, np.float32),
        }
    tw_s = tw_df.sort_values('TVT')
    s = visible_gr_shift_fit(
        kn['TVT_input'].values,
        kn['GR'].values,
        tw_s['TVT'].values.astype(float),
        tw_s['GR'].values.astype(float),
    )
    return {k: np.full(n_eval_rows, v, np.float32) for k, v in s.items()}
