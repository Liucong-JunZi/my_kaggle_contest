"""
Candidate stub: Per-formation segment b_well + TVT signals

Source: lightningv08/lb-7-776-rogii-ridge-sp
Stage: STUB ONLY — parent agent should review before integrating.

Builds the per-formation TVT family (6 formations × 5 segment-bias variants
+ aggregate features). Requires `feat_formation_plane_knn.FormationPlaneKNN`
to provide the formation surface estimates at known and eval (X,Y) points.

The features are added to a per-well feature dict; the LGB stack consumes them
along with PF/beam/NCC/anchored-offset signals.
"""
import numpy as np

FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


def seg_b_well(ktvt, kz, form_col):
    """Segment b_well: returns (b_full, b_early, b_mid, b_late, b_wls).

    bv = ktvt + kz - form_col (the "bias" against the formation surface)
    - b_full   = median of all rows
    - b_early  = median of first third
    - b_mid    = median of middle third
    - b_late   = median of last 50 (or full if shorter)
    - b_wls    = exp(0.02*i) tail-upweighted weighted mean
    """
    bv = ktvt + kz - form_col
    n = len(bv)
    b_full = float(np.median(bv))
    b_late = float(np.median(bv[max(0, n - 50):])) if n >= 5 else b_full
    t1, t2 = n // 3, 2 * n // 3
    b_early = float(np.median(bv[:max(1, t1)])) if t1 > 0 else b_full
    b_mid = float(np.median(bv[t1:max(t1 + 1, t2)])) if t2 > t1 else b_full
    w = np.exp(0.02 * np.arange(n))
    w /= w.sum()
    b_wls = float(np.dot(w, bv))
    return b_full, b_early, b_mid, b_late, b_wls


def add_formation_segment_features(formation_imputer, hw_df, last_known_tvt,
                                   self_wid_for_train=None):
    """Compute per-formation TVT signals + 5 segment biases.

    Args:
        formation_imputer: a FormationPlaneKNN instance, already fit.
        hw_df: horizontal well DataFrame.
        last_known_tvt: float anchor.
        self_wid_for_train: if training, pass the well id; the imputer will
                            mask self-well to avoid label leakage. **Even at training
                            time, force the imputer to run** — using real formation
                            cols at train and imputed at test causes a big train/test
                            distribution shift (medali1992 reports 6.32 → 0.27 ft gap
                            closure when this is unconditional).

    Returns:
        dict {feature_name: np.ndarray (eval_rows,)}
    """
    kn = hw_df[hw_df['TVT_input'].notna()]
    ev = hw_df[hw_df['TVT_input'].isna()]
    ktvt = kn['TVT_input'].to_numpy(np.float32)
    z_kn = kn['Z'].to_numpy(np.float32)
    z_ev = ev['Z'].to_numpy(np.float32)
    xy_kn = kn[['X', 'Y']].to_numpy(np.float64)
    xy_ev = ev[['X', 'Y']].to_numpy(np.float64)

    form_kn, _ = formation_imputer.impute(xy_kn, self_wid=self_wid_for_train)
    form_ev, knn_d = formation_imputer.impute(xy_ev, self_wid=self_wid_for_train)

    out = {}
    form_list = []
    form_rmse = {}
    for fi, fn in enumerate(FORMATIONS):
        b_full, b_early, b_mid, b_late, b_wls = seg_b_well(ktvt, z_kn, form_kn[:, fi])
        tvt_f = (-z_ev + form_ev[:, fi] + b_full).astype(np.float32)
        tvt_fw = (-z_ev + form_ev[:, fi] + b_wls).astype(np.float32)
        tvt_f50 = (-z_ev + form_ev[:, fi] + b_late).astype(np.float32)
        out[f'tvtF_{fn}_d'] = (tvt_f - last_known_tvt).astype(np.float32)
        out[f'tvtFw_{fn}_d'] = (tvt_fw - last_known_tvt).astype(np.float32)
        out[f'tvtF50_{fn}_d'] = (tvt_f50 - last_known_tvt).astype(np.float32)
        nh = len(ev)
        out[f'bw_{fn}'] = np.full(nh, b_full, np.float32)
        out[f'bww_{fn}'] = np.full(nh, b_wls, np.float32)
        out[f'bw50_{fn}'] = np.full(nh, b_late, np.float32)
        out[f'bw_early_{fn}'] = np.full(nh, b_early, np.float32)
        out[f'bw_mid_{fn}'] = np.full(nh, b_mid, np.float32)
        form_rmse[fn] = float(np.sqrt(np.mean((ktvt - (-z_kn + form_kn[:, fi] + b_full)) ** 2)))
        out[f'frm_rmse_{fn}'] = np.full(nh, form_rmse[fn], np.float32)
        form_list.append(tvt_f)

    fs = np.stack(form_list, 1)
    out['form_mean_d'] = (fs.mean(1) - last_known_tvt).astype(np.float32)
    out['form_std_d'] = fs.std(1).astype(np.float32)
    out['form_rng_d'] = (fs.max(1) - fs.min(1)).astype(np.float32)
    out['spatial_knn_dist'] = knn_d.astype(np.float32)

    return out
