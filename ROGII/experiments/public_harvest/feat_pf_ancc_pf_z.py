"""
Candidate stub: PF-ANCC + PF-Z (twin physical signal estimators)

Source: lightningv08/lb-7-776-rogii-ridge-sp; nihilisticneuralnet/9-251 DWT
Stage: STUB ONLY — parent agent should copy the actual numba JIT functions
verbatim from the source kernels.

The numba JIT functions (_pf_ancc, _pf_z, _resamp, _interp1) live in the
referenced kernels at:
    kernels_raw/lightningv08_lb-7-776-rogii-ridge-sp/lb-7-776-rogii-ridge-sp.code.txt
    lines ~398-552
"""
import numpy as np
import pandas as pd


# Hyperparameters (from LB7.776)
PF_N = 600
ANCC_N = 600
PF_MOM = 0.993
PF_VN = 0.005
PF_PN = 0.01
PF_GR_SIG_MIN = 10.0
PF_GR_SIG_MAX = 60.0
PF_GR_SIG_DEF = 30.0
PF_RESAMP = 0.5
PF_ROUGH_P = 0.2
PF_ROUGH_V = 0.003
PF_GR_WIN = 5
PF_GR_WT = 0.3
ANCC_ALPHA = 0.998
ANCC_RN = 0.002
ANCC_PN = 0.005
ANCC_IS = 0.3
ANCC_RP = 0.1
ANCC_RR = 0.001


def _gr_sig(hw_df, tw_tvt, tw_gr):
    """Compute the GR sigma scale used by the PF likelihood."""
    kn = hw_df[hw_df['TVT_input'].notna() & hw_df['GR'].notna()]
    if len(kn) < 20:
        return float(PF_GR_SIG_DEF)
    diff = kn['GR'].values - np.interp(kn['TVT_input'].values, tw_tvt, tw_gr)
    return float(np.clip(np.std(diff), PF_GR_SIG_MIN, PF_GR_SIG_MAX))


def _grid(tw_tvt, tw_gr, step=0.2):
    tmin = float(tw_tvt.min())
    tmax = float(tw_tvt.max())
    tvt_g = np.arange(tmin, tmax + step, step)
    return np.interp(tvt_g, tw_tvt, tw_gr).astype(np.float64), float(tmin), float(step)


def run_pf_ancc(hw_df, tw_tvt, tw_gr, N=ANCC_N):
    """Wrapper around _pf_ancc numba JIT.

    Returns (per_row_tvt: np.float32, per_row_std: np.float32) over the eval rows.
    """
    raise NotImplementedError(
        "Copy _pf_ancc numba function from "
        "kernels_raw/lightningv08_lb-7-776-rogii-ridge-sp/lb-7-776-rogii-ridge-sp.code.txt:398-498"
    )


def run_pf_z(hw_df, tw_tvt, tw_gr, N=PF_N):
    """Wrapper around _pf_z numba JIT."""
    raise NotImplementedError(
        "Copy _pf_z numba function from same source, lines 500-552"
    )


def add_pf_features(hw_df, tw_df, eval_idx, last_known_tvt):
    """Compute PF-ANCC and PF-Z features for one well's eval rows.

    Returns dict of feature arrays.
    """
    tw_s = tw_df.sort_values('TVT')
    tw_tvt = tw_s['TVT'].to_numpy(np.float32)
    tw_gr = tw_s['GR'].to_numpy(np.float32)

    pf_a, std_a = run_pf_ancc(hw_df, tw_tvt, tw_gr)
    pf_z, std_z = run_pf_z(hw_df, tw_tvt, tw_gr)

    has_z = (len(pf_z) == len(pf_a)) and (not np.any(np.isnan(pf_z)))
    pf_use = pf_a.astype(np.float32)

    out = {
        'pf_ancc':       pf_use,
        'pf_ancc_std':   std_a.astype(np.float32),
        'pf_ancc_delta': (pf_use - last_known_tvt).astype(np.float32),
    }
    if has_z:
        out['pf_z'] = pf_z.astype(np.float32)
        out['pf_z_delta'] = (pf_z - last_known_tvt).astype(np.float32)
        out['pf_vs_z'] = (pf_use - pf_z.astype(np.float32))
    else:
        nh = len(eval_idx)
        out['pf_z'] = np.full(nh, last_known_tvt, np.float32)
        out['pf_z_delta'] = np.zeros(nh, np.float32)
        out['pf_vs_z'] = np.zeros(nh, np.float32)
    return out
