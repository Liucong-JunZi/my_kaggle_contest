"""
Candidate stub: DTW (Sakoe-Chiba constrained) signal feature

Source: nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based
Stage: STUB ONLY — parent agent should review before integrating.

Contract for round_010 hill-climb feature pool:
    add_dtw_features(hw_df, tw_df, last_known_tvt) -> dict[str, np.ndarray of len(eval_rows)]

The dict returns per-row DTW signals to be concatenated into the per-well
feature matrix for the LGB stack. All keys prefixed `dtw_` to namespace cleanly.

Numba-jit functions are mandatory; without them this is intractable on a 723-well
training corpus (radii up to 200, full sequence length).
"""
import numpy as np
import pandas as pd

DTW_RADII = (20, 50, 100, 200)
DTW_STOCH_K = 12
DTW_STOCH_TEMP = 3.0
DTW_OFFS = np.array([-20, -10, -5, -2, 0, 2, 5, 10, 20], dtype=np.float32)


# Numba functions extracted verbatim from
# kernels_raw/nihilisticneuralnet_9-251-rogii-wellbore-geology-prediction-dwt-based/
# 9-251-rogii-wellbore-geology-prediction-dwt-based.code.txt lines 137-291
# Copy them in full when integrating; we list signatures only:
#   _dtw_sakoe_chiba(query, ref, radius) -> (D, pi, pj)
#   _dtw_path_to_tvt(pi, pj, tw_tvt, N) -> tvt_pred (float32)
#   _dtw_path_slope(pi, pj, N, smooth_win=5) -> slope (float32)
#   _dtw_stochastic_realizations(query, ref, radius, K, temperature) -> paths (K,N)


def run_dtw_multiscale(query_gr, tw_tvt, tw_gr, last_tvt, radii=DTW_RADII):
    """Cost-weighted ensemble of multi-radius DTW alignments.

    Returns:
        dtw_tvts   : dict r -> (N,) tvt_pred
        dtw_slopes : dict r -> (N,) path slope
        dtw_costs  : dict r -> normalized DTW cost
        dtw_ens    : (N,) cost-weighted ensemble
    """
    raise NotImplementedError("Copy from nihilisticneuralnet kernel; numba required")


def run_dtw_stochastic(query_gr, tw_tvt, tw_gr, last_tvt,
                       radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP):
    """Stochastic DTW via Gumbel-noise traceback. Returns mean, std, cv arrays."""
    raise NotImplementedError("Copy from nihilisticneuralnet kernel; numba required")


def add_dtw_features(hw_df, tw_df, last_known_tvt):
    """Top-level entry: compute all DTW features for one well's eval rows.

    Args:
        hw_df: horizontal well DataFrame (full, including known prefix and eval rows)
        tw_df: typewell DataFrame, sorted by TVT
        last_known_tvt: float, the anchor TVT at the prediction-start

    Returns:
        dict {feature_name: np.ndarray of length len(eval_rows)}
    """
    full_gr = hw_df['GR'].astype(float).interpolate(limit_direction='both').values.astype(np.float32)
    tw_tvt = tw_df['TVT'].to_numpy(np.float32)
    tw_gr = tw_df['GR'].to_numpy(np.float32)

    dtw_tvts, dtw_slopes, dtw_costs, dtw_ens = run_dtw_multiscale(
        full_gr, tw_tvt, tw_gr, last_known_tvt, radii=DTW_RADII)
    dtw_mean, dtw_std, dtw_cv = run_dtw_stochastic(
        full_gr, tw_tvt, tw_gr, last_known_tvt,
        radius=50, K=DTW_STOCH_K, temperature=DTW_STOCH_TEMP)

    ev_mask = hw_df['TVT_input'].isna().values
    ev_idx = np.where(ev_mask)[0]
    nh = len(ev_idx)
    out = {}
    out['dtw_ens_d'] = (dtw_ens[ev_idx] - last_known_tvt).astype(np.float32)
    out['dtw_mean_d'] = (dtw_mean[ev_idx] - last_known_tvt).astype(np.float32)
    out['dtw_std'] = dtw_std[ev_idx].astype(np.float32)
    out['dtw_cv'] = dtw_cv[ev_idx].astype(np.float32)

    slopes = []
    for r in DTW_RADII:
        out[f'dtw_tvt_{r}_d'] = (dtw_tvts[r][ev_idx] - last_known_tvt).astype(np.float32)
        out[f'dtw_slope_{r}'] = dtw_slopes[r][ev_idx].astype(np.float32)
        slopes.append(dtw_slopes[r][ev_idx])
    out['dtw_slope_mean'] = np.stack(slopes, 1).mean(1).astype(np.float32)

    cost_arr = np.array([dtw_costs[r] for r in DTW_RADII], np.float32)
    out['dtw_cost_min'] = np.full(nh, cost_arr.min(), np.float32)
    out['dtw_cost_range'] = np.full(nh, cost_arr.max() - cost_arr.min(), np.float32)

    # Anchored offset family
    hgr_ev = full_gr[ev_idx]
    for o in DTW_OFFS:
        out[f'tdtw{int(o)}'] = (hgr_ev - np.interp(dtw_ens[ev_idx] + o, tw_tvt, tw_gr)).astype(np.float32)

    return out
