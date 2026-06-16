"""128-seed log-likelihood-weighted Particle Filter for ROGII TVT prediction.

Pure numpy. No ML, no GPU. Lifted from
``experiments/round_009/r9_pf_only_submit_v2.py`` and packaged as an importable
module so the Kaggle entry-point (``ensemble_submit.py``) only owns I/O,
selector and the main loop.

Public API:
    run_pf_lik_ensemble_scales(hw, tw, n_particles=500, n_seeds=128)
        → dict[str, np.ndarray] keyed by ``pf_scale_<S>`` for S in PF_SCALES,
          plus ``pf_mean``. Each value is a per-row absolute-TVT vector aligned
          to ``hw`` (with the known prefix already filled in).

Constants ``PF_*`` mirror the v9 LB-tested config; do not edit without a
reason.
"""
from __future__ import annotations

import numpy as np

# PF params (from ajayrao43 v12 pure PF kernel, validated by r9 v2)
PF_N_PARTICLES = 500
PF_N_SEEDS = 128
PF_SCALES = (3.0, 5.0, 8.0, 12.0)
PF_INIT_SPREAD = 3.0  # wider init helps wells with abrupt TVT shift at PS

PF_MOM = 0.998
PF_VN = 0.002
PF_PN = 0.005
PF_RP = 0.1
PF_RR = 0.001
PF_RESAMP = 0.5


def run_particle_filter(hw, tw, n_particles=PF_N_PARTICLES, seed=42):
    """Single-seed PF. Returns (per-row TVT predictions, total log-likelihood)."""
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z = float(last["Z"])
    last_MD = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10.0, 60.0))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N = n_particles
    rng = np.random.default_rng(seed)
    ls = last_tvt + last_Z
    pos = ls + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w = np.ones(N) / N

    md_v = ev["MD"].values.astype(float)
    z_v = ev["Z"].values.astype(float)
    # GR interpolated before PF — critical for wells with high NaN fraction
    gr_interp = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean())
    gr_v = gr_interp.values.astype(float)[ev.index]

    out_vals = hw["TVT_input"].values.astype(float).copy()
    res = np.empty(len(ev))
    prev_MD = last_MD
    log_lik = 0.0

    for i in range(len(ev)):
        dm_step = max(md_v[i] - prev_MD, 1.0)
        rate = PF_MOM * rate + PF_VN * rng.standard_normal(N)
        pos = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.0))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < PF_RESAMP * N:
            cum = np.cumsum(w)
            u0 = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos = pos[idx] + PF_RP * rng.standard_normal(N)
            rate = rate[idx] + PF_RR * rng.standard_normal(N)
            w = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS):
    """128-seed log-likelihood-weighted PF ensemble at multiple temperatures.

    Returns a dict mapping scale-tag → per-row TVT vector. Keys:
      ``pf_scale_3`` / ``pf_scale_5`` / ``pf_scale_8`` / ``pf_scale_12`` and
      ``pf_mean`` (uniform-weighted mean for reference).
    """
    preds = []
    liks = []
    for s in range(n_seeds):
        p, ll = run_particle_filter(hw, tw, n_particles=n_particles, seed=s)
        preds.append(p)
        liks.append(ll)
    pred_arr = np.stack(preds, 0)
    liks = np.array(liks)
    liks_n = liks - liks.max()
    out = {}
    for scale in PF_SCALES:
        weights = np.exp(liks_n / float(scale))
        weights /= weights.sum()
        out[f"pf_scale_{scale:g}"] = (weights[:, None] * pred_arr).sum(0)
    out["pf_mean"] = pred_arr.mean(0)
    return out
