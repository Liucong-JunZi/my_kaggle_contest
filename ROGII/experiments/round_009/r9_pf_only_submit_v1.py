"""ROGII Kaggle submission — Pure PF (no ML).

Direct reproduction of ajayrao43/msd0110 LB-~8 pure PF kernel:
  128-seed log-likelihood-weighted PF ensemble (scale=5)
  init_spread=3.0 ft (wider initial particle spread)
  GR interpolated before PF (fills NaN gaps so PF always has observations)
  14-beam search ensemble
  Per-well selector (adaptive scale/beam/hold based on eval count & Z span)
  4th-degree polynomial robust projection

No ML model — pure particle filter physics.
Test wall: ~5-7h on Kaggle CPU for 100 hidden test wells.
"""

import os, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_DIR = None  # auto-detected below
TRAIN_DIR = None
TEST_DIR  = None
OUT_PATH  = None

# PF params (from ajayrao43 v12 pure PF kernel)
PF_N_PARTICLES = 500
PF_N_SEEDS     = 128
PF_SCALES      = (3.0, 5.0, 8.0, 12.0)
PF_INIT_SPREAD = 3.0   # wider init spread helps wells with abrupt TVT shift at PS

PF_MOM = 0.998
PF_VN  = 0.002
PF_PN  = 0.005
PF_RP  = 0.1
PF_RR  = 0.001
PF_RESAMP = 0.5

# Selector config (from Yaroslav/quanzhongji kernel — tuned on CV)
SELECTOR_N_EVAL_THRESHOLD  = 4840.0
SELECTOR_Z_SPAN_THRESHOLDS = (136.73, 185.5133333333342)
SELECTOR_BIN_VARIANTS = {
    0: "pf_scale_5_hold_0.2",
    1: "pf_scale_3_hold_0.15",
    2: "pf_scale_12_beam_0.2_hold_0.15",
    3: "pf_scale_5_hold_0.15",
    4: "pf_scale_5_beam_0.05_hold_0.05",
    5: "pf_scale_12_beam_0.2_hold_0.05",
}
SELECTOR_GLOBAL_VARIANT = "pf_scale_8_hold_0.2"

# 14 beam configs: (beam_size, motion_cost, error_scale, savgol_r)
BEAM_CONFIGS = [
    (10, 20.0, 144.0, 2),
    (10,  8.0,  64.0, 2),
    ( 8, 35.0, 220.0, 1),
    (10, 14.0,  90.0, 5),
    (20,  4.0,  36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0,  80.0, 4),
    (25,  6.0,  50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30,  8.0,  70.0, 2),
    (10, 50.0, 400.0, 0),
]


# ── Helpers ───────────────────────────────────────────────────────────────────
def _find_kaggle_input(slug="rogii-wellbore-geology-prediction"):
    for p in [f"/kaggle/input/competitions/{slug}", f"/kaggle/input/{slug}"]:
        if os.path.isdir(p): return p
    return None

def load_well(wid, split="train"):
    base = TRAIN_DIR if split == "train" else TEST_DIR
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))
    return hw, tw

def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    """Physical model: project from known formation contacts."""
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# ── Particle Filter (pure numpy, no numba) ────────────────────────────────────
def run_particle_filter(hw, tw, n_particles=PF_N_PARTICLES, seed=42):
    """Conservative PF. Returns (predictions_array, total_log_likelihood)."""
    tw_s   = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr  = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy(), 0.0

    last     = kn.iloc[-1]
    last_tvt = float(last["TVT_input"])
    last_Z   = float(last["Z"])
    last_MD  = float(last["MD"])

    tw_at_k = np.interp(kn["TVT_input"].values, tw_tvt, tw_gr)
    gs = float(np.clip(np.nanstd(kn["GR"].fillna(0).values - tw_at_k), 10., 60.))

    tail = kn.tail(30)
    dt = np.diff(tail["TVT_input"].values)
    dz = np.diff(tail["Z"].values)
    dm = np.diff(tail["MD"].values)
    m  = dm > 0
    ir = float(np.median((dt + dz)[m] / dm[m])) if m.sum() >= 3 else 0.0

    N   = n_particles
    rng = np.random.default_rng(seed)
    ls   = last_tvt + last_Z
    pos  = ls + PF_INIT_SPREAD * rng.standard_normal(N)
    rate = ir + 0.01 * rng.standard_normal(N)
    w    = np.ones(N) / N

    md_v = ev["MD"].values.astype(float)
    z_v  = ev["Z"].values.astype(float)
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
        pos  = pos + rate * dm_step + PF_PN * rng.standard_normal(N)
        tvt_p = pos - z_v[i]
        tvt_p = np.clip(tvt_p, tw_tvt[0] - 100, tw_tvt[-1] + 100)
        pos   = tvt_p + z_v[i]

        eg = np.interp(tvt_p, tw_tvt, tw_gr)
        d  = (gr_v[i] - eg) / gs
        lk = np.exp(-0.5 * np.minimum(d**2, 600.))
        lk = np.maximum(lk, 1e-300)
        avg_lk = float((w * lk).sum())
        log_lik += np.log(max(avg_lk, 1e-300))
        w = w * lk
        ws = w.sum()
        w = w / ws if ws > 0 else np.ones(N) / N

        n_eff = 1.0 / (w**2).sum()
        if n_eff < PF_RESAMP * N:
            cum = np.cumsum(w)
            u0  = rng.uniform(0, 1.0 / N)
            idx = np.clip(np.searchsorted(cum, u0 + np.arange(N) / N), 0, N - 1)
            pos  = pos[idx]  + PF_RP * rng.standard_normal(N)
            rate = rate[idx] + PF_RR * rng.standard_normal(N)
            w    = np.ones(N) / N

        res[i] = float(np.dot(w, pos - z_v[i]))
        prev_MD = md_v[i]

    out_vals[list(ev.index)] = res
    return out_vals, log_lik


def run_pf_lik_ensemble_scales(hw, tw, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS):
    """128-seed log-likelihood-weighted PF ensemble, cached across selector scales."""
    preds = []
    liks  = []
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


# ── Beam Search ───────────────────────────────────────────────────────────────
def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Vectorized beam search for TVT tracking via GR matching."""
    n  = len(hgr)
    nt = len(tw_tvt)
    if n == 0:
        return np.array([last_tvt])

    if r > 0 and n > max(3, 2 * r + 1):
        win = min(2 * r + 1, n if n % 2 == 1 else n - 1)
        sgr = savgol_filter(hgr, win, min(2, win - 1))
    else:
        sgr = hgr.copy()

    si = int(np.argmin(np.abs(tw_tvt - last_tvt)))

    MOVES = np.array([-2, -1, 0, 1, 2], dtype=np.int64)
    MC    = mc * np.array([2., 1., 0., 1., 2.])

    bidx  = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci])**2 / es
        tot  = bcost[:bn, None] + gr_e + MC[None, :]
        tot  = np.where(valid, tot, np.inf)

        ni_f  = ni.flatten()
        tot_f = tot.flatten()
        vf    = valid.flatten()
        ni_f  = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s  = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u  = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        if kept > 0:
            top  = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
            top  = top[np.argsort(tot_u[top])]
            bidx[:kept]  = ni_u[top]
            bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1] if kept > 0 else si
            bcost[kept:] = np.inf
        bn = kept if kept > 0 else 1

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    """Average 14 beam-search configs."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s  = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr  = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr    = gr_all[ev.index]

    beam_results = [beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
                    for (bs, mc, es, r) in BEAM_CONFIGS]

    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw["TVT_input"].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


# ── Selector ──────────────────────────────────────────────────────────────────
def selector_well_code(hw, eval_mask):
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name):
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name, pf_by_scale, tvt_beam, last_known_tvt):
    scale, beam_weight, hold_weight = parse_selector_variant(name)
    base = pf_by_scale.get(f"pf_scale_{scale:g}")
    if base is None:
        base = pf_by_scale.get("pf_scale_8",
               pf_by_scale.get("pf_scale_5",
               list(pf_by_scale.values())[0]))
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


# ── Projection ────────────────────────────────────────────────────────────────
def robust_polyfit(x, y, deg=4):
    """Robust 4th-degree polynomial fit with IRLS."""
    n = len(x)
    if n < deg + 2:
        return np.polyfit(x, y, min(deg, n - 1))
    w = np.ones(n)
    for _ in range(5):
        p = np.polyfit(x, y, deg, w=w)
        r = np.abs(y - np.polyval(p, x))
        m = np.median(r)
        if m < 1e-10:
            break
        w = np.clip(1 / (1 + (r / (6 * m))**2), 0.01, 1.0)
    return p


def projection_by_well(proj_df, values):
    """Apply per-well 4th-degree robust polynomial projection."""
    out = values.copy().astype(float)
    for wid, grp in proj_df.groupby("well"):
        idx = grp.index.to_numpy()
        frac = grp["frac"].values.astype(float)
        frac_2d = np.column_stack([frac, frac**2, frac**3, frac**4])
        y = values[idx]
        if len(y) < 10:
            coeffs = robust_polyfit(frac, y, deg=min(4, len(y) - 1))
            out[idx] = np.polyval(coeffs, frac)
        else:
            A = np.column_stack([np.ones(len(frac)), frac_2d])
            coeffs, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
            out[idx] = A @ coeffs
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    global INPUT_DIR, TRAIN_DIR, TEST_DIR, OUT_PATH

    INPUT_DIR = _find_kaggle_input() or "/kaggle/input/rogii-wellbore-geology-prediction"
    TRAIN_DIR = f"{INPUT_DIR}/train"
    TEST_DIR  = f"{INPUT_DIR}/test"
    on_kaggle = _find_kaggle_input() is not None
    OUT_PATH  = "/kaggle/working/submission.csv" if on_kaggle else \
                "/Users/liucong/code/kaggle/ROGII/results/round_009/submission_pf_only.csv"

    print(f"INPUT_DIR = {INPUT_DIR}")
    assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR missing: {TRAIN_DIR}"
    assert os.path.isdir(TEST_DIR),  f"TEST_DIR  missing: {TEST_DIR}"

    t0 = time.time()

    # Get test wells
    hw_files = sorted(os.listdir(TEST_DIR))
    test_wids = sorted(set(f.split("__")[0] for f in hw_files if f.endswith("__horizontal_well.csv")))
    print(f"Test wells: {len(test_wids)}")

    # Get train wells (for physical model on visible wells)
    train_wids = set()
    if os.path.isdir(TRAIN_DIR):
        hw_files_tr = sorted(os.listdir(TRAIN_DIR))
        train_wids = set(f.split("__")[0] for f in hw_files_tr if f.endswith("__horizontal_well.csv"))
    print(f"Train wells available: {len(train_wids)}")

    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    sample["well"]    = sample["id"].str[:8]
    sample["row_idx"] = sample["id"].str[9:].astype(int)

    rows = []
    n_processed = 0
    n_pf_failed = 0
    n_phys     = 0

    for wid in test_wids:
        print(f"\n[{n_processed + 1}/{len(test_wids)}] Processing {wid}...", end=" ")
        hw_te, tw_te = load_well(wid, "test")

        eval_mask = hw_te["TVT_input"].isna().to_numpy()
        n_eval = int(eval_mask.sum())

        # Physical model for visible (training) wells
        tvt_phys = None
        if wid in train_wids:
            try:
                hw_tr, tw_tr = load_well(wid, "train")
                hw_te["TVT_input"] = hw_tr["TVT_input"].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
                n_phys += 1
                print("phys", end=" ")
            except Exception as e:
                print(f"phys_fail({e})", end=" ")
                tvt_phys = None

        # Selector heuristic
        selector_code, selector_variant, sel_n_eval, sel_z_span = \
            selector_well_code(hw_te, eval_mask)

        # 128-seed PF ensemble
        try:
            tw_ref = tw_tr if "tw_tr" in dir() and tw_tr is not None else tw_te
            pf_by_scale = run_pf_lik_ensemble_scales(hw_te, tw_ref,
                            n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS)
            tvt_pf = pf_by_scale.get("pf_scale_8", list(pf_by_scale.values())[0])
            print(f"pf128", end=" ")
        except Exception as e:
            print(f"pf_fail({e})", end=" ")
            n_pf_failed += 1
            last_known = hw_te["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf = hw_te["TVT_input"].fillna(last_val).values.astype(float)
            pf_by_scale = {f"pf_scale_{sc:g}": tvt_pf.copy() for sc in PF_SCALES}

        # Beam search ensemble
        try:
            tw_ref = tw_tr if "tw_tr" in dir() and tw_tr is not None else tw_te
            tvt_beam = run_beam_ensemble(hw_te, tw_ref)
            print("beam", end=" ")
        except Exception as e:
            print(f"beam_fail({e})", end=" ")
            tvt_beam = tvt_pf.copy()

        # Selector combination
        last_known = hw_te["TVT_input"].dropna()
        last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
        tvt_selector = apply_selector_variant(selector_variant, pf_by_scale, tvt_beam, last_known_tvt)
        print(f"sel={selector_code}:{selector_variant}", end=" ")

        ws = sample[sample["well"] == wid]
        for _, row in ws.iterrows():
            ridx = int(row["row_idx"])
            if tvt_phys is not None:
                tvt_val = float(tvt_phys.iloc[ridx])
            else:
                tvt_val = float(tvt_selector[ridx])
            rows.append({"id": row["id"], "tvt": tvt_val})

        n_processed += 1
        print(f"→ {len(ws)} rows")

    sub = pd.DataFrame(rows)
    sub.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done: {len(sub)} rows in {elapsed/60:.1f} min ({elapsed/3600:.2f} h)")
    print(f"Wells: {n_processed} total | {n_phys} physical | {n_pf_failed} PF failures")
    print(f"Output: {OUT_PATH}")
    print(f"Mean TVT: {sub['tvt'].mean():.3f}")
    print(sub.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
