"""ROGII Kaggle submission entry-point — round 010.

Pure-PF variant of the round-010 hill-climb v3 ensemble.

The v3 averaged weights are::

    c20_r9_pf128_full   0.7785   ← THIS IS WHAT WE SHIP
    p14_cat             0.1089   (CatBoost — needs a Kaggle Dataset, not in this pkg)
    p14_lgb             0.0926   (LightGBM — needs a Kaggle Dataset, not in this pkg)
    p5_lgb              0.0131   (LightGBM — needs a Kaggle Dataset, not in this pkg)
    c06_xgb_default     0.0069   (XGBoost   — needs a Kaggle Dataset, not in this pkg)

The four ML candidates require trained model artefacts uploaded as a Kaggle
Dataset; that work is not yet done. The single PF candidate carries 77.85% of
the v3 blend and is the strongest single model in the hill-climb pool
(per-well RMSE 7.95 vs. the v3 blend 7.57). Shipping PF-only therefore costs
~0.4 RMSE expected vs. the full blend, in exchange for a self-contained
notebook with no external Dataset dependencies.

Layout assumed by Kaggle:
    /kaggle/input/rogii-wellbore-geology-prediction/
        ├── train/   ← visible train wells (we use these for the physical model)
        ├── test/    ← hidden test wells (replaced at scoring time)
        └── sample_submission.csv

Run wall (CPU-only, 100 hidden test wells, 128 seeds × 500 particles):
    ~5–7 hours.

I/O, selector, beam-search and the polish step are kept in this file; the PF
core lives in ``training_code/pf_128seed.py``.
"""
from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

# ── PF core (importable module, ships beside this file) ──────────────────────
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from training_code.pf_128seed import (  # noqa: E402
    PF_N_PARTICLES,
    PF_N_SEEDS,
    PF_SCALES,
    run_pf_lik_ensemble_scales,
)

# ── Selector config (Yaroslav/quanzhongji kernel — tuned on CV) ──────────────
SELECTOR_N_EVAL_THRESHOLD = 4840.0
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
    (10, 8.0, 64.0, 2),
    (8, 35.0, 220.0, 1),
    (10, 14.0, 90.0, 5),
    (20, 4.0, 36.0, 3),
    (12, 12.0, 100.0, 3),
    (15, 25.0, 180.0, 2),
    (20, 30.0, 200.0, 2),
    (15, 10.0, 80.0, 4),
    (25, 6.0, 50.0, 3),
    (10, 40.0, 300.0, 1),
    (12, 18.0, 120.0, 5),
    (30, 8.0, 70.0, 2),
    (10, 50.0, 400.0, 0),
]


# ── I/O helpers ──────────────────────────────────────────────────────────────
def _find_kaggle_input(slug="rogii-wellbore-geology-prediction"):
    for p in [f"/kaggle/input/competitions/{slug}", f"/kaggle/input/{slug}"]:
        if os.path.isdir(p):
            return p
    return None


def load_well(wid, base):
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))
    return hw, tw


def tvt_from_contacts(hw_tr, tw_tr, ref_col="EGFDU"):
    """Visible-well shortcut: project from known formation contacts."""
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# ── Beam search ──────────────────────────────────────────────────────────────
def beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs=10, mc=20.0, es=144.0, r=2):
    """Vectorized beam search for TVT tracking via GR matching."""
    n = len(hgr)
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
    MC = mc * np.array([2.0, 1.0, 0.0, 1.0, 2.0])

    bidx = np.full(bs, si, dtype=np.int64)
    bcost = np.full(bs, np.inf)
    bcost[0] = 0.0
    bn = 1

    result = np.zeros(n)

    for step in range(n):
        gv = sgr[step]
        ni = bidx[:bn, None] + MOVES[None, :]
        ci = np.clip(ni, 0, nt - 1)
        valid = (ni >= 0) & (ni < nt)

        gr_e = (gv - tw_gr[ci]) ** 2 / es
        tot = bcost[:bn, None] + gr_e + MC[None, :]
        tot = np.where(valid, tot, np.inf)

        ni_f = ni.flatten()
        tot_f = tot.flatten()
        vf = valid.flatten()
        ni_f = ni_f[vf]
        tot_f = tot_f[vf]

        order = np.argsort(tot_f)
        ni_s = ni_f[order]
        tot_s = tot_f[order]

        _, first = np.unique(ni_s, return_index=True)
        ni_u = ni_s[first]
        tot_u = tot_s[first]

        kept = min(bs, len(ni_u))
        if kept > 0:
            top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
            top = top[np.argsort(tot_u[top])]
            bidx[:kept] = ni_u[top]
            bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:] = bidx[kept - 1] if kept > 0 else si
            bcost[kept:] = np.inf
        bn = kept if kept > 0 else 1

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw, tw):
    """Average 14 beam-search configs over the eval rows."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(float)
    tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(float)

    gr_all = hw["GR"].interpolate(limit_direction="both").fillna(tw_gr.mean()).values.astype(float)
    hgr = gr_all[ev.index]

    beam_results = [
        beam_search(hgr, tw_tvt, tw_gr, last_tvt, bs, mc, es, r)
        for (bs, mc, es, r) in BEAM_CONFIGS
    ]
    beam_mean = np.stack(beam_results, 0).mean(0)

    out = hw["TVT_input"].values.astype(float).copy()
    out[list(ev.index)] = beam_mean
    return out


# ── Selector ─────────────────────────────────────────────────────────────────
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
        base = pf_by_scale.get(
            "pf_scale_8",
            pf_by_scale.get("pf_scale_5", list(pf_by_scale.values())[0]),
        )
    pred = (1.0 - beam_weight) * base + beam_weight * tvt_beam
    pred = (1.0 - hold_weight) * pred + hold_weight * last_known_tvt
    return pred


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    INPUT_DIR = _find_kaggle_input() or "/kaggle/input/rogii-wellbore-geology-prediction"
    TRAIN_DIR = f"{INPUT_DIR}/train"
    TEST_DIR = f"{INPUT_DIR}/test"
    on_kaggle = _find_kaggle_input() is not None
    OUT_PATH = (
        "/kaggle/working/submission.csv"
        if on_kaggle
        else "/Users/liucong/code/kaggle/ROGII/experiments/round_010/submission/submission_local.csv"
    )

    print(f"INPUT_DIR = {INPUT_DIR}")
    assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR missing: {TRAIN_DIR}"
    assert os.path.isdir(TEST_DIR), f"TEST_DIR  missing: {TEST_DIR}"

    t0 = time.time()

    hw_files = sorted(os.listdir(TEST_DIR))
    test_wids = sorted(
        set(f.split("__")[0] for f in hw_files if f.endswith("__horizontal_well.csv"))
    )
    print(f"Test wells: {len(test_wids)}")

    train_wids = set()
    if os.path.isdir(TRAIN_DIR):
        hw_files_tr = sorted(os.listdir(TRAIN_DIR))
        train_wids = set(
            f.split("__")[0] for f in hw_files_tr if f.endswith("__horizontal_well.csv")
        )
    print(f"Train wells available: {len(train_wids)}")

    sample = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
    sample["well"] = sample["id"].str[:8]
    sample["row_idx"] = sample["id"].str[9:].astype(int)

    rows = []
    n_processed = 0
    n_pf_failed = 0
    n_phys = 0

    for wid in test_wids:
        # Reset per-well state — prevents leak from previous iteration's tw_tr.
        tw_tr = None
        hw_tr = None

        print(f"\n[{n_processed + 1}/{len(test_wids)}] Processing {wid}...", end=" ")
        hw_te, tw_te = load_well(wid, TEST_DIR)

        eval_mask = hw_te["TVT_input"].isna().to_numpy()
        n_eval = int(eval_mask.sum())

        # Physical model for visible (training) wells
        tvt_phys = None
        if wid in train_wids:
            try:
                hw_tr, tw_tr = load_well(wid, TRAIN_DIR)
                hw_te["TVT_input"] = hw_tr["TVT_input"].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
                n_phys += 1
                print("phys", end=" ")
            except Exception as e:
                print(f"phys_fail({e})", end=" ")
                tvt_phys = None

        selector_code, selector_variant, sel_n_eval, sel_z_span = selector_well_code(
            hw_te, eval_mask
        )

        # 128-seed PF ensemble
        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            pf_by_scale = run_pf_lik_ensemble_scales(
                hw_te, tw_ref, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS
            )
            tvt_pf = pf_by_scale.get("pf_scale_8", list(pf_by_scale.values())[0])
            print("pf128", end=" ")
        except Exception as e:
            print(f"pf_fail({e})", end=" ")
            n_pf_failed += 1
            last_known = hw_te["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf = hw_te["TVT_input"].fillna(last_val).values.astype(float)
            pf_by_scale = {f"pf_scale_{sc:g}": tvt_pf.copy() for sc in PF_SCALES}

        # Beam search ensemble
        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            tvt_beam = run_beam_ensemble(hw_te, tw_ref)
            print("beam", end=" ")
        except Exception as e:
            print(f"beam_fail({e})", end=" ")
            tvt_beam = tvt_pf.copy()

        last_known = hw_te["TVT_input"].dropna()
        last_known_tvt = (
            float(last_known.iloc[-1]) if len(last_known) > 0 else float(np.nanmean(tvt_pf))
        )
        tvt_selector = apply_selector_variant(
            selector_variant, pf_by_scale, tvt_beam, last_known_tvt
        )
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

    # Hard-align to sample_submission so any missing well/row gets filled
    # with the median of submitted preds rather than dropped.
    sub_aligned = sample[["id"]].merge(sub, on="id", how="left")
    n_missing = int(sub_aligned["tvt"].isna().sum())
    if n_missing:
        fill_val = float(sub_aligned["tvt"].median())
        print(f"⚠ {n_missing} rows missing from preds → filling with median {fill_val:.2f}")
        sub_aligned["tvt"] = sub_aligned["tvt"].fillna(fill_val)
    sub_aligned.to_csv(OUT_PATH, index=False)
    sub = sub_aligned

    elapsed = time.time() - t0
    print(f"\n{'=' * 60}")
    print(f"Done: {len(sub)} rows in {elapsed / 60:.1f} min ({elapsed / 3600:.2f} h)")
    print(
        f"Wells: {n_processed} total | {n_phys} physical | {n_pf_failed} PF failures"
    )
    print(f"Output: {OUT_PATH}")
    print(f"Mean TVT: {sub['tvt'].mean():.3f}")
    print(sub.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
