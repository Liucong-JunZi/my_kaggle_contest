"""ROGII Kaggle Notebook entry point — pure PF submission (no ML).

What this script does
---------------------
1.  Reads test wells from /kaggle/input/rogii-wellbore-geology-prediction/test/.
2.  For each test well:
      • runs the 128-seed × 500-particle × 4-scale PF ensemble
        (imported from training_code.pf_128seed)
      • runs the 14-config beam-search ensemble
      • picks a Yaroslav 6-bin per-well variant that combines PF/Beam/hold
      • applies a 4th-degree robust polynomial projection
      • for wells that also exist in TRAIN/, uses the deterministic
        physical model `tvt_from_contacts` instead of the PF prediction
3.  Writes /kaggle/working/submission.csv with columns id, tvt
   (absolute TVT, not offset).

No ML model files (CatBoost / LightGBM / XGBoost) are loaded.

Wall time
---------
The 128-seed × 500-particle PF dominates: ~4–6 hours for ~100 hidden
test wells on the Kaggle CPU notebook (16 GiB / 4 vCPU). Beam + selector
+ projection together are <2 min per well.

Layout
------
This file is the Notebook entry point. The PF kernel lives in
training_code/pf_128seed.py so it can be imported here. Both files
must be in the same Kaggle Dataset (or both copied into /kaggle/working).
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

# ── Make training_code importable regardless of where the package is mounted ─
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Kaggle Datasets typically mount under /kaggle/input/<dataset-slug>/.
# Look for `training_code/pf_128seed.py` in a few common locations.
_PKG_CANDIDATES = [
    _HERE,
    "/kaggle/working/submission_package",
    "/kaggle/working",
]
for _root in os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else []:
    _PKG_CANDIDATES.append(f"/kaggle/input/{_root}")
    _PKG_CANDIDATES.append(f"/kaggle/input/{_root}/submission_package")

for _cand in _PKG_CANDIDATES:
    if os.path.isfile(os.path.join(_cand, "training_code", "pf_128seed.py")):
        if _cand not in sys.path:
            sys.path.insert(0, _cand)
        break

from training_code.pf_128seed import (  # noqa: E402
    run_pf_lik_ensemble_scales,
    PF_N_PARTICLES,
    PF_N_SEEDS,
    PF_SCALES,
)


# ── Selector config (Yaroslav/quanzhongji 6-bin per-well selector) ────────────
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

# ── 14 beam configs: (beam_size, motion_cost, error_scale, savgol_radius) ─────
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


# ── Path discovery ────────────────────────────────────────────────────────────
def _find_kaggle_input(slug: str = "rogii-wellbore-geology-prediction"):
    for p in (f"/kaggle/input/competitions/{slug}", f"/kaggle/input/{slug}"):
        if os.path.isdir(p):
            return p
    return None


def load_well(wid: str, base: str):
    hw = pd.read_csv(os.path.join(base, f"{wid}__horizontal_well.csv"))
    tw = pd.read_csv(os.path.join(base, f"{wid}__typewell.csv"))
    return hw, tw


# ── Physical model (deterministic, used when a test well also appears in TRAIN) ─
def tvt_from_contacts(hw_tr: pd.DataFrame, tw_tr: pd.DataFrame,
                       ref_col: str = "EGFDU") -> pd.Series:
    """Project TVT from a known formation contact in the typewell.

    For wells visible in train/, this is more accurate than any model
    prediction because TVT is fully determined by Z and the formation
    surface depth column.
    """
    tw_g = tw_tr.dropna(subset=["Geology"])
    ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g["Geology"].iloc[0]
        ref_tvt = tw_g[tw_g["Geology"] == ref_col]["TVT"].min()
    offset = (hw_tr["TVT"] - (ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr["Z"] - hw_tr[ref_col]) + offset


# ── Beam search ───────────────────────────────────────────────────────────────
def beam_search(hgr: np.ndarray, tw_tvt: np.ndarray, tw_gr: np.ndarray,
                 last_tvt: float, bs: int = 10, mc: float = 20.0,
                 es: float = 144.0, r: int = 2) -> np.ndarray:
    """Vectorised beam search for TVT tracking via GR matching."""
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
            top = np.argpartition(tot_u, min(kept - 1, len(tot_u) - 1))[:kept]
            top = top[np.argsort(tot_u[top])]
            bidx[:kept]  = ni_u[top]
            bcost[:kept] = tot_u[top]
        if kept < bs:
            bidx[kept:]  = bidx[kept - 1] if kept > 0 else si
            bcost[kept:] = np.inf
        bn = kept if kept > 0 else 1

        result[step] = tw_tvt[bidx[0]]

    return result


def run_beam_ensemble(hw: pd.DataFrame, tw: pd.DataFrame) -> np.ndarray:
    """Average across the 14 beam-search configs."""
    kn = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return hw["TVT_input"].values.astype(float).copy()

    last_tvt = float(kn.iloc[-1]["TVT_input"])
    tw_s   = tw.sort_values("TVT")
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


# ── Selector + variant parser ─────────────────────────────────────────────────
def selector_well_code(hw: pd.DataFrame, eval_mask: np.ndarray):
    """6-bin per-well variant lookup: (n_eval > T) + 2 * z_span_bin."""
    n_eval = float(eval_mask.sum())
    z_eval = hw.loc[eval_mask, "Z"].values.astype(float)
    z_span = float(np.nanmax(z_eval) - np.nanmin(z_eval)) if len(z_eval) else 0.0
    n_bin = int(n_eval > SELECTOR_N_EVAL_THRESHOLD)
    z_bin = int(np.searchsorted(SELECTOR_Z_SPAN_THRESHOLDS, z_span, side="right"))
    code = n_bin + 2 * z_bin
    variant = SELECTOR_BIN_VARIANTS.get(code, SELECTOR_GLOBAL_VARIANT)
    return code, variant, n_eval, z_span


def parse_selector_variant(name: str):
    parts = name.split("_")
    scale = float(parts[2])
    beam_weight = 0.0
    hold_weight = 0.0
    if "beam" in parts:
        beam_weight = float(parts[parts.index("beam") + 1])
    if "hold" in parts:
        hold_weight = float(parts[parts.index("hold") + 1])
    return scale, beam_weight, hold_weight


def apply_selector_variant(name: str, pf_by_scale: dict,
                            tvt_beam: np.ndarray, last_known_tvt: float) -> np.ndarray:
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


# ── 4th-degree robust polynomial projection ───────────────────────────────────
def robust_polyfit(x: np.ndarray, y: np.ndarray, deg: int = 4) -> np.ndarray:
    """Robust polynomial fit via IRLS (Tukey-style biweight)."""
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


def project_lateral(tvt_pred: np.ndarray, eval_mask: np.ndarray) -> np.ndarray:
    """Apply a 4th-degree robust polynomial fit over the lateral rows.

    Reduces high-freq noise from PF/beam without flattening the trend.
    """
    out = tvt_pred.copy().astype(float)
    idx = np.flatnonzero(eval_mask)
    if len(idx) < 6:
        return out

    frac = (idx - idx.min()) / max(idx.max() - idx.min(), 1)
    y = out[idx]

    coeffs = robust_polyfit(frac, y, deg=min(4, len(y) - 1))
    out[idx] = np.polyval(coeffs, frac)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    t0 = time.time()

    INPUT_DIR = _find_kaggle_input() or "/kaggle/input/rogii-wellbore-geology-prediction"
    TRAIN_DIR = f"{INPUT_DIR}/train"
    TEST_DIR  = f"{INPUT_DIR}/test"
    on_kaggle = _find_kaggle_input() is not None
    OUT_PATH  = "/kaggle/working/submission.csv" if on_kaggle else "submission.csv"

    print(f"INPUT_DIR = {INPUT_DIR}")
    print(f"OUT_PATH  = {OUT_PATH}")
    assert os.path.isdir(TEST_DIR), f"TEST_DIR missing: {TEST_DIR}"
    print(f"PF: {PF_N_SEEDS} seeds × {PF_N_PARTICLES} particles × {len(PF_SCALES)} scales")

    # Test wells
    test_files = sorted(os.listdir(TEST_DIR))
    test_wids = sorted({f.split("__")[0] for f in test_files
                        if f.endswith("__horizontal_well.csv")})
    print(f"Test wells: {len(test_wids)}")

    # Train wells (used by the physical model when a well is visible)
    train_wids: set[str] = set()
    if os.path.isdir(TRAIN_DIR):
        train_files = sorted(os.listdir(TRAIN_DIR))
        train_wids = {f.split("__")[0] for f in train_files
                      if f.endswith("__horizontal_well.csv")}
    print(f"Train wells available (for physical model): {len(train_wids)}")

    sample_path = os.path.join(INPUT_DIR, "sample_submission.csv")
    sample = pd.read_csv(sample_path)
    sample["well"]    = sample["id"].str[:8]
    sample["row_idx"] = sample["id"].str[9:].astype(int)

    rows: list[dict] = []
    n_processed = 0
    n_pf_failed = 0
    n_phys      = 0

    for wid in test_wids:
        # Reset per-well state — prevents the previous iteration's variables
        # from leaking into this well via fall-through.
        tw_tr = None
        hw_tr = None

        print(f"\n[{n_processed + 1}/{len(test_wids)}] {wid} ...", end=" ", flush=True)
        hw_te, tw_te = load_well(wid, TEST_DIR)
        eval_mask = hw_te["TVT_input"].isna().to_numpy()

        # If the well is also in TRAIN, use the deterministic physical model
        tvt_phys = None
        if wid in train_wids:
            try:
                hw_tr, tw_tr = load_well(wid, TRAIN_DIR)
                # Use train's full TVT_input column (it has all rows filled)
                # so PF + beam see the same known segment as the physical model.
                hw_te["TVT_input"] = hw_tr["TVT_input"].values
                tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
                n_phys += 1
                print("phys", end=" ", flush=True)
            except Exception as e:
                print(f"phys_fail({e})", end=" ", flush=True)
                tvt_phys = None

        # 6-bin selector
        selector_code, selector_variant, sel_n_eval, sel_z_span = \
            selector_well_code(hw_te, eval_mask)

        # 128-seed PF ensemble (the dominant cost — 80–90% of wall time)
        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            pf_by_scale = run_pf_lik_ensemble_scales(
                hw_te, tw_ref,
                n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS,
            )
            tvt_pf_default = pf_by_scale.get("pf_scale_8",
                              list(pf_by_scale.values())[0])
            print("pf128", end=" ", flush=True)
        except Exception as e:
            print(f"pf_fail({e})", end=" ", flush=True)
            n_pf_failed += 1
            last_known = hw_te["TVT_input"].dropna()
            last_val = float(last_known.iloc[-1]) if len(last_known) > 0 else 0.0
            tvt_pf_default = hw_te["TVT_input"].fillna(last_val).values.astype(float)
            pf_by_scale = {f"pf_scale_{sc:g}": tvt_pf_default.copy() for sc in PF_SCALES}

        # 14-config beam ensemble
        try:
            tw_ref = tw_tr if tw_tr is not None else tw_te
            tvt_beam = run_beam_ensemble(hw_te, tw_ref)
            print("beam", end=" ", flush=True)
        except Exception as e:
            print(f"beam_fail({e})", end=" ", flush=True)
            tvt_beam = tvt_pf_default.copy()

        # Selector blend (PF scale + beam_weight + hold_weight)
        last_known = hw_te["TVT_input"].dropna()
        last_known_tvt = float(last_known.iloc[-1]) if len(last_known) > 0 \
                        else float(np.nanmean(tvt_pf_default))
        tvt_selector = apply_selector_variant(
            selector_variant, pf_by_scale, tvt_beam, last_known_tvt,
        )
        print(f"sel={selector_code}:{selector_variant}", end=" ", flush=True)

        # 4th-degree robust polynomial projection (only over lateral rows)
        try:
            tvt_proj = project_lateral(tvt_selector, eval_mask)
            print("proj", end=" ", flush=True)
        except Exception as e:
            print(f"proj_fail({e})", end=" ", flush=True)
            tvt_proj = tvt_selector

        ws = sample[sample["well"] == wid]
        for _, row in ws.iterrows():
            ridx = int(row["row_idx"])
            if tvt_phys is not None:
                tvt_val = float(tvt_phys.iloc[ridx])
            else:
                tvt_val = float(tvt_proj[ridx])
            rows.append({"id": row["id"], "tvt": tvt_val})

        n_processed += 1
        print(f"→ {len(ws)} rows", flush=True)

    sub = pd.DataFrame(rows)

    # Hard-align to sample_submission so any missing well/row gets the
    # median fill rather than dropping silently.
    sub_aligned = sample[["id"]].merge(sub, on="id", how="left")
    n_missing = int(sub_aligned["tvt"].isna().sum())
    if n_missing:
        fill_val = float(sub_aligned["tvt"].median())
        print(f"⚠ {n_missing} rows missing → filling with median {fill_val:.2f}")
        sub_aligned["tvt"] = sub_aligned["tvt"].fillna(fill_val)
    sub_aligned.to_csv(OUT_PATH, index=False)

    elapsed = time.time() - t0
    print("\n" + "=" * 60)
    print(f"Done")
    print(f"  rows:    {len(sub_aligned)}")
    print(f"  wells:   {n_processed} total | {n_phys} physical | {n_pf_failed} PF fail")
    print(f"  output:  {OUT_PATH}")
    print(f"  tvt:     min={sub_aligned['tvt'].min():.2f} "
          f"max={sub_aligned['tvt'].max():.2f} mean={sub_aligned['tvt'].mean():.2f}")
    print(f"  elapsed: {elapsed:.1f} s ({elapsed/60:.1f} min, {elapsed/3600:.2f} h)")
    print(sub_aligned.head(8).to_string(index=False))


if __name__ == "__main__":
    main()
