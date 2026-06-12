"""R8 Phase 8: PF with 4 process-noise scales (low/med/high/extra-high).

The diagnostic showed worst-40 wells have 5× more TVT drift than best-40. Our
single PF underestimates the dip range. Solution: run PF with varied process
noise so the lower-PN runs track stable wells well, higher-PN runs track
high-dip wells well. Tree model blends.

Variants (only `_pf_z` re-run, 4 scales):
  pf_z_pn005 — PN=0.005 (low process noise; default)  -- like our existing pf_z but tighter
  pf_z_pn010 — PN=0.010 (default)
  pf_z_pn030 — PN=0.030 (3× looser, tracks bigger dip)
  pf_z_pn080 — PN=0.080 (8× looser, tracks aggressive dip)

Each run is one seed (seed=42 fixed). Takes ~10 min total.

Output: results/round_008/pf_multiscale.parquet
  columns: well, row_idx, pf_z_pn005, pf_z_pn010, pf_z_pn030, pf_z_pn080,
           pf_z_std_pn005..pn080
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from numba import njit

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")

# Import existing JIT _pf_z + helpers
sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/experiments/round_008")
from r8_pf_features import (
    _pf_z, _grid, _gr_sig,
    PF_MOM, PF_VN, PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP,
    PF_GR_WIN, PF_N, _warmup,
)


PN_SCALES = [0.005, 0.010, 0.030, 0.080]  # process noise variants
SCALE_LABELS = [f"pn{int(p*1000):03d}" for p in PN_SCALES]


def run_pf_z_with_pn(hw, tw_tvt, tw_gr, PN, N=PF_N):
    """Same as run_pf_z but with overridden PF_PN."""
    gs = _gr_sig(hw, tw_tvt, tw_gr)
    tw_s = pd.Series(tw_gr).rolling(PF_GR_WIN, center=True, min_periods=1).mean().values.astype(np.float32)
    kna = hw[hw["TVT_input"].notna()]
    ev = hw[hw["TVT_input"].isna()]
    if len(ev) == 0:
        return np.array([]), np.array([])
    dz_k = np.diff(kna["Z"].values); dvt = np.diff(kna["TVT_input"].values)
    dmd_k = np.diff(kna["MD"].values); m2 = dmd_k > 0
    if m2.sum() >= 10:
        vz = dz_k[m2]/dmd_k[m2]; vt = dvt[m2]/dmd_k[m2]
        A = np.column_stack([vz, np.ones_like(vz)])
        c, _, _, _ = np.linalg.lstsq(A, vt, rcond=None)
        beta, icpt, zsig = float(c[0]), float(c[1]), max(float(np.std(vt-(c[0]*vz+c[1]))), 0.001)
    else:
        beta, icpt, zsig = -1., 0., 0.1
    t2 = kna.tail(20)
    dvt2 = np.diff(t2["TVT_input"].values); dmd2 = np.diff(t2["MD"].values); m3 = dmd2 > 0
    iv = float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum() >= 3 else 0.
    gg, gmin, gst = _grid(tw_tvt, tw_gr)
    gs2, _, _ = _grid(tw_tvt, tw_s)
    gr_sm = hw["GR"].rolling(PF_GR_WIN, center=True, min_periods=1).mean()
    pts, std = _pf_z(
        ev["MD"].values.astype(np.float64), ev["Z"].values.astype(np.float64),
        ev["GR"].values.astype(np.float64),
        gr_sm.loc[ev.index].values.astype(np.float64),
        gg, gs2, gmin, gst,
        gs, float(kna["TVT_input"].iloc[-1]), iv, beta, icpt, zsig, N,
        PF_MOM, PF_VN, PN,           # ← overridden PN
        PF_GR_WT, PF_ROUGH_P, PF_ROUGH_V, PF_RESAMP,
    )
    return pts.astype(np.float32), std.astype(np.float32)


def main():
    print("=== R8 Phase 8: multi-PN PF features ===\n")
    print(f"PN scales: {dict(zip(SCALE_LABELS, PN_SCALES))}\n")

    print("[warmup]")
    np.random.seed(0); _warmup()

    all_wells = sorted({f.replace("__horizontal_well.csv", "")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})
    print(f"Wells: {len(all_wells)}\n")

    records = []
    n_ok = n_fail = 0
    t0 = time.time()
    for i, wid in enumerate(all_wells):
        try:
            hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
            tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        except Exception:
            n_fail += 1; continue
        if hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 10:
            n_fail += 1; continue
        tw_s = tw.sort_values("TVT")
        tw_tvt = tw_s["TVT"].values.astype(np.float64)
        tw_gr = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float64)
        if len(tw_tvt) < 10 or np.isnan(tw_gr).all():
            n_fail += 1; continue

        ev_idx = hw.index[hw["TVT_input"].isna()].values
        row = {"well": wid, "row_idx": ev_idx.astype(np.int32)}
        try:
            for label, PN in zip(SCALE_LABELS, PN_SCALES):
                np.random.seed(42)  # same seed across scales for reproducibility
                pts, std_ = run_pf_z_with_pn(hw, tw_tvt, tw_gr, PN=PN)
                if len(pts) != len(ev_idx):
                    raise ValueError(f"len mismatch: {len(pts)} vs {len(ev_idx)}")
                row[f"pf_z_{label}"]     = pts
                row[f"pf_z_std_{label}"] = std_
        except Exception as e:
            print(f"  ! {wid}: {e}"); n_fail += 1; continue

        records.append(pd.DataFrame(row))
        n_ok += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}, wells: {df['well'].nunique()}, cols: {df.shape[1]}")
    out = OUT_DIR / "pf_multiscale.parquet"
    df.to_parquet(out)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
