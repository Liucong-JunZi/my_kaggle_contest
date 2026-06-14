"""R8 Phase 12: Q-3D wellbore tortuosity features (Jing et al. 2022).

Uses Niccoli's vendored wellbore_tortuosity.py (mycarta/rogii-geosteering-toolkit, MIT).
Computes per-portion (TQG_incline, TQG_azimuth, TQG_Q3D, etc.) for each well's
horizontal section, then broadcasts to row-level via portion_start_md/end_md
membership lookup. Per-row features:
  q3d_T_incline, q3d_Gamma_incline, q3d_TQG_incline,
  q3d_T_azimuth, q3d_Gamma_azimuth, q3d_TQG_azimuth,
  q3d_TQG_Q3D, q3d_portion_idx_norm

Output: results/round_008/q3d_features.parquet (well, row_idx, ...features).
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from wellbore_tortuosity import compute_tortuosity_from_xyz

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT      = Path("/Users/liucong/code/kaggle/ROGII/results/round_008/q3d_features.parquet")

PORTION_LEN = 100.0  # ft (consistent with Jing recommendation for ~3 km laterals)
FEATS = ["T_incline","Gamma_incline","TQG_incline",
         "T_azimuth","Gamma_azimuth","TQG_azimuth","TQG_Q3D"]


def q3d_per_row(wid, data_dir=DATA_DIR, portion_length=PORTION_LEN):
    try:
        h = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
    except Exception:
        return None
    md = h["MD"].values; x = h["X"].values; y = h["Y"].values; z = h["Z"].values
    if len(md) < 30: return None
    # ROGII conventions: X-East, Y-North, Z-up (Z is negative for deeper)
    try:
        results, _, _ = compute_tortuosity_from_xyz(
            md, x, y, z,
            portion_length=portion_length,
            x_axis="east", y_axis="north", z_axis="up",
        )
    except Exception as e:
        return None
    if results is None or len(results) == 0:
        return None
    # Per-row: find which portion each row falls into
    starts = results["portion_start_md"].values
    ends   = results["portion_end_md"].values
    # Vectorized bucket lookup
    portion_idx = np.searchsorted(starts, md, side="right") - 1
    portion_idx = np.clip(portion_idx, 0, len(results)-1)
    out = {"well": wid, "row_idx": np.arange(len(md), dtype=np.int32)}
    for c in FEATS:
        v = results[c].values
        out[f"q3d_{c}"] = v[portion_idx].astype(np.float32)
    out["q3d_portion_idx_norm"] = (portion_idx / max(len(results)-1,1)).astype(np.float32)
    return pd.DataFrame(out)


def main():
    print("=== R8 Phase 12: Q-3D tortuosity features ===\n")
    t0 = time.time()
    wells = sorted({f.replace("__horizontal_well.csv","")
                    for f in os.listdir(DATA_DIR)
                    if f.endswith("__horizontal_well.csv")})
    print(f"  wells: {len(wells)}")
    rows = []; n_ok = n_fail = 0
    for i, wid in enumerate(wells):
        df = q3d_per_row(wid)
        if df is None:
            n_fail += 1
        else:
            rows.append(df); n_ok += 1
        if (i+1) % 100 == 0:
            print(f"  {i+1}/{len(wells)} | ok={n_ok} fail={n_fail} | {time.time()-t0:.0f}s", flush=True)
    out = pd.concat(rows, ignore_index=True)
    print(f"\nrows: {len(out):,}, wells: {out['well'].nunique()}, ok={n_ok} fail={n_fail}")
    print(f"Q-3D feature stats:")
    for c in [f"q3d_{f}" for f in FEATS]:
        v = out[c]
        print(f"  {c:25s}  min={v.min():.3f}  med={v.median():.3f}  max={v.max():.3f}  nan={v.isna().sum():,}")
    out.to_parquet(OUT)
    print(f"\n→ {OUT}  wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
