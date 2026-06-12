"""R8 Phase 4A: formation-contact physics features for all 723 train wells.

Implements `tvt_from_contacts` from the LB-7.776 kernel
(docs/lb-references/lb-7-776-rogii-ridge-sp.py:115). The physical model:

  Given a reference formation (e.g. EGFDU), the geologist annotated where
  that formation appears along the horizontal well (column 'EGFDU' = TVD of
  that formation contact at each horizontal row). The typewell knows the
  TVT of that formation (from tw.Geology). Then:

      tvt_phys[h] = ref_tvt - (Z[h] - hw[ref_col][h]) + offset

  where `offset` calibrates against the *known* part of the lateral.

  This is orthogonal to GR matching: it uses the geological surface
  picked by the steerer (column 'EGFDU' etc.) as a physical anchor,
  not GR similarity.

CV cleanliness: the offset is computed using only `TVT_input` (the known
prefix), NOT `TVT` (which would leak the lateral label). This differs from
the kernel which uses full `TVT` at test time (when full prefix is known).

For each formation in [ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA] we produce:
  tvtF_<col>          — physical TVT prediction at each lateral row
  tvtF_<col>_offset   — relative to last_known_tvt (for use as feature)

Output: results/round_008/contacts_features.parquet
"""
import os, sys, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]


def tvt_from_contacts_cv_safe(hw, tw, ref_col):
    """Per-well, per-formation physical TVT prediction.

    Returns (tvt_phys[n_rows], valid_bool) — predictions at every row of hw.
    valid=False if the formation column has too few rows or no typewell ref.

    Calibration uses only the *known* prefix (TVT_input non-NaN), matching
    what's available at test time. This makes the feature CV-clean: we can
    compute it for every well in train and reuse the exact same formula at
    test inference.
    """
    if ref_col not in hw.columns or "Geology" not in tw.columns:
        return None, False
    tw_g = tw.dropna(subset=["Geology"])
    g_rows = tw_g[tw_g["Geology"] == ref_col]
    if len(g_rows) == 0:
        return None, False
    ref_tvt = float(g_rows["TVT"].min())
    if np.isnan(ref_tvt):
        return None, False

    z = hw["Z"].values.astype(np.float64)
    rc = hw[ref_col].values.astype(np.float64)
    raw = ref_tvt - (z - rc)  # uncalibrated physical prediction

    # Calibrate offset against known segment only
    mask_known = hw["TVT_input"].notna().values & ~np.isnan(rc) & ~np.isnan(z)
    if mask_known.sum() < 10:
        return None, False
    known_tvt = hw["TVT_input"].values[mask_known]
    offset = float(np.nanmean(known_tvt - raw[mask_known]))
    if np.isnan(offset):
        return None, False

    return (raw + offset).astype(np.float32), True


def main():
    print("=== R8 Phase 4A: formation contact features ===\n")

    all_wells = sorted({f.replace("__horizontal_well.csv", "")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})
    print(f"Wells: {len(all_wells)}")
    print(f"Formations: {FORMS}\n")

    records = []
    n_ok = n_fail = 0
    coverage = {f: 0 for f in FORMS}
    t0 = time.time()
    for i, wid in enumerate(all_wells):
        try:
            hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
            tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        except Exception:
            n_fail += 1; continue

        if hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 10:
            n_fail += 1; continue

        # last_known for offsetting
        kn_idx = np.flatnonzero(hw["TVT_input"].notna().values)
        if len(kn_idx) < 10:
            n_fail += 1; continue
        last_known_tvt = float(hw["TVT_input"].iloc[kn_idx[-1]])

        # Lateral rows
        ev_mask = hw["TVT_input"].isna().values
        ev_idx = np.flatnonzero(ev_mask)
        if len(ev_idx) == 0:
            n_fail += 1; continue

        row = {"well": wid, "row_idx": ev_idx.astype(np.int32)}
        ok_one = False
        for f in FORMS:
            pred, valid = tvt_from_contacts_cv_safe(hw, tw, f)
            if valid:
                coverage[f] += 1
                row[f"tvtF_{f}"]        = pred[ev_idx].astype(np.float32)
                row[f"tvtF_{f}_offset"] = (pred[ev_idx] - last_known_tvt).astype(np.float32)
                ok_one = True
            else:
                # Fill NaN — LightGBM handles NaN natively
                row[f"tvtF_{f}"]        = np.full(len(ev_idx), np.nan, dtype=np.float32)
                row[f"tvtF_{f}_offset"] = np.full(len(ev_idx), np.nan, dtype=np.float32)
        if not ok_one:
            n_fail += 1; continue

        # Mean across available formations (ignore nan)
        offs_stack = np.stack(
            [row[f"tvtF_{f}_offset"] for f in FORMS], axis=1
        )  # (n_ev, 6)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            row["tvtF_mean_offset"] = np.nanmean(offs_stack, axis=1).astype(np.float32)
            row["tvtF_std_offset"]  = np.nanstd(offs_stack, axis=1).astype(np.float32)

        records.append(pd.DataFrame(row))
        n_ok += 1

        if (i + 1) % 100 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s", flush=True)

    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} fail={n_fail}")
    print("Per-formation coverage:")
    for f in FORMS:
        print(f"  {f:7s}: {coverage[f]}/{len(all_wells)} "
              f"({100*coverage[f]/len(all_wells):.1f}%)")

    df = pd.concat(records, ignore_index=True)
    print(f"\nRows: {len(df):,}  Wells: {df['well'].nunique()}  Cols: {df.shape[1]}")
    out_path = OUT_DIR / "contacts_features.parquet"
    df.to_parquet(out_path)
    print(f"→ {out_path}")


if __name__ == "__main__":
    main()
