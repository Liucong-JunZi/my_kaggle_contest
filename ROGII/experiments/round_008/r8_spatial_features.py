"""R8 Phase 10: spatial-neighbor TVT consensus features.

Per the forum (Matteo Niccoli #702919, Amged Alfaqih #699289, Tucker Arrants):
> The 10→14 RMSE gap is NOT about better GR matching — it's about adding a
> SECOND independent information channel: spatial structure from neighboring
> wells.

Approach (no leakage, test-safe):
  For each (X, Y) location in lateral:
    1. Query cKDTree of all 773 train-well known-segment last (X,Y,Z,TVT_input)
    2. Find K=5 nearest neighbors (excluding self)
    3. For each neighbor: compute the neighbor's TVT at its closest match
       in OUR (X, Y) — using its full TVT_input curve interpolated by MD
    4. Feature = median / mean / std of neighbor TVTs at this row

Critical: this uses ONLY `TVT_input` from neighbors (their KNOWN segment).
At test time, our test wells have TVT_input known on the prefix, but their
*lateral* (where we predict) has only X, Y, Z known — exactly the same as
training-time inference. So this is leak-free.

Output: results/round_008/spatial_features.parquet
  Columns: well, row_idx, neighbor_tvt_median, neighbor_tvt_mean,
           neighbor_tvt_std, neighbor_count_valid, neighbor_dist_min
"""
import os, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
K_NEIGHBORS = 5
K_QUERY = 200  # wide query to ensure K non-self neighbors after filter
# No hard distance cutoff — let GBDT decide via dist_min feature


def load_well_known(wid, data_dir):
    """Load a well's full (X, Y, TVT_input) for valid known rows.

    Returns DataFrame with [x, y, tvt] for known segment ONLY (TVT_input not NaN).
    """
    try:
        h = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
    except Exception:
        return None
    mask = h["TVT_input"].notna()
    if mask.sum() < 10:
        return None
    return pd.DataFrame({
        "x":   h["X"].values[mask],
        "y":   h["Y"].values[mask],
        "tvt": h["TVT_input"].values[mask],
    })


def main():
    print("=== R8 Phase 10: spatial-neighbor TVT features ===\n")
    t0 = time.time()

    print("[1/3] Loading all known-segment data from 773 train wells")
    all_wells = sorted({f.replace("__horizontal_well.csv", "")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})

    # Build a global pool: every known row tagged with its source well
    global_pts = []
    for wid in all_wells:
        df = load_well_known(wid, DATA_DIR)
        if df is None:
            continue
        df["well"] = wid
        global_pts.append(df)
    pool = pd.concat(global_pts, ignore_index=True)
    print(f"  pool size: {len(pool):,} rows from {pool['well'].nunique()} wells")

    pool_xy = pool[["x", "y"]].values
    pool_tvt = pool["tvt"].values
    pool_well = pool["well"].to_numpy(dtype=object)  # plain ndarray for 2D indexing
    tree = cKDTree(pool_xy)
    print(f"  cKDTree built in {time.time() - t0:.0f}s\n")

    print("[2/3] Querying neighbors for each lateral row")
    records = []
    n_ok = n_fail = 0
    t1 = time.time()
    for i, wid in enumerate(all_wells):
        try:
            h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
        except Exception:
            n_fail += 1; continue
        ev_mask = h["TVT_input"].isna().values
        if ev_mask.sum() == 0:
            n_fail += 1; continue
        ev_idx = np.flatnonzero(ev_mask)
        ev_xy = h.loc[ev_mask, ["X", "Y"]].values

        # Query K + buffer in case some neighbors are from same well
        K_query = K_QUERY
        dist, idx = tree.query(ev_xy, k=K_query, workers=-1)

        # Mask self-well neighbors (avoid leak from row's own known prefix)
        is_self = (pool_well[idx] == wid)
        # Replace self matches with infinity so they don't get picked
        dist_masked = np.where(is_self, np.inf, dist)

        # Take K best (smallest dist) — sort each row
        order = np.argsort(dist_masked, axis=1)
        idx_sorted   = np.take_along_axis(idx,         order, axis=1)
        dist_sorted  = np.take_along_axis(dist_masked, order, axis=1)
        idx_top  = idx_sorted[:, :K_NEIGHBORS]
        dist_top = dist_sorted[:, :K_NEIGHBORS]
        # No hard cutoff — keep all returned values; dist_top encodes how
        # far the nearest non-self neighbor is. GBDT can learn from dist.
        valid = np.isfinite(dist_top)  # only inf when even K_QUERY all self
        tvt_top = pool_tvt[idx_top]   # (n_ev, K)

        # Compute median / mean / std (per row, masked to valid)
        med = np.full(len(ev_idx), np.nan, np.float32)
        mean = np.full(len(ev_idx), np.nan, np.float32)
        std = np.full(len(ev_idx), np.nan, np.float32)
        n_valid = valid.sum(axis=1)
        d_min = dist_top[:, 0].astype(np.float32)
        for r in range(len(ev_idx)):
            v = valid[r]
            if v.sum() == 0:
                continue
            t = tvt_top[r, v]
            med[r]  = float(np.median(t))
            mean[r] = float(np.mean(t))
            std[r]  = float(np.std(t)) if v.sum() > 1 else 0.0

        records.append(pd.DataFrame({
            "well": wid,
            "row_idx": ev_idx.astype(np.int32),
            "neighbor_tvt_median": med,
            "neighbor_tvt_mean":   mean,
            "neighbor_tvt_std":    std,
            "neighbor_count":      n_valid.astype(np.int8),
            "neighbor_dist_min":   d_min,
        }))
        n_ok += 1
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t1
            eta = elapsed / (i + 1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} fail={n_fail} | "
                  f"elapsed={elapsed:.0f}s eta={eta:.0f}s", flush=True)

    print(f"\nDone neighbor lookup in {time.time()-t1:.0f}s | ok={n_ok} fail={n_fail}")
    df = pd.concat(records, ignore_index=True)
    print(f"Rows: {len(df):,}, wells: {df['well'].nunique()}")
    print(f"NaN in median (no valid neighbor): {df['neighbor_tvt_median'].isna().sum():,}")
    print(f"Mean neighbor count: {df['neighbor_count'].mean():.1f}")
    print(f"Mean dist_min: {df['neighbor_dist_min'].mean():.0f} ft")
    print(f"\n[3/3] Saving")
    out = OUT_DIR / "spatial_features.parquet"
    df.to_parquet(out)
    print(f"→ {out}")


if __name__ == "__main__":
    main()
