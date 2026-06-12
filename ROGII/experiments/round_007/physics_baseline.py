"""R7 sanity check: physics-only baseline.

For each well in cfg-img-medium val + cfg-img-full val (same wells),
compute tvt_phys[h] = last_known_tvt + cumsum(-dz)[h] from the raw CSV
and measure RMSE on the lateral segment.

If RMSE < 25, then the SegFormer 15.84 baseline is below physics; SDF
post-processing was carrying us. If RMSE >> 25, the lateral really does
move; SDF is necessary.
"""
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/src")
from gen_images import VAL_IDS, DATA_DIR

results = []
for wid in VAL_IDS:
    h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
    tvt = h["TVT"].values
    tvt_inp = h["TVT_input"].values
    z = h["Z"].values

    # Lateral region: TVT_input NaN
    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0:
        continue

    # Last known row
    last_known_idx = np.flatnonzero(~mask_lat)[-1]
    last_known_tvt = tvt_inp[last_known_idx]
    last_known_z = z[last_known_idx]

    # tvt_phys[h] = last_known_tvt - (z[h] - last_known_z)  (== last_known + cumsum(-dz))
    tvt_phys = last_known_tvt - (z - last_known_z)

    # Eval on lateral rows only
    lat_idx = np.flatnonzero(mask_lat)
    pred = tvt_phys[lat_idx]
    true = tvt[lat_idx]
    valid = ~np.isnan(true)
    if valid.sum() == 0:
        continue
    rmse = float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2)))
    results.append((wid, rmse, valid.sum()))

rmses = np.array([r[1] for r in results])
weights = np.array([r[2] for r in results])
mean_rmse = float(rmses.mean())
median_rmse = float(np.median(rmses))
weighted_rmse = float(np.sqrt(np.average(rmses ** 2, weights=weights)))

print(f"Physics baseline on {len(results)} curated val wells:")
print(f"  mean RMSE       : {mean_rmse:.2f}")
print(f"  median RMSE     : {median_rmse:.2f}")
print(f"  rms-weighted RMSE: {weighted_rmse:.2f}")
print(f"  max RMSE         : {rmses.max():.2f}")
print(f"  min RMSE         : {rmses.min():.2f}")

# Same against random-50 split
sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/experiments/round_006")
from gen_random_val import random_split
_, randval_ids = random_split()

results_r = []
for wid in randval_ids:
    h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
    tvt = h["TVT"].values
    tvt_inp = h["TVT_input"].values
    z = h["Z"].values
    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0:
        continue
    last_known_idx = np.flatnonzero(~mask_lat)[-1]
    last_known_tvt = tvt_inp[last_known_idx]
    last_known_z = z[last_known_idx]
    tvt_phys = last_known_tvt - (z - last_known_z)
    lat_idx = np.flatnonzero(mask_lat)
    pred = tvt_phys[lat_idx]
    true = tvt[lat_idx]
    valid = ~np.isnan(true)
    if valid.sum() == 0:
        continue
    rmse = float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2)))
    results_r.append((wid, rmse, valid.sum()))

rmses_r = np.array([r[1] for r in results_r])
print(f"\nPhysics baseline on {len(results_r)} RANDOM val wells:")
print(f"  mean RMSE       : {rmses_r.mean():.2f}")
print(f"  median RMSE     : {np.median(rmses_r):.2f}")
print(f"  max RMSE        : {rmses_r.max():.2f}")
print(f"  min RMSE        : {rmses_r.min():.2f}")

# All 773 wells
all_wells = sorted({f.replace("__horizontal_well.csv","") for f in os.listdir(DATA_DIR) if f.endswith("__horizontal_well.csv")})
print(f"\nPhysics baseline on ALL {len(all_wells)} wells:")
results_a = []
for wid in all_wells:
    try:
        h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
    except Exception:
        continue
    tvt = h["TVT"].values
    tvt_inp = h["TVT_input"].values
    z = h["Z"].values
    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0:
        continue
    last_known_idx = np.flatnonzero(~mask_lat)[-1]
    last_known_tvt = tvt_inp[last_known_idx]
    last_known_z = z[last_known_idx]
    tvt_phys = last_known_tvt - (z - last_known_z)
    lat_idx = np.flatnonzero(mask_lat)
    pred = tvt_phys[lat_idx]
    true = tvt[lat_idx]
    valid = ~np.isnan(true)
    if valid.sum() == 0:
        continue
    rmse = float(np.sqrt(np.mean((pred[valid] - true[valid]) ** 2)))
    results_a.append((wid, rmse, valid.sum()))

rmses_a = np.array([r[1] for r in results_a])
print(f"  n={len(results_a)} mean={rmses_a.mean():.2f} median={np.median(rmses_a):.2f} max={rmses_a.max():.2f}")
