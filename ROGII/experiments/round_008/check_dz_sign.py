"""Sanity check: is dtvt ≈ -dz, or +dz, or neither?

For 10 wells, on lateral rows:
  diff_tvt = TVT[r] - TVT[r-1]
  diff_z   = Z[r]   - Z[r-1]
correlate them and check signs.

Forum (msg 3462931): dtvt = -dz. Our cumsum(-dz) baseline at 107 RMSE
contradicts this. Either the formula assumes a different Z convention,
or the lateral has Z drift that dwarfs TVT change.
"""
import os, numpy as np, pandas as pd
DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"

wells = sorted({f.replace("__horizontal_well.csv","")
                for f in os.listdir(DATA_DIR)
                if f.endswith("__horizontal_well.csv")})[:20]

print(f"{'well':12s} {'corr(dtvt,dz)':>14s} {'mean(dtvt/dz)':>14s} "
      f"{'std(dtvt)':>10s} {'std(dz)':>10s} "
      f"{'lat_len':>8s} {'z_drift':>10s} {'tvt_drift':>10s}")

agg = []
for wid in wells:
    h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
    tvt = h["TVT"].values
    z = h["Z"].values
    mask_lat = np.isnan(h["TVT_input"].values)
    lat = np.flatnonzero(mask_lat)
    if len(lat) < 50: continue
    tvt_l = tvt[lat]
    z_l = z[lat]
    valid = ~np.isnan(tvt_l)
    tvt_l = tvt_l[valid]; z_l = z_l[valid]
    if len(tvt_l) < 50: continue

    dtvt = np.diff(tvt_l)
    dz = np.diff(z_l)
    keep = ~(np.isnan(dtvt) | np.isnan(dz))
    dtvt, dz = dtvt[keep], dz[keep]
    if len(dtvt) < 20: continue

    corr = np.corrcoef(dtvt, dz)[0,1]
    # ratio: how much tvt moves per unit z (only where dz nonzero)
    nz = np.abs(dz) > 1e-4
    ratio = (dtvt[nz] / dz[nz]).mean() if nz.sum() else np.nan
    z_drift = z_l[-1] - z_l[0]
    tvt_drift = tvt_l[-1] - tvt_l[0]
    print(f"{wid:12s} {corr:>14.3f} {ratio:>14.3f} "
          f"{dtvt.std():>10.3f} {dz.std():>10.3f} "
          f"{len(lat):>8d} {z_drift:>10.2f} {tvt_drift:>10.2f}")
    agg.append((corr, ratio, z_drift, tvt_drift))

agg = np.array(agg)
print(f"\nover {len(agg)} wells:")
print(f"  mean corr(dtvt, dz)    = {agg[:,0].mean():+.3f}")
print(f"  median corr            = {np.median(agg[:,0]):+.3f}")
print(f"  mean ratio dtvt/dz     = {agg[:,1].mean():+.3f}  (forum claims -1.0)")
print(f"  mean Z drift (ft)      = {agg[:,2].mean():.1f}")
print(f"  mean TVT drift (ft)    = {agg[:,3].mean():.1f}")
print(f"  ratio of drifts (TVT/Z)= {agg[:,3].mean()/agg[:,2].mean():+.3f}")
