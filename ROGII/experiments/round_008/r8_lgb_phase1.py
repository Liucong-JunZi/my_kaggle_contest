"""R8 Phase 1: LightGBM with correct target + full corpus + GroupKFold-5.

Target = TVT - last_known_tvt (relative offset, per forum insights INSIGHTS_UPDATE.md).
Train on full 723 wells. Validate via GroupKFold-5 on well_id.

Features (all test-safe — only use TVT_input known segment, not lateral TVT):
  Geometry:
    - md_offset_from_last  (MD - last_known_md)
    - z_rel                (Z - last_known_z)  ← THE dominant signal (dtvt ≈ -dz)
    - x_rel, y_rel
    - cumsum_neg_dz_from_last  (≈ -dz cumulative since last known)
  Geometric tangents (hengck23 msg 3467823, the "magic" channel):
    - sin_dmd_dz, cos_dmd_dz  (well dip)
    - sin_dx_dy, cos_dx_dy    (geology direction)
  Formation residuals (per-formation Z gap, 6 features):
    - z_minus_ancc, z_minus_astnu, z_minus_astnl,
      z_minus_egfdu, z_minus_egfdl, z_minus_buda
  GR (32% NaN handled):
    - gr_smooth_51, gr_std_20
    - gr_rolling_mean_{5,21,51,101} (4)
    - gr_rolling_std_{5,21,51,101}  (4)
    - gr_diff_from_last
  Per-well static:
    - last_known_tvt, last_known_z, last_known_gr
    - n_known_rows, n_lateral_rows (well shape)

Target: tvt[lat] - last_known_tvt
Eval: per-well RMSE (mean of well-level rmses) — apples-to-apples with our 13.67 baseline.
Plus flat RMSE (matches Kaggle).
"""
import os, sys, json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
OUT_DIR  = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FORMATION_COLS = ["ANCC", "ASTNU", "ASTNL", "EGFDU", "EGFDL", "BUDA"]
ROLLING_WINS = [5, 21, 51, 101]


def safe_savgol(x, win=51, order=2):
    if len(x) <= win:
        return x.copy()
    return savgol_filter(x, win, order)


def extract_well(wid):
    """Per-well: extract per-lateral-row feature dict. Returns DataFrame or None."""
    try:
        h = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
    except Exception:
        return None
    if len(h) < 50:
        return None

    md = h["MD"].values
    x = h["X"].values
    y = h["Y"].values
    z = h["Z"].values
    tvt_inp = h["TVT_input"].values
    tvt = h["TVT"].values
    gr_raw = h["GR"].values

    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0:
        return None
    known = ~mask_lat
    if known.sum() < 10:
        return None

    last_idx = np.flatnonzero(known)[-1]
    last_tvt = float(tvt_inp[last_idx])
    last_z = float(z[last_idx])
    last_md = float(md[last_idx])
    last_x = float(x[last_idx])
    last_y = float(y[last_idx])

    # GR clean + smooth
    gr_clean = pd.Series(gr_raw).interpolate(limit_direction="both").bfill().ffill().values
    if np.all(np.isnan(gr_clean)):
        gr_clean = np.zeros_like(z)
    gr_smooth51 = safe_savgol(gr_clean, 51, 2)
    last_gr = float(gr_smooth51[last_idx])

    # Rolling stats (computed on full series, used at lateral indices)
    gr_s = pd.Series(gr_clean)
    rolls = {}
    for w in ROLLING_WINS:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"gr_mean_{w}"] = r.mean().values
        rolls[f"gr_std_{w}"] = r.std().fillna(0).values

    # Geometric tangents — independent of TVT (no leak)
    dmd = np.gradient(md)
    dz = np.gradient(z)
    dx = np.gradient(x)
    dy = np.gradient(y)
    norm_md_z = np.sqrt(dmd**2 + dz**2) + 1e-8
    norm_x_y = np.sqrt(dx**2 + dy**2) + 1e-8
    sin_dmd_dz = dz / norm_md_z
    cos_dmd_dz = dmd / norm_md_z
    sin_dx_dy = dy / norm_x_y
    cos_dx_dy = dx / norm_x_y

    # cumsum(-dz) since last_known — the dominant physics signal
    neg_dz_from_last = np.zeros_like(z)
    for i in range(last_idx + 1, len(z)):
        neg_dz_from_last[i] = neg_dz_from_last[i-1] + (-(z[i] - z[i-1]))

    # Per-formation Z residuals
    form_residuals = {}
    for col in FORMATION_COLS:
        if col in h.columns:
            form_residuals[f"z_minus_{col.lower()}"] = (z - h[col].values).astype(np.float32)
        else:
            form_residuals[f"z_minus_{col.lower()}"] = np.zeros_like(z, dtype=np.float32)

    # Build per-row records for lateral rows only
    lat_idx_arr = np.flatnonzero(mask_lat)
    n_known = int(known.sum())
    n_lat = int(mask_lat.sum())

    records = []
    for r in lat_idx_arr:
        target = tvt[r] - last_tvt
        if np.isnan(target):
            continue  # some wells have NaN TVT in lateral (shouldn't but be safe)

        rec = {
            "well": wid,
            "row_idx": int(r),
            # Geometry
            "md_offset": float(md[r] - last_md),
            "z_rel": float(z[r] - last_z),
            "x_rel": float(x[r] - last_x),
            "y_rel": float(y[r] - last_y),
            "cumsum_neg_dz": float(neg_dz_from_last[r]),
            # Tangents
            "sin_dmd_dz": float(sin_dmd_dz[r]),
            "cos_dmd_dz": float(cos_dmd_dz[r]),
            "sin_dx_dy": float(sin_dx_dy[r]),
            "cos_dx_dy": float(cos_dx_dy[r]),
            # GR
            "gr_smooth": float(gr_smooth51[r]),
            "gr_diff_from_last": float(gr_smooth51[r] - last_gr),
            # Per-well static (LightGBM uses these as well-id-like signals)
            "last_known_tvt": last_tvt,
            "last_known_z": last_z,
            "last_known_gr": last_gr,
            "n_known_rows": n_known,
            "n_lateral_rows": n_lat,
            "row_position_norm": float((r - last_idx) / max(n_lat, 1)),
            # Target
            "target": float(target),
        }
        for k, arr in rolls.items():
            rec[k] = float(arr[r])
        for k, arr in form_residuals.items():
            rec[k] = float(arr[r])

        records.append(rec)

    return pd.DataFrame(records)


def main():
    t0 = time.time()

    all_wells = sorted({f.replace("__horizontal_well.csv","")
                        for f in os.listdir(DATA_DIR)
                        if f.endswith("__horizontal_well.csv")})
    print(f"Pool: {len(all_wells)} wells")

    print("\n[1/3] Extracting features...")
    dfs = []
    n_fail = 0
    for i, wid in enumerate(all_wells):
        df_w = extract_well(wid)
        if df_w is None or len(df_w) == 0:
            n_fail += 1
            continue
        dfs.append(df_w)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(all_wells)} | dfs={len(dfs)} fails={n_fail}")
    df = pd.concat(dfs, ignore_index=True)
    print(f"  done in {time.time()-t0:.1f}s | rows={len(df):,} wells={df['well'].nunique()} fails={n_fail}")
    df.to_parquet(OUT_DIR / "features_full.parquet")

    feature_cols = [c for c in df.columns if c not in {"well", "row_idx", "target"}]
    print(f"  features ({len(feature_cols)}): {feature_cols}\n")

    # ── Baseline check: physics + last_known only ──
    print("[2/3] Baseline reference (no model):")
    # last_known_tvt itself = predict last_known (target=0 always) → RMSE = sqrt(mean(target^2))
    flat_baseline = float(root_mean_squared_error(df["target"], np.zeros(len(df))))
    cum_baseline = float(root_mean_squared_error(df["target"], df["cumsum_neg_dz"]))
    print(f"  last_known         (predict target=0)  : flat RMSE = {flat_baseline:.3f}")
    print(f"  last_known + cumsum(-dz)               : flat RMSE = {cum_baseline:.3f}")

    # Per-well baselines
    def perwell(pred, label):
        out = []
        for w, g in df.groupby("well"):
            idx = g.index.values
            r = float(np.sqrt(np.mean((g["target"].values - pred[idx])**2)))
            out.append(r)
        arr = np.array(out)
        print(f"  {label:50s} per-well: mean={arr.mean():.2f} median={np.median(arr):.2f}")
        return arr
    perwell(np.zeros(len(df)), "last_known")
    perwell(df["cumsum_neg_dz"].values, "last_known + cumsum(-dz)")

    # ── LightGBM CV ──
    print("\n[3/3] LightGBM GroupKFold-5...")
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(df))
    fold_rmses = []
    feature_importances = np.zeros(len(feature_cols))

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df["target"], groups=df["well"])):
        X_tr = df.iloc[tr_idx][feature_cols]
        X_va = df.iloc[va_idx][feature_cols]
        y_tr = df.iloc[tr_idx]["target"]
        y_va = df.iloc[va_idx]["target"]

        model = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.02, num_leaves=127,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
            colsample_bytree=0.8, subsample=0.85, subsample_freq=5,
            verbose=-1, n_jobs=-1,
        )
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = model.predict(X_va)
        oof[va_idx] = pred
        feature_importances += model.feature_importances_
        rmse = float(root_mean_squared_error(y_va, pred))
        fold_rmses.append(rmse)
        n_tr_wells = df.iloc[tr_idx]["well"].nunique()
        n_va_wells = df.iloc[va_idx]["well"].nunique()
        print(f"  fold {fold+1}: RMSE={rmse:.3f} | "
              f"train={n_tr_wells}w/{len(tr_idx):,}r val={n_va_wells}w/{len(va_idx):,}r | "
              f"best_iter={model.best_iteration_}")

    overall_flat = float(root_mean_squared_error(df["target"], oof))
    print(f"\n  OOF flat RMSE: {overall_flat:.3f}  | fold-mean: {np.mean(fold_rmses):.3f}")
    perwell_oof = perwell(oof, "LightGBM OOF")

    # Feature importance ranking
    feature_importances /= 5
    fi = sorted(zip(feature_cols, feature_importances), key=lambda x: -x[1])
    print("\n  Top-15 feature importances:")
    for f, v in fi[:15]:
        print(f"    {f:40s} {v:.0f}")

    # Save metrics
    metrics = {
        "n_wells": int(df["well"].nunique()),
        "n_rows": int(len(df)),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "baseline_last_known_flat": flat_baseline,
        "baseline_last_known_plus_cumsum_dz_flat": cum_baseline,
        "oof_flat_rmse": overall_flat,
        "oof_perwell_mean": float(perwell_oof.mean()),
        "oof_perwell_median": float(np.median(perwell_oof)),
        "fold_rmses": fold_rmses,
        "feature_importances_avg": dict(zip(feature_cols, feature_importances.tolist())),
        "wall_time_sec": round(time.time()-t0, 1),
    }
    with open(OUT_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # Save oof for later ensemble
    df_oof = df[["well", "row_idx", "target"]].copy()
    df_oof["oof"] = oof
    df_oof.to_parquet(OUT_DIR / "oof_preds.parquet")

    print(f"\n=== Summary (target = TVT - last_known_tvt) ===")
    print(f"  baseline last_known         : flat {flat_baseline:.2f}")
    print(f"  baseline + cumsum(-dz)      : flat {cum_baseline:.2f}")
    print(f"  LightGBM OOF                : flat {overall_flat:.2f} | per-well {perwell_oof.mean():.2f}")
    print(f"\n=== Comparison ===")
    print(f"  Our SegFormer cfg-img-medium (curated val, biased): raw 15.84 / anc 13.67")
    print(f"  Public PF baseline LB                              : ~8.86")
    print(f"  This R8 LGB (full 723 wells, GroupKFold-5)         : {perwell_oof.mean():.2f} (per-well)")
    print(f"\n  → {OUT_DIR}/metrics.json")


if __name__ == "__main__":
    main()
