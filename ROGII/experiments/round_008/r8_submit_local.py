"""R8 Submit: build submission CSV from clean stack (no formation leak).

Pipeline:
  1. Load clean OOF parquet to know exact features
  2. For each test well: regenerate base features + PF + Beam
  3. Train final LightGBM on all 723 train wells (full corpus, no holdout)
  4. Predict on test, add last_known_tvt back, write submission.csv

Test-safe: uses only [MD, X, Y, Z, GR, TVT_input] + typewell. Zero formation
columns. Run this against the 3-well sample test_dir first to verify, then
package as Kaggle notebook for real submission.
"""
import os, sys, time, warnings, json
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import lightgbm as lgb

# Reuse feature builders
sys.path.insert(0, "/Users/liucong/code/kaggle/ROGII/experiments/round_008")
from r8_pf_features   import run_pf_ancc, run_pf_z, _warmup as _pf_warmup
from r8_beam_features import run_beam_all, _warmup as _beam_warmup

TRAIN_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"
TEST_DIR  = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/test"
OUT_DIR   = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
OUT_DIR.mkdir(parents=True, exist_ok=True)

ROLLING_WINS = [5, 21, 51, 101]
# These cols are derived from formation columns (label leak); never include.
LEAK_COLS = {"z_minus_ancc","z_minus_astnu","z_minus_astnl",
             "z_minus_egfdu","z_minus_egfdl","z_minus_buda"}


def safe_savgol(x, win=51, order=2):
    if len(x) <= win: return x.copy()
    return savgol_filter(x, win, order)


def extract_base_features(wid, data_dir):
    """Per-well: extract per-lateral-row base features (no formation leak).

    Returns: (DataFrame rows, last_known_tvt). df has 25 columns matching
    the clean Phase-1 minus formation `z_minus_*`.
    """
    h = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
    if len(h) < 50: return None, None
    md = h["MD"].values; x = h["X"].values; y = h["Y"].values; z = h["Z"].values
    tvt_inp = h["TVT_input"].values
    gr_raw  = h["GR"].values

    mask_lat = np.isnan(tvt_inp)
    if mask_lat.sum() == 0: return None, None
    known = ~mask_lat
    if known.sum() < 10: return None, None

    last_idx = np.flatnonzero(known)[-1]
    last_tvt = float(tvt_inp[last_idx])
    last_z   = float(z[last_idx])
    last_md  = float(md[last_idx])
    last_x   = float(x[last_idx])
    last_y   = float(y[last_idx])

    gr_clean = (pd.Series(gr_raw).interpolate(limit_direction="both")
                                  .bfill().ffill().values)
    if np.all(np.isnan(gr_clean)): gr_clean = np.zeros_like(z)
    gr_smooth51 = safe_savgol(gr_clean, 51, 2)
    last_gr     = float(gr_smooth51[last_idx])

    gr_s = pd.Series(gr_clean)
    rolls = {}
    for w in ROLLING_WINS:
        r = gr_s.rolling(w, center=True, min_periods=1)
        rolls[f"gr_mean_{w}"] = r.mean().values
        rolls[f"gr_std_{w}"]  = r.std().fillna(0).values

    dmd = np.gradient(md); dz = np.gradient(z); dx = np.gradient(x); dy = np.gradient(y)
    nmz = np.sqrt(dmd**2 + dz**2) + 1e-8; nxy = np.sqrt(dx**2 + dy**2) + 1e-8
    sin_dmd_dz = dz/nmz; cos_dmd_dz = dmd/nmz
    sin_dx_dy  = dy/nxy; cos_dx_dy  = dx/nxy

    neg_dz_from_last = np.zeros_like(z)
    for i in range(last_idx + 1, len(z)):
        neg_dz_from_last[i] = neg_dz_from_last[i-1] + (-(z[i] - z[i-1]))

    lat_idx = np.flatnonzero(mask_lat)
    n_known = int(known.sum()); n_lat = int(mask_lat.sum())
    rows = []
    for r in lat_idx:
        rec = {
            "well": wid, "row_idx": int(r),
            "md_offset": float(md[r] - last_md),
            "z_rel": float(z[r] - last_z),
            "x_rel": float(x[r] - last_x),
            "y_rel": float(y[r] - last_y),
            "cumsum_neg_dz": float(neg_dz_from_last[r]),
            "sin_dmd_dz": float(sin_dmd_dz[r]),
            "cos_dmd_dz": float(cos_dmd_dz[r]),
            "sin_dx_dy":  float(sin_dx_dy[r]),
            "cos_dx_dy":  float(cos_dx_dy[r]),
            "gr_smooth":  float(gr_smooth51[r]),
            "gr_diff_from_last": float(gr_smooth51[r] - last_gr),
            "last_known_tvt": last_tvt,
            "last_known_z":   last_z,
            "last_known_gr":  last_gr,
            "n_known_rows":  n_known,
            "n_lateral_rows": n_lat,
            "row_position_norm": float((r - last_idx) / max(n_lat, 1)),
        }
        for k, arr in rolls.items(): rec[k] = float(arr[r])
        rows.append(rec)
    return pd.DataFrame(rows), last_tvt


def extract_pf(wid, data_dir):
    hw = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
    tw = pd.read_csv(f"{data_dir}/{wid}__typewell.csv")
    if hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 10:
        return None
    tw_s = tw.sort_values("TVT")
    tw_tvt = tw_s["TVT"].values.astype(np.float64)
    tw_gr  = tw_s["GR"].fillna(tw_s["GR"].mean()).values.astype(np.float64)
    np.random.seed(42)
    a, a_std = run_pf_ancc(hw, tw_tvt, tw_gr)
    z, z_std = run_pf_z(hw, tw_tvt, tw_gr)
    ev_idx = hw.index[hw["TVT_input"].isna()].values
    if len(ev_idx) != len(a): return None
    return pd.DataFrame({
        "well": wid, "row_idx": ev_idx.astype(np.int32),
        "pf_ancc": a, "pf_ancc_std": a_std,
        "pf_z":    z, "pf_z_std":    z_std,
    })


def extract_beam(wid, data_dir):
    hw = pd.read_csv(f"{data_dir}/{wid}__horizontal_well.csv")
    tw = pd.read_csv(f"{data_dir}/{wid}__typewell.csv")
    if hw["TVT_input"].notna().sum() < 10 or hw["TVT_input"].isna().sum() < 10:
        return None
    paths, ev_idx = run_beam_all(hw, tw)
    if paths is None: return None
    return pd.DataFrame({
        "well": wid, "row_idx": ev_idx.astype(np.int32),
        "beam_mean": paths.mean(axis=1).astype(np.float32),
        "beam_std":  paths.std(axis=1).astype(np.float32),
        "beam_med":  np.median(paths, axis=1).astype(np.float32),
        "beam_range": (paths.max(axis=1)-paths.min(axis=1)).astype(np.float32),
        "beam_cons": paths[:, 0],
        "beam_sm5":  paths[:, 3],
    })


def add_derived(df):
    """PF + Beam derived offsets / disagreements — match training pipeline."""
    df = df.copy()
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5 * (df["pf_ancc_offset"] + df["pf_z_offset"])
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]
    return df


def build_well_features(wid, data_dir):
    base, _ = extract_base_features(wid, data_dir)
    if base is None: return None
    pf = extract_pf(wid, data_dir)
    if pf is None: return None
    bm = extract_beam(wid, data_dir)
    if bm is None: return None
    df = base.merge(pf, on=["well","row_idx"]).merge(bm, on=["well","row_idx"])
    return add_derived(df)


def load_training_corpus():
    """Reload the precomputed clean stack via parquet (fast)."""
    print("  loading cached parquets ...")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features.parquet")
    bm   = pd.read_parquet(OUT_DIR/"beam_features.parquet")
    df = base.merge(pf, on=["well","row_idx"], how="left") \
             .merge(bm, on=["well","row_idx"], how="left")
    df = df.dropna(subset=["pf_ancc","beam_mean"]).reset_index(drop=True)
    df = add_derived(df)
    return df


def main():
    t0 = time.time()
    print("=== R8 Submit: clean LGB pipeline ===\n")

    print("[1/4] Warming up numba ...")
    np.random.seed(0); _pf_warmup(); _beam_warmup()

    print("\n[2/4] Loading training corpus + fitting final LGB on full 723 wells")
    train = load_training_corpus()
    # CLEAN feature list — must match r8_clean.log run
    DROP = {"well","row_idx","target",
            "pf_ancc","pf_z","beam_mean","beam_std","beam_med","beam_range",
            "beam_cons","beam_sm5"} | LEAK_COLS
    feat_cols = [c for c in train.columns if c not in DROP]
    print(f"  features ({len(feat_cols)}): {feat_cols}")
    print(f"  train rows: {len(train):,}, wells: {train['well'].nunique()}")

    # We don't have a held-out, fit on all 723 with a sensible n_estimators
    # (use mean best_iter from clean CV: ~1500 across folds).
    model = lgb.LGBMRegressor(
        n_estimators=2500, learning_rate=0.02, num_leaves=127,
        min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
        colsample_bytree=0.8, subsample=0.85, subsample_freq=5,
        verbose=-1, n_jobs=-1,
    )
    model.fit(train[feat_cols], train["target"])
    print(f"  fitted in {time.time()-t0:.0f}s")

    print("\n[3/4] Building features for test wells")
    t1 = time.time()
    test_wells = sorted({f.replace("__horizontal_well.csv","")
                         for f in os.listdir(TEST_DIR)
                         if f.endswith("__horizontal_well.csv")})
    print(f"  test wells: {len(test_wells)}")

    dfs = []
    for wid in test_wells:
        df_w = build_well_features(wid, TEST_DIR)
        if df_w is None:
            print(f"  ! {wid}: feature build failed"); continue
        dfs.append(df_w)
        print(f"  {wid}: {len(df_w):6d} rows")
    test = pd.concat(dfs, ignore_index=True)
    print(f"  total test rows: {len(test):,} | {time.time()-t1:.0f}s")

    print("\n[4/4] Predict + write submission")
    # Predict relative offset, add last_known_tvt back
    pred_offset = model.predict(test[feat_cols])
    pred_tvt    = test["last_known_tvt"].values + pred_offset
    sub = pd.DataFrame({
        "id":  test["well"] + "_" + test["row_idx"].astype(str),
        "tvt": pred_tvt.astype(np.float32),
    })

    # Compare to sample_submission.csv to confirm format
    sample = pd.read_csv(
        "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/sample_submission.csv"
    )
    print(f"  sample sub: {len(sample)} rows")
    print(f"  our    sub: {len(sub)} rows")
    if len(sub) != len(sample):
        missing = set(sample["id"]) - set(sub["id"])
        extra   = set(sub["id"]) - set(sample["id"])
        print(f"  ⚠ mismatch — missing {len(missing)}, extra {len(extra)}")
        if missing:
            print(f"    first missing: {list(missing)[:3]}")
    # Re-order to match sample
    sub = sample[["id"]].merge(sub, on="id", how="left")
    if sub["tvt"].isna().any():
        print(f"  ⚠ {sub['tvt'].isna().sum()} NaN preds — filling with last_known_tvt-ish 0")
        sub["tvt"] = sub["tvt"].fillna(0.0)

    sub_path = OUT_DIR / "submission_sample.csv"
    sub.to_csv(sub_path, index=False)
    print(f"\n  → {sub_path}")
    print(f"  preview:")
    print(sub.head(8).to_string(index=False))
    print(f"  ...")
    print(sub.tail(3).to_string(index=False))
    print(f"\n  pred stats: min={pred_tvt.min():.1f}, max={pred_tvt.max():.1f}, "
          f"mean={pred_tvt.mean():.1f}, median={np.median(pred_tvt):.1f}")
    print(f"  wall time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
