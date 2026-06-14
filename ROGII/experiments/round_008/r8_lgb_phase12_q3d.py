"""R8 Phase 12: v4 36-feat stack + Q-3D tortuosity (43 feats).

Compare against Phase 5 (clean 36 feat → CV per-well 10.07 / flat 13.01)
to see if Q-3D adds anything. Forum claim: -0.107 RMSE.
"""
import time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

OUT = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
LEAK = {"z_minus_ancc","z_minus_astnu","z_minus_astnl",
        "z_minus_egfdu","z_minus_egfdl","z_minus_buda"}

def perwell(df, pred, label):
    rmses = []
    for w, g in df.groupby("well"):
        idx = g.index.values
        rmses.append(np.sqrt(np.mean((g["target"].values - pred[idx])**2)))
    a = np.array(rmses)
    print(f"  {label:50s} per-well={a.mean():.3f} med={np.median(a):.3f}")
    return a

def main():
    t0 = time.time()
    print("=== R8 Phase 12: v4 36-feat + Q-3D (43 feat) ===\n")
    base = pd.read_parquet(OUT/"features_full.parquet")
    pf   = pd.read_parquet(OUT/"pf_features.parquet")
    bm   = pd.read_parquet(OUT/"beam_features.parquet")
    q3d  = pd.read_parquet(OUT/"q3d_features.parquet")
    df = (base.merge(pf,on=["well","row_idx"],how="left")
              .merge(bm,on=["well","row_idx"],how="left")
              .merge(q3d,on=["well","row_idx"],how="left"))
    df = df.dropna(subset=["pf_ancc","beam_mean"]).reset_index(drop=True)
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5*(df["pf_ancc_offset"] + df["pf_z_offset"])
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]
    DROP = ({"well","row_idx","target",
             "pf_ancc","pf_z","beam_mean","beam_std","beam_med","beam_range",
             "beam_cons","beam_sm5"} | LEAK)
    feat_cols = [c for c in df.columns if c not in DROP]
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}")
    print(f"  Q-3D feats included: {[c for c in feat_cols if c.startswith('q3d_')]}\n")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("[1/3] LightGBM GKF-5")
    lgb_oof = np.zeros(len(df))
    fis = np.zeros(len(feat_cols))
    for fold, (tr,va) in enumerate(folds):
        m = lgb.LGBMRegressor(
            n_estimators=3000, learning_rate=0.02, num_leaves=127,
            min_child_samples=50, reg_alpha=0.1, reg_lambda=0.1,
            colsample_bytree=0.8, subsample=0.85, subsample_freq=5,
            verbose=-1, n_jobs=-1,
        )
        m.fit(df.iloc[tr][feat_cols], df.iloc[tr]["target"],
              eval_set=[(df.iloc[va][feat_cols], df.iloc[va]["target"])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        lgb_oof[va] = m.predict(df.iloc[va][feat_cols])
        fis += m.feature_importances_
        r = root_mean_squared_error(df.iloc[va]["target"], lgb_oof[va])
        print(f"  fold {fold+1}: RMSE={r:.3f} | best_iter={m.best_iteration_}", flush=True)
    lgb_flat = root_mean_squared_error(df["target"], lgb_oof)
    lgb_pw = perwell(df, lgb_oof, "LGB OOF (P12)")
    print(f"  → flat {lgb_flat:.3f}\n")
    fis /= 5
    print("  Top-15 features (with Q-3D rank):")
    ranked = sorted(zip(feat_cols, fis), key=lambda x:-x[1])
    for i, (f, v) in enumerate(ranked[:15]):
        print(f"    {i+1:2d}. {f:30s} {v:.0f}")
    print("  Q-3D feature ranks:")
    for i, (f, v) in enumerate(ranked):
        if f.startswith("q3d_"):
            print(f"    rank {i+1}: {f:30s} {v:.0f}")

    print("\n[2/3] CatBoost GKF-5")
    cat_oof = np.zeros(len(df))
    for fold, (tr,va) in enumerate(folds):
        m = CatBoostRegressor(
            iterations=3000, learning_rate=0.05, depth=8,
            l2_leaf_reg=3.0, subsample=0.85, rsm=0.8,
            early_stopping_rounds=100,
            loss_function="RMSE", eval_metric="RMSE",
            verbose=False, thread_count=-1, random_seed=42,
            bootstrap_type="Bernoulli",
        )
        m.fit(df.iloc[tr][feat_cols], df.iloc[tr]["target"],
              eval_set=(df.iloc[va][feat_cols], df.iloc[va]["target"]),
              use_best_model=True)
        cat_oof[va] = m.predict(df.iloc[va][feat_cols])
        r = root_mean_squared_error(df.iloc[va]["target"], cat_oof[va])
        print(f"  fold {fold+1}: RMSE={r:.3f} | best_iter={m.tree_count_}", flush=True)
    cat_flat = root_mean_squared_error(df["target"], cat_oof)
    cat_pw = perwell(df, cat_oof, "CAT OOF (P12)")
    print(f"  → flat {cat_flat:.3f}\n")

    print("[3/3] Blends (vs Phase 5: pw 10.07, flat 13.01)")
    best = None; best_pw = 1e9
    for w in [0.3,0.4,0.5,0.6,0.7]:
        bl = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], bl)
        pw = perwell(df, bl, f"{w:.1f}·LGB + {1-w:.1f}·CAT  flat={flat:.3f}")
        if pw.mean() < best_pw:
            best_pw = pw.mean(); best = (w, bl, flat, pw)
    bw, bbl, bflat, bpw = best

    # Now blend with PF (v8 final approach)
    print("\n[3b/3] Blend with PF heuristic (v8 final)")
    base_last = df["last_known_tvt"].values
    y_abs = df["target"].values + base_last
    pf_ancc_abs = df["pf_ancc_offset"].values + base_last  # = pf_ancc original
    ml_abs = bbl + base_last
    for w in [0.7,0.75,0.8,0.85,0.9,1.0]:
        p = w*ml_abs + (1-w)*pf_ancc_abs
        rm = []
        for wn, g in df.groupby("well"):
            idx = g.index.values
            rm.append(np.sqrt(np.mean((y_abs[idx]-p[idx])**2)))
        rm = np.array(rm)
        flat = np.sqrt(np.mean((y_abs - p)**2))
        print(f"  ML×{w:.2f} + PF_ancc×{1-w:.2f}: per-well={rm.mean():.3f} med={np.median(rm):.3f} flat={flat:.3f}")

    out_df = df[["well","row_idx","target"]].copy()
    out_df["lgb_oof_p12"] = lgb_oof
    out_df["cat_oof_p12"] = cat_oof
    out_df["blend_p12"]   = bbl
    out_df.to_parquet(OUT/"oof_phase12.parquet")
    print(f"\nwall: {time.time()-t0:.0f}s → {OUT}/oof_phase12.parquet")


if __name__ == "__main__":
    main()
