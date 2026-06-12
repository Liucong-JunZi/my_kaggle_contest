"""R8 Phase 11: LGB+CAT with multi-PN PF + spatial neighbor features.

Joins all features:
  - Phase 1 base (no leak)
  - PF (single, ANCC + Z)
  - Beam (14-config ensemble stats)
  - Multi-PN PF (4 scales)
  - Spatial neighbors (K=5, no hard cutoff)

Compares Phase 5 (no multi-PN, no spatial), Phase 9 (multi-PN), Phase 11 (all).
"""
import json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb
from catboost import CatBoostRegressor

OUT_DIR = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
LEAK_COLS = {"z_minus_ancc","z_minus_astnu","z_minus_astnl",
             "z_minus_egfdu","z_minus_egfdl","z_minus_buda"}
PN_LABELS = ["pn005","pn010","pn030","pn080"]


def perwell(df, pred, label):
    rmses = []
    for w, g in df.groupby("well"):
        idx = g.index.values
        rmses.append(np.sqrt(np.mean((g["target"].values - pred[idx])**2)))
    arr = np.array(rmses)
    print(f"  {label:40s} per-well mean={arr.mean():.2f} median={np.median(arr):.2f}")
    return arr


def main():
    t0 = time.time()
    print("=== R8 Phase 11: All features (P9 + spatial) ===\n")
    print("[1/5] Load + join all features")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features.parquet")
    bm   = pd.read_parquet(OUT_DIR/"beam_features.parquet")
    pfm  = pd.read_parquet(OUT_DIR/"pf_multiscale.parquet")
    sp   = pd.read_parquet(OUT_DIR/"spatial_features.parquet")
    df = (base.merge(pf,  on=["well","row_idx"], how="left")
              .merge(bm,  on=["well","row_idx"], how="left")
              .merge(pfm, on=["well","row_idx"], how="left")
              .merge(sp,  on=["well","row_idx"], how="left"))
    df = df.dropna(subset=["pf_ancc","beam_mean","pf_z_pn005"]).reset_index(drop=True)
    # Spatial may have NaN for isolated wells — keep them, GBDT handles NaN

    # PF + beam + multi-PN derived (same as Phase 9)
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5*(df["pf_ancc_offset"] + df["pf_z_offset"])
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]
    for pn in PN_LABELS:
        df[f"pf_z_{pn}_offset"] = df[f"pf_z_{pn}"] - df["last_known_tvt"]
    pn_offset_cols = [f"pf_z_{pn}_offset" for pn in PN_LABELS]
    df["pf_pn_span"]  = df[pn_offset_cols].max(axis=1) - df[pn_offset_cols].min(axis=1)
    df["pf_pn_mean"]  = df[pn_offset_cols].mean(axis=1)
    df["pf_pn_std"]   = df[pn_offset_cols].std(axis=1)
    df["pf_high_vs_default"] = df["pf_z_pn080_offset"] - df["pf_z_pn010_offset"]
    df["pf_low_vs_default"]  = df["pf_z_pn005_offset"] - df["pf_z_pn010_offset"]

    # Spatial-derived (relative to last_known_tvt)
    # NB: neighbor_tvt_median is in absolute TVT; offset is mostly noise but std is signal
    df["neighbor_tvt_offset"] = df["neighbor_tvt_median"] - df["last_known_tvt"]
    df["neighbor_count"] = df["neighbor_count"].astype(np.float32)  # was int8
    # log dist (raw is huge range)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df["neighbor_dist_log"] = np.log1p(df["neighbor_dist_min"].clip(0, 1e6).fillna(1e6))

    DROP = ({"well","row_idx","target",
             "pf_ancc","pf_z","beam_mean","beam_std","beam_med","beam_range",
             "beam_cons","beam_sm5",
             "neighbor_tvt_median","neighbor_tvt_mean","neighbor_dist_min"}
            | LEAK_COLS
            | {f"pf_z_{pn}" for pn in PN_LABELS})
    feat_cols = [c for c in df.columns if c not in DROP]
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("\n[2/5] LightGBM GKF-5")
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
    lgb_pw = perwell(df, lgb_oof, "LGB OOF (P11)")
    print(f"  → flat {lgb_flat:.3f}\n")
    fis /= 5
    print("  Top-15 features:")
    for f, v in sorted(zip(feat_cols, fis), key=lambda x:-x[1])[:15]:
        print(f"    {f:35s} {v:.0f}")

    print("\n[3/5] CatBoost GKF-5")
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
    cat_pw = perwell(df, cat_oof, "CAT OOF (P11)")
    print(f"  → flat {cat_flat:.3f}\n")

    print("[4/5] Blends")
    best_blend = None; best_pw_mean = 1e9
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], blend)
        pw = perwell(df, blend, f"{w:.1f}·LGB + {1-w:.1f}·CAT")
        print(f"    flat {flat:.3f}")
        if pw.mean() < best_pw_mean:
            best_pw_mean = pw.mean(); best_blend = (w, blend, flat, pw)

    bw, bblend, bflat, bpw = best_blend
    print(f"\n[5/5] Compare progression:")
    print(f"  Phase 5 baseline (no multi-PN, no spatial):       per-well 10.07  flat 13.01  → LB 11.52")
    print(f"  Phase 9 (with multi-PN, no spatial):              per-well 10.05  flat 13.07")
    print(f"  Phase 11 LGB alone:                               per-well {lgb_pw.mean():.2f}  flat {lgb_flat:.3f}")
    print(f"  Phase 11 CAT alone:                               per-well {cat_pw.mean():.2f}  flat {cat_flat:.3f}")
    print(f"  Phase 11 best blend ({bw:.1f}·LGB+{1-bw:.1f}·CAT):  per-well {bpw.mean():.2f}  flat {bflat:.3f}")

    out_df = df[["well","row_idx","target"]].copy()
    out_df["lgb_oof_p11"] = lgb_oof
    out_df["cat_oof_p11"] = cat_oof
    out_df["blend_p11"]   = bblend
    out_df.to_parquet(OUT_DIR / "oof_phase11.parquet")
    print(f"\nwall: {time.time()-t0:.0f}s")
    print(f"→ {OUT_DIR}/oof_phase11.parquet")


if __name__ == "__main__":
    main()
