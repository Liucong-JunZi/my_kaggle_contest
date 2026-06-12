"""R8 Phase 7: LGB + CAT + Ridge with NCC features added.

Joins:
  - Phase-1 base features (without leak z_minus_*)
  - PF features
  - Beam features
  - NCC features (new in Phase 7)

Compares:
  - Phase 5 baseline (LGB+CAT 0.4/0.6 blend on 36 feats)
  - Phase 7 (same blend on 36 + NCC features)
"""
import json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
from sklearn.linear_model import Ridge
import lightgbm as lgb
from catboost import CatBoostRegressor

OUT_DIR = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
LEAK_COLS = {"z_minus_ancc","z_minus_astnu","z_minus_astnl",
             "z_minus_egfdu","z_minus_egfdl","z_minus_buda"}


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
    print("=== R8 Phase 7: LGB+CAT with NCC features ===\n")

    print("[1/5] Load + join all features (base + PF + Beam + NCC)")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features.parquet")
    bm   = pd.read_parquet(OUT_DIR/"beam_features.parquet")
    nc   = pd.read_parquet(OUT_DIR/"ncc_features.parquet")
    df = base.merge(pf, on=["well","row_idx"], how="left") \
             .merge(bm, on=["well","row_idx"], how="left") \
             .merge(nc, on=["well","row_idx"], how="left")
    df = df.dropna(subset=["pf_ancc","beam_mean","ncc_ens"]).reset_index(drop=True)

    # PF-derived
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5*(df["pf_ancc_offset"] + df["pf_z_offset"])
    # Beam-derived
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]
    # NCC-derived
    df["ncc8_offset"]   = df["ncc8"]   - df["last_known_tvt"]
    df["ncc15_offset"]  = df["ncc15"]  - df["last_known_tvt"]
    df["ncc25_offset"]  = df["ncc25"]  - df["last_known_tvt"]
    df["ncc_ens_offset"] = df["ncc_ens"] - df["last_known_tvt"]
    df["ncc_vs_pf"]      = df["ncc_ens_offset"] - df["pf_mean_offset"]
    df["ncc_vs_beam"]    = df["ncc_ens_offset"] - df["beam_mean_offset"]

    DROP = ({"well","row_idx","target",
             "pf_ancc","pf_z","beam_mean","beam_std","beam_med","beam_range",
             "beam_cons","beam_sm5",
             "ncc8","ncc15","ncc25","ncc_ens"} | LEAK_COLS)
    feat_cols = [c for c in df.columns if c not in DROP]
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}")

    # NCC-only check
    print("\n[2/5] NCC alone (no model):")
    for col in ["ncc_ens_offset","ncc8_offset","ncc15_offset","ncc25_offset"]:
        pred = df[col].values
        flat = float(root_mean_squared_error(df["target"], pred))
        print(f"  {col:25s} flat RMSE = {flat:.2f}")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("\n[3/5] LightGBM GKF-5 with NCC")
    lgb_oof = np.zeros(len(df))
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
        r = root_mean_squared_error(df.iloc[va]["target"], lgb_oof[va])
        print(f"  fold {fold+1}: RMSE={r:.3f} | best_iter={m.best_iteration_}", flush=True)
    lgb_flat = root_mean_squared_error(df["target"], lgb_oof)
    lgb_pw = perwell(df, lgb_oof, "LGB OOF (P7)")
    print(f"  → flat {lgb_flat:.3f}\n")

    print("[4/5] CatBoost GKF-5 with NCC")
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
    cat_pw = perwell(df, cat_oof, "CAT OOF (P7)")
    print(f"  → flat {cat_flat:.3f}\n")

    print("[5/5] Blends")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], blend)
        pw = perwell(df, blend, f"{w:.1f}·LGB + {1-w:.1f}·CAT")
        print(f"    flat {flat:.3f}")

    print("\n=== Compare to Phase 5 (no NCC) ===")
    print(f"  Phase 5: 0.4·LGB + 0.6·CAT       per-well 10.07  flat 13.01")
    print(f"  Phase 7: LGB alone                per-well {lgb_pw.mean():.2f}  flat {lgb_flat:.3f}")
    print(f"  Phase 7: CAT alone                per-well {cat_pw.mean():.2f}  flat {cat_flat:.3f}")

    # Save oof + best blend
    out_df = df[["well","row_idx","target"]].copy()
    out_df["lgb_oof_p7"] = lgb_oof
    out_df["cat_oof_p7"] = cat_oof
    out_df.to_parquet(OUT_DIR / "oof_phase7.parquet")
    print(f"\nwall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
