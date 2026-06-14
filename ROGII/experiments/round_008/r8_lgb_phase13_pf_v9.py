"""R8 Phase 13: GKF-5 OOF on the v4 36-feat clean stack with PF v9.

PF v9 changes (vs Phase 5's pf_features.parquet):
  - GR pre-interpolated over the FULL well (kernel pattern), not just ev rows
  - ANCC_IS = 0.3 → 4.5 (sp45 patch)
  - Returns cumulative log-likelihood (pf_ancc_ll, pf_z_ll) — usable as features

Compares to:
  Phase 5 (old PF):  per-well 10.07, flat 13.01
  v8 + PF blend:     per-well 9.91,  flat 12.95
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

OUT_DIR   = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
LEAK_COLS = {"z_minus_ancc","z_minus_astnu","z_minus_astnl",
             "z_minus_egfdu","z_minus_egfdl","z_minus_buda"}


def perwell(df, pred, label):
    rmses = []
    for w, g in df.groupby("well"):
        idx = g.index.values
        rmses.append(np.sqrt(np.mean((g["target"].values - pred[idx])**2)))
    arr = np.array(rmses)
    print(f"  {label:46s} per-well mean={arr.mean():.3f} median={np.median(arr):.3f}")
    return arr


def main():
    t0 = time.time()
    print("=== R8 Phase 13: PF v9 (GR pre-interp + sp45 + log-lik) ===\n")

    print("[1/4] Load + join features")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features_v9.parquet")
    bm   = pd.read_parquet(OUT_DIR/"beam_features.parquet")
    df = base.merge(pf, on=["well","row_idx"], how="left") \
             .merge(bm, on=["well","row_idx"], how="left")
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
             "beam_cons","beam_sm5"} | LEAK_COLS)
    feat_cols = [c for c in df.columns if c not in DROP]
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}")
    new_feats = [c for c in feat_cols if c.endswith("_ll")]
    print(f"  new PF features: {new_feats}")
    print(f"  pf_ancc_ll: med={df['pf_ancc_ll'].median():.1f}  std={df['pf_ancc_ll'].std():.1f}")
    print(f"  pf_z_ll   : med={df['pf_z_ll'].median():.1f}  std={df['pf_z_ll'].std():.1f}\n")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("[2/4] LightGBM GKF-5 (v4 hyperparams)")
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
    lgb_pw = perwell(df, lgb_oof, "LGB OOF")
    print(f"  → flat {lgb_flat:.3f}\n")

    # Save LGB feature importance
    fi = pd.DataFrame({"feat": feat_cols, "imp": m.booster_.feature_importance(importance_type="gain")})
    fi = fi.sort_values("imp", ascending=False)
    print("  Top-15 LGB importances (last fold):")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feat']:30s} {row['imp']:>12.0f}")
    print("  Bottom-5:")
    for _, row in fi.tail(5).iterrows():
        print(f"    {row['feat']:30s} {row['imp']:>12.0f}")
    print()

    print("[3/4] CatBoost GKF-5 (v4 hyperparams)")
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
    cat_pw = perwell(df, cat_oof, "CAT OOF")
    print(f"  → flat {cat_flat:.3f}\n")

    print("[4/4] Blends + PF heuristic blend")
    # ML blend sweep
    best_w_ml, best_pw_ml = None, 1e9
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ml = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], ml)
        pw = perwell(df, ml, f"{w:.1f}·LGB + {1-w:.1f}·CAT")
        print(f"    flat {flat:.3f}")
        if pw.mean() < best_pw_ml:
            best_pw_ml = pw.mean(); best_w_ml = w
    # Heuristic blend on best ML: ml_tvt = last_known + ml_offset; pf_tvt = pf_ancc
    ml_best = best_w_ml*lgb_oof + (1-best_w_ml)*cat_oof
    ml_tvt = df["last_known_tvt"].values + ml_best
    pf_tvt = df["pf_ancc"].values
    target_tvt = df["last_known_tvt"].values + df["target"].values
    print(f"\n  Best ML: w_lgb={best_w_ml:.2f}  per-well {best_pw_ml:.3f}")
    print("  PF heuristic blend on best ML:")
    for a in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35]:
        blend_tvt = (1-a)*ml_tvt + a*pf_tvt
        offset_pred = blend_tvt - df["last_known_tvt"].values
        flat = root_mean_squared_error(df["target"], offset_pred)
        pw = perwell(df, offset_pred, f"{1-a:.2f}·ML + {a:.2f}·PF_ancc")
        print(f"    flat {flat:.3f}")

    print("\n=== Comparison ===")
    print(f"  Phase  5 (old PF) : per-well 10.07  flat 13.01")
    print(f"  Phase 13 (PF v9)  : per-well {best_pw_ml:.3f} (ML only) flat {root_mean_squared_error(df['target'], ml_best):.3f}")
    print(f"\n  wall time: {time.time()-t0:.0f}s")

    # Save OOFs
    out_df = df[["well","row_idx","target","last_known_tvt","pf_ancc"]].copy()
    out_df["lgb_oof"] = lgb_oof
    out_df["cat_oof"] = cat_oof
    out_df.to_parquet(OUT_DIR / "oof_phase13.parquet")
    print(f"  → {OUT_DIR}/oof_phase13.parquet")

    metrics = {
        "n_features": len(feat_cols),
        "new_features": new_feats,
        "lgb":  {"flat": float(lgb_flat),
                 "perwell_mean": float(lgb_pw.mean()),
                 "perwell_median": float(np.median(lgb_pw))},
        "cat":  {"flat": float(cat_flat),
                 "perwell_mean": float(cat_pw.mean()),
                 "perwell_median": float(np.median(cat_pw))},
        "best_ml_w_lgb": best_w_ml,
        "best_ml_perwell": float(best_pw_ml),
        "wall_sec": round(time.time()-t0, 1),
    }
    with open(OUT_DIR / "metrics_phase13.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
