"""R8 Phase 14B: GKF-5 OOF on Phase 5 stack + multi-seed PF ensemble features.

Compares:
  Phase 5 baseline   : per-well 10.07, flat 13.01  (single-seed PF + beam, 36 feats)
  v8 Kaggle (LB)     : per-well 9.91 / flat 12.95  (P5 ML + heuristic 0.75/0.25 PF blend → LB 11.383)
  Phase 14B (this)   : P5 36 feats + 5 PF-ensemble features (s3/s5/s8/s12/mean)

The new pf_ens_* columns are *raw TVT*, like pf_ancc. We add them as offsets
(`pf_ens_s12_offset = pf_ens_s12 - last_known_tvt`). They're orthogonal to
single-seed pf_ancc because they come from likelihood-weighted seed mixtures.
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
ENS_COLS  = ["pf_ens_s3","pf_ens_s5","pf_ens_s8","pf_ens_s12","pf_ens_mean"]


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
    print("=== R8 Phase 14B: P5 stack + 16-seed PF likelihood ensemble ===\n")

    print("[1/4] Load + join")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features.parquet")          # original single-seed
    bm   = pd.read_parquet(OUT_DIR/"beam_features.parquet")
    ens  = pd.read_parquet(OUT_DIR/"pf_ensemble.parquet")          # NEW: 16-seed ensemble

    df = (base.merge(pf,  on=["well","row_idx"], how="left")
              .merge(bm,  on=["well","row_idx"], how="left")
              .merge(ens, on=["well","row_idx"], how="left"))
    df = df.dropna(subset=["pf_ancc","beam_mean","pf_ens_s12"]).reset_index(drop=True)

    # Standard P5 derived features
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5*(df["pf_ancc_offset"] + df["pf_z_offset"])
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]

    # NEW: ensemble offsets + disagreement
    for c in ENS_COLS:
        df[f"{c}_offset"] = df[c] - df["last_known_tvt"]
    df["pf_ens_vs_ancc"]   = df["pf_ens_s12_offset"] - df["pf_ancc_offset"]
    df["pf_ens_scale_disag"] = df["pf_ens_s3_offset"] - df["pf_ens_s12_offset"]

    DROP = ({"well","row_idx","target",
             "pf_ancc","pf_z","beam_mean","beam_std","beam_med","beam_range",
             "beam_cons","beam_sm5"} | set(ENS_COLS) | LEAK_COLS)
    feat_cols = [c for c in df.columns if c not in DROP]
    new_feats = [c for c in feat_cols if c.startswith("pf_ens")]
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}")
    print(f"  new ensemble features ({len(new_feats)}): {new_feats}\n")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("[2/4] LightGBM GKF-5")
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

    fi = pd.DataFrame({"feat": feat_cols, "imp": m.booster_.feature_importance(importance_type="gain")})
    fi = fi.sort_values("imp", ascending=False)
    print("  Top-15 LGB importances (last fold):")
    for _, row in fi.head(15).iterrows():
        print(f"    {row['feat']:30s} {row['imp']:>14.0f}")
    print()

    print("[3/4] CatBoost GKF-5")
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

    print("[4/4] ML blend + heuristic blends")
    best_w_ml, best_pw_ml = None, 1e9
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        ml = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], ml)
        pw = perwell(df, ml, f"{w:.1f}·LGB + {1-w:.1f}·CAT")
        print(f"    flat {flat:.3f}")
        if pw.mean() < best_pw_ml:
            best_pw_ml = pw.mean(); best_w_ml = w

    ml_best = best_w_ml*lgb_oof + (1-best_w_ml)*cat_oof
    ml_tvt  = df["last_known_tvt"].values + ml_best
    pf_anchor_tvt = df["pf_ancc"].values
    pf_ens_tvt    = df["pf_ens_s12"].values   # try ensemble in heuristic blend too

    print(f"\n  Best ML: w_lgb={best_w_ml:.2f}  per-well {best_pw_ml:.3f}")
    print("\n  Heuristic blend with single-seed PF (pf_ancc):")
    for a in [0.10, 0.15, 0.20, 0.25, 0.30]:
        blend = (1-a)*ml_tvt + a*pf_anchor_tvt
        offset = blend - df["last_known_tvt"].values
        flat = root_mean_squared_error(df["target"], offset)
        perwell(df, offset, f"{1-a:.2f}·ML + {a:.2f}·PF_ancc")
        print(f"    flat {flat:.3f}")
    print("\n  Heuristic blend with PF ensemble (pf_ens_s12):")
    for a in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        blend = (1-a)*ml_tvt + a*pf_ens_tvt
        offset = blend - df["last_known_tvt"].values
        flat = root_mean_squared_error(df["target"], offset)
        perwell(df, offset, f"{1-a:.2f}·ML + {a:.2f}·PF_ens_s12")
        print(f"    flat {flat:.3f}")

    print("\n=== Comparison ===")
    print(f"  Phase 5 (single-seed only)  : per-well 10.07  flat 13.01")
    print(f"  v8 (P5 + 0.75/0.25 blend)   : per-well 9.91   flat 12.95  (LB 11.383)")
    print(f"  Phase 14B (ML only)         : per-well {best_pw_ml:.3f}  flat {root_mean_squared_error(df['target'], ml_best):.3f}")
    print(f"\n  wall: {time.time()-t0:.0f}s")

    out_df = df[["well","row_idx","target","last_known_tvt","pf_ancc","pf_ens_s12"]].copy()
    out_df["lgb_oof"] = lgb_oof
    out_df["cat_oof"] = cat_oof
    out_df.to_parquet(OUT_DIR / "oof_phase14.parquet")
    print(f"  → {OUT_DIR}/oof_phase14.parquet")

    metrics = {
        "n_features": len(feat_cols),
        "new_features": new_feats,
        "lgb": {"flat": float(lgb_flat),
                "perwell_mean": float(lgb_pw.mean()),
                "perwell_median": float(np.median(lgb_pw))},
        "cat": {"flat": float(cat_flat),
                "perwell_mean": float(cat_pw.mean()),
                "perwell_median": float(np.median(cat_pw))},
        "best_ml_w_lgb": best_w_ml,
        "best_ml_perwell": float(best_pw_ml),
        "wall_sec": round(time.time()-t0, 1),
    }
    with open(OUT_DIR / "metrics_phase14.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
