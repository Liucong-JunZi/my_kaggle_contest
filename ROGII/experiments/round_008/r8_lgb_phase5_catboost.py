"""R8 Phase 5: CatBoost + Ridge blend on the clean 36-feature stack.

Goal: stack diverse-bias model on top of LightGBM OOF predictions to
squeeze ~0.1-0.3 ft per-well RMSE.

Pipeline:
  1. Reload cached parquets (Phase-1 base + PF + Beam), join, derive offsets
  2. Drop formation leak (z_minus_*) → 36 clean features
  3. CatBoost GKF-5 → cat_oof
  4. LightGBM GKF-5 (re-fit, identical to Phase-3 clean) → lgb_oof
  5. Ridge stack (target ~ cat_oof + lgb_oof) → ridge_oof
  6. Compare all three + simple 50/50 blend
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

OUT_DIR   = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
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
    print("=== R8 Phase 5: CatBoost + Ridge stack ===\n")

    print("[1/5] Load + join features")
    base = pd.read_parquet(OUT_DIR/"features_full.parquet")
    pf   = pd.read_parquet(OUT_DIR/"pf_features.parquet")
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
    print(f"  rows: {len(df):,}, wells: {df['well'].nunique()}, features: {len(feat_cols)}\n")

    gkf = GroupKFold(n_splits=5)
    folds = list(gkf.split(df, df["target"], groups=df["well"]))

    print("[2/5] LightGBM GKF-5 (reproduce clean stack)")
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

    print("[3/5] CatBoost GKF-5")
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

    print("[4/5] Blends")
    # Simple averages
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        blend = w*lgb_oof + (1-w)*cat_oof
        flat = root_mean_squared_error(df["target"], blend)
        perwell(df, blend, f"{w:.1f}·LGB + {1-w:.1f}·CAT")
        print(f"    flat {flat:.3f}")
    # Per-fold Ridge stack — fit Ridge inside each fold using ONLY train-fold OOFs
    # ⚠ Ridge stack must NOT use va-fold OOFs in its training set.
    print("\n[5/5] Per-fold Ridge stack on (lgb, cat) OOFs")
    ridge_oof = np.zeros(len(df))
    for fold, (tr,va) in enumerate(folds):
        X_st = np.column_stack([lgb_oof[tr], cat_oof[tr]])
        r = Ridge(alpha=1.0)
        r.fit(X_st, df.iloc[tr]["target"])
        X_va = np.column_stack([lgb_oof[va], cat_oof[va]])
        ridge_oof[va] = r.predict(X_va)
        print(f"  fold {fold+1}: w_lgb={r.coef_[0]:.3f} w_cat={r.coef_[1]:.3f} int={r.intercept_:.3f}")
    ridge_flat = root_mean_squared_error(df["target"], ridge_oof)
    ridge_pw = perwell(df, ridge_oof, "RIDGE STACK OOF")
    print(f"  → flat {ridge_flat:.3f}\n")

    print("=== Final ===")
    print(f"  Predict last_known        : per-well 12.81  flat 15.91")
    print(f"  LGB clean (P3)            : per-well {lgb_pw.mean():.2f}  flat {lgb_flat:.3f}")
    print(f"  CAT clean                 : per-well {cat_pw.mean():.2f}  flat {cat_flat:.3f}")
    print(f"  RIDGE(LGB,CAT) stack      : per-well {ridge_pw.mean():.2f}  flat {ridge_flat:.3f}")
    print(f"  Public PF baseline LB     : ~8.86  (corresponds to ~10 flat)")
    print(f"\n  wall time: {time.time()-t0:.0f}s")

    # Save OOFs for later 3-way stack
    out_df = df[["well","row_idx","target"]].copy()
    out_df["lgb_oof"]   = lgb_oof
    out_df["cat_oof"]   = cat_oof
    out_df["ridge_oof"] = ridge_oof
    out_df.to_parquet(OUT_DIR / "oof_phase5.parquet")
    print(f"  → {OUT_DIR}/oof_phase5.parquet")

    metrics = {
        "lgb":   {"flat": float(lgb_flat),
                  "perwell_mean": float(lgb_pw.mean()),
                  "perwell_median": float(np.median(lgb_pw))},
        "cat":   {"flat": float(cat_flat),
                  "perwell_mean": float(cat_pw.mean()),
                  "perwell_median": float(np.median(cat_pw))},
        "ridge": {"flat": float(ridge_flat),
                  "perwell_mean": float(ridge_pw.mean()),
                  "perwell_median": float(np.median(ridge_pw))},
        "wall_sec": round(time.time()-t0, 1),
    }
    with open(OUT_DIR / "metrics_phase5.json", "w") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
