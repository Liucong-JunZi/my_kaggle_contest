"""R8 Phase 2B: PF + LightGBM stack.

Joins PF predictions (results/round_008/pf_features.parquet) with the
Phase-1 31-feature matrix (features_full.parquet), evaluates PF alone as a
sanity check (should beat per-well 10.29), then refits LightGBM with the
augmented feature set under the same GroupKFold-5 protocol.

Key feature engineering on the PF outputs:
  pf_ancc_offset    = pf_ancc - last_known_tvt    (relative, like target)
  pf_z_offset       = pf_z    - last_known_tvt
  pf_disagreement   = pf_ancc - pf_z              (uncertainty proxy)
  + pf_ancc_std, pf_z_std (raw)
"""
import json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

OUT_DIR = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
FEAT_FILE = OUT_DIR / "features_full.parquet"
PF_FILE   = OUT_DIR / "pf_features.parquet"


def perwell(df, pred, label):
    out = []
    for w, g in df.groupby("well"):
        idx = g.index.values
        r = float(np.sqrt(np.mean((g["target"].values - pred[idx]) ** 2)))
        out.append(r)
    arr = np.array(out)
    print(f"  {label:50s} per-well: mean={arr.mean():.2f} median={np.median(arr):.2f}")
    return arr


def main():
    t0 = time.time()
    print("=== R8 Phase 2B: PF + LightGBM ===\n")

    print("[1/4] Load features + PF preds ...")
    df = pd.read_parquet(FEAT_FILE)
    pf = pd.read_parquet(PF_FILE)
    print(f"  feat : {len(df):,} rows × {df.shape[1]} cols, {df['well'].nunique()} wells")
    print(f"  pf   : {len(pf):,} rows × {pf.shape[1]} cols, {pf['well'].nunique()} wells")

    # Join on (well, row_idx)
    df = df.merge(pf, on=["well", "row_idx"], how="left")
    n_missing = df[["pf_ancc", "pf_z"]].isna().any(axis=1).sum()
    print(f"  joined: {len(df):,} rows | rows missing PF: {n_missing:,}")
    if n_missing:
        # Drop them; downstream LGB needs clean PF
        df = df.dropna(subset=["pf_ancc", "pf_z"]).reset_index(drop=True)
        print(f"  after drop: {len(df):,} rows, {df['well'].nunique()} wells")

    # Derived PF features (all relative, no leak)
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5 * (df["pf_ancc_offset"] + df["pf_z_offset"])

    print()

    # ── [2/4] Sanity check: PF alone as predictor ──
    print("[2/4] PF as standalone predictor (no model):")
    for name, col in [
        ("pf_ancc_offset",  "pf_ancc_offset"),
        ("pf_z_offset",     "pf_z_offset"),
        ("pf_mean_offset",  "pf_mean_offset"),
    ]:
        pred = df[col].values
        flat = float(root_mean_squared_error(df["target"], pred))
        print(f"  {name:30s} flat RMSE = {flat:.3f}")
        perwell(df, pred, name)
    print()

    # ── [3/4] LightGBM: same hp as Phase 1, augmented features ──
    phase1_cols = [c for c in df.columns if c not in {
        "well", "row_idx", "target",
        "pf_ancc", "pf_ancc_std", "pf_z", "pf_z_std",
        "pf_ancc_offset", "pf_z_offset", "pf_disagreement", "pf_mean_offset",
    }]
    pf_cols = ["pf_ancc_offset", "pf_z_offset", "pf_ancc_std", "pf_z_std",
               "pf_disagreement", "pf_mean_offset"]

    feat_sets = {
        "Phase1 (31 feats)":        phase1_cols,
        "Phase2 (31 + 6 PF feats)": phase1_cols + pf_cols,
    }

    results = {}
    for label, feature_cols in feat_sets.items():
        print(f"[3/4] LightGBM GroupKFold-5: {label} ({len(feature_cols)} feats)")
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(df))
        fold_rmses = []
        fi = np.zeros(len(feature_cols))
        for fold, (tr_idx, va_idx) in enumerate(
            gkf.split(df, df["target"], groups=df["well"])
        ):
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
            fi += model.feature_importances_
            r = float(root_mean_squared_error(y_va, pred))
            fold_rmses.append(r)
            print(f"  fold {fold+1}: RMSE={r:.3f} | best_iter={model.best_iteration_}")
        fi /= 5
        oof_flat = float(root_mean_squared_error(df["target"], oof))
        oof_pw = perwell(df, oof, "OOF")
        print(f"  → flat RMSE {oof_flat:.3f} | fold-mean {np.mean(fold_rmses):.3f}\n")
        results[label] = {
            "oof_flat": oof_flat,
            "perwell_mean":   float(oof_pw.mean()),
            "perwell_median": float(np.median(oof_pw)),
            "fold_rmses": fold_rmses,
            "top_importances": sorted(
                zip(feature_cols, fi.tolist()), key=lambda x: -x[1]
            )[:20],
        }
        results[label + "_oof"] = oof

    # ── [4/4] Compare ──
    print("=== Final comparison (per-well mean RMSE) ===")
    print(f"  Predict last_known                : 12.81  (R8 P1 baseline)")
    for label in feat_sets:
        r = results[label]
        print(f"  {label:35s} : {r['perwell_mean']:.2f}  "
              f"(flat {r['oof_flat']:.2f}, median {r['perwell_median']:.2f})")
    print(f"  Public PF baseline LB             : ~8.86")
    print(f"  LB top 1                          : ~5.99")

    # Save metrics + OOF
    metrics = {label: {k: v for k, v in r.items() if k != "fold_rmses" or True}
               for label, r in results.items() if not label.endswith("_oof")}
    metrics["wall_time_sec"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "metrics_phase2.json", "w") as f:
        json.dump(metrics, f, indent=2)
    df_oof = df[["well", "row_idx", "target"]].copy()
    df_oof["oof_phase2"] = results["Phase2 (31 + 6 PF feats)_oof"]
    df_oof.to_parquet(OUT_DIR / "oof_phase2.parquet")
    print(f"\n→ {OUT_DIR}/metrics_phase2.json")

    # Show top-20 importance for Phase2
    print("\nTop-20 feature importance (Phase 2):")
    for f, v in results["Phase2 (31 + 6 PF feats)"]["top_importances"]:
        print(f"  {f:40s} {v:.0f}")


if __name__ == "__main__":
    main()
