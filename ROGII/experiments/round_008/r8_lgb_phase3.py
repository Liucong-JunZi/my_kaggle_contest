"""R8 Phase 3B: PF + Beam + LightGBM stack (37+5 = 42 feats).

Loads Phase-1 features (31) + PF features (6 derived) + Beam features
(7 raw) → checks beam-alone sanity → fits LightGBM under GroupKFold-5.

Beam features (offsets relative to last_known_tvt, to match target):
  beam_mean_offset, beam_med_offset, beam_cons_offset, beam_sm5_offset
  beam_std, beam_range                                  (uncertainty proxies)
  beam_vs_pf  = beam_mean_offset - pf_mean_offset       (disagreement)
"""
import json, time, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import root_mean_squared_error
import lightgbm as lgb

OUT_DIR   = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
FEAT_FILE = OUT_DIR / "features_full.parquet"
PF_FILE   = OUT_DIR / "pf_features.parquet"
BEAM_FILE = OUT_DIR / "beam_features.parquet"


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
    print("=== R8 Phase 3B: PF + Beam + LightGBM ===\n")

    print("[1/4] Load + join all features ...")
    df = pd.read_parquet(FEAT_FILE)
    pf = pd.read_parquet(PF_FILE)
    bm = pd.read_parquet(BEAM_FILE)
    print(f"  base : {len(df):,} rows × {df.shape[1]} cols")
    print(f"  pf   : {len(pf):,} rows × {pf.shape[1]} cols")
    print(f"  beam : {len(bm):,} rows × {bm.shape[1]} cols")

    df = df.merge(pf, on=["well", "row_idx"], how="left")
    df = df.merge(bm, on=["well", "row_idx"], how="left")
    n_missing = df[["pf_ancc", "beam_mean"]].isna().any(axis=1).sum()
    if n_missing:
        df = df.dropna(subset=["pf_ancc", "beam_mean"]).reset_index(drop=True)
    print(f"  joined: {len(df):,} rows × {df.shape[1]} cols "
          f"| dropped {n_missing} | {df['well'].nunique()} wells")

    # PF-derived (mirror Phase 2)
    df["pf_ancc_offset"]  = df["pf_ancc"] - df["last_known_tvt"]
    df["pf_z_offset"]     = df["pf_z"]    - df["last_known_tvt"]
    df["pf_disagreement"] = df["pf_ancc"] - df["pf_z"]
    df["pf_mean_offset"]  = 0.5 * (df["pf_ancc_offset"] + df["pf_z_offset"])

    # Beam-derived
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]
    print()

    # ── [2/4] Standalone beam check ──
    print("[2/4] Beam standalone predictors:")
    for col in ["beam_mean_offset", "beam_med_offset",
                "beam_cons_offset", "beam_sm5_offset"]:
        pred = df[col].values
        flat = float(root_mean_squared_error(df["target"], pred))
        print(f"  {col:30s} flat RMSE = {flat:.3f}")
        perwell(df, pred, col)
    # 50/50 beam_mean + pf_mean
    blend = 0.5 * (df["beam_mean_offset"] + df["pf_mean_offset"]).values
    flat = float(root_mean_squared_error(df["target"], blend))
    print(f"\n  beam_mean+pf_mean 50/50        flat RMSE = {flat:.3f}")
    perwell(df, blend, "beam_mean+pf_mean 50/50")
    print()

    # ── [3/4] LightGBM with all features ──
    phase1_cols = [c for c in df.columns if c not in {
        "well", "row_idx", "target",
        "pf_ancc", "pf_ancc_std", "pf_z", "pf_z_std",
        "pf_ancc_offset", "pf_z_offset", "pf_disagreement", "pf_mean_offset",
        "beam_mean", "beam_std", "beam_med", "beam_range",
        "beam_cons", "beam_sm5",
        "beam_mean_offset", "beam_med_offset", "beam_cons_offset",
        "beam_sm5_offset", "beam_vs_pf",
    }]
    pf_cols = ["pf_ancc_offset", "pf_z_offset", "pf_ancc_std", "pf_z_std",
               "pf_disagreement", "pf_mean_offset"]
    beam_cols = ["beam_mean_offset", "beam_med_offset",
                 "beam_cons_offset", "beam_sm5_offset",
                 "beam_std", "beam_range", "beam_vs_pf"]

    feat_sets = {
        "P2 (31+6 PF)":        phase1_cols + pf_cols,
        "P3 (31+6 PF+7 Beam)": phase1_cols + pf_cols + beam_cols,
    }

    results = {}
    for label, feature_cols in feat_sets.items():
        print(f"[3/4] LightGBM GKF-5: {label} ({len(feature_cols)} feats)")
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
            print(f"  fold {fold+1}: RMSE={r:.3f} | best_iter={model.best_iteration_}",
                  flush=True)
        fi /= 5
        oof_flat = float(root_mean_squared_error(df["target"], oof))
        oof_pw = perwell(df, oof, "OOF")
        print(f"  → flat {oof_flat:.3f} | fold-mean {np.mean(fold_rmses):.3f}\n")
        results[label] = {
            "oof_flat": oof_flat,
            "perwell_mean":   float(oof_pw.mean()),
            "perwell_median": float(np.median(oof_pw)),
            "fold_rmses": fold_rmses,
            "top_importances": sorted(
                zip(feature_cols, fi.tolist()), key=lambda x: -x[1]
            )[:25],
        }
        results[label + "_oof"] = oof

    # ── [4/4] Compare ──
    print("=== Final comparison (per-well mean RMSE) ===")
    print(f"  Predict last_known        : 12.81")
    print(f"  Phase 1 (31)              : 10.29")
    for label in feat_sets:
        r = results[label]
        print(f"  {label:35s} : {r['perwell_mean']:.2f}  "
              f"(flat {r['oof_flat']:.2f}, median {r['perwell_median']:.2f})")
    print(f"  Public PF baseline LB     : ~8.86")
    print(f"  LB top 1                  : ~5.99")

    metrics = {label: {k: v for k, v in r.items()}
               for label, r in results.items() if not label.endswith("_oof")}
    metrics["wall_time_sec"] = round(time.time() - t0, 1)
    with open(OUT_DIR / "metrics_phase3.json", "w") as f:
        json.dump(metrics, f, indent=2)
    df_oof = df[["well", "row_idx", "target"]].copy()
    df_oof["oof_phase3"] = results["P3 (31+6 PF+7 Beam)_oof"]
    df_oof.to_parquet(OUT_DIR / "oof_phase3.parquet")
    print(f"\n→ {OUT_DIR}/metrics_phase3.json")

    print("\nTop-25 feature importance (P3):")
    for f_, v in results["P3 (31+6 PF+7 Beam)"]["top_importances"]:
        print(f"  {f_:40s} {v:.0f}")


if __name__ == "__main__":
    main()
