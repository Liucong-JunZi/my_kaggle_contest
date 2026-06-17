#!/usr/bin/env python3
"""Import 5 ravaghi pretrained models' OOF as round_010 hill-climb candidates.

Each ravaghi koolbox Trainer has m.oof_preds (5-fold GroupKFold). We wrap them
into our standard candidates parquet schema. Note: ravaghi's fold split (sklearn
default GroupKFold) differs from our sha256-hash fold map — but since hillclimb.py
is fold-aware (each fold N hill-climbs on the OTHER 4 folds), the OOF rows are still
honest predictions for those rows from ravaghi's perspective.

Output:
  results/candidates/c50_ravaghi_lgb1.parquet  (perwell ~8.26)
  results/candidates/c51_ravaghi_lgb2.parquet  (perwell ~8.06)
  results/candidates/c52_ravaghi_lgb3.parquet  (perwell ~8.05)
  results/candidates/c53_ravaghi_cat1.parquet  (perwell ~8.19)
  results/candidates/c54_ravaghi_cat2.parquet  (perwell ~8.20)
"""
import sys, time, joblib
from pathlib import Path
import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.metrics import perwell_rmse, flat_rmse
from shared.oof_writer import write_oof

RAVAGHI_DIR = Path("/Users/liucong/code/kaggle/ROGII/experiments/public_resources/datasets_raw/ravaghi-artifacts")

MODELS = [
    ("c50_ravaghi_lgb1", "lgbm", RAVAGHI_DIR / "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl",
     {"lr": 0.030, "num_leaves": 255, "n_est": 5000, "source": "ravaghi/wellbore-geology-prediction-artifacts"}),
    ("c51_ravaghi_lgb2", "lgbm", RAVAGHI_DIR / "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl",
     {"lr": 0.00934, "num_leaves": 64, "n_est": 10000, "source": "ravaghi/wellbore-geology-prediction-artifacts"}),
    ("c52_ravaghi_lgb3", "lgbm", RAVAGHI_DIR / "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl",
     {"lr": 0.00934, "num_leaves": 64, "n_est": 10000, "seed": 29,
      "source": "ravaghi/wellbore-geology-prediction-artifacts"}),
    ("c53_ravaghi_cat1", "catboost", RAVAGHI_DIR / "models/catboost-1/catboostregressor_trainer_20260526193740.pkl",
     {"depth": 7, "iter": 8000, "lr": 0.02, "source": "ravaghi/wellbore-geology-prediction-artifacts"}),
    ("c54_ravaghi_cat2", "catboost", RAVAGHI_DIR / "models/catboost-2/catboostregressor_trainer_20260526194838.pkl",
     {"depth": 7, "iter": 8000, "lr": 0.03, "source": "ravaghi/wellbore-geology-prediction-artifacts"}),
]


def main():
    t0 = time.time()
    print("=== Import ravaghi pretrained OOF as round_010 candidates ===\n")

    # Load id→(well, row_idx, target, fold) mapping from joined_features
    print("[1/3] Loading our joined_features (truth source for fold/order)...")
    df_jo = pd.read_parquet(ROUND_DIR / "results/joined_features.parquet",
                            columns=["well", "row_idx", "target", "fold"])
    print(f"  joined rows: {len(df_jo):,}  wells: {df_jo['well'].nunique()}")

    # Load ravaghi train.csv id mapping (well, row_idx in ravaghi order)
    print("\n[2/3] Loading ravaghi train.csv (well/id only — fast)...")
    t1 = time.time()
    df_rav = pd.read_csv(RAVAGHI_DIR / "data/train.csv",
                         usecols=["well", "id", "target"], low_memory=False)
    df_rav["row_idx"] = df_rav["id"].str.split("_").str[1].astype("int32")
    print(f"  rows: {len(df_rav):,}  wall: {time.time()-t1:.0f}s")

    # Sanity: target alignment
    df_check = df_rav.merge(df_jo, on=["well", "row_idx"], suffixes=("_rav", "_jo"))
    assert len(df_check) == len(df_jo), f"row mismatch: ravaghi={len(df_rav)} joined={len(df_jo)} merged={len(df_check)}"
    diff = (df_check["target_rav"] - df_check["target_jo"]).abs().mean()
    print(f"  target alignment: mean abs diff {diff:.6f} (must be tiny)")
    assert diff < 0.01, f"target mismatch too large: {diff}"

    # Build re-index: ravaghi row order → joined row order
    # We'll align by (well, row_idx) join.
    print("\n[3/3] Wrapping each model's OOF...")
    y = df_jo["target"].values.astype(np.float32)
    wells = df_jo["well"].values

    # Map: ravaghi position i → joined position; build via merge with index
    df_rav_idx = df_rav[["well", "row_idx"]].reset_index().rename(columns={"index": "rav_idx"})
    df_jo_idx  = df_jo[["well", "row_idx"]].reset_index().rename(columns={"index": "jo_idx"})
    map_df = df_rav_idx.merge(df_jo_idx, on=["well", "row_idx"])
    rav_to_jo = np.full(len(df_rav), -1, dtype=np.int64)
    rav_to_jo[map_df["rav_idx"].values] = map_df["jo_idx"].values
    assert (rav_to_jo >= 0).all(), "alignment failure"

    for cid, ctype, mpath, hparams in MODELS:
        print(f"\n  → {cid}")
        m = joblib.load(mpath)
        oof_rav = m.oof_preds.astype(np.float32)
        assert len(oof_rav) == len(df_rav), f"OOF length mismatch: {len(oof_rav)} vs {len(df_rav)}"

        # Reorder ravaghi-order OOF → joined-order
        oof_jo = np.empty(len(df_jo), dtype=np.float32)
        oof_jo[rav_to_jo] = oof_rav

        pw = perwell_rmse(y, oof_jo, wells)
        fl = flat_rmse(y, oof_jo)
        print(f"    perwell={pw:.4f}  flat={fl:.4f}")

        df_oof = pd.DataFrame({
            "well":     df_jo["well"].values,
            "row_idx":  df_jo["row_idx"].values.astype(np.int32),
            "fold":     df_jo["fold"].values.astype(np.int8),
            "target":   y,
            "oof_pred": oof_jo,
        })
        out = write_oof(
            candidate_id   = cid,
            df_oof         = df_oof,
            candidate_type = ctype,
            features_used  = list(m.estimators[0].feature_names_in_) if hasattr(m.estimators[0], "feature_names_in_") else ["__pretrained_ravaghi__"],
            hyperparams    = hparams,
            seed           = hparams.get("seed", 42),
            train_time_sec = 0.0,  # pretrained — no training time
            extra_meta     = {
                "import_source":  str(mpath),
                "ravaghi_overall_score": float(m.overall_score),
                "ravaghi_fold_scores":   [float(s) for s in m.fold_scores],
                "ravaghi_cv":            "sklearn GroupKFold(5) — DIFFERENT from our hash fold map",
                "note": "OOF ports honest per-row preds; hillclimb fold-aware step is unaffected",
            },
        )
        print(f"    → {out}")

    print(f"\nTotal wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
