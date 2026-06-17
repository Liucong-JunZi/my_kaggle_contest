#!/usr/bin/env python3
"""Audit imported ravaghi OOF candidates against raw artifact Trainer.oof_preds."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
sys.path.insert(0, str(ROUND_DIR))
# Required for unpickling ravaghi Trainer artifacts.
sys.path.insert(0, str(ROUND_DIR / "submission_package/ravaghi_ridge_kernel"))

from shared.metrics import flat_rmse, perwell_rmse

RAVAGHI_DIR = REPO_DIR / "experiments/public_resources/datasets_raw/ravaghi-artifacts"
OUT_DIR = ROUND_DIR / "results/audits"

MODELS = [
    ("c50_ravaghi_lgb1", RAVAGHI_DIR / "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl"),
    ("c51_ravaghi_lgb2", RAVAGHI_DIR / "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl"),
    ("c52_ravaghi_lgb3", RAVAGHI_DIR / "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl"),
    ("c53_ravaghi_cat1", RAVAGHI_DIR / "models/catboost-1/catboostregressor_trainer_20260526193740.pkl"),
    ("c54_ravaghi_cat2", RAVAGHI_DIR / "models/catboost-2/catboostregressor_trainer_20260526194838.pkl"),
]


def describe(x):
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def main():
    print("=== Ravaghi OOF import audit ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df_jo = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    df_rav = pd.read_csv(RAVAGHI_DIR / "data/train.csv", usecols=["well", "id", "target"], low_memory=False)
    df_rav["row_idx"] = df_rav["id"].str.split("_").str[1].astype("int32")

    df_check = df_rav.merge(df_jo, on=["well", "row_idx"], suffixes=("_rav", "_jo"), how="inner")
    target_diff = (df_check["target_rav"] - df_check["target_jo"]).abs()
    mapping = {
        "joined_rows": int(len(df_jo)),
        "ravaghi_rows": int(len(df_rav)),
        "merged_rows": int(len(df_check)),
        "joined_unique_keys": int(df_jo[["well", "row_idx"]].drop_duplicates().shape[0]),
        "ravaghi_unique_keys": int(df_rav[["well", "row_idx"]].drop_duplicates().shape[0]),
        "target_abs_diff_mean": float(target_diff.mean()),
        "target_abs_diff_max": float(target_diff.max()),
    }

    df_rav_idx = df_rav[["well", "row_idx"]].reset_index().rename(columns={"index": "rav_idx"})
    df_jo_idx = df_jo[["well", "row_idx"]].reset_index().rename(columns={"index": "jo_idx"})
    map_df = df_rav_idx.merge(df_jo_idx, on=["well", "row_idx"], how="inner")
    rav_to_jo = np.full(len(df_rav), -1, dtype=np.int64)
    rav_to_jo[map_df["rav_idx"].values] = map_df["jo_idx"].values

    y = df_jo["target"].values.astype(np.float32)
    wells = df_jo["well"].values
    model_reports = []

    for cid, mpath in MODELS:
        print(f"\n→ {cid}")
        trainer = joblib.load(mpath)
        oof_rav = np.asarray(trainer.oof_preds, dtype=np.float32)
        oof_jo = np.empty(len(df_jo), dtype=np.float32)
        oof_jo[rav_to_jo] = oof_rav

        cand = pd.read_parquet(ROUND_DIR / f"results/candidates/{cid}.parquet")
        pred_diff = np.abs(cand["oof_pred"].values.astype(np.float32) - oof_jo)
        target_cand_diff = np.abs(cand["target"].values.astype(np.float32) - y)
        key_match = bool(
            cand["well"].equals(df_jo["well"])
            and cand["row_idx"].astype("int32").equals(df_jo["row_idx"].astype("int32"))
        )

        pw = perwell_rmse(y, oof_jo, wells)
        fl = flat_rmse(y, oof_jo)
        report = {
            "candidate_id": cid,
            "trainer_path": str(mpath),
            "trainer_oof_len": int(len(oof_rav)),
            "candidate_rows": int(len(cand)),
            "candidate_key_order_matches_joined": key_match,
            "pred_abs_diff_mean": float(pred_diff.mean()),
            "pred_abs_diff_max": float(pred_diff.max()),
            "candidate_target_abs_diff_mean": float(target_cand_diff.mean()),
            "candidate_target_abs_diff_max": float(target_cand_diff.max()),
            "perwell_rmse": float(pw),
            "flat_rmse": float(fl),
            "trainer_overall_score": float(getattr(trainer, "overall_score", np.nan)),
            "trainer_fold_scores": [float(x) for x in getattr(trainer, "fold_scores", [])],
            "target_stats": describe(y),
            "pred_stats": describe(oof_jo),
        }
        report["pass"] = bool(
            mapping["merged_rows"] == mapping["joined_rows"] == mapping["ravaghi_rows"]
            and mapping["joined_unique_keys"] == mapping["joined_rows"]
            and mapping["ravaghi_unique_keys"] == mapping["ravaghi_rows"]
            # ravaghi train.csv stores target rounded to ~1e-3, while joined_features
            # keeps our canonical float target. Candidate parquets must still match
            # joined_features exactly; the raw ravaghi CSV target may differ by <0.001.
            and mapping["target_abs_diff_max"] < 1e-3
            and key_match
            and report["pred_abs_diff_max"] < 1e-6
            and report["candidate_target_abs_diff_max"] < 1e-6
            and 7.5 <= pw <= 8.7
        )
        model_reports.append(report)
        print(f"  perwell={pw:.4f} flat={fl:.4f} pred_diff_max={report['pred_abs_diff_max']:.3g} pass={report['pass']}")

    overall_pass = bool(all(r["pass"] for r in model_reports))
    out = {
        "pass": overall_pass,
        "mapping": mapping,
        "models": model_reports,
    }
    out_path = OUT_DIR / "ravaghi_oof_import_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nMapping: {json.dumps(mapping, indent=2)}")
    print(f"\nOVERALL: {'PASS' if overall_pass else 'FAIL'}")
    print(f"Report: {out_path}")
    if not overall_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
