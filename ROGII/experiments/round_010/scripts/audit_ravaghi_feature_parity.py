#!/usr/bin/env python3
"""Audit ravaghi artifact model feature names/order and predict behavior."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
sys.path.insert(0, str(ROUND_DIR / "submission_package/ravaghi_ridge_kernel"))

RAVAGHI_DIR = REPO_DIR / "experiments/public_resources/datasets_raw/ravaghi-artifacts"
OUT_DIR = ROUND_DIR / "results/audits"

MODELS = [
    ("c50_ravaghi_lgb1", RAVAGHI_DIR / "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl"),
    ("c51_ravaghi_lgb2", RAVAGHI_DIR / "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl"),
    ("c52_ravaghi_lgb3", RAVAGHI_DIR / "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl"),
    ("c53_ravaghi_cat1", RAVAGHI_DIR / "models/catboost-1/catboostregressor_trainer_20260526193740.pkl"),
    ("c54_ravaghi_cat2", RAVAGHI_DIR / "models/catboost-2/catboostregressor_trainer_20260526194838.pkl"),
]
NON_FEATURE_COLS = {"well", "id", "target", "TVT", "row_idx", "fold", "oof_pred"}


def feature_names(est):
    if hasattr(est, "feature_names_in_"):
        return list(est.feature_names_in_)
    if hasattr(est, "feature_names_"):
        return list(est.feature_names_)
    if hasattr(est, "feature_name_"):
        return list(est.feature_name_)
    if hasattr(est, "booster_"):
        return list(est.booster_.feature_name())
    return []


def predict_with_trainer(trainer, X):
    return np.mean([est.predict(X) for est in trainer.estimators], axis=0)


def main():
    print("=== Ravaghi feature parity audit ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use the original artifact train.csv as canonical feature matrix.
    train = pd.read_csv(RAVAGHI_DIR / "data/train.csv", low_memory=False)
    artifact_features = [c for c in train.columns if c not in NON_FEATURE_COLS]
    subset_idx = np.linspace(0, len(train) - 1, num=200, dtype=np.int64)

    reports = []
    for cid, mpath in MODELS:
        print(f"\n→ {cid}")
        trainer = joblib.load(mpath)
        est_reports = []
        expected = feature_names(trainer.estimators[0])
        X_expected = train[expected].iloc[subset_idx] if expected else None
        trainer_pred = predict_with_trainer(trainer, X_expected) if expected else None

        for i, est in enumerate(trainer.estimators):
            names = feature_names(est)
            same_as_artifact_prefix = names == artifact_features[: len(names)]
            same_as_first = names == expected
            missing = [c for c in names if c not in train.columns]
            extra = [c for c in artifact_features if c not in names]
            pred_diff = None
            shuffled_diff = None

            if names:
                X = train[names].iloc[subset_idx]
                pred_named = est.predict(X)
                pred_np = est.predict(X.values)
                pred_diff = float(np.max(np.abs(np.asarray(pred_named) - np.asarray(pred_np))))

                if len(names) > 1:
                    X_bad = X[list(reversed(names))]
                    try:
                        pred_bad = est.predict(X_bad)
                        shuffled_diff = float(np.max(np.abs(np.asarray(pred_named) - np.asarray(pred_bad))))
                    except Exception as exc:
                        shuffled_diff = f"error:{type(exc).__name__}"

            est_reports.append({
                "fold": i,
                "estimator_type": type(est).__name__,
                "n_features": len(names),
                "same_order_as_first_estimator": same_as_first,
                "same_order_as_artifact_train_prefix": same_as_artifact_prefix,
                "missing_features": missing[:20],
                "n_missing_features": len(missing),
                "n_artifact_extra_features": len(extra),
                "dataframe_vs_numpy_pred_max_abs_diff": pred_diff,
                "reversed_column_order_pred_max_abs_diff": shuffled_diff,
            })

        expected_stats = {}
        if expected:
            stats = train[expected].agg(["mean", "std", "min", "max"]).T
            expected_stats = {
                "first_features": expected[:10],
                "last_features": expected[-10:],
                "n_nan_total": int(train[expected].isna().sum().sum()),
                "n_inf_total": int(np.isinf(train[expected].select_dtypes(include=[np.number]).values).sum()),
                "feature_stats_sample": {
                    k: {kk: float(vv) for kk, vv in stats.loc[k].to_dict().items()}
                    for k in expected[:5]
                },
            }

        report = {
            "candidate_id": cid,
            "trainer_path": str(mpath),
            "n_estimators": len(trainer.estimators),
            "artifact_train_feature_count": len(artifact_features),
            "expected_feature_count": len(expected),
            "expected_order_matches_artifact_train_prefix": expected == artifact_features[: len(expected)],
            "trainer_predict_subset_stats": {
                "mean": float(np.mean(trainer_pred)) if trainer_pred is not None else None,
                "std": float(np.std(trainer_pred)) if trainer_pred is not None else None,
                "min": float(np.min(trainer_pred)) if trainer_pred is not None else None,
                "max": float(np.max(trainer_pred)) if trainer_pred is not None else None,
            },
            "feature_stats": expected_stats,
            "estimators": est_reports,
        }
        report["pass"] = bool(
            report["expected_feature_count"] == 195
            and report["expected_order_matches_artifact_train_prefix"]
            and all(e["same_order_as_first_estimator"] for e in est_reports)
            and all(e["n_missing_features"] == 0 for e in est_reports)
            and all(
                e["dataframe_vs_numpy_pred_max_abs_diff"] is not None
                and e["dataframe_vs_numpy_pred_max_abs_diff"] < 1e-9
                for e in est_reports
            )
        )
        reports.append(report)
        print(
            f"  features={len(expected)} order_ok={report['expected_order_matches_artifact_train_prefix']} "
            f"df_vs_np_max={max(e['dataframe_vs_numpy_pred_max_abs_diff'] or 0 for e in est_reports):.3g} "
            f"pass={report['pass']}"
        )

    out = {
        "pass": bool(all(r["pass"] for r in reports)),
        "artifact_train_rows": int(len(train)),
        "artifact_train_columns": int(len(train.columns)),
        "reports": reports,
    }
    out_path = OUT_DIR / "ravaghi_feature_parity_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nOVERALL: {'PASS' if out['pass'] else 'FAIL'}")
    print(f"Report: {out_path}")
    if not out["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
