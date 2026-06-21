#!/usr/bin/env python3
"""Export run_v13 local artifacts for Kaggle dataset inference."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import joblib
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
ARTIFACT_DIR = ROUND_DIR / "submission_package" / "run_v13_local_artifact_dataset"
RAVAGHI_SRC = REPO_DIR / "experiments/public_resources/datasets_raw/ravaghi-artifacts"
# RUN_JSON may be a plain hillclimb result (averaged_weights) or a pipeline_eval
# result carrying a full "spec". Override via env ROGII_RUN_JSON.
RUN_JSON = Path(os.environ.get(
    "ROGII_RUN_JSON",
    str(ROUND_DIR / "results/hillclimb_runs/run_v12_hidden_safe_pf_ravaghi.json"),
))
RUN_OOF = RUN_JSON.with_name(RUN_JSON.stem + "_oof.parquet")
C20_OOF = ROUND_DIR / "results/candidates/c20_r9_pf128_full.parquet"

# Unpickling the public ravaghi trainers needs the shim module on sys.path.
sys.path.insert(0, str(ROUND_DIR / "submission_package" / "run_v12_hidden_safe_kernel"))

RAVAGHI_MODELS = {
    "c50_ravaghi_lgb1": ("lightgbm", "models/lightgbm-1/lgbmregressor_trainer_20260526182612.pkl"),
    "c51_ravaghi_lgb2": ("lightgbm", "models/lightgbm-2/lgbmregressor_trainer_20260526190415.pkl"),
    "c52_ravaghi_lgb3": ("lightgbm", "models/lightgbm-3/lgbmregressor_trainer_20260526192806.pkl"),
    "c53_ravaghi_cat1": ("catboost", "models/catboost-1/catboostregressor_trainer_20260526193740.pkl"),
    "c54_ravaghi_cat2": ("catboost", "models/catboost-2/catboostregressor_trainer_20260526194838.pkl"),
}

NATIVE_PREDICT_CODE = r'''from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor


def load_json(path):
    return json.loads(Path(path).read_text())


def predict_component(model_dir, features):
    model_dir = Path(model_dir)
    meta = load_json(model_dir / "meta.json")
    feature_names = load_json(model_dir / "feature_names.json")
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"{model_dir.name}: missing {len(missing)} features, first={missing[:10]}")

    X = features[feature_names]
    preds = []
    if meta["model_type"] == "lightgbm":
        for path in sorted(model_dir.glob("fold_*.txt")):
            booster = lgb.Booster(model_file=str(path))
            preds.append(booster.predict(X))
    elif meta["model_type"] == "catboost":
        for path in sorted(model_dir.glob("fold_*.cbm")):
            model = CatBoostRegressor()
            model.load_model(str(path))
            preds.append(model.predict(X))
    else:
        raise ValueError(f"unknown model_type: {meta['model_type']}")

    if not preds:
        raise ValueError(f"{model_dir}: no fold models found")
    out = np.mean(np.vstack(preds), axis=0).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"{model_dir.name}: non-finite predictions")
    return out
'''


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True).strip()
    except Exception:
        return "unknown"


def reset_dir(path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)


def feature_names_from_estimator(est):
    if hasattr(est, "feature_names_in_"):
        return [str(x) for x in list(est.feature_names_in_)]
    if hasattr(est, "feature_names_"):
        return [str(x) for x in list(est.feature_names_)]
    if hasattr(est, "booster_"):
        return [str(x) for x in est.booster_.feature_name()]
    raise ValueError(f"Cannot extract feature names from {type(est)}")


def export_trainer(candidate_id, model_type, rel_path):
    src_path = RAVAGHI_SRC / rel_path
    if not src_path.exists():
        raise FileNotFoundError(src_path)
    trainer = joblib.load(src_path)
    estimators = list(getattr(trainer, "estimators", []))
    if not estimators:
        raise ValueError(f"{candidate_id}: trainer has no estimators")

    out_dir = ARTIFACT_DIR / "models" / "ravaghi" / candidate_id
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_names = feature_names_from_estimator(estimators[0])
    (out_dir / "feature_names.json").write_text(json.dumps(feature_names, indent=2))

    model_files = []
    for i, est in enumerate(estimators):
        names = feature_names_from_estimator(est)
        if names != feature_names:
            raise ValueError(f"{candidate_id}: fold {i} feature order differs")
        if model_type == "lightgbm":
            path = out_dir / f"fold_{i}.txt"
            est.booster_.save_model(str(path))
        elif model_type == "catboost":
            path = out_dir / f"fold_{i}.cbm"
            est.save_model(str(path))
        else:
            raise ValueError(model_type)
        model_files.append(path.name)

    meta = {
        "candidate_id": candidate_id,
        "model_type": model_type,
        "prediction_reducer": "mean",
        "n_fold_models": len(model_files),
        "model_files": model_files,
        "feature_names_file": "feature_names.json",
        "source_path": str(src_path),
        "source_sha256": sha256(src_path),
        "trainer_overall_score": float(getattr(trainer, "overall_score", float("nan"))),
        "trainer_fold_scores": [float(x) for x in getattr(trainer, "fold_scores", [])],
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main():
    print(f"Exporting artifacts to {ARTIFACT_DIR}")
    reset_dir(ARTIFACT_DIR)
    for sub in ["src", "models/ravaghi", "metrics", "oof"]:
        (ARTIFACT_DIR / sub).mkdir(parents=True, exist_ok=True)

    (ARTIFACT_DIR / "dataset-metadata.json").write_text(json.dumps({
        "title": "ROGII run v13 local artifacts",
        "id": "smartorz/rogii-run-v13-local-artifacts",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2) + "\n")

    shutil.copy2(ROUND_DIR / "submission_package/training_code/pf_128seed.py", ARTIFACT_DIR / "src/pf_features.py")
    shutil.copy2(
        ROUND_DIR / "submission_package/run_v12_hidden_safe_kernel/ravaghi_features.py",
        ARTIFACT_DIR / "src/ravaghi_features.py",
    )
    shutil.copy2(ROUND_DIR / "shared/pipeline.py", ARTIFACT_DIR / "src/pipeline.py")
    (ARTIFACT_DIR / "src/native_predict.py").write_text(NATIVE_PREDICT_CODE)

    exported = {}
    for cid, (model_type, rel_path) in RAVAGHI_MODELS.items():
        print(f"  exporting {cid}")
        exported[cid] = export_trainer(cid, model_type, rel_path)

    if not RUN_JSON.exists():
        raise FileNotFoundError(RUN_JSON)
    run = json.loads(RUN_JSON.read_text())
    if "spec" in run:
        # pipeline_eval run: carries the full pipeline spec verbatim.
        spec = run["spec"]
    else:
        # plain hillclimb run: blend-only, no pp/sg.
        spec = {
            "label": run["label"],
            "fuser": run.get("fuser", "hillclimb"),
            "prediction_space": "offset",
            "final_output_space": "absolute_tvt",
            "weights": run["averaged_weights"],
            "pf_offset_component": "c20_r9_pf128_full",
            "pp_params": None,
            "sg_params": None,
        }
    weights = spec["weights"]
    (ARTIFACT_DIR / "ensemble_weights.json").write_text(json.dumps(spec, indent=2))
    shutil.copy2(RUN_JSON, ARTIFACT_DIR / "metrics" / RUN_JSON.name)
    if RUN_OOF.exists():
        shutil.copy2(RUN_OOF, ARTIFACT_DIR / "oof" / RUN_OOF.name)
    shutil.copy2(C20_OOF, ARTIFACT_DIR / "oof" / C20_OOF.name)

    manifest = {
        "artifact_version": "run_v13_local_artifacts_v1",
        "git_commit": git_commit(),
        "hidden_safe": True,
        "uses_public_test_predictions": False,
        "prediction_space": "offset",
        "final_output_space": "absolute_tvt",
        "weights_file": "ensemble_weights.json",
        "metrics_file": f"metrics/{RUN_JSON.name}",
        "pf_source": {
            "module": "src/pf_features.py",
            "candidate_id": "c20_r9_pf128_full",
            "select_scale": 8,
            "note": "PF module is packaged locally; verify exact parity before submission.",
        },
        "ravaghi_models": exported,
        "files_sha256": {},
    }
    for p in sorted(ARTIFACT_DIR.rglob("*")):
        if p.is_file() and p.name != "artifact_manifest.json":
            manifest["files_sha256"][str(p.relative_to(ARTIFACT_DIR))] = sha256(p)
    (ARTIFACT_DIR / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2))
    print("done")


if __name__ == "__main__":
    main()
