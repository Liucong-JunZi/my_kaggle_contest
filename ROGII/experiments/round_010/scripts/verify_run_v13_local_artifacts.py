#!/usr/bin/env python3
"""Verify run_v13 local artifact dataset before Kaggle upload."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
ARTIFACT_DIR = ROUND_DIR / "submission_package" / "run_v13_local_artifact_dataset"
KERNEL_DIR = ROUND_DIR / "submission_package" / "run_v13_local_artifacts_kernel"


def fail(msg):
    raise SystemExit(f"FAIL: {msg}")


def main():
    print("=== Verify run_v13 local artifacts ===")
    for rel in ["dataset-metadata.json", "artifact_manifest.json", "ensemble_weights.json", "src/pf_features.py", "src/ravaghi_features.py", "src/native_predict.py", "src/pipeline.py"]:
        path = ARTIFACT_DIR / rel
        print(f"  check {rel}: {path.exists()}")
        if not path.exists():
            fail(f"missing {rel}")

    banned_names = {"submission.csv", "test_base_predictions.csv", "test_base_predictions.csv.gz"}
    banned_suffixes = {".pkl", ".pickle", ".joblib"}
    bad = []
    for path in ARTIFACT_DIR.rglob("*"):
        if not path.is_file():
            continue
        if path.name in banned_names or path.suffix in banned_suffixes:
            bad.append(str(path.relative_to(ARTIFACT_DIR)))
    if bad:
        fail(f"hidden-unsafe or pickled files found: {bad[:20]}")
    print("  no public-test CSV or pickle artifacts")

    inf = (KERNEL_DIR / "inference_kernel.py").read_text()
    for token in ["joblib.load", "pickle.load", "test_base_predictions", "V10_DIR", "V50_DIR"]:
        if token in inf:
            fail(f"inference contains banned token: {token}")
    if "predict_pipeline" not in inf:
        fail("inference must use predict_pipeline")
    print("  inference uses predict_pipeline, no banned loaders/public-test references")

    spec = json.loads((ARTIFACT_DIR / "ensemble_weights.json").read_text())
    weights = spec["weights"]
    if not weights or not all(v > 0 for v in weights.values()):
        fail(f"bad weights spec: {weights}")
    print(f"  weights OK: {weights}")
    print(f"  pp_params={spec.get('pp_params')} sg_params={spec.get('sg_params')}")

    sys.path.insert(0, str(ARTIFACT_DIR / "src"))
    import native_predict
    import ravaghi_features

    local_input = REPO_DIR / "rogii-wellbore-geology-prediction"
    rav = ravaghi_features.build_test_features(local_input)
    if len(rav) == 0:
        fail("ravaghi_features returned no rows")
    print(f"  ravaghi local features: rows={len(rav):,} cols={len(rav.columns)}")

    for model_dir in sorted((ARTIFACT_DIR / "models/ravaghi").iterdir()):
        if not model_dir.is_dir():
            continue
        pred = native_predict.predict_component(model_dir, rav)
        if len(pred) != len(rav) or not np.isfinite(pred).all():
            fail(f"bad predictions for {model_dir.name}")
        print(f"    {model_dir.name:18s} mean={pred.mean():.3f} std={pred.std():.3f}")

    sample = pd.read_csv(local_input / "sample_submission.csv")
    if len(sample) != 14151 or not sample["id"].is_unique:
        fail("unexpected local sample_submission")
    print("  sample_submission OK")
    print("PASS")


if __name__ == "__main__":
    main()
