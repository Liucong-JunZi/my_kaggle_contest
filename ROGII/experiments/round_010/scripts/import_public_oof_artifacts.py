#!/usr/bin/env python3
"""Import aligned public OOF artifacts as round_010 candidates."""
import io
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.metrics import flat_rmse, perwell_rmse
from shared.oof_writer import write_oof

PILKWANG_DIR = REPO_DIR / "experiments/public_resources/datasets_raw/pilkwang_rogii-model-package"
HAMATZ_OOF = REPO_DIR / "experiments/public_resources/datasets_raw/hamatz_rogii-dual-models-pub/pipeline_b/exp_b_oof.pkl"

PILKWANG_MODELS = [
    ("c70_pilkwang_xgb", "xgboost", "oof/xgb_oof.npy"),
    ("c71_pilkwang_catboost", "catboost", "oof/catboost_oof.npy"),
    ("c72_pilkwang_hgb", "histgb", "oof/hgb_oof.npy"),
    ("c73_pilkwang_lgb", "lightgbm", "oof/lgb_oof.npy"),
    ("c74_pilkwang_tcn", "sequence_tcn", "oof/sequence_tcn_oof.npy"),
    ("c75_pilkwang_blend", "blend", "oof/blend_oof.npy"),
    ("c76_pilkwang_blend_pp", "blend_postprocess", "oof/blend_oof_postprocessed.npy"),
]

HAMATZ_MODELS = [
    ("c77_hamatz_b_meta", "blend", "meta_oof"),
    ("c78_hamatz_b_lgb0", "lightgbm", "model_oofs/lgb_b0"),
    ("c79_hamatz_b_lgb1", "lightgbm", "model_oofs/lgb_b1"),
    ("c80_hamatz_b_lgb2", "lightgbm", "model_oofs/lgb_b2"),
]


def load_hamatz(path):
    raw = Path(path).read_bytes()
    return joblib.load(io.BytesIO(raw))


def hamatz_get(obj, key):
    if "/" not in key:
        return obj[key]
    a, b = key.split("/", 1)
    return obj[a][b]


def main():
    t0 = time.time()
    print("=== Import aligned public OOF artifacts ===")
    df = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    y = df["target"].to_numpy(np.float32)
    wells = df["well"].values
    imported = []

    def write_candidate(cid, arr, ctype, features_used, hyperparams, extra_meta):
        arr = np.asarray(arr, dtype=np.float32)
        if arr.shape != (len(df),):
            raise ValueError(f"{cid}: bad shape {arr.shape}, expected {(len(df),)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{cid}: non-finite predictions")
        pw = perwell_rmse(y, arr, wells)
        fl = flat_rmse(y, arr)
        print(f"  {cid:24s} perwell={pw:.4f} flat={fl:.4f} std={arr.std():.4f}")
        df_oof = pd.DataFrame({
            "well": df["well"].values,
            "row_idx": df["row_idx"].values.astype(np.int32),
            "fold": df["fold"].values.astype(np.int8),
            "target": y,
            "oof_pred": arr,
        })
        out = write_oof(
            candidate_id=cid,
            df_oof=df_oof,
            candidate_type=ctype,
            features_used=features_used,
            hyperparams=hyperparams,
            seed=0,
            train_time_sec=0.0,
            extra_meta=extra_meta,
        )
        imported.append(str(out))

    print("\n[pilkwang]")
    gt = pd.read_parquet(PILKWANG_DIR / "oof/train_gt.parquet", columns=["well_id", "row_index", "target_delta_from_last_known"])
    assert len(gt) == len(df)
    assert gt["well_id"].astype(str).equals(df["well"].astype(str))
    assert gt["row_index"].astype(np.int32).equals(df["row_idx"].astype(np.int32))
    target_diff = np.max(np.abs(gt["target_delta_from_last_known"].to_numpy(np.float32) - y))
    print(f"  alignment target max_abs_diff={target_diff:.6g}")
    assert target_diff < 1e-5
    manifest = json.loads((PILKWANG_DIR / "metadata/model_package_manifest.json").read_text())
    blend_cfg = json.loads((PILKWANG_DIR / "stacking/blend_config.json").read_text())
    pp_cfg = json.loads((PILKWANG_DIR / "postprocess/postprocess_config.json").read_text())
    for cid, ctype, rel in PILKWANG_MODELS:
        arr = np.load(PILKWANG_DIR / rel)
        write_candidate(
            cid,
            arr,
            ctype,
            ["pilkwang_rogii_model_package", rel],
            {"source": "pilkwang/rogii-model-package", "path": rel},
            {
                "import_source": str(PILKWANG_DIR / rel),
                "manifest": manifest,
                "blend_config": blend_cfg if "blend" in cid else {},
                "postprocess_config": pp_cfg if cid.endswith("_pp") else {},
                "note": "OOF ports honest per-row preds; source CV fold ids may differ from round_010 fold map.",
            },
        )

    print("\n[hamatz]")
    obj = load_hamatz(HAMATZ_OOF)
    for cid, ctype, key in HAMATZ_MODELS:
        arr = hamatz_get(obj, key)
        write_candidate(
            cid,
            arr,
            ctype,
            ["hamatz_pipeline_b", key],
            {"source": "hamatz/rogii-dual-models-pub", "key": key, "meta_score": float(obj.get("meta_score", np.nan))},
            {
                "import_source": str(HAMATZ_OOF),
                "note": "Positionally aligned to joined_features; predictions are offset-space OOF from source GroupKFold.",
            },
        )

    summary = {"imported": imported, "wall_seconds": time.time() - t0}
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "public_oof_artifacts_import.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nImported {len(imported)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
