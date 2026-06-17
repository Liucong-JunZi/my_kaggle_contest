#!/usr/bin/env python3
"""Import thbdh5765 v11 fresh ravaghi-style OOF arrays."""
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

V11_DIR = REPO_DIR / "experiments/public_resources/datasets_raw/thbdh5765_rogii-v11-fresh-artifacts"
MODELS = [
    ("c81_v11fresh_lgb1", "lightgbm", "lightgbm-1"),
    ("c82_v11fresh_lgb2", "lightgbm", "lightgbm-2"),
    ("c83_v11fresh_lgb3", "lightgbm", "lightgbm-3"),
    ("c84_v11fresh_cat1", "catboost", "catboost-1"),
    ("c85_v11fresh_cat2", "catboost", "catboost-2"),
    ("c86_v11fresh_cat3", "catboost", "catboost-3"),
]


def main():
    t0 = time.time()
    print("=== Import v11 fresh OOF candidates ===")
    df = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    y = df["target"].to_numpy(np.float32)
    wells = df["well"].values
    features = json.loads((V11_DIR / "features.json").read_text())
    scores = json.loads((V11_DIR / "scores.json").read_text())
    manifest = json.loads((V11_DIR / "manifest.json").read_text())
    imported = []

    for cid, ctype, key in MODELS:
        path = V11_DIR / "models" / key / "oof_preds.pkl"
        arr = np.asarray(joblib.load(path), dtype=np.float32)
        if arr.shape != (len(df),):
            raise ValueError(f"{cid}: bad shape {arr.shape}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{cid}: non-finite predictions")
        pw = perwell_rmse(y, arr, wells)
        fl = flat_rmse(y, arr)
        print(f"  {cid:22s} perwell={pw:.4f} flat={fl:.4f} std={arr.std():.4f}")
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
            features_used=["thbdh5765_v11_fresh_artifacts"] + features,
            hyperparams={"source": "thbdh5765/rogii-v11-fresh-artifacts", "model_key": key},
            seed=0,
            train_time_sec=0.0,
            extra_meta={
                "import_source": str(path),
                "manifest": manifest,
                "v11_overall_score": scores.get("overall_scores", {}).get(key),
                "v11_fold_scores": scores.get("fold_scores", {}).get(key),
                "note": "OOF array is positionally aligned to joined_features; source GroupKFold ids differ from round_010 fold map.",
            },
        )
        imported.append(str(out))

    summary = {"imported": imported, "wall_seconds": time.time() - t0}
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v11_fresh_import.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nImported {len(imported)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
