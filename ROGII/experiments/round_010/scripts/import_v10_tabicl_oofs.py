#!/usr/bin/env python3
"""Import thbdh5765 v10 artifact OOF predictions as round_010 candidates."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
REPO_DIR = ROUND_DIR.parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.metrics import flat_rmse, perwell_rmse
from shared.oof_writer import write_oof

V10_DIR = REPO_DIR / "experiments/public_resources/datasets_raw/thbdh5765_rogii-v10-fresh-artifacts"
DIAG_DIR = V10_DIR / "diagnostics"

STACK_WEIGHTS = {
    "lgb123": 0.0,
    "lgb42": 0.0,
    "lgb7": 0.0,
    "cb42": 0.0,
    "cb7": 0.0422716960310936,
    "cb123": 0.09353841096162796,
    "tabicl_A": 0.7117284536361694,
    "tabicl_B": 0.038138117641210556,
}


def main():
    t0 = time.time()
    print("=== Import v10 TabICL artifact OOF candidates ===")

    df_jo = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    meta = pd.read_csv(DIAG_DIR / "oof_val_meta.csv", low_memory=False)
    meta["row_idx"] = meta["id"].str.split("_").str[1].astype("int32")

    keys = json.loads((DIAG_DIR / "prediction_keys.json").read_text())
    z = np.load(DIAG_DIR / "oof_val_predictions.npz")
    preds = z["predictions"].astype(np.float32)
    target = z["target"].astype(np.float32)
    assert preds.shape == (len(meta), len(keys)), (preds.shape, len(meta), len(keys))
    assert len(target) == len(meta)

    df_check = meta[["well", "row_idx", "target"]].merge(
        df_jo, on=["well", "row_idx"], suffixes=("_v10", "_jo"), how="inner"
    )
    diff = (df_check["target_v10"] - df_check["target_jo"]).abs()
    print(f"rows: v10={len(meta):,} joined={len(df_jo):,} merged={len(df_check):,}")
    print(f"target alignment: mean_abs={diff.mean():.6g} max_abs={diff.max():.6g}")
    assert len(df_check) == len(df_jo) == len(meta)
    assert meta[["well", "row_idx"]].drop_duplicates().shape[0] == len(meta)
    assert df_jo[["well", "row_idx"]].drop_duplicates().shape[0] == len(df_jo)
    assert diff.max() < 1e-3

    v10_idx = meta[["well", "row_idx"]].reset_index().rename(columns={"index": "v10_idx"})
    jo_idx = df_jo[["well", "row_idx"]].reset_index().rename(columns={"index": "jo_idx"})
    map_df = v10_idx.merge(jo_idx, on=["well", "row_idx"], how="inner")
    v10_to_jo = np.full(len(meta), -1, dtype=np.int64)
    v10_to_jo[map_df["v10_idx"].values] = map_df["jo_idx"].values
    assert (v10_to_jo >= 0).all()

    y = df_jo["target"].values.astype(np.float32)
    wells = df_jo["well"].values
    imported = []

    def write_candidate(cid, pred_jo, ctype, extra_meta):
        pw = perwell_rmse(y, pred_jo, wells)
        fl = flat_rmse(y, pred_jo)
        print(f"  {cid:20s} perwell={pw:.4f} flat={fl:.4f} std={np.std(pred_jo):.4f}")
        df_oof = pd.DataFrame({
            "well": df_jo["well"].values,
            "row_idx": df_jo["row_idx"].values.astype(np.int32),
            "fold": df_jo["fold"].values.astype(np.int8),
            "target": y,
            "oof_pred": pred_jo.astype(np.float32),
        })
        out = write_oof(
            candidate_id=cid,
            df_oof=df_oof,
            candidate_type=ctype,
            features_used=["thbdh5765_v10_fresh_artifacts", extra_meta.get("source_key", cid)],
            hyperparams={"source": "thbdh5765/rogii-v10-fresh-artifacts", **extra_meta},
            seed=0,
            train_time_sec=0.0,
            extra_meta={"import_source": str(DIAG_DIR / "oof_val_predictions.npz"), **extra_meta},
        )
        imported.append(str(out))

    pred_jo_by_key = {}
    for j, key in enumerate(keys):
        pred_jo = np.empty(len(df_jo), dtype=np.float32)
        pred_jo[v10_to_jo] = preds[:, j]
        pred_jo_by_key[key] = pred_jo
        write_candidate(f"v10_{key}", pred_jo, "public_v10_artifact", {"source_key": key})

    stack = np.zeros(len(df_jo), dtype=np.float32)
    for key, w in STACK_WEIGHTS.items():
        stack += np.float32(w) * pred_jo_by_key[key]
    write_candidate("v10_saved_stack", stack, "public_v10_stack", {"source_key": "saved_stack", "weights": STACK_WEIGHTS})

    summary = {
        "keys": keys,
        "stack_weights": STACK_WEIGHTS,
        "imported": imported,
        "wall_seconds": time.time() - t0,
    }
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "v10_tabicl_import.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nImported {len(imported)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
