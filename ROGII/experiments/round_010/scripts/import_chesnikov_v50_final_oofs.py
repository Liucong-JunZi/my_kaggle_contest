#!/usr/bin/env python3
"""Import chesnikov v50 finalizer OOF predictions from full phaseA artifacts."""
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

SRC_DIR = (
    REPO_DIR
    / "experiments/public_resources/datasets_raw/annoyingbrine_v50-wellbore-phasea-artifacts"
    / "v50_wellbore-phaseA-artifacts"
)

MODELS = [
    ("c96_chesnikov_v50_hc_raw", "v50_hillclimb_raw", "hc_delta"),
    ("c97_chesnikov_v50_hc_pp", "v50_hillclimb_postprocess", "hc_pp_delta"),
]


def main():
    t0 = time.time()
    print("=== Import chesnikov v50 finalizer OOF candidates ===")
    oof_path = SRC_DIR / "oof_A.csv"
    sub_path = SRC_DIR / "submission.csv"
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)
    if not sub_path.exists():
        raise FileNotFoundError(sub_path)

    df = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    base = df.copy()
    base["id"] = base["well"].astype(str) + "_" + base["row_idx"].astype(str)
    y = base["target"].to_numpy(np.float32)
    wells = base["well"].values

    usecols = ["id", "target_delta", "last_known_tvt", "hc_delta", "hc_pp_tvt"]
    oof = pd.read_csv(oof_path, usecols=usecols)
    oof["hc_pp_delta"] = oof["hc_pp_tvt"].astype(np.float32) - oof["last_known_tvt"].astype(np.float32)
    aligned = base[["id"]].merge(oof, on="id", how="left", validate="one_to_one")
    missing = int(aligned["target_delta"].isna().sum())
    print(f"  rows={len(aligned):,} missing={missing}")
    if missing:
        raise ValueError(f"missing v50 finalizer OOF rows: {missing}")
    target_diff = np.max(np.abs(aligned["target_delta"].to_numpy(np.float32) - y))
    print(f"  target max_abs_diff={target_diff:.6g}")
    if target_diff > 1e-2:
        raise ValueError(f"target mismatch: {target_diff}")

    imported = []
    for cid, ctype, col in MODELS:
        arr = aligned[col].to_numpy(np.float32)
        if not np.isfinite(arr).all():
            raise ValueError(f"{cid}: non-finite predictions")
        pw = perwell_rmse(y, arr, wells)
        fl = flat_rmse(y, arr)
        print(f"  {cid:30s} perwell={pw:.4f} flat={fl:.4f} std={arr.std():.4f}")
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
            features_used=["chesnikov_v50_full_phaseA_oof_A", col],
            hyperparams={
                "source": "annoyingbrine/v50-wellbore-phasea-artifacts",
                "oof_column": col,
                "oof_path": str(oof_path),
                "submission_path": str(sub_path) if col == "hc_pp_delta" else "",
            },
            seed=0,
            train_time_sec=0.0,
            extra_meta={
                "import_source": str(oof_path),
                "test_submission_source": str(sub_path) if col == "hc_pp_delta" else "",
                "note": "Offset-space OOF from v50 finalizer. c97_hc_pp corresponds to the downloaded v50 submission.csv after adding last_known_tvt.",
            },
        )
        imported.append(str(out))

    summary = {"imported": imported, "wall_seconds": time.time() - t0}
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chesnikov_v50_final_import.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nImported {len(imported)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
