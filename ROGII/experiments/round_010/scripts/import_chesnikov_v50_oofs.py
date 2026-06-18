#!/usr/bin/env python3
"""Import chesnikov v50 phaseA/phaseB public OOF arrays."""
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

SRC_DIR = (
    REPO_DIR
    / "experiments/public_resources/datasets_raw/chesnikovleonid_rogii-v50-dwt-phaseb-modelonly"
    / "v50_modelonly_artifacts/v50_wellbore-phaseA-artifacts"
)
TRAIN_DIR = REPO_DIR / "rogii-wellbore-geology-prediction/train"

MODELS = [
    ("c87_chesnikov_v50_lgb1", "lightgbm", "lightgbm-1"),
    ("c88_chesnikov_v50_lgb2", "lightgbm", "lightgbm-2"),
    ("c89_chesnikov_v50_lgb3", "lightgbm", "lightgbm-3"),
    ("c90_chesnikov_v50_cat1", "catboost", "catboost-1"),
    ("c91_chesnikov_v50_cat2", "catboost", "catboost-2"),
    ("c92_chesnikov_v50_cat3", "catboost", "catboost-3"),
    ("c93_chesnikov_v50_phaseb_lgb_fast", "lightgbm_phaseb", "phaseB-lgb-fast"),
    ("c94_chesnikov_v50_phaseb_lgb_slow", "lightgbm_phaseb", "phaseB-lgb-slow"),
    ("c95_chesnikov_v50_phaseb_cb", "catboost_phaseb", "phaseB-cb"),
]


def build_lateral_index() -> pd.DataFrame:
    rows = []
    for path in sorted(TRAIN_DIR.glob("*__horizontal_well.csv")):
        well = path.name.split("__", 1)[0]
        hw = pd.read_csv(path, usecols=["TVT", "TVT_input"])
        mask = hw["TVT_input"].isna().to_numpy()
        if not mask.any():
            continue
        idx = np.flatnonzero(mask).astype(np.int32)
        known = hw.loc[~hw["TVT_input"].isna(), "TVT_input"]
        if known.empty:
            raise ValueError(f"{well}: no known TVT_input rows")
        last_known = float(known.iloc[-1])
        target = hw.loc[mask, "TVT"].to_numpy(np.float32) - np.float32(last_known)
        rows.append(pd.DataFrame({"well": well, "row_idx": idx, "target_from_csv": target}))
    return pd.concat(rows, ignore_index=True)


def main():
    t0 = time.time()
    print("=== Import chesnikov v50 OOF candidates ===")
    if not SRC_DIR.exists():
        raise FileNotFoundError(SRC_DIR)

    df = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "target", "fold"],
    )
    y = df["target"].to_numpy(np.float32)
    wells = df["well"].values

    print("  verifying source OOF row order against sorted train lateral rows")
    lat = build_lateral_index()
    if len(lat) != len(df):
        raise ValueError(f"lateral row count mismatch: {len(lat)} vs {len(df)}")
    if not lat["well"].astype(str).equals(df["well"].astype(str)):
        raise ValueError("well order mismatch vs joined_features")
    if not lat["row_idx"].astype(np.int32).equals(df["row_idx"].astype(np.int32)):
        raise ValueError("row_idx order mismatch vs joined_features")
    target_diff = np.max(np.abs(lat["target_from_csv"].to_numpy(np.float32) - y))
    print(f"  alignment target max_abs_diff={target_diff:.6g}")
    if target_diff > 1e-2:
        raise ValueError(f"target mismatch: {target_diff}")

    readme = SRC_DIR / "README.md"
    log_path = SRC_DIR / "local_finalizer.log"
    imported = []

    for cid, ctype, key in MODELS:
        path = SRC_DIR / "models" / key / "oof_preds.pkl"
        arr = np.asarray(joblib.load(path), dtype=np.float32)
        if arr.shape != (len(df),):
            raise ValueError(f"{cid}: bad shape {arr.shape}, expected {(len(df),)}")
        if not np.isfinite(arr).all():
            raise ValueError(f"{cid}: non-finite predictions")
        pw = perwell_rmse(y, arr, wells)
        fl = flat_rmse(y, arr)
        print(f"  {cid:34s} perwell={pw:.4f} flat={fl:.4f} std={arr.std():.4f}")
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
            features_used=["chesnikovleonid_rogii_v50_phaseb_modelonly", key],
            hyperparams={
                "source": "chesnikovleonid/rogii-v50-dwt-phaseb-modelonly",
                "model_key": key,
                "oof_path": str(path),
            },
            seed=0,
            train_time_sec=0.0,
            extra_meta={
                "import_source": str(path),
                "readme": readme.read_text(errors="replace") if readme.exists() else "",
                "local_finalizer_log_excerpt": log_path.read_text(errors="replace")[-4000:] if log_path.exists() else "",
                "note": "OOF arrays align to sorted train lateral rows and are offset-space predictions. Test inference is not packaged because exact 259-feature builder/cache is not included in this model-only dataset.",
            },
        )
        imported.append(str(out))

    summary = {"imported": imported, "wall_seconds": time.time() - t0}
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chesnikov_v50_import.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nImported {len(imported)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
