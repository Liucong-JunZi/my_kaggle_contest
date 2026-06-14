"""Universal training driver — runs any candidate that exposes the
contract (CANDIDATE_ID, CANDIDATE_TYPE, get_features, fit_fold, predict).

Usage:
    python orchestrator/train_one.py c01_lgb_default
    python orchestrator/train_one.py c01_lgb_default --seed 7

It loads the joined feature df + global fold map, runs 5-fold OOF using the
candidate module's fit_fold, calls write_oof with full metadata.
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.data_loader import load_joined
from shared.metrics import perwell_rmse, flat_rmse
from shared.oof_writer import write_oof


def run(candidate_id: str, seed_override: int | None = None):
    mod = importlib.import_module(f"candidates.{candidate_id}")

    cid    = mod.CANDIDATE_ID
    ctype  = mod.CANDIDATE_TYPE
    seed   = seed_override if seed_override is not None else getattr(mod, "DEFAULT_SEED", 42)
    hparams = getattr(mod, "HYPERPARAMS", {})

    print(f"=== Train candidate: {cid} ({ctype}, seed={seed}) ===")

    df_full = load_joined()
    X, feat_cols = mod.get_features(df_full)
    y = df_full["target"].values.astype(np.float32)
    folds = df_full["fold"].values
    wells = df_full["well"].values

    print(f"  rows={len(df_full):,}  features={len(feat_cols)}  wells={df_full['well'].nunique()}")

    oof = np.zeros(len(df_full), dtype=np.float32)
    fold_metrics = {}
    t0 = time.time()
    for fold in range(5):
        tr_mask = folds != fold
        va_mask = folds == fold
        X_tr = X[tr_mask] if hasattr(X, "iloc") else X[tr_mask.nonzero()[0]]
        X_va = X[va_mask] if hasattr(X, "iloc") else X[va_mask.nonzero()[0]]
        if hasattr(X, "iloc"):
            X_tr = X.iloc[tr_mask.nonzero()[0]]
            X_va = X.iloc[va_mask.nonzero()[0]]
        y_tr = y[tr_mask]; y_va = y[va_mask]

        t_fold = time.time()
        model = mod.fit_fold(X_tr, y_tr, X_va, y_va, seed)
        pred  = mod.predict(model, X_va)
        oof[va_mask] = pred.astype(np.float32)

        fold_pw = perwell_rmse(y_va, pred, wells[va_mask])
        fold_flat = flat_rmse(y_va, pred)
        fold_metrics[f"fold{fold}"] = {"perwell": float(fold_pw), "flat": float(fold_flat)}
        print(f"  fold {fold}: perwell={fold_pw:.3f}  flat={fold_flat:.3f}  "
              f"t={time.time()-t_fold:.0f}s", flush=True)

    train_time = time.time() - t0
    overall_pw = perwell_rmse(y, oof, wells)
    overall_flat = flat_rmse(y, oof)
    print(f"\n  → OVERALL  perwell={overall_pw:.3f}  flat={overall_flat:.3f}  "
          f"({train_time:.0f}s)")

    df_oof = pd.DataFrame({
        "well": df_full["well"].values,
        "row_idx": df_full["row_idx"].values,
        "fold": folds,
        "target": y,
        "oof_pred": oof,
    })
    out = write_oof(
        candidate_id    = cid,
        df_oof          = df_oof,
        candidate_type  = ctype,
        features_used   = feat_cols,
        hyperparams     = hparams,
        seed            = seed,
        train_time_sec  = train_time,
        fold_metrics    = fold_metrics,
    )
    print(f"  → {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate_id", help="e.g. c01_lgb_default")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    run(args.candidate_id, args.seed)
