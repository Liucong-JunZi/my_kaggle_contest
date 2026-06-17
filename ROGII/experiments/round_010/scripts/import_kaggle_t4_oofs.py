#!/usr/bin/env python3
"""Import T4-trained MLP OOFs (c60-c64) as round_010 hill-climb candidates.

The Kaggle kernel `smartorz/rogii-t4-train-mlp` writes one parquet per candidate
to /kaggle/working/<cid>.parquet, with the canonical schema:
  well, row_idx, fold, target, oof_pred (all matching joined_features row order).

Pull them with `kaggle kernels output`, then run this importer to register as
round_010 candidates.

Pre-flight:
  cd experiments/round_010
  mkdir -p t4_out
  kaggle kernels output smartorz/rogii-t4-train-mlp -p t4_out

Then:
  /Users/liucong/miniconda3/bin/python3 scripts/import_kaggle_t4_oofs.py

Each cid gets validated:
  (a) row count == joined_features row count (3,783,989)
  (b) target matches joined_features.target exactly (kernel reads same file)
  (c) oof_pred is finite everywhere
  (d) printed perwell/flat match what the kernel reported
"""
import sys, time, json
from pathlib import Path
import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.metrics import perwell_rmse, flat_rmse
from shared.oof_writer import write_oof

T4_OUT_DIR = ROUND_DIR / "t4_out"

# Match candidate_specs in train_t4_source.py
CANDIDATES = [
    ("c60_mlp_s42",        "torch_mlp",    {"arch": "mlp",    "seed": 42,   "epochs": 30, "lr": 1e-3, "loss": "mse",   "batch": 4096}),
    ("c61_mlp_s7",         "torch_mlp",    {"arch": "mlp",    "seed": 7,    "epochs": 30, "lr": 1e-3, "loss": "mse",   "batch": 4096}),
    ("c62_mlp_s2024",      "torch_mlp",    {"arch": "mlp",    "seed": 2024, "epochs": 30, "lr": 1e-3, "loss": "mse",   "batch": 4096}),
    ("c63_mlp_huber_s42",  "torch_mlp",    {"arch": "mlp",    "seed": 42,   "epochs": 30, "lr": 1e-3, "loss": "huber", "batch": 4096}),
    ("c64_resmlp_s42",     "torch_resmlp", {"arch": "resmlp", "seed": 42,   "epochs": 40, "lr": 1e-3, "loss": "mse",   "batch": 4096}),
]


def main():
    t0 = time.time()
    print(f"=== Import T4 MLP OOFs as round_010 candidates ===\n")
    print(f"Source dir: {T4_OUT_DIR}")
    print(f"Files present: {sorted(p.name for p in T4_OUT_DIR.glob('*.parquet'))}")

    # Ground-truth row order + folds + target
    print(f"\n[1/2] Loading joined_features as ground-truth row order...")
    df_jo = pd.read_parquet(ROUND_DIR / "results" / "joined_features.parquet",
                            columns=["well", "row_idx", "target", "fold"])
    print(f"  {len(df_jo):,} rows  {df_jo['well'].nunique()} wells")
    y_truth     = df_jo["target"].values.astype(np.float32)
    wells_truth = df_jo["well"].values
    fold_truth  = df_jo["fold"].values.astype(np.int8)

    # Optional kernel summary (printed self-check)
    summary_path = T4_OUT_DIR / "summary.json"
    kernel_metrics = {}
    if summary_path.exists():
        with open(summary_path) as f:
            for r in json.load(f):
                kernel_metrics[r["cid"]] = (r["perwell"], r["flat"])
        print(f"  kernel summary loaded: {list(kernel_metrics)}")

    print(f"\n[2/2] Importing each candidate...")
    for cid, ctype, hparams in CANDIDATES:
        path = T4_OUT_DIR / f"{cid}.parquet"
        if not path.exists():
            print(f"  ⚠  SKIP {cid} — file not found at {path}")
            continue
        df_in = pd.read_parquet(path)
        # Strict checks
        assert len(df_in) == len(df_jo), f"{cid}: row count {len(df_in)} != {len(df_jo)}"
        # Order should match because kernel read the SAME parquet — cross-check
        # the (well, row_idx) sequence.
        if not (df_in["well"].values == wells_truth).all():
            raise RuntimeError(f"{cid}: well column order mismatch")
        if not (df_in["row_idx"].values.astype(np.int32) == df_jo["row_idx"].values.astype(np.int32)).all():
            raise RuntimeError(f"{cid}: row_idx column order mismatch")
        # Target should be byte-identical
        target_diff = np.abs(df_in["target"].values - y_truth).max()
        assert target_diff < 1e-5, f"{cid}: target mismatch max={target_diff}"
        # fold should match (kernel uses the same parquet's fold column)
        if not (df_in["fold"].values.astype(np.int8) == fold_truth).all():
            raise RuntimeError(f"{cid}: fold column mismatch")
        oof = df_in["oof_pred"].values.astype(np.float32)
        if not np.isfinite(oof).all():
            n_bad = (~np.isfinite(oof)).sum()
            raise RuntimeError(f"{cid}: {n_bad} non-finite predictions")

        pw = perwell_rmse(y_truth, oof, wells_truth)
        fl = flat_rmse(y_truth, oof)
        krm = kernel_metrics.get(cid, (None, None))
        if krm[0] is not None:
            assert abs(pw - krm[0]) < 1e-3, f"{cid}: perwell mismatch importer={pw:.6f} kernel={krm[0]:.6f}"
        print(f"  {cid:25s}  perwell={pw:.4f}  flat={fl:.4f}  "
              f"(kernel: {krm[0] if krm[0] is None else f'{krm[0]:.4f}'})")

        df_oof = pd.DataFrame({
            "well":     wells_truth,
            "row_idx":  df_jo["row_idx"].values.astype(np.int32),
            "fold":     fold_truth,
            "target":   y_truth,
            "oof_pred": oof,
        })
        out = write_oof(
            candidate_id   = cid,
            df_oof         = df_oof,
            candidate_type = ctype,
            features_used  = ["__t4_mlp_feat_set_v14__"],  # 43 hard-coded feats; placeholder
            hyperparams    = hparams,
            seed           = hparams.get("seed", 42),
            train_time_sec = 0.0,
            extra_meta     = {
                "import_source":   str(path),
                "kernel_id":       "smartorz/rogii-t4-train-mlp",
                "trained_on":      "Kaggle T4 GPU",
                "feat_count":      43,
                "note":            "PyTorch MLP, fold-aligned (sha256 hash); honest 5-fold OOF",
            },
        )
        print(f"    → {out}")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
