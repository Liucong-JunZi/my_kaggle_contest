#!/usr/bin/env python3
"""R10: r9 v2 PF 128-seed OOF generator.

Imports the pure-numpy PF 128-seed ensemble from round_009's submission
script and runs it on all 773 train wells, producing a candidate OOF
parquet in results/candidates/ matching the round_010 schema.

PF is inherently OOF (per-well, sees only its own typewell + hw). No fold
looping needed — we run once, assign fold from global map, done.

Estimated wall: ~2h (9s/well × 773 wells, pure numpy, no numba).
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
R8_DIR = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
R9_DIR = Path("/Users/liucong/code/kaggle/ROGII/experiments/round_009")

sys.path.insert(0, str(ROUND_DIR))
sys.path.insert(0, str(R9_DIR))

# Import r9's PF infra
from r9_pf_only_submit_v2 import (
    run_pf_lik_ensemble_scales,
    PF_N_PARTICLES,
    PF_N_SEEDS,
    PF_SCALES,
)

DATA_DIR = "/Users/liucong/code/kaggle/ROGII/rogii-wellbore-geology-prediction/train"

from shared.data_loader import load_joined
from shared.oof_writer import write_oof


def main():
    t0 = time.time()
    print("=== R10: PF 128-seed OOF generation ===\n")
    print(f"PF_N_PARTICLES={PF_N_PARTICLES}  PF_N_SEEDS={PF_N_SEEDS}")
    print(f"PF_SCALES={PF_SCALES}\n")

    # Get canonical row order + fold map from joined dataset
    base = load_joined()[["well", "row_idx", "target", "last_known_tvt", "fold"]]
    print(f"Base rows: {len(base):,}  wells: {base['well'].nunique()}")

    all_wells = sorted(base["well"].unique())
    print(f"Wells to process: {len(all_wells)}\n")

    n_ok = n_skip = 0
    oof = np.full(len(base), np.nan, dtype=np.float32)
    t_start = time.time()

    for i, wid in enumerate(all_wells):
        try:
            hw = pd.read_csv(f"{DATA_DIR}/{wid}__horizontal_well.csv")
            tw = pd.read_csv(f"{DATA_DIR}/{wid}__typewell.csv")
        except Exception as e:
            print(f"  ! {wid}: load fail ({e})")
            n_skip += 1
            continue

        # Get lateral row indices
        ev_mask = hw["TVT_input"].isna().values
        if ev_mask.sum() == 0:
            n_skip += 1
            continue

        try:
            pf_by_scale = run_pf_lik_ensemble_scales(
                hw, tw, n_particles=PF_N_PARTICLES, n_seeds=PF_N_SEEDS,
            )
        except Exception as e:
            print(f"  ! {wid}: PF fail ({e})")
            n_skip += 1
            continue

        # Use scale 8 as default (kernel's default for selector fallback)
        # PF_SCALES is (3.0, 5.0, 8.0, 12.0) so scale 8 = pf_scale_8
        pf_pred = pf_by_scale.get("pf_scale_8")
        if pf_pred is None:
            pf_pred = list(pf_by_scale.values())[0]

        # Map to joined dataframe rows
        last_known_tvt = float(hw["TVT_input"].dropna().iloc[-1])
        pf_offset = pf_pred - last_known_tvt  # relative target convention

        # Find matching rows in base dataframe
        wid_mask = base["well"] == wid
        ev_row_indices = hw.index[ev_mask].values

        # Align by row_idx
        row_sub = base.loc[wid_mask, "row_idx"].values
        # Build index mapping
        for j, ridx in enumerate(ev_row_indices):
            hit = np.where(row_sub == ridx)[0]
            if len(hit) == 1:
                abs_idx = base.index[wid_mask].values[hit[0]]
                oof[abs_idx] = pf_offset[ridx]

        n_ok += 1

        if (i + 1) % 50 == 0:
            elapsed = time.time() - t_start
            eta = elapsed / max(i + 1, 1) * (len(all_wells) - i - 1)
            print(f"  {i+1}/{len(all_wells)} | ok={n_ok} skip={n_skip} | "
                  f"{elapsed:.0f}s eta={eta:.0f}s", flush=True)

    n_nan = int(np.isnan(oof).sum())
    print(f"\nDone in {time.time()-t0:.0f}s | ok={n_ok} skip={n_skip} | NaN in OOF: {n_nan}")

    if n_nan > 0:
        print(f"  ⚠ {n_nan} rows missing PF predictions — filling with last_known_tvt offset (0.0)")
        oof[np.isnan(oof)] = 0.0

    df_oof = pd.DataFrame({
        "well":     base["well"].values,
        "row_idx":  base["row_idx"].values.astype(np.int32),
        "fold":     base["fold"].values.astype(np.int8),
        "target":   base["target"].values.astype(np.float32),
        "oof_pred": oof.astype(np.float32),
    })

    # Quick metrics
    from shared.metrics import perwell_rmse, flat_rmse
    pw = perwell_rmse(df_oof["target"].values, df_oof["oof_pred"].values, df_oof["well"].values)
    fl = flat_rmse(df_oof["target"].values, df_oof["oof_pred"].values)
    print(f"  per-well RMSE: {pw:.3f}")
    print(f"  flat RMSE:     {fl:.3f}")

    out = write_oof(
        candidate_id   = "c20_r9_pf128_full",
        df_oof         = df_oof,
        candidate_type = "pf_128seed",
        features_used  = ["__pf_128_seed_ensemble_numpy"],
        hyperparams    = {"n_particles": PF_N_PARTICLES, "n_seeds": PF_N_SEEDS,
                          "scales": [float(s) for s in PF_SCALES],
                          "init_spread": 3.0, "mom": 0.998, "vn": 0.002, "pn": 0.005},
        seed           = 0,
        train_time_sec = round(time.time() - t0, 1),
        extra_meta     = {"source": "imported_from_round_009_v2",
                          "select_scale": 8},
    )
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()