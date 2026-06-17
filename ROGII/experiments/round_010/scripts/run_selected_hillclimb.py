#!/usr/bin/env python3
"""Run fast hillclimb on a selected candidate subset."""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from orchestrator.hillclimb_fast import encode_wells, hillclimb_one_fold, perwell_rmse_fast, to_weights
from shared.oof_writer import load_oof

SETS = {
    "runv7_plus_best_public": [
        "c20_r9_pf128_full",
        "v10_tabicl_A", "v10_tabicl_B",
        "c50_ravaghi_lgb1", "c51_ravaghi_lgb2", "c52_ravaghi_lgb3", "c53_ravaghi_cat1", "c54_ravaghi_cat2",
        "c76_pilkwang_blend_pp", "c75_pilkwang_blend", "c74_pilkwang_tcn", "c71_pilkwang_catboost",
        "c77_hamatz_b_meta",
        "c86_v11fresh_cat3", "c85_v11fresh_cat2",
    ],
    "runv7_plus_pilkwang_only": [
        "c20_r9_pf128_full",
        "v10_tabicl_A", "v10_tabicl_B",
        "c50_ravaghi_lgb1", "c51_ravaghi_lgb2", "c52_ravaghi_lgb3", "c53_ravaghi_cat1", "c54_ravaghi_cat2",
        "c76_pilkwang_blend_pp", "c75_pilkwang_blend", "c74_pilkwang_tcn", "c71_pilkwang_catboost",
    ],
    "runv7_plus_pilkwang_pp": [
        "c20_r9_pf128_full",
        "v10_tabicl_A", "v10_tabicl_B",
        "c50_ravaghi_lgb1", "c51_ravaghi_lgb2", "c52_ravaghi_lgb3", "c53_ravaghi_cat1", "c54_ravaghi_cat2",
        "c76_pilkwang_blend_pp",
    ],
}


def run(label, cid_list):
    t0 = time.time()
    print(f"=== Selected hillclimb: {label} ===")
    pool = {}
    base_df = None
    for cid in cid_list:
        df, meta = load_oof(cid)
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        pool[cid] = df["oof_pred"].to_numpy(np.float64)
        print(f"  {cid:26s} perwell={meta.get('perwell_oof', float('nan')):.4f}")

    target = base_df["target"].to_numpy(np.float64)
    wells = base_df["well"].values
    folds = base_df["fold"].to_numpy(int)
    n_folds = int(folds.max()) + 1
    final = np.zeros(len(target), dtype=np.float64)
    fold_histories, fold_weights = [], []

    for fold in range(n_folds):
        tr = folds != fold
        va = folds == fold
        codes_tr, n_wells_tr, counts_tr = encode_wells(wells[tr])
        codes_va, n_wells_va, counts_va = encode_wells(wells[va])
        hist, _ = hillclimb_one_fold(
            {k: v[tr] for k, v in pool.items()}, target[tr], codes_tr, n_wells_tr, counts_tr,
            cid_list, verbose_fold=fold,
        )
        fw = to_weights(hist)
        fold_histories.append(hist)
        fold_weights.append(fw)
        pred = np.zeros(int(va.sum()), dtype=np.float64)
        for cid, w in fw.items():
            pred += w * pool[cid][va]
        final[va] = pred
        score = perwell_rmse_fast(target[va], pred, codes_va, n_wells_va, counts_va)
        print(f"  → fold {fold} val perwell={score:.4f} ({len(hist)} steps)")
        for cid, w in sorted(fw.items(), key=lambda x: -x[1])[:8]:
            print(f"      {cid:26s} {w:.4f}")

    codes_all, n_wells_all, counts_all = encode_wells(wells)
    overall = perwell_rmse_fast(target, final, codes_all, n_wells_all, counts_all)
    avg = {cid: float(np.mean([fw.get(cid, 0.0) for fw in fold_weights])) for cid in set().union(*[set(fw) for fw in fold_weights])}
    avg = dict(sorted(((k, v) for k, v in avg.items() if v > 1e-4), key=lambda x: -x[1]))
    print(f"\n=== Honest OOF: per-well RMSE = {overall:.6f} ===")
    for cid, w in list(avg.items())[:12]:
        print(f"  {cid:26s} {w:.4f}")
    out_dir = ROUND_DIR / "results/hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label,
        "candidate_ids": cid_list,
        "perwell_oof": float(overall),
        "fold_histories": fold_histories,
        "averaged_weights": avg,
        "wall_seconds": time.time() - t0,
    }
    (out_dir / f"{label}.json").write_text(json.dumps(summary, indent=2))
    df_out = base_df.copy()
    df_out["oof_pred"] = final.astype(np.float32)
    df_out.to_parquet(out_dir / f"{label}_oof.parquet")
    print(f"  → {out_dir / f'{label}.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    run(args.label or args.set, SETS[args.set])


if __name__ == "__main__":
    main()
