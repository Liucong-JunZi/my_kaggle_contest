#!/usr/bin/env python3
"""Fold-honest full-pipeline ensemble driver.

Generalizes run_selected_hillclimb.py + stacking_nnls.py into one driver with a
switchable fusion layer (hillclimb | nnls | ridge), optional in-fold apply_pp
grid search, and optional Savitzky-Golay smoothing. Reports a baseline ladder
with BOTH flat and per-well RMSE so OOF↔LB comparison is interpretable.

All stages run via shared/pipeline.py — the same module Kaggle inference uses.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from orchestrator.hillclimb_fast import encode_wells, hillclimb_one_fold, to_weights
from orchestrator.stacking_nnls import fit_nnls_fold
from shared.metrics import flat_rmse, perwell_rmse
from shared import pipeline as P
from shared.oof_writer import load_oof

SETS = {
    "runv12_pf_rava": [
        "c20_r9_pf128_full",
        "c50_ravaghi_lgb1", "c51_ravaghi_lgb2", "c52_ravaghi_lgb3",
        "c53_ravaghi_cat1", "c54_ravaghi_cat2",
    ],
}

PP_GRID = [
    {"alpha": a, "tau": t, "w_pf": w}
    for a in [0.98, 0.99, 1.0, 1.01, 1.02]
    for t in [35, 50, 65, 85, 105, 130, 170, 220]
    for w in [0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.16]
]


def fit_fusion(fuser, pool_tr, target_tr, cid_list, codes_tr, n_wells_tr, counts_tr, fold):
    """Return dict {cid: weight} for the chosen fusion method."""
    if fuser == "hillclimb":
        hist, _ = hillclimb_one_fold(pool_tr, target_tr, codes_tr, n_wells_tr,
                                     counts_tr, cid_list, verbose_fold=fold)
        return to_weights(hist)
    if fuser == "nnls":
        return fit_nnls_fold(pool_tr, target_tr, cid_list, normalize=True)
    raise ValueError(f"unknown fuser: {fuser}")


def run(label, cid_list, fuser="hillclimb", use_pp=False, use_sg=False, pf_cid="c20_r9_pf128_full"):
    t0 = time.time()
    print(f"=== pipeline_eval: {label} | fuser={fuser} pp={use_pp} sg={use_sg} ===")

    pool = {}
    base_df = None
    for cid in cid_list:
        df, meta = load_oof(cid)
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        pool[cid] = df["oof_pred"].to_numpy(np.float64)
        print(f"  {cid:26s} perwell={meta.get('perwell_oof', float('nan')):.4f} flat={meta.get('flat_oof', float('nan')):.4f}")

    # md_since from joined_features (md_offset), aligned by (well,row_idx)
    from shared.data_loader import load_joined
    joined = load_joined()[["well", "row_idx", "md_offset"]]
    base_df = base_df.merge(joined, on=["well", "row_idx"], how="left", validate="one_to_one")
    assert base_df["md_offset"].notna().all(), "md_offset join produced NaN"

    target = base_df["target"].to_numpy(np.float64)
    wells = base_df["well"].values
    folds = base_df["fold"].to_numpy(int)
    md_since = base_df["md_offset"].to_numpy(np.float64)
    last_known = np.zeros(len(target))  # offset space: absolute add is identity for scoring
    n_folds = int(folds.max()) + 1
    codes_all, n_wells_all, counts_all = encode_wells(wells)
    pf_off_all = pool[pf_cid]

    # Stage outputs accumulated across folds (held-out only)
    stages = {"raw": np.zeros(len(target)), "pp": np.zeros(len(target)), "sg": np.zeros(len(target))}
    fold_weights, fold_pp = [], []

    for fold in range(n_folds):
        tr = folds != fold
        va = folds == fold
        codes_tr, n_wells_tr, counts_tr = encode_wells(wells[tr])
        fw = fit_fusion(fuser, {k: v[tr] for k, v in pool.items()}, target[tr],
                        cid_list, codes_tr, n_wells_tr, counts_tr, fold)
        fold_weights.append(fw)

        comps_va = {k: v[va] for k, v in pool.items()}
        raw_va = P.blend_offsets(comps_va, fw)
        stages["raw"][va] = raw_va

        pp_va = raw_va
        chosen_pp = None
        if use_pp:
            # In-fold grid search on TRAIN rows only (no leakage)
            comps_tr = {k: v[tr] for k, v in pool.items()}
            raw_tr = P.blend_offsets(comps_tr, fw)
            codes_tr2, nw_tr2, cnt_tr2 = codes_tr, n_wells_tr, counts_tr
            best = (perwell_rmse(target[tr], raw_tr, wells[tr]), None)
            for params in PP_GRID:
                d = P.apply_pp(raw_tr, pf_off_all[tr], md_since[tr], **params)
                sc = perwell_rmse(target[tr], d, wells[tr])
                if sc < best[0] - 1e-9:
                    best = (sc, params)
            chosen_pp = best[1]
            if chosen_pp is not None:
                pp_va = P.apply_pp(raw_va, pf_off_all[va], md_since[va], **chosen_pp)
        fold_pp.append(chosen_pp)
        stages["pp"][va] = pp_va

        sg_va = pp_va
        if use_sg:
            codes_va_local, _, _ = encode_wells(wells[va])
            sg_va = P.sg_smooth_offsets(pp_va, codes_va_local)
        stages["sg"][va] = sg_va

        cv, nv, ctv = encode_wells(wells[va])
        print(f"  → fold {fold}: raw pw={perwell_rmse(target[va], raw_va, wells[va]):.4f} "
              f"pp={perwell_rmse(target[va], pp_va, wells[va]):.4f} "
              f"sg={perwell_rmse(target[va], sg_va, wells[va]):.4f} pp_params={chosen_pp}")

    # ── Baseline ladder (flat + perwell) ──
    def metrics(pred):
        return {"flat": flat_rmse(target, pred), "perwell": perwell_rmse(target, pred, wells)}

    ladder = {
        "hold_last": metrics(np.zeros(len(target))),
        "raw_blend": metrics(stages["raw"]),
    }
    final = stages["raw"]
    if use_pp:
        ladder["+apply_pp"] = metrics(stages["pp"]); final = stages["pp"]
    if use_sg:
        ladder["+sg_smooth"] = metrics(stages["sg"]); final = stages["sg"]

    print("\n=== Stage ladder (flat / perwell) ===")
    for name, m in ladder.items():
        print(f"  {name:14s} flat={m['flat']:.4f}  perwell={m['perwell']:.4f}")

    avg = {cid: float(np.mean([fw.get(cid, 0.0) for fw in fold_weights]))
           for cid in set().union(*[set(fw) for fw in fold_weights])}
    avg = dict(sorted(((k, v) for k, v in avg.items() if v > 1e-4), key=lambda x: -x[1]))

    # Aggregate per-fold pp choices into one global pp setting (mean of numeric params).
    global_pp = None
    if use_pp:
        chosen = [p for p in fold_pp if p]
        if chosen:
            global_pp = {
                "alpha": float(np.mean([p["alpha"] for p in chosen])),
                "tau": float(np.mean([p["tau"] for p in chosen])),
                "w_pf": float(np.mean([p["w_pf"] for p in chosen])),
            }
    global_sg = {"sg_w": 17, "sg_p": 3} if use_sg else None

    # Full pipeline spec — consumed verbatim by Kaggle inference (ensemble_weights.json).
    spec = {
        "label": label,
        "fuser": fuser,
        "prediction_space": "offset",
        "final_output_space": "absolute_tvt",
        "weights": avg,
        "pf_offset_component": pf_cid,
        "pp_params": global_pp,
        "sg_params": global_sg,
    }

    out_dir = ROUND_DIR / "results/hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label, "fuser": fuser, "use_pp": use_pp, "use_sg": use_sg,
        "candidate_ids": cid_list,
        "perwell_oof": ladder[list(ladder)[-1]]["perwell"],
        "flat_oof": ladder[list(ladder)[-1]]["flat"],
        "stage_ladder": ladder,
        "averaged_weights": avg,
        "fold_pp_params": fold_pp,
        "spec": spec,
        "wall_seconds": time.time() - t0,
    }
    (out_dir / f"{label}.json").write_text(json.dumps(summary, indent=2))
    df_out = base_df[["well", "row_idx", "fold", "target"]].copy()
    df_out["oof_pred"] = final.astype(np.float32)
    df_out.to_parquet(out_dir / f"{label}_oof.parquet")
    print(f"\n  → {out_dir / f'{label}.json'}")
    print(f"  weights: {avg}")
    print(f"  global pp={global_pp} sg={global_sg}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=SETS, required=True)
    ap.add_argument("--label", default=None)
    ap.add_argument("--fuser", choices=["hillclimb", "nnls"], default="hillclimb")
    ap.add_argument("--pp", action="store_true")
    ap.add_argument("--sg", action="store_true")
    args = ap.parse_args()
    label = args.label or f"{args.set}_{args.fuser}{'_pp' if args.pp else ''}{'_sg' if args.sg else ''}"
    run(label, SETS[args.set], fuser=args.fuser, use_pp=args.pp, use_sg=args.sg)


if __name__ == "__main__":
    main()
