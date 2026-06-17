"""NNLS stacking — non-negative least squares blend over the candidate pool.

Same 5-fold honest OOF protocol as hillclimb_fast, but each fold fits NNLS on
the held-in train rows instead of greedy hill-climbing. NNLS gives every
candidate a chance: a weak-but-diverse model that hill climb refuses to pick
can still get a small non-zero weight if it reduces residual variance.

Output schema mirrors hillclimb_fast for drop-in comparison.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.oof_writer import load_oof, list_candidates


def encode_wells(wells: np.ndarray):
    codes, uniq = pd.factorize(wells, sort=False)
    n_wells = len(uniq)
    counts = np.bincount(codes, minlength=n_wells).astype(np.float64)
    return codes.astype(np.int32), n_wells, counts


def perwell_rmse_fast(target, pred, codes, n_wells, counts):
    diff = target - pred
    ss = np.bincount(codes, weights=diff * diff, minlength=n_wells)
    return float(np.sqrt(ss / counts).mean())


def fit_nnls_fold(pool_tr, target_tr, cid_list, normalize=False):
    """Solve min ||Aw - y||^2 s.t. w >= 0.  Returns dict {cid: weight}."""
    A = np.stack([pool_tr[cid] for cid in cid_list], axis=1)  # (n_train, n_cand)
    w, _ = nnls(A, target_tr)
    if normalize and w.sum() > 1e-9:
        w = w / w.sum()
    return {cid: float(weight) for cid, weight in zip(cid_list, w) if weight > 1e-6}


def run(label=None, perwell_threshold=15.0, normalize=False):
    t0 = time.time()
    if label is None:
        label = time.strftime("nnls_%Y%m%d_%H%M%S")
    print(f"=== NNLS stacking: {label}  (normalize={normalize}) ===")

    cidx = list_candidates()
    cidx = cidx[cidx["perwell_oof"] < perwell_threshold].reset_index(drop=True)
    print(f"\nUsing {len(cidx)} candidates (perwell < {perwell_threshold})")

    pool = {}
    base_df = None
    for _, row in cidx.iterrows():
        df, _ = load_oof(row["candidate_id"])
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        pool[row["candidate_id"]] = df["oof_pred"].values.astype(np.float64)

    target = base_df["target"].values.astype(np.float64)
    wells_str = base_df["well"].values
    folds = base_df["fold"].values.astype(int)
    n_folds = int(folds.max()) + 1
    cid_list = list(pool.keys())

    fold_weights = []
    final_va_pred = np.zeros(len(target), dtype=np.float64)

    for fold in range(n_folds):
        tr_mask = folds != fold
        va_mask = folds == fold
        target_tr = target[tr_mask]
        pool_tr = {k: v[tr_mask] for k, v in pool.items()}
        codes_va, n_wells_va, counts_va = encode_wells(wells_str[va_mask])

        t_f = time.time()
        fw = fit_nnls_fold(pool_tr, target_tr, cid_list, normalize=normalize)
        fold_weights.append(fw)

        va_ensemble = np.zeros(int(va_mask.sum()), dtype=np.float64)
        for cid, w in fw.items():
            va_ensemble += w * pool[cid][va_mask]
        final_va_pred[va_mask] = va_ensemble

        fold_val_pw = perwell_rmse_fast(target[va_mask], va_ensemble,
                                        codes_va, n_wells_va, counts_va)
        # Print top picks (sorted by weight)
        top = sorted(fw.items(), key=lambda x: -x[1])
        wsum = sum(w for _, w in top)
        print(f"  → fold {fold} val perwell={fold_val_pw:.4f}  "
              f"({len(top)} non-zero, sum_w={wsum:.4f})  ({time.time()-t_f:.1f}s)")
        for cid, w in top[:8]:
            print(f"      {cid:25s}  {w:.4f}")

    # Cross-fold averaged weights
    all_cids = set().union(*[set(fw.keys()) for fw in fold_weights])
    avg_w = {cid: float(np.mean([fw.get(cid, 0.0) for fw in fold_weights]))
             for cid in all_cids}
    avg_w = {k: v for k, v in sorted(avg_w.items(), key=lambda x: -x[1]) if v > 1e-4}

    codes_all, n_wells_all, counts_all = encode_wells(wells_str)
    overall_pw = perwell_rmse_fast(target, final_va_pred, codes_all, n_wells_all, counts_all)
    print(f"\n=== Honest OOF: per-well RMSE = {overall_pw:.4f} ===")
    print("Averaged weights (top 15):")
    for k, v in list(avg_w.items())[:15]:
        print(f"  {k:25s}  {v:.4f}")

    out_dir = ROUND_DIR / "results" / "hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label,
        "method": "nnls",
        "normalize": normalize,
        "perwell_oof": float(overall_pw),
        "n_candidates_pool": len(cid_list),
        "fold_weights": fold_weights,
        "averaged_weights": avg_w,
        "wall_seconds": time.time() - t0,
    }
    json_out = out_dir / f"{label}.json"
    parq_out = out_dir / f"{label}_oof.parquet"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    df_out = base_df.copy()
    df_out["oof_pred"] = final_va_pred.astype(np.float32)
    df_out.to_parquet(parq_out)
    print(f"\n  → {json_out}")
    print(f"  → {parq_out}")
    print(f"  total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--normalize", action="store_true",
                    help="Re-normalize weights to sum=1 (typical convex blend)")
    args = ap.parse_args()
    run(label=args.label, perwell_threshold=args.threshold, normalize=args.normalize)
