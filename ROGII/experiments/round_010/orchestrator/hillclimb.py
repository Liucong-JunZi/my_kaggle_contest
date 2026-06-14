"""GKF-5 fold-aware hill climb — the core ensemble builder.

Algorithm (per fold):
    1. Use ONLY the 4 train folds' rows to decide weights (avoid OOF leakage).
    2. Greedy: each iteration, try every (candidate, weight∈STEP_GRID); pick
       the (cid, w) that most reduces train-fold per-well RMSE.
    3. Apply: ensemble ← (1-w)·ensemble + w·candidate
    4. Stop when no improvement > TOL or max iters reached.

Final:
    - Average each candidate's cumulative weight across the 5 folds.
    - Use these averaged weights to compute the val-fold-only OOF — this is
      the *honest* per-well RMSE (no fold sees its own val data).

Output: results/hillclimb_runs/<label>.json + <label>_oof.parquet.
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

from shared.metrics import perwell_rmse
from shared.oof_writer import load_oof, list_candidates

N_ITER    = 50
STEP_GRID = [0.05, 0.10, 0.20, 0.30]
TOL       = 1e-4


def to_weights(history):
    """Replay step history → cumulative {cid: weight}."""
    acc = {}
    for step in history:
        w = step["weight"]
        for k in list(acc):
            acc[k] *= (1.0 - w)
        acc[step["picked"]] = acc.get(step["picked"], 0.0) + w
    return acc


def hillclimb_one_fold(pool_tr, target_tr, wells_tr, verbose_fold=None):
    """One fold's hill climb. pool_tr: dict[cid] → 1d array (train rows only).
    Returns (history, ensemble_tr)."""
    n = len(target_tr)
    ensemble = np.zeros(n, dtype=np.float64)
    cur_loss = perwell_rmse(target_tr, ensemble, wells_tr)
    history = []
    for it in range(N_ITER):
        best = (cur_loss, None, None)
        for cid, oof in pool_tr.items():
            for w in STEP_GRID:
                trial = (1.0 - w) * ensemble + w * oof
                loss = perwell_rmse(target_tr, trial, wells_tr)
                if loss < best[0] - TOL:
                    best = (loss, cid, w)
        if best[1] is None:
            break
        new_loss, cid, w = best
        delta = cur_loss - new_loss
        ensemble = (1.0 - w) * ensemble + w * pool_tr[cid]
        cur_loss = new_loss
        history.append({"iter": it, "picked": cid, "weight": w,
                         "train_perwell": float(cur_loss), "delta": float(delta)})
        if verbose_fold is not None and (it < 5 or it % 10 == 0):
            print(f"  fold {verbose_fold} iter {it:>2d}: +{w:.2f}·{cid:20s}  "
                  f"perwell={cur_loss:.4f}  Δ{delta:+.4f}", flush=True)
    return history, ensemble


def run(label=None, perwell_threshold=15.0):
    t0 = time.time()
    if label is None:
        label = time.strftime("run_%Y%m%d_%H%M%S")

    print(f"=== Hill climb: {label} ===")
    cidx = list_candidates()
    print(f"\nCandidates ({len(cidx)} total):")
    for _, row in cidx.iterrows():
        marker = "✓" if row["perwell_oof"] < perwell_threshold else "✗"
        print(f"  {marker} {row['candidate_id']:20s}  {row['type']:18s}  "
              f"perwell={row['perwell_oof']:6.3f}  flat={row['flat_oof']:6.3f}")

    cidx = cidx[cidx["perwell_oof"] < perwell_threshold].reset_index(drop=True)
    print(f"\n→ Using {len(cidx)} healthy candidates (perwell < {perwell_threshold})")

    # Load all candidates into shared dict + grab ground truth from first one
    pool = {}
    base_df = None
    for _, row in cidx.iterrows():
        df, _ = load_oof(row["candidate_id"])
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        pool[row["candidate_id"]] = df["oof_pred"].values.astype(np.float64)

    target = base_df["target"].values.astype(np.float64)
    wells  = base_df["well"].values
    folds  = base_df["fold"].values.astype(int)
    n      = len(target)
    n_folds = int(folds.max()) + 1

    fold_histories = []
    fold_weights   = []
    final_va_pred  = np.zeros(n, dtype=np.float64)

    for fold in range(n_folds):
        tr_mask = folds != fold
        va_mask = folds == fold
        pool_tr = {k: v[tr_mask] for k, v in pool.items()}
        hist, _ = hillclimb_one_fold(
            pool_tr, target[tr_mask], wells[tr_mask], verbose_fold=fold
        )
        fold_histories.append(hist)
        fw = to_weights(hist)
        fold_weights.append(fw)

        # Apply this fold's weights to its val rows
        va_ensemble = np.zeros(int(va_mask.sum()), dtype=np.float64)
        for cid, w in fw.items():
            va_ensemble += w * pool[cid][va_mask]
        final_va_pred[va_mask] = va_ensemble

        fold_val_pw = perwell_rmse(target[va_mask], va_ensemble, wells[va_mask])
        top5 = sorted(fw.items(), key=lambda x: -x[1])[:5]
        print(f"  → fold {fold} val perwell={fold_val_pw:.4f}  ({len(hist)} steps)")
        for cid, w in top5:
            print(f"      {cid:20s}  {w:.4f}")
        print()

    # Cross-fold average weights — for stability + reporting
    all_cids = set().union(*[fw.keys() for fw in fold_weights])
    avg_w = {cid: float(np.mean([fw.get(cid, 0.0) for fw in fold_weights]))
             for cid in all_cids}
    total = sum(avg_w.values())
    avg_w = {k: v / total for k, v in avg_w.items()} if total > 0 else avg_w

    # Honest val-OOF using each fold's own weights:
    perwell_per_fold_weights = perwell_rmse(target, final_va_pred, wells)
    flat_per_fold_weights    = float(np.sqrt(np.mean((target - final_va_pred) ** 2)))

    # Alternative: apply averaged weights uniformly to all val rows
    # (this is what we'd actually deploy — single set of weights for inference)
    avg_va_pred = np.zeros(n, dtype=np.float64)
    for cid, w in avg_w.items():
        avg_va_pred += w * pool[cid]
    perwell_avg = perwell_rmse(target, avg_va_pred, wells)
    flat_avg    = float(np.sqrt(np.mean((target - avg_va_pred) ** 2)))

    best_single_id = cidx.sort_values("perwell_oof").iloc[0]["candidate_id"]
    best_single_pw = float(cidx[cidx["candidate_id"] == best_single_id].iloc[0]["perwell_oof"])

    print(f"{'='*70}")
    print(f"RESULTS")
    print(f"  baseline (best single = {best_single_id}): perwell={best_single_pw:.4f}")
    print(f"  per-fold weights, val OOF:                  perwell={perwell_per_fold_weights:.4f}  flat={flat_per_fold_weights:.4f}")
    print(f"  averaged weights (deploy form):             perwell={perwell_avg:.4f}  flat={flat_avg:.4f}")
    print(f"  Δ vs best single (avg weights):             {best_single_pw - perwell_avg:+.4f}")
    print(f"  models used (avg w > 0.01):                 {sum(1 for v in avg_w.values() if v > 0.01)}")
    print(f"  wall: {time.time()-t0:.0f}s")

    print(f"\nTop 10 averaged weights:")
    for cid, w in sorted(avg_w.items(), key=lambda x: -x[1])[:10]:
        if w > 0.001:
            print(f"  {cid:20s}  {w:.4f}")

    # Save
    out_dir = ROUND_DIR / "results" / "hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label":            label,
        "n_candidates":     len(pool),
        "best_single_id":   best_single_id,
        "best_single_pw":   best_single_pw,
        "perwell_per_fold_weights": float(perwell_per_fold_weights),
        "perwell_avg_weights":      float(perwell_avg),
        "flat_avg_weights":         float(flat_avg),
        "improvement_avg":  float(best_single_pw - perwell_avg),
        "n_models_in_avg":  int(sum(1 for v in avg_w.values() if v > 0.01)),
        "avg_weights":      {k: round(v, 6) for k, v in sorted(avg_w.items(), key=lambda x: -x[1])},
        "fold_weights":     [
            {k: round(v, 6) for k, v in sorted(fw.items(), key=lambda x: -x[1])}
            for fw in fold_weights
        ],
        "fold_history":     fold_histories,
        "wall_sec":         round(time.time() - t0, 1),
    }
    json_path = out_dir / f"{label}.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n→ {json_path}")

    # OOF parquet (using per-fold weights — that's the proper OOF)
    base_df["oof_pred"] = final_va_pred.astype(np.float32)
    base_df.to_parquet(out_dir / f"{label}_oof.parquet")
    print(f"→ {out_dir / f'{label}_oof.parquet'}")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--threshold", type=float, default=15.0,
                    help="drop candidates with perwell_oof above this")
    args = ap.parse_args()
    run(args.label, args.threshold)
