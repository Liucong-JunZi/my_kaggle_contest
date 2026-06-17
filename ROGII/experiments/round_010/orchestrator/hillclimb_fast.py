"""Fast hill climb — vectorized perwell RMSE via integer well codes + bincount.

Drop-in replacement for orchestrator/hillclimb.py with the same output schema
but ~50-100× faster per iteration on large pools. The original perwell_rmse
uses pandas groupby+apply over 3M rows — that's the bottleneck.

Speedup: encode wells once to int32 codes; for each trial compute squared
residual sum per well via np.bincount, then perwell_rmse = mean(sqrt(ss/n_per_well)).
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

from shared.oof_writer import load_oof, list_candidates

N_ITER    = 50
STEP_GRID = [0.05, 0.10, 0.20, 0.30]
TOL       = 1e-4


def encode_wells(wells: np.ndarray):
    """str array → (codes int32, n_wells int, well_counts float64 [n_wells])."""
    codes, uniq = pd.factorize(wells, sort=False)
    n_wells = len(uniq)
    counts = np.bincount(codes, minlength=n_wells).astype(np.float64)
    return codes.astype(np.int32), n_wells, counts


def perwell_rmse_fast(target: np.ndarray, pred: np.ndarray,
                      codes: np.ndarray, n_wells: int, counts: np.ndarray) -> float:
    """Vectorized perwell RMSE: mean over wells of sqrt(MSE_per_well)."""
    diff = target - pred
    ss = np.bincount(codes, weights=diff * diff, minlength=n_wells)
    return float(np.sqrt(ss / counts).mean())


def to_weights(history):
    acc = {}
    for step in history:
        w = step["weight"]
        for k in list(acc):
            acc[k] *= (1.0 - w)
        acc[step["picked"]] = acc.get(step["picked"], 0.0) + w
    return acc


def hillclimb_one_fold(pool_tr, target_tr, codes_tr, n_wells_tr, counts_tr,
                       cid_list, verbose_fold=None):
    n = len(target_tr)
    ensemble = np.zeros(n, dtype=np.float64)
    cur_loss = perwell_rmse_fast(target_tr, ensemble, codes_tr, n_wells_tr, counts_tr)
    history = []
    t_iter = time.time()
    for it in range(N_ITER):
        best = (cur_loss, None, None)
        # Vectorize trials over candidates and weights
        for cid in cid_list:
            oof = pool_tr[cid]
            for w in STEP_GRID:
                trial = (1.0 - w) * ensemble + w * oof
                loss = perwell_rmse_fast(trial.astype(np.float64) if trial.dtype != np.float64 else trial,
                                         target_tr * 0 + 0,  # dummy — not used; we passed target as 'target' separately
                                         codes_tr, n_wells_tr, counts_tr) if False else \
                       perwell_rmse_fast(target_tr, trial, codes_tr, n_wells_tr, counts_tr)
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
            dt = time.time() - t_iter; t_iter = time.time()
            print(f"  fold {verbose_fold} iter {it:>2d}: +{w:.2f}·{cid:24s}  "
                  f"perwell={cur_loss:.4f}  Δ{delta:+.4f}  ({dt:.1f}s)", flush=True)
    return history, ensemble


def run(label=None, perwell_threshold=15.0):
    t0 = time.time()
    if label is None:
        label = time.strftime("run_%Y%m%d_%H%M%S")

    print(f"=== Fast Hill climb: {label} ===")
    cidx = list_candidates()
    print(f"\nCandidates ({len(cidx)} total):")
    for _, row in cidx.iterrows():
        marker = "✓" if row["perwell_oof"] < perwell_threshold else "✗"
        print(f"  {marker} {row['candidate_id']:24s}  {row['type']:18s}  "
              f"perwell={row['perwell_oof']:6.3f}  flat={row['flat_oof']:6.3f}")

    cidx = cidx[cidx["perwell_oof"] < perwell_threshold].reset_index(drop=True)
    print(f"\n→ Using {len(cidx)} healthy candidates (perwell < {perwell_threshold})")

    pool = {}
    base_df = None
    for _, row in cidx.iterrows():
        df, _ = load_oof(row["candidate_id"])
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        pool[row["candidate_id"]] = df["oof_pred"].values.astype(np.float64)
    print(f"  pool loaded in {time.time()-t0:.0f}s")

    target = base_df["target"].values.astype(np.float64)
    wells_str = base_df["well"].values
    folds = base_df["fold"].values.astype(int)
    n_folds = int(folds.max()) + 1
    cid_list = list(pool.keys())

    fold_histories = []
    fold_weights   = []
    final_va_pred  = np.zeros(len(target), dtype=np.float64)

    for fold in range(n_folds):
        tr_mask = folds != fold
        va_mask = folds == fold
        codes_tr, n_wells_tr, counts_tr = encode_wells(wells_str[tr_mask])
        codes_va, n_wells_va, counts_va = encode_wells(wells_str[va_mask])
        target_tr = target[tr_mask]
        pool_tr = {k: v[tr_mask] for k, v in pool.items()}

        hist, _ = hillclimb_one_fold(
            pool_tr, target_tr, codes_tr, n_wells_tr, counts_tr,
            cid_list, verbose_fold=fold,
        )
        fold_histories.append(hist)
        fw = to_weights(hist); fold_weights.append(fw)

        va_ensemble = np.zeros(int(va_mask.sum()), dtype=np.float64)
        for cid, w in fw.items():
            va_ensemble += w * pool[cid][va_mask]
        final_va_pred[va_mask] = va_ensemble

        fold_val_pw = perwell_rmse_fast(target[va_mask], va_ensemble,
                                        codes_va, n_wells_va, counts_va)
        top5 = sorted(fw.items(), key=lambda x: -x[1])[:5]
        print(f"  → fold {fold} val perwell={fold_val_pw:.4f}  ({len(hist)} steps)")
        for cid, w in top5:
            print(f"      {cid:24s}  {w:.4f}")

    # Cross-fold averaged weights
    all_cids = set().union(*[set(fw.keys()) for fw in fold_weights])
    avg_w = {cid: float(np.mean([fw.get(cid, 0.0) for fw in fold_weights]))
             for cid in all_cids}
    avg_w = {k: v for k, v in sorted(avg_w.items(), key=lambda x: -x[1]) if v > 1e-4}

    codes_all, n_wells_all, counts_all = encode_wells(wells_str)
    overall_pw = perwell_rmse_fast(target, final_va_pred, codes_all, n_wells_all, counts_all)
    print(f"\n=== Honest OOF: per-well RMSE = {overall_pw:.4f} ===")
    print("Averaged weights (top 10):")
    for k, v in list(avg_w.items())[:10]:
        print(f"  {k:24s}  {v:.4f}")

    out_dir = ROUND_DIR / "results" / "hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label,
        "perwell_oof": float(overall_pw),
        "n_candidates_pool": len(cid_list),
        "fold_histories": fold_histories,
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
    args = ap.parse_args()
    run(label=args.label, perwell_threshold=args.threshold)
