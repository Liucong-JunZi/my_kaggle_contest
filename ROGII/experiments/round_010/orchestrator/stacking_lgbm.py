"""Two-stage LightGBM stacker over first-stage OOF candidates.

Honest protocol: for meta fold f, train the meta-model only on rows with
fold != f using first-stage OOF predictions as features, then predict fold f.
The first-stage predictions are already OOF, so this avoids training the
second-stage model on rows it predicts.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.oof_writer import load_oof, list_candidates


PRESETS = {
    "v1": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_data_in_leaf": 2000,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l1": 0.0,
        "lambda_l2": 5.0,
        "max_depth": 5,
        "verbosity": -1,
        "seed": 20260617,
        "num_threads": -1,
    },
    "regularized": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "min_data_in_leaf": 8000,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.7,
        "bagging_freq": 1,
        "lambda_l1": 0.5,
        "lambda_l2": 20.0,
        "max_depth": 4,
        "verbosity": -1,
        "seed": 20260617,
        "num_threads": -1,
    },
    "linearish": {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.03,
        "num_leaves": 7,
        "min_data_in_leaf": 20000,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "lambda_l1": 1.0,
        "lambda_l2": 50.0,
        "max_depth": 3,
        "verbosity": -1,
        "seed": 20260617,
        "num_threads": -1,
    },
}


def encode_wells(wells: np.ndarray):
    codes, uniq = pd.factorize(wells, sort=False)
    counts = np.bincount(codes, minlength=len(uniq)).astype(np.float64)
    return codes.astype(np.int32), len(uniq), counts


def perwell_rmse_fast(target, pred, codes, n_wells, counts):
    diff = target - pred
    ss = np.bincount(codes, weights=diff * diff, minlength=n_wells)
    return float(np.sqrt(ss / counts).mean())


def row_weights_for_perwell(wells):
    codes, _, counts = encode_wells(wells)
    return (1.0 / counts[codes]).astype(np.float32)


def load_pool(perwell_threshold, top_n=None, add_extra=True):
    cidx = list_candidates()
    cidx = cidx[cidx["perwell_oof"] < perwell_threshold].reset_index(drop=True)
    if top_n is not None:
        cidx = cidx.sort_values("perwell_oof").head(top_n).reset_index(drop=True)
    print(f"Using {len(cidx)} candidates (perwell < {perwell_threshold}, top_n={top_n})")

    base_df = None
    cols = []
    names = []
    for _, row in cidx.iterrows():
        cid = row["candidate_id"]
        df, _ = load_oof(cid)
        if base_df is None:
            base_df = df[["well", "row_idx", "fold", "target"]].copy()
        cols.append(df["oof_pred"].values.astype(np.float32))
        names.append(cid)
    X = np.stack(cols, axis=1).astype(np.float32)

    if add_extra:
        # Cheap derived prediction-only features. These don't introduce leakage and
        # help trees model row-wise disagreement/magnitude without many splits.
        extra = np.stack([
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            np.median(X, axis=1),
        ], axis=1).astype(np.float32)
        extra_names = ["pred_mean", "pred_std", "pred_min", "pred_max", "pred_median"]
        X = np.concatenate([X, extra], axis=1)
        names = names + extra_names
    return base_df, X, names


def run(label, preset, perwell_threshold, num_boost_round, early_stopping_rounds, use_weights,
        top_n=None, add_extra=True):
    t0 = time.time()
    params = PRESETS[preset].copy()
    print(f"=== LGBM stacker: {label} preset={preset} weights={use_weights} ===")
    print(json.dumps(params, indent=2))

    base_df, X, feature_names = load_pool(perwell_threshold, top_n=top_n, add_extra=add_extra)
    target = base_df["target"].values.astype(np.float32)
    wells = base_df["well"].values
    folds = base_df["fold"].values.astype(int)
    n_folds = int(folds.max()) + 1
    final_pred = np.zeros(len(target), dtype=np.float32)
    fold_metrics = {}
    fold_best_iters = {}
    fold_importances = []

    print(f"Matrix: rows={X.shape[0]:,} cols={X.shape[1]} load_time={time.time()-t0:.1f}s")

    for fold in range(n_folds):
        tr = folds != fold
        va = folds == fold
        w_tr = row_weights_for_perwell(wells[tr]) if use_weights else None
        w_va = row_weights_for_perwell(wells[va]) if use_weights else None

        dtr = lgb.Dataset(X[tr], label=target[tr], weight=w_tr,
                          feature_name=feature_names, free_raw_data=False)
        dva = lgb.Dataset(X[va], label=target[va], weight=w_va,
                          feature_name=feature_names, reference=dtr,
                          free_raw_data=False)
        callbacks = [
            lgb.early_stopping(early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]
        t_f = time.time()
        model = lgb.train(
            params,
            dtr,
            num_boost_round=num_boost_round,
            valid_sets=[dtr, dva],
            valid_names=["train", "valid"],
            callbacks=callbacks,
        )
        pred = model.predict(X[va], num_iteration=model.best_iteration).astype(np.float32)
        final_pred[va] = pred

        codes_va, n_va, counts_va = encode_wells(wells[va])
        pw = perwell_rmse_fast(target[va], pred, codes_va, n_va, counts_va)
        fl = float(np.sqrt(((target[va] - pred) ** 2).mean()))
        fold_metrics[f"fold_{fold}"] = {"perwell": pw, "flat": fl, "n": int(va.sum())}
        fold_best_iters[f"fold_{fold}"] = int(model.best_iteration)
        fold_importances.append(model.feature_importance(importance_type="gain"))
        print(f"  → fold {fold}: perwell={pw:.4f} flat={fl:.4f} "
              f"best_iter={model.best_iteration} ({time.time()-t_f:.1f}s)")

    codes_all, n_all, counts_all = encode_wells(wells)
    overall_pw = perwell_rmse_fast(target, final_pred, codes_all, n_all, counts_all)
    overall_fl = float(np.sqrt(((target - final_pred) ** 2).mean()))
    print(f"\n=== Honest OOF: perwell={overall_pw:.4f} flat={overall_fl:.4f} ===")

    imp = np.mean(np.vstack(fold_importances), axis=0)
    top_imp = sorted(zip(feature_names, imp), key=lambda x: -x[1])[:20]
    print("Top importances:")
    for name, val in top_imp:
        print(f"  {name:25s} {val:.1f}")

    out_dir = ROUND_DIR / "results" / "hillclimb_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "label": label,
        "method": "lgbm_stacker",
        "preset": preset,
        "perwell_oof": float(overall_pw),
        "flat_oof": float(overall_fl),
        "n_features": len(feature_names),
        "feature_names": feature_names,
        "top_n": top_n,
        "add_extra": add_extra,
        "params": params,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "use_perwell_weights": use_weights,
        "fold_metrics": fold_metrics,
        "fold_best_iters": fold_best_iters,
        "top_importances": [(k, float(v)) for k, v in top_imp],
        "wall_seconds": time.time() - t0,
    }
    json_out = out_dir / f"{label}.json"
    parq_out = out_dir / f"{label}_oof.parquet"
    with open(json_out, "w") as f:
        json.dump(summary, f, indent=2)
    df_out = base_df.copy()
    df_out["oof_pred"] = final_pred
    df_out.to_parquet(parq_out)
    print(f"\n  → {json_out}")
    print(f"  → {parq_out}")
    print(f"  total wall: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default=None)
    ap.add_argument("--preset", choices=sorted(PRESETS), default="regularized")
    ap.add_argument("--threshold", type=float, default=15.0)
    ap.add_argument("--rounds", type=int, default=3000)
    ap.add_argument("--early", type=int, default=100)
    ap.add_argument("--no-weights", action="store_true",
                    help="Do not use inverse-per-well row weights")
    ap.add_argument("--top-n", type=int, default=None,
                    help="Use only top N candidates by perwell_oof")
    ap.add_argument("--no-extra", action="store_true",
                    help="Do not add pred_mean/std/min/max/median derived features")
    args = ap.parse_args()
    label = args.label or f"lgbm_stack_{args.preset}"
    run(label, args.preset, args.threshold, args.rounds, args.early,
        use_weights=not args.no_weights, top_n=args.top_n, add_extra=not args.no_extra)
