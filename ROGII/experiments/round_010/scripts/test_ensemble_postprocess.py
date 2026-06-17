#!/usr/bin/env python3
"""Test LB-7.776-style postprocess on a completed hillclimb OOF ensemble."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from orchestrator.hillclimb_fast import encode_wells, perwell_rmse_fast


def apply_pp(raw_offset, pf_offset, md_since, alpha=1.0, tau=85.0, w_pf=0.09):
    ramp = 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)
    return alpha * ((1.0 - w_pf) * raw_offset + w_pf * pf_offset) * ramp


def smooth_by_well(df, values, window=17, polyorder=3):
    out = np.asarray(values, dtype=np.float64).copy()
    for _, idx in df.groupby("well", sort=False).indices.items():
        n = len(idx)
        if n < window:
            continue
        win = window if window % 2 == 1 else window - 1
        if win > n:
            win = n if n % 2 == 1 else n - 1
        if win > polyorder:
            out[idx] = savgol_filter(out[idx], window_length=win, polyorder=polyorder, mode="interp")
    return out


def score(label, target, pred, codes, n_wells, counts):
    pw = perwell_rmse_fast(target, pred, codes, n_wells, counts)
    fl = float(np.sqrt(np.mean((target - pred) ** 2)))
    print(f"{label:30s} perwell={pw:.6f} flat={fl:.6f} pred_std={np.std(pred):.6f}")
    return {"perwell": float(pw), "flat": fl, "pred_std": float(np.std(pred))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="run_v6_with_pp7776")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=85.0)
    ap.add_argument("--w-pf", type=float, default=0.09)
    ap.add_argument("--sg-window", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    args = ap.parse_args()

    oof_path = ROUND_DIR / f"results/hillclimb_runs/{args.run}_oof.parquet"
    if not oof_path.exists():
        raise FileNotFoundError(oof_path)

    df = pd.read_parquet(oof_path)
    feat = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "md_offset", "pf_ancc_offset", "pf_ens_s12_offset"],
    )
    if not (df["well"].equals(feat["well"]) and df["row_idx"].astype("int32").equals(feat["row_idx"].astype("int32"))):
        raise ValueError("row order mismatch")

    target = df["target"].values.astype(np.float64)
    raw = df["oof_pred"].values.astype(np.float64)
    pf = feat["pf_ancc_offset"].fillna(feat["pf_ens_s12_offset"]).values.astype(np.float64)
    md = feat["md_offset"].values.astype(np.float64)

    codes, n_wells, counts = encode_wells(df["well"].values)
    results = {}
    print(f"=== Test final postprocess on {args.run} ===")
    results["raw"] = score("raw", target, raw, codes, n_wells, counts)

    pp = apply_pp(raw, pf, md, alpha=args.alpha, tau=args.tau, w_pf=args.w_pf)
    results["pp"] = score("apply_pp", target, pp, codes, n_wells, counts)

    pp_sg = smooth_by_well(df, pp, window=args.sg_window, polyorder=args.sg_poly)
    results["pp_sg"] = score("apply_pp + sg", target, pp_sg, codes, n_wells, counts)

    raw_sg = smooth_by_well(df, raw, window=args.sg_window, polyorder=args.sg_poly)
    results["raw_sg"] = score("raw + sg", target, raw_sg, codes, n_wells, counts)

    out = {
        "run": args.run,
        "params": vars(args),
        "results": results,
    }
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.run}_final_postprocess_audit.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
