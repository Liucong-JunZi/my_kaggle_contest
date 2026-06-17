#!/usr/bin/env python3
"""Create LB-7.776-style postprocessed OOF candidate variants.

The public ridge-sp kernels apply a fixed postprocess to offset predictions:
blend a model offset with PF-ANCC offset, ramp it away from the known segment,
then optionally smooth per well. This script treats those transforms as new OOF
candidates so hillclimb can evaluate them honestly.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

from shared.oof_writer import load_oof, write_oof

DEFAULT_CANDIDATES = [
    "c20_r9_pf128_full",
    "c50_ravaghi_lgb1",
    "c51_ravaghi_lgb2",
    "c52_ravaghi_lgb3",
    "c53_ravaghi_cat1",
    "c54_ravaghi_cat2",
]


def apply_pp(raw_offset, pf_offset, md_since, alpha=1.0, tau=85.0, w_pf=0.09):
    ramp = 1.0 - np.exp(-np.maximum(md_since, 0.0) / tau)
    return alpha * ((1.0 - w_pf) * raw_offset + w_pf * pf_offset) * ramp


def smooth_by_well(df, pred_col, window=17, polyorder=3):
    out = np.empty(len(df), dtype=np.float32)
    for _, idx in df.groupby("well", sort=False).indices.items():
        vals = df.iloc[idx][pred_col].values.astype(np.float64)
        n = len(vals)
        if n >= window:
            win = window if window % 2 == 1 else window - 1
            if win > n:
                win = n if n % 2 == 1 else n - 1
            if win > polyorder:
                vals = savgol_filter(vals, window_length=win, polyorder=polyorder, mode="interp")
        out[idx] = vals.astype(np.float32)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", default=DEFAULT_CANDIDATES)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=85.0)
    ap.add_argument("--w-pf", type=float, default=0.09)
    ap.add_argument("--sg-window", type=int, default=17)
    ap.add_argument("--sg-poly", type=int, default=3)
    args = ap.parse_args()

    t0 = time.time()
    print("=== Create postprocessed OOF candidates ===")
    print(json.dumps(vars(args), indent=2))

    feat = pd.read_parquet(
        ROUND_DIR / "results/joined_features.parquet",
        columns=["well", "row_idx", "md_offset", "last_known_tvt", "pf_ancc_offset", "pf_ens_s12_offset"],
    )
    pf_offset = feat["pf_ancc_offset"].fillna(feat["pf_ens_s12_offset"]).values.astype(np.float32)
    md_since = feat["md_offset"].values.astype(np.float32)

    made = []
    for cid in args.candidates:
        print(f"\n→ {cid}")
        df, meta = load_oof(cid)
        if not (
            df["well"].equals(feat["well"])
            and df["row_idx"].astype("int32").equals(feat["row_idx"].astype("int32"))
        ):
            raise ValueError(f"row order mismatch for {cid}")

        pp = apply_pp(
            df["oof_pred"].values.astype(np.float32),
            pf_offset,
            md_since,
            alpha=args.alpha,
            tau=args.tau,
            w_pf=args.w_pf,
        ).astype(np.float32)
        base = df[["well", "row_idx", "fold", "target"]].copy()
        base["oof_pred"] = pp
        pp_id = f"{cid}_pp7776"
        out = write_oof(
            candidate_id=pp_id,
            df_oof=base,
            candidate_type="postprocess",
            features_used=[cid, "pf_ancc_offset", "md_offset", "lb7776_apply_pp"],
            hyperparams={"alpha": args.alpha, "tau": args.tau, "w_pf": args.w_pf},
            seed=int(meta.get("seed", 0) if meta else 0),
            train_time_sec=0.0,
            extra_meta={"source_candidate": cid, "postprocess": "lb7776_apply_pp"},
        )
        print(f"  wrote {pp_id} → {out}")
        made.append(pp_id)

        tmp = base[["well", "row_idx", "fold", "target", "oof_pred"]].copy()
        tmp["oof_pred"] = smooth_by_well(tmp, "oof_pred", window=args.sg_window, polyorder=args.sg_poly)
        sg_id = f"{cid}_pp7776_sg17"
        out = write_oof(
            candidate_id=sg_id,
            df_oof=tmp,
            candidate_type="postprocess_sg",
            features_used=[cid, "pf_ancc_offset", "md_offset", "lb7776_apply_pp", "savgol"],
            hyperparams={
                "alpha": args.alpha,
                "tau": args.tau,
                "w_pf": args.w_pf,
                "sg_window": args.sg_window,
                "sg_poly": args.sg_poly,
            },
            seed=int(meta.get("seed", 0) if meta else 0),
            train_time_sec=0.0,
            extra_meta={"source_candidate": cid, "postprocess": "lb7776_apply_pp_savgol"},
        )
        print(f"  wrote {sg_id} → {out}")
        made.append(sg_id)

    summary = {
        "made": made,
        "params": vars(args),
        "wall_seconds": time.time() - t0,
    }
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "postprocessed_oof_candidates.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\nMade {len(made)} candidates in {time.time()-t0:.0f}s")
    print(f"Summary: {out_path}")


if __name__ == "__main__":
    main()
