#!/usr/bin/env python3
"""Analyze complementarity of imported public candidates against run_v7 OOF."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
CANDIDATES = [
    "c70_pilkwang_xgb", "c71_pilkwang_catboost", "c72_pilkwang_hgb",
    "c73_pilkwang_lgb", "c74_pilkwang_tcn", "c75_pilkwang_blend",
    "c76_pilkwang_blend_pp", "c77_hamatz_b_meta", "c78_hamatz_b_lgb0",
    "c79_hamatz_b_lgb1", "c80_hamatz_b_lgb2", "c81_v11fresh_lgb1",
    "c82_v11fresh_lgb2", "c83_v11fresh_lgb3", "c84_v11fresh_cat1",
    "c85_v11fresh_cat2", "c86_v11fresh_cat3",
]


def perwell_rmse(y, p, wells):
    return float(pd.DataFrame({"w": wells, "d": (y - p) ** 2}).groupby("w").d.mean().pow(0.5).mean())


def main():
    base = pd.read_parquet(ROUND_DIR / "results/hillclimb_runs/run_v7_with_v10_tabicl_oof.parquet")
    y = base["target"].to_numpy(np.float64)
    rv7 = base["oof_pred"].to_numpy(np.float64)
    wells = base["well"].values
    rv7_pw = perwell_rmse(y, rv7, wells)
    rv7_res = y - rv7
    rows = []
    for cid in CANDIDATES:
        path = ROUND_DIR / f"results/candidates/{cid}.parquet"
        if not path.exists():
            continue
        p = pd.read_parquet(path, columns=["oof_pred"])["oof_pred"].to_numpy(np.float64)
        e = y - p
        best_score, best_alpha = rv7_pw, 0.0
        for alpha in np.linspace(0.01, 0.30, 30):
            score = perwell_rmse(y, (1.0 - alpha) * rv7 + alpha * p, wells)
            if score < best_score:
                best_score, best_alpha = score, float(alpha)
        rows.append({
            "candidate_id": cid,
            "solo_perwell": perwell_rmse(y, p, wells),
            "pred_corr_with_run_v7": float(np.corrcoef(rv7, p)[0, 1]),
            "err_corr_with_run_v7": float(np.corrcoef(rv7_res, e)[0, 1]),
            "best_grid_blend_perwell": best_score,
            "best_grid_alpha": best_alpha,
            "gain_vs_run_v7": rv7_pw - best_score,
        })
    rows = sorted(rows, key=lambda r: -r["gain_vs_run_v7"])
    out = {"baseline_run_v7_perwell": rv7_pw, "rows": rows}
    out_dir = ROUND_DIR / "results/audits"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "public_candidate_complementarity.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(f"run_v7={rv7_pw:.6f}")
    for r in rows:
        print(f"{r['candidate_id']:26s} solo={r['solo_perwell']:.3f} best={r['best_grid_blend_perwell']:.4f} "
              f"alpha={r['best_grid_alpha']:.2f} gain={r['gain_vs_run_v7']:.4f}")
    print(f"Report: {out_path}")


if __name__ == "__main__":
    main()
