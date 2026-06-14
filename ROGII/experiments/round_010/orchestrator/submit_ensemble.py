#!/usr/bin/env python3
"""Generate Kaggle submission.csv from hill climb weights + candidate predictions.

Usage:
    python orchestrator/submit_ensemble.py --weights results/hillclimb_runs/run_20260614_legacy8.json
    python orchestrator/submit_ensemble.py --weights results/hillclimb_runs/run_v2.json --output submission.csv
    python orchestrator/submit_ensemble.py --weights results/hillclimb_runs/run_v2.json --dry-run

The script:
  1. Loads averaged weights from a hill climb JSON (the `avg_weights` key).
  2. Loads each candidate's prediction parquet from results/candidates/.
  3. Blends predictions as a weighted sum.
  4. Converts from offset-space (TVT - last_known_tvt) to absolute TVT.
  5. Writes submission.csv with columns: id, tvt.

Prediction columns are auto-detected: `oof_pred` (OOF parquets) or `pred`
(full-test parquets). Use --pred-col to override.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND_DIR))

CANDIDATE_DIR = ROUND_DIR / "results" / "candidates"
JOINED_PATH   = ROUND_DIR / "results" / "joined_features.parquet"
TEST_DIR      = ROUND_DIR / ".." / ".." / "rogii-wellbore-geology-prediction" / "test"


def load_weights(path: str) -> dict[str, float]:
    """Load hill climb weights JSON → {candidate_id: weight}.

    Accepts either the full hill climb output (reads ``avg_weights`` key)
    or a bare {id: weight} dict.  Always normalises to sum = 1.
    """
    with open(path) as f:
        data = json.load(f)
    weights = data.get("avg_weights", data)
    total = sum(weights.values())
    if total <= 0:
        print("WARNING: weights sum to zero — using uniform weights")
        n = len(weights)
        weights = {k: 1.0 / n for k in weights}
    elif abs(total - 1.0) > 1e-6:
        weights = {k: v / total for k, v in weights.items()}
    return weights


def load_candidate_prediction(
    candidate_id: str,
    pred_col: str | None = None,
) -> pd.DataFrame:
    """Load a candidate parquet; return (well, row_idx, pred)."""
    p = CANDIDATE_DIR / f"{candidate_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(
            f"Candidate parquet not found: {p}\n"
            f"    Candidates available: {len(list(CANDIDATE_DIR.glob('*.parquet')))}"
        )

    df = pd.read_parquet(p)

    # ---- determine prediction column ----
    if pred_col is not None:
        col = pred_col
    elif "oof_pred" in df.columns:
        col = "oof_pred"
    elif "pred" in df.columns:
        col = "pred"
    else:
        raise ValueError(
            f"No prediction column found in {p}.\n"
            f"    Columns: {list(df.columns)}\n"
            f"    Use --pred-col to specify."
        )

    required = {"well", "row_idx", col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {p}: {missing}")

    return df[["well", "row_idx", col]].rename(columns={col: "pred"}).copy()


def main() -> None:
    t0 = time.time()

    ap = argparse.ArgumentParser(
        description="Generate Kaggle submission from hill climb weights + candidate predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--weights", required=True, help="Path to hill climb JSON (uses avg_weights key)")
    ap.add_argument("--output", default="submission.csv", help="Output submission.csv path")
    ap.add_argument("--pred-col", default=None,
                    help="Prediction column name (auto-detect: oof_pred → pred)")
    ap.add_argument("--abs-tvt", action="store_true",
                    help="Predictions are already absolute TVT (skip offset→abs conversion)")
    ap.add_argument("--test-wells", nargs="*", default=None,
                    help="Space-separated list of test well IDs (e.g. 000d7d20 00bbac68 00e12e8b). "
                         "Defaults to auto-detect from rogii-wellbore-geology-prediction/test/.")
    ap.add_argument("--all-wells", action="store_true",
                    help="Include ALL wells in output (not just test wells). Overrides --test-wells.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print blend info without writing output")
    args = ap.parse_args()

    # ---- 1. Load weights ----
    print(f"Loading weights from {args.weights}")
    weights = load_weights(args.weights)
    print(f"  {len(weights)} candidates (normalised, sum=1):")
    for cid, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"    {cid:24s}  {w:.4f}")

    # ---- 2. Load each candidate's predictions ----
    blended: pd.DataFrame | None = None
    for cid, w in weights.items():
        sys.stdout.write(f"  Loading {cid} ... ")
        sys.stdout.flush()
        try:
            df = load_candidate_prediction(cid, args.pred_col)
        except FileNotFoundError as e:
            print(f"\n  WARNING: {e} — skipping")
            continue

        print(f"{len(df):,} rows")

        if blended is None:
            blended = df[["well", "row_idx"]].copy()
            blended["pred"] = np.zeros(len(df), dtype=np.float64)

        blended["pred"] += w * df["pred"].values

    if blended is None:
        print("ERROR: no candidates could be loaded — nothing to blend.")
        sys.exit(1)

    n = len(blended)
    print(f"\nBlend: {n:,} rows from {len(weights)} candidates")

    # ---- 2b. Filter to test wells only (unless --all-wells) ----
    if args.all_wells:
        print("  Outputting ALL wells (--all-wells)")
    else:
        test_wells = args.test_wells
        if test_wells is None:
            # Auto-detect test wells from the test directory
            test_wells = sorted({
                p.name.split("__")[0]
                for p in TEST_DIR.glob("*.csv")
            })
            print(f"  Auto-detected {len(test_wells)} test wells from {TEST_DIR}")
        pre = len(blended)
        blended = blended[blended["well"].isin(test_wells)].copy()
        n = len(blended)
        skipped = pre - n
        print(f"  Filtered: {n:,} rows ({skipped:,} training rows removed)")
        if n == 0:
            print(f"  WARNING: no rows matched test wells {test_wells}")
            print(f"  Available wells: {sorted(blended['well'].unique()[:10])}...")
            sys.exit(1)

    # ---- 3. Convert offset predictions → absolute TVT ----
    if not args.abs_tvt:
        print("Converting offset predictions → absolute TVT ...")
        joined = pd.read_parquet(
            JOINED_PATH, columns=["well", "row_idx", "last_known_tvt"]
        )
        m = blended.merge(joined, on=["well", "row_idx"], how="left")
        missing_lkt = m["last_known_tvt"].isna().sum()
        if missing_lkt > 0:
            print(f"  WARNING: {missing_lkt:,} rows missing last_known_tvt — filling with 0")
            m["last_known_tvt"] = m["last_known_tvt"].fillna(0.0)
        tvt = (m["pred"].values + m["last_known_tvt"].values).astype(np.float32)
    else:
        tvt = blended["pred"].values.astype(np.float32)

    # ---- 4. Build submission dataframe ----
    submission = pd.DataFrame({
        "id": blended["well"].astype(str) + "_" + blended["row_idx"].astype(str),
        "tvt": tvt,
    })

    print(f"\nSubmission: {len(submission):,} rows")
    print(f"  Columns: {list(submission.columns)}")
    print(f"  tvt range: [{submission['tvt'].min():.2f}, {submission['tvt'].max():.2f}]")
    print(f"  tvt mean:  {submission['tvt'].mean():.2f}")
    print(f"  Unique wells: {submission['id'].str.split('_').str[0].nunique()}")

    # ---- 5. Write or dry-run ----
    if args.dry_run:
        print("\n[Dry-run] Not writing output.\n")
    else:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        submission.to_csv(out_path, index=False)
        print(f"\n\u2192 {out_path.resolve()}")

    # ---- 6. Summary ----
    print(f"\nBlend composition:")
    for cid, w in sorted(weights.items(), key=lambda x: -x[1]):
        print(f"  {cid:24s}  {w:.4f}")
    print(f"\nWall time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
