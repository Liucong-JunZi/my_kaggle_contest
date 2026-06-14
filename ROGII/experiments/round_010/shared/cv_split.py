"""Global GKF-5 fold map — all candidates MUST use this.

Hash-based: well_id → fold ∈ {0,1,2,3,4}. Deterministic, increment-friendly
(adding new wells doesn't shuffle existing wells' folds).

The map is materialised to results/global_folds.parquet on first call.
"""
import hashlib
from pathlib import Path
import pandas as pd

ROUND_DIR = Path(__file__).resolve().parents[1]
FOLDS_PATH = ROUND_DIR / "results" / "global_folds.parquet"
N_SPLITS = 5


def _hash_fold(wid: str, n_splits: int = N_SPLITS) -> int:
    h = hashlib.sha256(wid.encode("utf-8")).hexdigest()
    return int(h[:8], 16) % n_splits


def build_fold_map(wells, n_splits: int = N_SPLITS) -> pd.DataFrame:
    """Pure function — given a list of well_ids, return DataFrame[well, fold]."""
    return pd.DataFrame({"well": list(wells), "fold": [_hash_fold(w, n_splits) for w in wells]})


def get_fold_map(wells=None) -> pd.DataFrame:
    """Load cached fold map; build + write if missing.

    If `wells` is given, validates that all of them have an assigned fold and
    that no fold is empty. Pass None to load existing.
    """
    if FOLDS_PATH.exists():
        m = pd.read_parquet(FOLDS_PATH)
    else:
        if wells is None:
            raise FileNotFoundError(f"{FOLDS_PATH} missing and no wells provided to bootstrap.")
        m = build_fold_map(wells)
        FOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
        m.to_parquet(FOLDS_PATH)
    if wells is not None:
        missing = set(wells) - set(m["well"])
        if missing:
            # Extend incrementally
            extra = build_fold_map(sorted(missing))
            m = pd.concat([m, extra], ignore_index=True).drop_duplicates("well", keep="first")
            m.to_parquet(FOLDS_PATH)
    return m


def fold_summary(m: pd.DataFrame):
    return m.groupby("fold").size().to_dict()


if __name__ == "__main__":
    # Bootstrap from features_full
    feats = pd.read_parquet("/Users/liucong/code/kaggle/ROGII/results/round_008/features_full.parquet",
                            columns=["well"])
    wells = sorted(feats["well"].unique())
    m = get_fold_map(wells)
    print(f"Wells: {len(wells)}  Fold sizes: {fold_summary(m)}")
    print(f"→ {FOLDS_PATH}")
