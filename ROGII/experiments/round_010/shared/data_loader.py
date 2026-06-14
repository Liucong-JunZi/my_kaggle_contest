"""Joined feature dataframe — single source of truth for all candidates.

Loads features_full + pf_features + beam_features + pf_ensemble, derives the
v8/v14 standard offset features, and joins the global fold map. The result is
cached to results/joined_features.parquet on first build.

All candidates start by calling `load_joined()` — guarantees identical row
order and feature alignment across all OOFs.
"""
from pathlib import Path
import numpy as np
import pandas as pd

R8_DIR    = Path("/Users/liucong/code/kaggle/ROGII/results/round_008")
ROUND_DIR = Path(__file__).resolve().parents[1]
CACHE     = ROUND_DIR / "results" / "joined_features.parquet"

LEAK_COLS = {"z_minus_ancc", "z_minus_astnu", "z_minus_astnl",
             "z_minus_egfdu", "z_minus_egfdl", "z_minus_buda"}


def _build():
    base = pd.read_parquet(R8_DIR / "features_full.parquet")
    pf   = pd.read_parquet(R8_DIR / "pf_features.parquet")
    bm   = pd.read_parquet(R8_DIR / "beam_features.parquet")
    ens  = pd.read_parquet(R8_DIR / "pf_ensemble.parquet")

    df = (base.merge(pf,  on=["well", "row_idx"], how="left")
              .merge(bm,  on=["well", "row_idx"], how="left")
              .merge(ens, on=["well", "row_idx"], how="left"))
    df = df.dropna(subset=["pf_ancc", "beam_mean", "pf_ens_s12"]).reset_index(drop=True)

    # Standard derived offset features (matches v8/v14)
    df["pf_ancc_offset"]   = df["pf_ancc"]   - df["last_known_tvt"]
    df["pf_z_offset"]      = df["pf_z"]      - df["last_known_tvt"]
    df["pf_disagreement"]  = df["pf_ancc"]   - df["pf_z"]
    df["pf_mean_offset"]   = 0.5 * (df["pf_ancc_offset"] + df["pf_z_offset"])
    df["beam_mean_offset"] = df["beam_mean"] - df["last_known_tvt"]
    df["beam_med_offset"]  = df["beam_med"]  - df["last_known_tvt"]
    df["beam_cons_offset"] = df["beam_cons"] - df["last_known_tvt"]
    df["beam_sm5_offset"]  = df["beam_sm5"]  - df["last_known_tvt"]
    df["beam_vs_pf"]       = df["beam_mean_offset"] - df["pf_mean_offset"]

    for c in ["pf_ens_s3", "pf_ens_s5", "pf_ens_s8", "pf_ens_s12", "pf_ens_mean"]:
        df[f"{c}_offset"] = df[c] - df["last_known_tvt"]
    df["pf_ens_vs_ancc"]     = df["pf_ens_s12_offset"] - df["pf_ancc_offset"]
    df["pf_ens_scale_disag"] = df["pf_ens_s3_offset"]  - df["pf_ens_s12_offset"]

    # Join fold
    from shared.cv_split import get_fold_map
    folds = get_fold_map(wells=df["well"].unique().tolist())
    df = df.merge(folds, on="well", how="left")
    assert df["fold"].notna().all(), "Fold join produced NaN"
    df["fold"] = df["fold"].astype(np.int8)

    df.to_parquet(CACHE)
    return df


def load_joined() -> pd.DataFrame:
    if CACHE.exists():
        return pd.read_parquet(CACHE)
    return _build()


# Standard feature set definitions — candidates can pick one or build their own
def feature_set_v14() -> list:
    """The 43-feat stack used in Phase 14B (v13 LB-submitted)."""
    df = load_joined()
    drop = ({"well", "row_idx", "target", "fold",
             "pf_ancc", "pf_z", "beam_mean", "beam_std", "beam_med", "beam_range",
             "beam_cons", "beam_sm5",
             "pf_ens_s3", "pf_ens_s5", "pf_ens_s8", "pf_ens_s12", "pf_ens_mean"}
            | LEAK_COLS)
    return [c for c in df.columns if c not in drop]


def feature_set_base_only() -> list:
    """Geometry + GR rolling only — no PF/beam. Forces model to learn pure path."""
    feats = feature_set_v14()
    return [c for c in feats if not (c.startswith("pf_") or c.startswith("beam_"))]


def feature_set_pf_beam_only() -> list:
    """Only PF + beam offsets — no geometry, no GR rolling."""
    feats = feature_set_v14()
    return [c for c in feats if c.startswith("pf_") or c.startswith("beam_")
            or c in {"last_known_tvt", "last_known_z", "last_known_gr",
                     "n_known_rows", "n_lateral_rows", "row_position_norm"}]


if __name__ == "__main__":
    df = load_joined()
    print(f"rows: {len(df):,}  wells: {df['well'].nunique()}  cols: {len(df.columns)}")
    print(f"  fold sizes: {df.groupby('fold').size().to_dict()}")
    print(f"  feature_set_v14: {len(feature_set_v14())} feats")
    print(f"  feature_set_base_only: {len(feature_set_base_only())} feats")
    print(f"  feature_set_pf_beam_only: {len(feature_set_pf_beam_only())} feats")
