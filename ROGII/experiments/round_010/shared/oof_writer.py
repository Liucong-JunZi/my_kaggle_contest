"""OOF parquet writer + loader with strict schema validation.

Schema (one row per lateral row):
    well_id    str
    row_idx    int32
    fold       int8     (0..4, MUST match global fold map)
    target     float32  (relative offset, TVT - last_known_tvt)
    oof_pred   float32  (OOF prediction in offset space)

Metadata stored as parquet kv:
    candidate_id, candidate_type, features_used, hyperparams, seed,
    train_time_sec, perwell_oof, flat_oof, feature_set_hash

Reading: load_oof(candidate_id) → (df, meta).
Writing: write_oof(candidate_id, df, **kwargs) → checks schema, computes
metrics, attaches metadata.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

ROUND_DIR = Path(__file__).resolve().parents[1]
OOF_DIR   = ROUND_DIR / "results" / "candidates"
OOF_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_COLS = {"well", "row_idx", "fold", "target", "oof_pred"}


def _hash_feats(features: list) -> str:
    return hashlib.sha256(",".join(sorted(features)).encode()).hexdigest()[:16]


def write_oof(
    candidate_id: str,
    df_oof: pd.DataFrame,
    candidate_type: str,
    features_used: list,
    hyperparams: dict,
    seed: int,
    train_time_sec: float,
    fold_metrics: dict | None = None,
    extra_meta: dict | None = None,
) -> Path:
    """Validate, compute metrics, write parquet with metadata."""
    missing = REQUIRED_COLS - set(df_oof.columns)
    if missing:
        raise ValueError(f"OOF df missing required cols: {missing}")
    df = df_oof[list(REQUIRED_COLS)].copy()
    df["row_idx"] = df["row_idx"].astype(np.int32)
    df["fold"]    = df["fold"].astype(np.int8)
    df["target"]  = df["target"].astype(np.float32)
    df["oof_pred"]= df["oof_pred"].astype(np.float32)

    # Compute aggregate metrics
    from shared.metrics import perwell_rmse, flat_rmse
    perwell = perwell_rmse(df["target"].values, df["oof_pred"].values, df["well"].values)
    flat    = flat_rmse(df["target"].values, df["oof_pred"].values)

    meta = {
        "candidate_id":     candidate_id,
        "candidate_type":   candidate_type,
        "features_used":    list(features_used),
        "hyperparams":      hyperparams,
        "seed":             int(seed),
        "train_time_sec":   float(train_time_sec),
        "perwell_oof":      float(perwell),
        "flat_oof":         float(flat),
        "fold_metrics":     fold_metrics or {},
        "feature_set_hash": _hash_feats(features_used),
        "n_rows":           int(len(df)),
        "n_wells":          int(df["well"].nunique()),
        "written_at":       time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    if extra_meta:
        meta.update(extra_meta)

    table = pa.Table.from_pandas(df)
    table = table.replace_schema_metadata({b"r10_meta": json.dumps(meta).encode()})

    out = OOF_DIR / f"{candidate_id}.parquet"
    pq.write_table(table, out)
    return out


def load_oof(candidate_id: str) -> tuple[pd.DataFrame, dict]:
    p = OOF_DIR / f"{candidate_id}.parquet"
    if not p.exists():
        raise FileNotFoundError(p)
    table = pq.read_table(p)
    meta = {}
    if table.schema.metadata and b"r10_meta" in table.schema.metadata:
        meta = json.loads(table.schema.metadata[b"r10_meta"])
    return table.to_pandas(), meta


def list_candidates() -> pd.DataFrame:
    rows = []
    for p in sorted(OOF_DIR.glob("*.parquet")):
        cid = p.stem
        try:
            _, meta = load_oof(cid)
            rows.append({
                "candidate_id":   cid,
                "type":           meta.get("candidate_type", "?"),
                "perwell_oof":    meta.get("perwell_oof"),
                "flat_oof":       meta.get("flat_oof"),
                "n_features":     len(meta.get("features_used", [])),
                "seed":           meta.get("seed"),
                "train_time_sec": meta.get("train_time_sec"),
                "feat_hash":      meta.get("feature_set_hash", "?")[:8],
            })
        except Exception as e:
            rows.append({"candidate_id": cid, "type": "ERROR", "perwell_oof": None,
                         "flat_oof": None, "n_features": None, "seed": None,
                         "train_time_sec": None, "feat_hash": str(e)[:20]})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = list_candidates()
    if len(df) == 0:
        print("No candidates yet.")
    else:
        print(df.to_string(index=False))
