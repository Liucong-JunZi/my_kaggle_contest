from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor


def load_json(path):
    return json.loads(Path(path).read_text())


def predict_component(model_dir, features):
    model_dir = Path(model_dir)
    meta = load_json(model_dir / "meta.json")
    feature_names = load_json(model_dir / "feature_names.json")
    missing = [c for c in feature_names if c not in features.columns]
    if missing:
        raise ValueError(f"{model_dir.name}: missing {len(missing)} features, first={missing[:10]}")

    X = features[feature_names]
    preds = []
    if meta["model_type"] == "lightgbm":
        for path in sorted(model_dir.glob("fold_*.txt")):
            booster = lgb.Booster(model_file=str(path))
            preds.append(booster.predict(X))
    elif meta["model_type"] == "catboost":
        for path in sorted(model_dir.glob("fold_*.cbm")):
            model = CatBoostRegressor()
            model.load_model(str(path))
            preds.append(model.predict(X))
    else:
        raise ValueError(f"unknown model_type: {meta['model_type']}")

    if not preds:
        raise ValueError(f"{model_dir}: no fold models found")
    out = np.mean(np.vstack(preds), axis=0).astype(np.float32)
    if not np.isfinite(out).all():
        raise ValueError(f"{model_dir.name}: non-finite predictions")
    return out
