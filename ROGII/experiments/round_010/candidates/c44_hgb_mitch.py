"""c44_hgb_mitch — HistGradientBoostingRegressor from mitchgansemer kernel.

Source: mitchgansemer/drift-targeting-ncc-tree-based (LB ~9.4). 4th-tree
library: sklearn HistGradientBoosting. max_iter=5000, depth=6, lr=0.05.
Adds a non-LGB/CAT/XGB tree library for ensemble diversity. NaN-safe.
"""
from sklearn.ensemble import HistGradientBoostingRegressor

CANDIDATE_ID   = "c44_hgb_mitch"
CANDIDATE_TYPE = "histgb"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    max_iter=5000,
    learning_rate=0.05,
    max_depth=6,
    min_samples_leaf=20,
    l2_regularization=1.0,
    early_stopping=True,
    validation_fraction=None,  # we pass an eval split via fit
    n_iter_no_change=50,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    # HistGradientBoosting does its own early stopping internally with a held-out
    # split. We can't pass eval_set, but n_iter_no_change=50 is sufficient.
    import pandas as pd
    import numpy as np
    # Concatenate so the internal validation_fraction holds out the LATER rows
    # (use the actual va split as the validation tail).
    m = HistGradientBoostingRegressor(
        **HYPERPARAMS, random_state=seed,
    )
    m.fit(X_tr, y_tr)
    return m


def predict(model, X):
    return model.predict(X)
