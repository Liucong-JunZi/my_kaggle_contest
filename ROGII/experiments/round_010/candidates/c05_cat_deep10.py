"""c05_cat_deep10 — CatBoost depth=10 + larger iterations.

Phase 14B CAT fold 3 hit best_iter=2844 (near 3000 cap) — capacity-bound.
Push to 5000 iter + depth 10.
"""
from catboost import CatBoostRegressor

CANDIDATE_ID   = "c05_cat_deep10"
CANDIDATE_TYPE = "catboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    iterations=5000,
    learning_rate=0.03,
    depth=10,
    l2_leaf_reg=5.0,
    subsample=0.8,
    rsm=0.7,
    early_stopping_rounds=200,
    loss_function="RMSE",
    eval_metric="RMSE",
    bootstrap_type="Bernoulli",
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = CatBoostRegressor(
        **HYPERPARAMS,
        random_seed=seed,
        verbose=False,
        thread_count=-1,
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return m


def predict(model, X):
    return model.predict(X)
