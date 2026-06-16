"""c33_cat_super_lr025 — romantamrazov/super CAT depth=7 lr=0.025.

Source: romantamrazov/rogii-super-solution-lb-top-3. CAT with subsample=0.75
(vs 0.8). The kernel notes this is 'slower, better convergence' vs lr=0.035.
"""
from catboost import CatBoostRegressor

CANDIDATE_ID   = "c33_cat_super_lr025"
CANDIDATE_TYPE = "catboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    iterations=8000,
    depth=7,
    l2_leaf_reg=2.0,
    min_data_in_leaf=15,
    border_count=254,
    loss_function="RMSE",
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=300,
    learning_rate=0.025,
    subsample=0.75,
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
