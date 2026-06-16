"""c35_cat_pilkwang_d8 — pilkwang/target-free CAT depth=8 lr=0.04.

Source: pilkwang/rogii-target-free-tvt-geosteering. Deeper CAT than the
LB7.776 stack (d=8 vs d=7), iterations=5000. Adds a different tree-shape
profile to the pool than c25/c26/c33.
"""
from catboost import CatBoostRegressor

CANDIDATE_ID   = "c35_cat_pilkwang_d8"
CANDIDATE_TYPE = "catboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    iterations=5000,
    depth=8,
    l2_leaf_reg=3.0,
    loss_function="RMSE",
    eval_metric="RMSE",
    od_type="Iter",
    od_wait=200,
    learning_rate=0.04,
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
