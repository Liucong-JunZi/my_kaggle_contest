"""c26_cat_d7_s123_lr030 — CatBoost depth=7 lr=0.030 seed=123 (LB 7.776 second).

Source: lightningv08/lb-7-776-rogii-ridge-sp. Second of two CAT configs in the
Ridge stack. Same shape as c25 but lr=0.030, seed=123.
"""
from catboost import CatBoostRegressor

CANDIDATE_ID   = "c26_cat_d7_s123_lr030"
CANDIDATE_TYPE = "catboost"
DEFAULT_SEED   = 123

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
    learning_rate=0.030,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = CatBoostRegressor(
        **HYPERPARAMS,
        random_seed=DEFAULT_SEED,
        verbose=False,
        thread_count=-1,
    )
    m.fit(X_tr, y_tr, eval_set=(X_va, y_va), use_best_model=True)
    return m


def predict(model, X):
    return model.predict(X)
