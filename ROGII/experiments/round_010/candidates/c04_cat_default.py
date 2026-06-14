"""c04_cat_default — CatBoost with Phase 14B production hyperparams."""
from catboost import CatBoostRegressor

CANDIDATE_ID   = "c04_cat_default"
CANDIDATE_TYPE = "catboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    iterations=3000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    subsample=0.85,
    rsm=0.8,
    early_stopping_rounds=100,
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
