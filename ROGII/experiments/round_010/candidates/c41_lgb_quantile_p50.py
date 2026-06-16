"""c41_lgb_quantile_p50 — LGB quantile loss (median) for diversity.

Quantile=0.5 is L1-equivalent (median regression). Robust to outliers in the
target tail (rare wells with large drift). Pairs with c03 (Huber) and c46
(MAE) to span loss-function space.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c41_lgb_quantile_p50"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.8,
    subsample=0.85,
    subsample_freq=5,
    objective="quantile",
    alpha=0.5,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = lgb.LGBMRegressor(**HYPERPARAMS, random_state=seed,
                          verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
