"""c03_lgb_huber — LightGBM with Huber loss (robust to outliers).

Outliers in the target (rare wells with abrupt TVT shifts) currently dominate
the L2 loss. Huber down-weights large residuals, should help fold-4-style
hard wells without sacrificing the easy majority.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c03_lgb_huber"
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
    objective="huber",
    alpha=0.9,
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
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
