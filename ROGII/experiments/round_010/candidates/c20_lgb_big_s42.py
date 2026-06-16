"""c20_lgb_big_s42 — LGB "big" config from LB 7.776 stack (Ridge-SP).

Source: lightningv08/lb-7-776-rogii-ridge-sp + nihilisticneuralnet/9-251 (DWT).
This is the *first* of three LGB configs they ensemble via Ridge.
num_leaves=255, lr=0.030, reg_lambda=3.0. seed=42.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c20_lgb_big_s42"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    boosting_type="gbdt",
    num_leaves=255,
    min_child_samples=15,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.8,
    reg_lambda=3.0,
    reg_alpha=0.05,
    objective="regression",
    max_bin=255,
    learning_rate=0.030,
    n_estimators=5000,
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
          callbacks=[lgb.early_stopping(250, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
