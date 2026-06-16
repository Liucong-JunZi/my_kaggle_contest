"""c34_lgb_pilkwang_d127 — pilkwang/target-free-tvt-geosteering LGB.

Source: pilkwang/rogii-target-free-tvt-geosteering. Uses the 'mid' LGB shape
(num_leaves=127, lr=0.04, reg_lambda=5) — similar to c27 but with shallower
tree count and lr; and lower lr_alpha=0.1 so different bias/variance tradeoff.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c34_lgb_pilkwang_d127"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    boosting_type="gbdt",
    num_leaves=127,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.1,
    objective="regression",
    learning_rate=0.04,
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
          callbacks=[lgb.early_stopping(150, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
