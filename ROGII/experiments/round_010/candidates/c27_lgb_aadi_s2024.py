"""c27_lgb_aadi_s2024 — LGB num_leaves=255 lr=0.035 seed=2024.

Source: aadigupta1601/rogii-10-239 (LB 10.239). The "4th LGB seed" of their
7-model Ridge stack. Higher reg (reg_lambda=5, reg_alpha=0.1).
"""
import lightgbm as lgb

CANDIDATE_ID   = "c27_lgb_aadi_s2024"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 2024

HYPERPARAMS = dict(
    boosting_type="gbdt",
    num_leaves=255,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=5.0,
    reg_alpha=0.1,
    objective="regression",
    learning_rate=0.035,
    n_estimators=6000,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = lgb.LGBMRegressor(**HYPERPARAMS, random_state=DEFAULT_SEED,
                          verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
