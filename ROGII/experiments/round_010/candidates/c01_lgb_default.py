"""c01_lgb_default — LightGBM with Phase 14B production hyperparams.

This reproduces the v13 LGB component as a freshly-trained candidate. Should
produce perwell ~9.58 (matching p14_lgb but trained from scratch under the
new fold map). If perwell deviates >0.1 from p14_lgb, the new fold map is
materially different from the old GroupKFold and we should investigate.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c01_lgb_default"
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
