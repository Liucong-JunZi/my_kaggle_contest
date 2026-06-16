"""c38_lgb_dwt_s7 — DWT kernel LGB seed=7 lr=0.020 n_est=8000.

Source: nihilisticneuralnet/9-251 (LB 9.251). LGB[1] of their triplet —
slowest lr (0.020), seed=7. Pairs with c30/c22 for ensemble diversity.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c38_lgb_dwt_s7"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 7

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
    learning_rate=0.020,
    n_estimators=8000,
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
          callbacks=[lgb.early_stopping(250, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
