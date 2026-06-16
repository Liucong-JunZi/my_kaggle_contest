"""c32_lgb_super_lr025 — romantamrazov/super-solution-lb-top-3 LGB.

Source: romantamrazov/rogii-super-solution-lb-top-3 (LB top-3). Variant of
LGB stack with subsample=0.75, colsample_bytree=0.75 (vs 0.8) — slightly
more aggressive bagging. Used in their 3-config triplet at lr=0.025.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c32_lgb_super_lr025"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    boosting_type="gbdt",
    num_leaves=255,
    min_child_samples=15,
    subsample=0.75,
    subsample_freq=1,
    colsample_bytree=0.75,
    reg_lambda=3.0,
    reg_alpha=0.05,
    objective="regression",
    max_bin=255,
    learning_rate=0.025,
    n_estimators=8000,
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
