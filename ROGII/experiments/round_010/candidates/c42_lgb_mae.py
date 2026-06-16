"""c42_lgb_mae — LGB with MAE (L1) loss.

Pure L1 — different gradient profile than Huber (which transitions to L2 in
the centre). Good for the heavy tail of the target distribution.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c42_lgb_mae"
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
    objective="mae",
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
