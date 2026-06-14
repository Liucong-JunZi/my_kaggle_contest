"""c02_lgb_deep — LightGBM with deeper trees + more regularisation."""
import lightgbm as lgb

CANDIDATE_ID   = "c02_lgb_deep"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=5000,
    learning_rate=0.01,
    num_leaves=255,
    min_child_samples=20,
    reg_alpha=0.3,
    reg_lambda=0.3,
    colsample_bytree=0.7,
    subsample=0.8,
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
          callbacks=[lgb.early_stopping(200, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
