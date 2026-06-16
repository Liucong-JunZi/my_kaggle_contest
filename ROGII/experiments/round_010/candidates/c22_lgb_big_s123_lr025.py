"""c22_lgb_big_s123_lr025 — LGB "big" config, lr=0.025 seed=123.

Source: nihilisticneuralnet/9-251-dwt + svanikkolli/aeroridge-engine-v2.
Third of the LGB triplet (the version with lr=0.025 from the DWT kernel).
"""
import lightgbm as lgb

CANDIDATE_ID   = "c22_lgb_big_s123_lr025"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 123

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
    learning_rate=0.025,
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
