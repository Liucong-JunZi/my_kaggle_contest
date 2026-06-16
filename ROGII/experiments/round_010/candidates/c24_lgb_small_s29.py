"""c24_lgb_small_s29 — LGB "small-deep" config, second seed.

Source: lightningv08/lb-7-776-rogii-ridge-sp. Same Optuna-tuned hyperparams
as c23 but seed=29 (the kernel ensembles both seeds).
"""
import lightgbm as lgb

CANDIDATE_ID   = "c24_lgb_small_s29"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 29

HYPERPARAMS = dict(
    boosting_type="gbdt",
    num_leaves=64,
    min_child_samples=40,
    min_child_weight=0.24081152127177283,
    subsample=0.47437582748953966,
    subsample_freq=1,
    colsample_bytree=0.39283351290380497,
    reg_lambda=95.75401894533888,
    reg_alpha=10.788188919840913,
    objective="regression",
    learning_rate=0.00934485794382918,
    n_estimators=10000,
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
          callbacks=[lgb.early_stopping(300, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
