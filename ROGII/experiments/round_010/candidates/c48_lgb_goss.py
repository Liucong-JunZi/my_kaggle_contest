"""c48_lgb_goss — LGB with GOSS sampling (gradient-based one-side sampling).

GOSS keeps high-gradient samples and randomly subsamples low-gradient ones.
Different bias profile from gbdt; kernel diversity contributor.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c48_lgb_goss"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    boosting_type="goss",
    n_estimators=4000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.8,
    top_rate=0.2,
    other_rate=0.1,
    objective="regression",
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
