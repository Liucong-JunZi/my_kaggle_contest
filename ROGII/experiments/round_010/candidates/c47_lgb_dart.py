"""c47_lgb_dart — LGB with DART boosting (drop-out trees) for diversity.

DART tends to underfit slightly but gives strongly different predictions
from gbdt — a known low-correlation contributor in tree ensembles.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c47_lgb_dart"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    boosting_type="dart",
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=127,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.8,
    subsample=0.85,
    subsample_freq=5,
    drop_rate=0.1,
    skip_drop=0.5,
    objective="regression",
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    # DART does not support early stopping in LightGBM. Just fit n_estimators.
    m = lgb.LGBMRegressor(**HYPERPARAMS, random_state=seed,
                          verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
    return m


def predict(model, X):
    return model.predict(X)
