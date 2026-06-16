"""c28_xgb_cdeotte_starter — cdeotte/xgb-starter-cv-15 hyperparams.

Source: cdeotte/xgb-starter-cv-15 (CV 15.x). Compact starter; depth=5, lr=0.035,
n_estimators=450. Differs from c06 (depth=8). Adds shallow-tree diversity.
"""
import xgboost as xgb

CANDIDATE_ID   = "c28_xgb_cdeotte_starter"
CANDIDATE_TYPE = "xgboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=450,
    learning_rate=0.035,
    max_depth=5,
    min_child_weight=20,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_lambda=4.0,
    reg_alpha=0.05,
    tree_method="hist",
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = xgb.XGBRegressor(
        **HYPERPARAMS,
        random_state=seed,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def predict(model, X):
    return model.predict(X)
