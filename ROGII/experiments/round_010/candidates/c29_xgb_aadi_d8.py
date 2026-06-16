"""c29_xgb_aadi_d8 — XGB depth=8 lr=0.04 reg_lambda=10 (aadigupta 10.239 stack).

Source: aadigupta1601/rogii-10-239. Heavier XGB (depth=8, lr=0.04, reg_lambda=10).
Adds a fundamentally different XGB profile to the pool than c06/c28.
"""
import xgboost as xgb

CANDIDATE_ID   = "c29_xgb_aadi_d8"
CANDIDATE_TYPE = "xgboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=6000,
    learning_rate=0.04,
    max_depth=8,
    min_child_weight=10,
    subsample=0.75,
    colsample_bytree=0.85,
    reg_lambda=10.0,
    reg_alpha=0.5,
    tree_method="hist",
    objective="reg:squarederror",
    eval_metric="rmse",
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
        early_stopping_rounds=200,
        verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def predict(model, X):
    return model.predict(X)
