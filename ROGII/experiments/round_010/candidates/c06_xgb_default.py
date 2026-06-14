"""c06_xgb_default — XGBoost as a 3rd GBDT lib for diversity."""
import xgboost as xgb

CANDIDATE_ID   = "c06_xgb_default"
CANDIDATE_TYPE = "xgboost"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=3000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=10,
    reg_alpha=0.1,
    reg_lambda=1.0,
    colsample_bytree=0.8,
    subsample=0.85,
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
        early_stopping_rounds=100,
        verbosity=0,
    )
    m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return m


def predict(model, X):
    return model.predict(X)
