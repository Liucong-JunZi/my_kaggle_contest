"""c46_rf — RandomForestRegressor on v14 features.

Same role as c45 but with bootstrapping + RF default split. Different
randomness profile from ExtraTrees — keeping both is standard hill-climb
practice. Less likely to be picked alone but contributes diversity.
"""
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CANDIDATE_ID   = "c46_rf"
CANDIDATE_TYPE = "randomforest"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=400,
    max_depth=20,
    min_samples_leaf=20,
    max_features=0.5,
    n_jobs=-1,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(**HYPERPARAMS, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def predict(model, X):
    return model.predict(X)
