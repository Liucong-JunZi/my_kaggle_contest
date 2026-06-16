"""c45_etr — ExtraTreesRegressor on v14 features.

Bagging (no boosting) as orthogonal pool member. Less effective alone but
contributes negative correlation to the ridge meta-stack — the rationale
behind Ravaghi's hill-climb kernel which keeps high-variance learners around.
"""
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

CANDIDATE_ID   = "c45_etr"
CANDIDATE_TYPE = "extratrees"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=600,
    max_depth=20,
    min_samples_leaf=20,
    max_features=0.6,
    n_jobs=-1,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("etr", ExtraTreesRegressor(**HYPERPARAMS, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def predict(model, X):
    return model.predict(X)
