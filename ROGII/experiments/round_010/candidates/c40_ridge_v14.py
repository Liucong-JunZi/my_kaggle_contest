"""c40_ridge_v14 — Ridge regression on v14 features (positive weights, alpha=1.66).

Source: lightningv08/lb-7-776-rogii-ridge-sp ridge_params dict. Used as the
META-learner there; here we use it as a pure linear baseline on the same
43 features. Scaling is REQUIRED for Ridge — v14 features span many decades.
Wrapped in a Pipeline(StandardScaler, Ridge).
"""
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

CANDIDATE_ID   = "c40_ridge_v14"
CANDIDATE_TYPE = "ridge"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    alpha=1.6602834637650032,
    tol=0.0005030247295617308,
    positive=True,
    fit_intercept=True,
)


def get_features(df):
    from shared.data_loader import feature_set_v14
    feat_cols = feature_set_v14()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    # Ridge has no early stopping; eval set is ignored.
    pipe = Pipeline([
        ("scale", StandardScaler(with_mean=True, with_std=True)),
        ("ridge", Ridge(**HYPERPARAMS, random_state=seed)),
    ])
    # Imputer is needed because v14 has NaNs in lag/lead/rolling features.
    from sklearn.impute import SimpleImputer
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(**HYPERPARAMS, random_state=seed)),
    ])
    pipe.fit(X_tr, y_tr)
    return pipe


def predict(model, X):
    return model.predict(X)
