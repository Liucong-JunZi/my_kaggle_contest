"""c08_lgb_pf_beam_only — LGB on PF + beam offsets only.

The complement to c07: forces model to learn corrections to PF/beam without
any geometry. Together with c07 they span the feature space. Hill climb will
likely keep both.
"""
import lightgbm as lgb

CANDIDATE_ID   = "c08_lgb_pf_beam_only"
CANDIDATE_TYPE = "lightgbm"
DEFAULT_SEED   = 42

HYPERPARAMS = dict(
    n_estimators=3000,
    learning_rate=0.02,
    num_leaves=127,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=0.1,
    colsample_bytree=0.8,
    subsample=0.85,
    subsample_freq=5,
)


def get_features(df):
    from shared.data_loader import feature_set_pf_beam_only
    feat_cols = feature_set_pf_beam_only()
    return df[feat_cols], feat_cols


def fit_fold(X_tr, y_tr, X_va, y_va, seed):
    m = lgb.LGBMRegressor(**HYPERPARAMS, random_state=seed,
                          verbose=-1, n_jobs=-1)
    m.fit(X_tr, y_tr,
          eval_set=[(X_va, y_va)],
          callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def predict(model, X):
    return model.predict(X)
