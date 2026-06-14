"""
Candidate stub: TabICL regressor wrapping for round_010 stack

Source: kojimar/rogii-inference-stack-with-pf-beam-and-tabicl
Stage: STUB ONLY — parent agent should review before integrating.

Requires:
    pip install tabicl  (or load from rogii-tabicl-mirror dataset)
    GPU recommended
Provides a "fit_fold + predict" wrapper matching round_010's candidate contract.
"""
import numpy as np

# Per fold, build a context (sub-sampled OOF training rows) and call TabICLRegressor.
# Heavy memory; chunk inference at 50k rows.

CANDIDATE_ID = 'tabicl_v1'
DEFAULT_FEATURE_SUBSET = None  # parent should specify; ~50-80 of the 172


def get_features(df):
    """Return the feature subset name list for TabICL.

    Falls back to all numeric columns if DEFAULT_FEATURE_SUBSET is None.
    """
    if DEFAULT_FEATURE_SUBSET is not None:
        return DEFAULT_FEATURE_SUBSET
    return [c for c in df.columns
            if c not in ('well_id', 'id', 'target_residual', 'target_tvt', 'baseline_tvt')
            and df[c].dtype.kind in 'fi']


def fit_fold(X_train, y_train, X_val, y_val, *, model_path, seed=42, n_estimators=4,
             context_size=10_000):
    """Fit TabICLRegressor on a sub-sampled context from the train fold.

    Args:
        X_train, y_train: full train fold
        model_path: path to tabicl-regressor-*.ckpt
        seed, n_estimators: TabICL ensemble setup
        context_size: number of train rows to use as context (memory limit)

    Returns:
        a callable `predict(X_test) -> np.ndarray`.
    """
    from tabicl import TabICLRegressor

    # Sub-sample context if too large
    n = len(X_train)
    if n > context_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, context_size, replace=False)
        X_ctx, y_ctx = X_train.iloc[idx].values, y_train.iloc[idx].values
    else:
        X_ctx, y_ctx = X_train.values, y_train.values

    reg = TabICLRegressor(
        model_path=model_path,
        device='cuda',
        random_state=seed,
        n_estimators=n_estimators,
        n_jobs=1,
        verbose=False,
        use_amp='auto',
        batch_size=4,
    )
    reg.fit(X_ctx, y_ctx)

    def predict(X):
        chunk = 50_000
        out = np.empty(len(X), np.float32)
        Xv = X.values if hasattr(X, 'values') else X
        for i in range(0, len(Xv), chunk):
            e = min(i + chunk, len(Xv))
            out[i:e] = reg.predict(Xv[i:e]).astype(np.float32)
        return out

    return predict
