import os
import joblib
import numpy as np
from sklearn.base import clone

class Trainer:
    def __init__(self, estimator=None, task="regression", metric=None, cv=None,
                 cv_args=None, use_early_stopping=False, verbose=True, save=False,
                 save_path=None, metric_precision=5, metric_threshold=None,
                 metric_args=None, **kwargs):
        self.estimator = estimator
        self.estimator_name = estimator.__class__.__name__.lower() if estimator is not None else None
        self.task = task
        self.metric = metric
        self.metric_name = getattr(metric, "__name__", str(metric)) if metric is not None else None
        self.cv = cv
        self.cv_args = cv_args or {}
        self.use_early_stopping = use_early_stopping
        self.verbose = verbose
        self.save = save
        self.save_path = save_path
        self.metric_precision = metric_precision
        self.metric_threshold = metric_threshold
        self.metric_args = metric_args or {}
        self.estimators = []
        self.fold_scores = []
        self.oof_preds = None
        self.overall_score = None
        self.is_fitted = False

    def _score(self, y_true, y_pred):
        if self.metric is None:
            return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))
        return float(self.metric(y_true, y_pred, **self.metric_args))

    def fit(self, X, y, fit_args=None):
        fit_args = fit_args or {}
        X_arr = X
        y_arr = np.asarray(y)
        self.oof_preds = np.zeros(len(y_arr), dtype=float)
        groups = self.cv_args.get("groups")
        split_iter = self.cv.split(X_arr, y_arr, groups=groups) if self.cv is not None else [(np.arange(len(y_arr)), np.arange(len(y_arr)))]
        self.estimators = []
        self.fold_scores = []
        for fold, (tr_idx, va_idx) in enumerate(split_iter):
            model = clone(self.estimator)
            kwargs = dict(fit_args)
            # Keep only fit kwargs that broadly work; public artifact notebooks pass
            # eval callbacks for LGB/CAT only when artifacts are absent. Ridge meta
            # fit passes no kwargs. This compatibility shim is intentionally small.
            try:
                model.fit(X_arr.iloc[tr_idx] if hasattr(X_arr, 'iloc') else X_arr[tr_idx], y_arr[tr_idx], **kwargs)
            except TypeError:
                model.fit(X_arr.iloc[tr_idx] if hasattr(X_arr, 'iloc') else X_arr[tr_idx], y_arr[tr_idx])
            pred = model.predict(X_arr.iloc[va_idx] if hasattr(X_arr, 'iloc') else X_arr[va_idx])
            self.oof_preds[va_idx] = pred
            self.estimators.append(model)
            self.fold_scores.append(self._score(y_arr[va_idx], pred))
        self.overall_score = self._score(y_arr, self.oof_preds)
        self.is_fitted = True
        if self.save and self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            joblib.dump(self, os.path.join(self.save_path, f"{self.estimator_name}_trainer.pkl"))
        return self

    def predict(self, X):
        if not getattr(self, "estimators", None):
            if hasattr(self, "estimator") and self.estimator is not None:
                return self.estimator.predict(X)
            raise AttributeError("Trainer has no estimators")
        return np.mean([m.predict(X) for m in self.estimators], axis=0)
