import numpy as np
from scipy.optimize import minimize

class Climber:
    def __init__(self, objective="minimize", eval_metric=None, allow_negative_weights=True,
                 precision=0.001, score_decimal_places=3, n_jobs=-1, use_gpu=False, **kwargs):
        self.objective = objective
        self.eval_metric = eval_metric
        self.allow_negative_weights = allow_negative_weights
        self.precision = precision
        self.score_decimal_places = score_decimal_places
        self.n_jobs = n_jobs
        self.use_gpu = use_gpu
        self.weights_ = None
        self.best_score = None
        self.columns_ = None

    def _score(self, y, p):
        if self.eval_metric is None:
            return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(p)) ** 2)))
        return float(self.eval_metric(y, p))

    def fit(self, X, y):
        A = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = A.shape[1]
        self.columns_ = list(getattr(X, 'columns', range(n)))
        x0 = np.ones(n) / max(n, 1)
        def obj(w):
            return self._score(y, A @ w)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
        bounds = None if self.allow_negative_weights else [(0, 1)] * n
        res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=constraints,
                       options={'maxiter': 1000, 'ftol': 1e-10})
        w = res.x if res.success else x0
        if self.precision:
            w = np.round(w / self.precision) * self.precision
            if abs(w.sum()) > 1e-12:
                w = w / w.sum()
        self.weights_ = w
        self.best_score = self._score(y, A @ w)
        print("Climber weights:", dict(zip(self.columns_, self.weights_)))
        print("Climber best_score:", self.best_score)
        return self

    def predict(self, X):
        A = np.asarray(X, dtype=float)
        return A @ self.weights_
