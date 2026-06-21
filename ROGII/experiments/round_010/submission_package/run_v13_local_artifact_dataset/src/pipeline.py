"""Canonical end-to-end ensemble pipeline.

ONE module used identically for OOF scoring and Kaggle test inference, so the
exact stages/params proven offline are what runs on Kaggle. All components are
offset-space (target = TVT - last_known_tvt). The final absolute TVT is
`last_known_tvt + offset`.

Stages (each optional / switchable):
  1. blend     — weighted sum over offset components (normalized by sum of weights)
  2. ridge meta — optional linear stack over components (fit elsewhere, applied here)
  3. apply_pp  — PF blend + exponential MD ramp + alpha scale (ravaghi LB-7.776)
  4. sg_smooth — per-well Savitzky-Golay smoothing

apply_pp / sg_smooth are verbatim ports of
experiments/public_resources/ensemble_weights/ridge_pp_smooth.md.
"""
from __future__ import annotations

import numpy as np

try:
    from scipy.signal import savgol_filter
except Exception:  # pragma: no cover - scipy always present in this project
    savgol_filter = None


def blend_offsets(components: dict, weights: dict) -> np.ndarray:
    """Weighted sum of offset components, normalized by sum(weights)."""
    wsum = float(sum(weights.values()))
    if wsum <= 0:
        raise ValueError(f"non-positive weight sum: {wsum}")
    n = len(next(iter(components.values())))
    out = np.zeros(n, dtype=np.float64)
    for cid, w in weights.items():
        if cid not in components:
            raise KeyError(f"missing component: {cid}")
        out += (float(w) / wsum) * components[cid].astype(np.float64)
    return out


def apply_pp(blend_off: np.ndarray, pf_off: np.ndarray, md_since: np.ndarray,
             alpha: float, tau: float, w_pf: float) -> np.ndarray:
    """PF blend + exponential MD ramp + alpha scale (offset space)."""
    d = blend_off * (1.0 - w_pf) + pf_off * w_pf
    if tau:
        d = d * (1.0 - np.exp(-np.maximum(md_since, 0.0) / float(tau)))
    return d * float(alpha)


def sg_smooth_offsets(off: np.ndarray, well_codes: np.ndarray,
                      sg_w: int = 17, sg_p: int = 3) -> np.ndarray:
    """Per-well Savitzky-Golay smoothing. Rows must be grouped by well in order."""
    if savgol_filter is None:
        return off
    out = off.astype(np.float64).copy()
    for code in np.unique(well_codes):
        idx = np.flatnonzero(well_codes == code)
        v = out[idx]
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            out[idx] = savgol_filter(v, wl, sg_p)
    return out


def predict_pipeline(components: dict, weights: dict, last_known_tvt: np.ndarray,
                     md_since: np.ndarray, well_codes: np.ndarray,
                     pf_offset: np.ndarray | None = None,
                     pp_params: dict | None = None,
                     sg_params: dict | None = None,
                     return_absolute: bool = True) -> np.ndarray:
    """Run the full pipeline. Returns absolute TVT (or offset if return_absolute=False)."""
    off = blend_offsets(components, weights)

    if pp_params is not None:
        if pf_offset is None:
            raise ValueError("apply_pp requires pf_offset")
        off = apply_pp(off, pf_offset.astype(np.float64), md_since,
                       pp_params["alpha"], pp_params["tau"], pp_params["w_pf"])

    if sg_params is not None:
        off = sg_smooth_offsets(off, well_codes,
                                sg_params.get("sg_w", 17), sg_params.get("sg_p", 3))

    if return_absolute:
        return last_known_tvt.astype(np.float64) + off
    return off


# ── Ridge meta-stacker (optional fusion layer) ───────────────────────────────
def ridge_meta_fit(oof_matrix: np.ndarray, y: np.ndarray, alpha: float = 1.66,
                   positive: bool = True, fit_intercept: bool = True):
    """Fit a Ridge meta-stacker over component OOF predictions."""
    from sklearn.linear_model import Ridge
    model = Ridge(alpha=alpha, positive=positive, fit_intercept=fit_intercept, tol=5e-4)
    model.fit(oof_matrix, y)
    return model


def ridge_meta_predict(model, matrix: np.ndarray) -> np.ndarray:
    return model.predict(matrix).astype(np.float64)
