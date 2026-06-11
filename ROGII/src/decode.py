"""Decoding & post-processing for SDF predictions → TVT.

Standard pipeline (R4-A established):
    sdf_pred (B, T, H) → subpixel argmin → t_tvt lookup → partial anchor (α=0.75)
                                                        → masked RMSE

Functions are numpy-based, vectorized across batch.
"""

import numpy as np

DEFAULT_ALPHA = 0.75  # R4-A1+A2 best on cfg-img-medium val (14.28 vs 15.89 baseline)


def decode_sdf_to_tvt(sdf_abs, t_tvt, subpixel=True):
    """Decode |sdf| (B, T, H) → tvt_pred (B, H) via per-column argmin.

    If subpixel=True, fit parabola through (idx-1, idx, idx+1) for sub-grid
    precision, then linearly interpolate t_tvt.
    """
    B, T, H = sdf_abs.shape
    idx = sdf_abs.argmin(axis=1)
    if not subpixel:
        return np.take_along_axis(t_tvt[:, :, None], idx[:, None, :], axis=1).squeeze(1)

    idx_c = np.clip(idx, 1, T - 2)
    bi = np.arange(B)[:, None]; hi = np.arange(H)[None, :]
    s_m = sdf_abs[bi, idx_c - 1, hi]
    s_0 = sdf_abs[bi, idx_c,     hi]
    s_p = sdf_abs[bi, idx_c + 1, hi]
    denom = s_m - 2 * s_0 + s_p
    denom = np.where(np.abs(denom) < 1e-8, 1e-8, denom)
    delta = np.clip(0.5 * (s_m - s_p) / denom, -0.5, 0.5)
    cont_idx = idx_c.astype(np.float64) + delta

    floor = np.clip(np.floor(cont_idx).astype(np.int64), 0, T - 2)
    frac = cont_idx - floor
    t_low = np.take_along_axis(t_tvt, floor, axis=1)
    t_hi = np.take_along_axis(t_tvt, floor + 1, axis=1)
    return t_low + frac * (t_hi - t_low)


def anchor_known_segment(tvt_pred, tvt_known, mask_known, alpha=DEFAULT_ALPHA):
    """Per-well partial bias correction using a known TVT segment.

    Args:
        tvt_pred: (B, H) predictions
        tvt_known: (B, H_H) ground-truth TVT on the known (history) segment
        mask_known: (B, H_H) validity mask for the known segment
        alpha: shrinkage in [0,1]. 0 = no correction, 1 = full bias subtraction.
               R4-A2 swept alpha; 0.75 best on cfg-img-medium 50-well val.

    Returns:
        (B, H) corrected predictions.
    """
    B, H = tvt_pred.shape
    H_H = tvt_known.shape[1]
    out = tvt_pred.copy()
    for b in range(B):
        valid = mask_known[b] > 0.5
        if valid.sum() < 5:
            continue
        bias = (tvt_pred[b, :H_H][valid] - tvt_known[b][valid]).mean()
        out[b] = tvt_pred[b] - alpha * bias
    return out


def masked_rmse(tvt_pred, tvt_true, mask):
    """Per-well masked RMSE → (B,). Aggregate with .mean() or np.median()."""
    sq = (tvt_pred - tvt_true) ** 2 * mask
    return np.sqrt(sq.sum(axis=1) / np.clip(mask.sum(axis=1), 1, None))
