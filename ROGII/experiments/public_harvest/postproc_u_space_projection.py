"""
Candidate stub: U-space (TVT+Z) anchor-relative robust polynomial projection

Source: pilkwang/rogii-target-free-tvt-geosteering
Stage: STUB ONLY — parent agent should review before integrating.

Post-processing layer (NOT a feature). Applied AFTER the LGB stack and PP step,
BEFORE submission.

Pipeline placement:
    raw_blend -> sg_smooth (or instead of sg_smooth) -> u_space_project -> submission
"""
import numpy as np


def robust_polyfit_iterated(x, y, deg=4, c=2.0, iters=4):
    """Iteratively reweighted least squares with Cauchy-like weighting.

    c: clipping scale (units of MAD). Larger c -> closer to OLS.
    """
    w = np.ones_like(x, dtype=float)
    coeffs = np.polyfit(x, y, deg)
    for _ in range(iters):
        resid = y - np.polyval(coeffs, x)
        mad = np.median(np.abs(resid - np.median(resid))) + 1e-9
        # Cauchy-like weights
        u = resid / (c * mad)
        w = 1.0 / (1.0 + u * u)
        coeffs = np.polyfit(x, y, deg, w=w)
    return coeffs


def u_space_project_well(tvt_blend, md, z, last_known_tvt, last_known_md, last_eval_md,
                         last_known_z=None, deg=4, beta=0.75):
    """Project a single well's TVT through anchor-relative U-space and back.

    Args:
        tvt_blend: (n_eval,) the current per-row TVT predictions for the well
        md, z:    (n_eval,) eval-row MD and Z
        last_known_tvt, last_known_md, last_known_z: float anchors

    Returns:
        (n_eval,) projected TVT predictions.
    """
    if last_known_z is None:
        # If unknown, fall back to z[0] - (tvt_blend[0] - last_known_tvt) approx
        last_known_z = z[0]

    A_w = last_known_tvt + last_known_z
    U = tvt_blend + z - A_w
    s_denom = max(last_eval_md - last_known_md, 1.0)
    s = (md - last_known_md) / s_denom

    coeffs = robust_polyfit_iterated(s, U, deg=deg, c=2.0, iters=4)
    U_proj = np.polyval(coeffs, s)
    T_proj = A_w + U_proj - z

    return ((1.0 - beta) * tvt_blend + beta * T_proj).astype(np.float32)


def apply_u_space_projection(submission_df, well_id_col='well',
                             pred_col='tvt', md_col='MD', z_col='Z',
                             last_known_lookup=None, deg=4, beta=0.75):
    """Apply per-well U-space projection to a long submission DataFrame.

    last_known_lookup: dict[well_id] = (last_known_tvt, last_known_md, last_known_z, last_eval_md)
    """
    out = submission_df.copy()
    for wid, g in submission_df.groupby(well_id_col, sort=False):
        if wid not in last_known_lookup:
            continue
        last_tvt, last_md, last_z, last_eval_md = last_known_lookup[wid]
        proj = u_space_project_well(
            g[pred_col].values.astype(float),
            g[md_col].values.astype(float),
            g[z_col].values.astype(float),
            last_tvt, last_md, last_eval_md, last_z,
            deg=deg, beta=beta,
        )
        out.loc[g.index, pred_col] = proj
    return out
