"""
wellbore_tortuosity.py

Quasi-three-dimensional (Q-3D) wellbore tortuosity evaluation.

Reference:
    Jing, J., Ye, W., Cao, C., Ran, X. (2022). Actual wellbore tortuosity
    evaluation using a new quasi-three-dimensional approach. Petroleum, 8,
    118-127. doi:10.1016/j.petlm.2021.03.008

Summary
-------
DLS (dogleg severity) measures curvature at a single survey interval and misses
the cumulative effect of oscillating, high-frequency tortuosity along the
horizontal section — the kind that increases drillstring contact force, torque,
and drag even when individual DLS readings are low. The tortuosity density (TD)
index improves on DLS but still treats each curved segment independently.

The Q-3D method (Jing et al. 2022) addresses this by:
  1. Projecting the 3D trajectory onto two independent 2D planes (vertical
     undulation and lateral wander), each of which can be analysed separately.
  2. Decomposing each plane into peak-valley segments that capture one full
     convex or concave arc — so frequency (number of reversals) and amplitude
     (arc excess over chord) both contribute to the index.
  3. Adding a deflection angle index (Gamma) that penalises abrupt direction
     changes between adjacent segments, which the paper shows correlates with
     drillstring elastic energy and wellbore contact force (Fig. 5, §3.1).
  4. Combining both planes via RSS to obtain a single Q-3D index per portion.

Pipeline
--------
Stage 1  minimum_curvature()        — survey stations → N, E, TVD, DLS
Stage 2  project_to_planes()        — 3D trajectory → inclined-plane and
                                       azimuth-plane 2D curves
Stage 3  peak_valley_decompose()    — locate peaks/valleys, place breakpoints
         _tortuosity_portion()      — compute T, Gamma, TQG per portion per plane
Stage 4  compute_tortuosity()       — orchestrate stages 1-3, combine planes
                                       via Eq. 13, return results DataFrame
"""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

_TAN_89_DEG = 57.29  # tan(89°), cap for near-perpendicular chord pairs


# ---------------------------------------------------------------------------
# Stage 1 — Minimum curvature
# ---------------------------------------------------------------------------

def minimum_curvature(md, inc, azi):
    """
    Compute 3D wellbore trajectory using the minimum curvature method.

    Parameters
    ----------
    md  : array-like, measured depth (m)
    inc : array-like, inclination (degrees)
    azi : array-like, azimuth (degrees)

    Returns
    -------
    pd.DataFrame with columns:
        MD, INC, AZI, N, E, TVD, DLS
    DLS is in degrees per 30 m.

    Notes
    -----
    This is the ISCWSA-standard minimum curvature method. The ratio factor RF
    = (2/beta) * tan(beta/2) maps the two-endpoint tangent vectors onto the arc
    of a circle of constant curvature. RF → 1 as beta → 0 (straight segment),
    smoothly handling the degenerate case without a branch.
    """
    md  = np.asarray(md,  dtype=float)
    inc = np.asarray(inc, dtype=float)
    azi = np.asarray(azi, dtype=float)

    n_stations = len(md)
    N   = np.zeros(n_stations)
    E   = np.zeros(n_stations)
    TVD = np.zeros(n_stations)
    DLS = np.zeros(n_stations)

    inc_r = np.radians(inc)
    azi_r = np.radians(azi)

    for i in range(1, n_stations):
        dmd = md[i] - md[i - 1]
        I1, I2 = inc_r[i - 1], inc_r[i]
        A1, A2 = azi_r[i - 1], azi_r[i]

        # Dogleg angle (radians)
        cos_beta = np.cos(I2 - I1) - np.sin(I1) * np.sin(I2) * (1.0 - np.cos(A2 - A1))
        cos_beta = np.clip(cos_beta, -1.0, 1.0)
        beta = np.arccos(cos_beta)

        if beta < 1e-10:
            RF = 1.0
        else:
            RF = (2.0 / beta) * np.tan(beta / 2.0)

        dN   = (dmd / 2.0) * (np.sin(I1) * np.cos(A1) + np.sin(I2) * np.cos(A2)) * RF
        dE   = (dmd / 2.0) * (np.sin(I1) * np.sin(A1) + np.sin(I2) * np.sin(A2)) * RF
        dTVD = (dmd / 2.0) * (np.cos(I1) + np.cos(I2)) * RF

        N[i]   = N[i - 1]   + dN
        E[i]   = E[i - 1]   + dE
        TVD[i] = TVD[i - 1] + dTVD

        # DLS in deg / 30 m
        DLS[i] = np.degrees(beta) / dmd * 30.0 if dmd > 0 else 0.0

    return pd.DataFrame({
        "MD":  md,
        "INC": inc,
        "AZI": azi,
        "N":   N,
        "E":   E,
        "TVD": TVD,
        "DLS": DLS,
    })


# ---------------------------------------------------------------------------
# Stage 2 — Projection onto inclined and azimuth planes
# ---------------------------------------------------------------------------

def project_to_planes(survey_df, hz_start_md=None):
    """
    Project the 3D trajectory onto the inclined plane and azimuth plane.

    The inclined plane captures vertical undulation (TVD vs horizontal dist).
    The azimuth plane captures lateral wander (lateral offset vs horizontal dist).

    Parameters
    ----------
    survey_df   : DataFrame from minimum_curvature()
    hz_start_md : float, MD at which horizontal section begins (m).
                  Used only to compute the mean azimuth reference direction.
                  If None, uses the full well.

    Returns
    -------
    incline_x : ndarray, cumulative horizontal displacement (m)
    incline_y : ndarray, TVD (m)
    azimuth_x : ndarray, same cumulative horizontal displacement (m)
    azimuth_y : ndarray, lateral offset perpendicular to mean azimuth (m)

    Notes
    -----
    The two planes capture orthogonal sources of tortuosity:
      - Inclined plane (TVD vs horiz-dist): vertical undulation — the well
        dipping above and below its target depth.
      - Azimuth plane (lateral offset vs horiz-dist): lateral wander — the
        well drifting left and right of its planned azimuth.

    Coordinate convention: azimuth is measured clockwise from North.
    Along-azimuth unit vector in the (N, E) frame: (cos(azi), sin(azi)).
    Right-perpendicular unit vector (positive = right looking downhole):
    (-sin(azi), cos(azi)).
    """
    N   = survey_df["N"].values
    E   = survey_df["E"].values
    TVD = survey_df["TVD"].values
    MD  = survey_df["MD"].values
    AZI = survey_df["AZI"].values

    # Cumulative horizontal displacement (step-wise)
    dN = np.diff(N, prepend=N[0])
    dE = np.diff(E, prepend=E[0])
    dH = np.sqrt(dN**2 + dE**2)
    dH[0] = 0.0
    horiz_dist = np.cumsum(dH)

    # Mean azimuth of horizontal section (or full well)
    if hz_start_md is not None:
        mask = MD >= hz_start_md
    else:
        mask = np.ones(len(MD), dtype=bool)

    hz_azi = AZI[mask]
    # Circular mean azimuth
    mean_azi_rad = np.arctan2(
        np.mean(np.sin(np.radians(hz_azi))),
        np.mean(np.cos(np.radians(hz_azi)))
    )

    # Decompose (N, E) into along-azimuth and perpendicular components
    cos_a = np.cos(mean_azi_rad)
    sin_a = np.sin(mean_azi_rad)
    # Azimuth is clockwise from North.
    # Along-azimuth unit vector in (N, E) frame: (cos(azi), sin(azi))
    # Perpendicular (right-looking-downhole):    (-sin(azi), cos(azi))
    along_N = np.cos(mean_azi_rad)
    along_E = np.sin(mean_azi_rad)
    perp_N  = -np.sin(mean_azi_rad)
    perp_E  =  np.cos(mean_azi_rad)

    lateral_offset = N * perp_N + E * perp_E

    return horiz_dist, TVD, horiz_dist, lateral_offset


# ---------------------------------------------------------------------------
# Stage 3a — Peak-valley decomposition
# ---------------------------------------------------------------------------

def peak_valley_decompose(x, y, prominence=0.1, min_distance=2):
    """
    Find breakpoint indices between segments defined by peaks and valleys.

    Breakpoints are placed at the midpoint (by arc-length) between each
    adjacent peak-valley pair (Jing et al. Eq. 4).

    Parameters
    ----------
    x           : ndarray, x-coordinates of the projection curve
    y           : ndarray, y-coordinates
    prominence  : float, minimum prominence for peak detection (same units as y)
    min_distance: int, minimum samples between peaks

    Returns
    -------
    breakpoint_indices : list of int, indices into x/y arrays
                         Always starts with 0 and ends with len(x)-1.

    Notes
    -----
    Each segment in the Peak-Valley decomposition captures exactly one convex
    or concave arc — from one breakpoint (midway between two extrema) to the
    next. This means each segment contains one peak or one valley near its
    centre, so the arc-excess and chord-slope metrics in _tortuosity_portion()
    reflect a single physical undulation rather than a mix of multiple ones.

    The `prominence` parameter controls what counts as a "real" undulation.
    Too low: survey-measurement noise registers as peaks, inflating the index.
    Too high: genuine low-amplitude tortuosity is ignored. A value around
    0.1 m works for horizontal wells where TVD amplitude is on the order of
    metres and survey resolution is ~0.1 m.
    """
    peaks,  _ = find_peaks( y, prominence=prominence, distance=min_distance)
    valleys, _ = find_peaks(-y, prominence=prominence, distance=min_distance)

    # Merge and sort all extrema
    extrema = np.sort(np.concatenate([peaks, valleys]))

    if len(extrema) < 2:
        # No meaningful peaks/valleys — return only endpoints
        return [0, len(x) - 1]

    # Breakpoint between extrema[i] and extrema[i+1]: midpoint by arc-length
    # Compute cumulative arc-length along the curve
    dx = np.diff(x)
    dy = np.diff(y)
    seg_len = np.sqrt(dx**2 + dy**2)
    arc_len = np.concatenate([[0.0], np.cumsum(seg_len)])

    breakpoints = [0]
    for i in range(len(extrema) - 1):
        i0, i1 = extrema[i], extrema[i + 1]
        # Arc-length midpoint between i0 and i1
        target = (arc_len[i0] + arc_len[i1]) / 2.0
        # Find index closest to this arc-length within [i0, i1]
        sub_arc = arc_len[i0: i1 + 1]
        local_idx = np.argmin(np.abs(sub_arc - target))
        bp = i0 + local_idx
        if bp != breakpoints[-1]:
            breakpoints.append(int(bp))

    if breakpoints[-1] != len(x) - 1:
        breakpoints.append(len(x) - 1)

    return breakpoints


# ---------------------------------------------------------------------------
# Stage 3b-e — Portion tortuosity
# ---------------------------------------------------------------------------

def _arc_length_between(x, y, idx_start, idx_end):
    """Sum of Euclidean distances between consecutive points from idx_start to idx_end."""
    xs = x[idx_start: idx_end + 1]
    ys = y[idx_start: idx_end + 1]
    return float(np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2)))


def _chord_length(x, y, idx_start, idx_end):
    """Straight-line distance between endpoints."""
    return float(np.sqrt((x[idx_end] - x[idx_start])**2 + (y[idx_end] - y[idx_start])**2))


def _chord_slope(x, y, idx_start, idx_end):
    """Slope (dy/dx) of the chord; returns large value for vertical chords."""
    dx = x[idx_end] - x[idx_start]
    dy = y[idx_end] - y[idx_start]
    if abs(dx) < 1e-12:
        return np.sign(dy) * 1e12
    return dy / dx


def _tortuosity_portion(x, y, bp_indices):
    """
    Compute tortuosity T, deflection angle index Gamma, and combined TQG
    for one portion, given breakpoint indices within x/y.

    Parameters
    ----------
    x, y        : ndarray, full projection curve coordinates
    bp_indices  : list of int, breakpoint indices for THIS portion (subset)

    Returns
    -------
    T_val   : float, tortuosity index (Eq. 7)
    Gamma   : float, deflection angle index (Eq. 10-11)
    TQG     : float, combined index (Eq. 12)
    """
    n_segs = len(bp_indices) - 1  # number of segments = n in the paper

    if n_segs < 1:
        return 0.0, 0.0, 0.0

    arc_lengths   = []
    chord_lengths = []
    slopes        = []

    for q in range(n_segs):
        i0, i1 = bp_indices[q], bp_indices[q + 1]
        lsq = _arc_length_between(x, y, i0, i1)
        lcq = _chord_length(x, y, i0, i1)
        arc_lengths.append(lsq)
        chord_lengths.append(lcq)
        slopes.append(_chord_slope(x, y, i0, i1))

    Lp = sum(arc_lengths)
    if Lp < 1e-12:
        return 0.0, 0.0, 0.0

    # Eq. 7 — tortuosity index
    # (lsq/lcq - 1) is the fractional arc excess: zero for a straight segment,
    # positive for a curved one. (lsq/Lp) is the length-weight so longer
    # segments contribute proportionally more to the portion total.
    T_sum = 0.0
    for lsq, lcq in zip(arc_lengths, chord_lengths):
        if lcq < 1e-9:
            continue
        T_sum += (lsq / lcq - 1.0) * (lsq / Lp)
    # n_segs amplifies by frequency: more segments in the same portion length
    # means higher-frequency oscillation, which increases torque and drag.
    T_val = n_segs * Lp * T_sum

    # Eq. 10-11 — deflection angle index
    # The deflection angle between adjacent chord lines captures abrupt
    # direction changes that increase drillstring elastic energy and contact
    # force even when amplitude and frequency are identical (paper Fig. 5, §3.1).
    if n_segs < 2:
        Gamma = 0.0
    else:
        deflections = []
        for q in range(n_segs - 1):
            k1, k2 = slopes[q], slopes[q + 1]
            denom = 1.0 + k1 * k2
            if abs(denom) < 1e-12:
                # Nearly perpendicular chords — cap at tan(89°)
                tan_theta = _TAN_89_DEG
            else:
                tan_theta = min(abs(k2 - k1) / abs(denom), _TAN_89_DEG)
            deflections.append(tan_theta)
        m = len(deflections)
        Gamma = Lp * (1.0 / m) * sum(deflections)

    # Eq. 12
    TQG = np.sqrt(T_val**2 + Gamma**2)

    return T_val, Gamma, TQG


# ---------------------------------------------------------------------------
# Main computation function
# ---------------------------------------------------------------------------

def _compute_from_survey_df(survey_df, portion_length, hz_start_md,
                            prominence, min_peak_distance):
    """
    Stages 2-3 of the Q-3D pipeline given a populated survey DataFrame.

    Internal helper. ``survey_df`` must contain MD, INC, AZI, N, E, TVD
    columns (DLS optional; not used downstream). Used by both
    ``compute_tortuosity`` (which derives the survey via minimum_curvature)
    and ``compute_tortuosity_from_xyz`` (which derives it from cartesian
    trajectory via finite differences).
    """
    md = survey_df["MD"].values

    inc_x, inc_y, azi_x, azi_y = project_to_planes(survey_df, hz_start_md)

    # Global peak-valley decomposition on each plane
    bp_inc = peak_valley_decompose(inc_x, inc_y,
                                   prominence=prominence,
                                   min_distance=min_peak_distance)
    bp_azi = peak_valley_decompose(azi_x, azi_y,
                                   prominence=prominence,
                                   min_distance=min_peak_distance)

    # Divide well into portions by measured depth
    md_start = md[0]
    md_end   = md[-1]
    portion_starts = np.arange(md_start, md_end, portion_length)

    rows = []
    for ps in portion_starts:
        pe = min(ps + portion_length, md_end)

        # Indices within this MD portion
        mask = (md >= ps) & (md <= pe)
        idx_portion = np.where(mask)[0]
        if len(idx_portion) < 2:
            continue

        i_start = idx_portion[0]
        i_end   = idx_portion[-1]

        # Breakpoints within this portion for each plane
        def portion_bps(global_bps):
            # Keep only BPs inside [i_start, i_end]; always include endpoints
            inner = [b for b in global_bps if i_start <= b <= i_end]
            pts = sorted(set([i_start] + inner + [i_end]))
            return pts

        bps_inc_p = portion_bps(bp_inc)
        bps_azi_p = portion_bps(bp_azi)

        T_i, G_i, TQG_i = _tortuosity_portion(inc_x, inc_y, bps_inc_p)
        T_a, G_a, TQG_a = _tortuosity_portion(azi_x, azi_y, bps_azi_p)

        # Eq. 13 — RSS combination assumes the inclined and azimuth planes
        # contribute independently to total tortuosity (orthogonal sources).
        TQG_Q3D = np.sqrt(TQG_i**2 + TQG_a**2)

        rows.append({
            "portion_start_md": ps,
            "portion_end_md":   pe,
            "T_incline":        T_i,
            "Gamma_incline":    G_i,
            "TQG_incline":      TQG_i,
            "T_azimuth":        T_a,
            "Gamma_azimuth":    G_a,
            "TQG_azimuth":      TQG_a,
            "TQG_Q3D":          TQG_Q3D,
        })

    results_df = pd.DataFrame(rows)

    projections = {
        "incline_x": inc_x,
        "incline_y": inc_y,
        "azimuth_x": azi_x,
        "azimuth_y": azi_y,
        "bp_incline": bp_inc,
        "bp_azimuth": bp_azi,
    }

    return results_df, survey_df, projections


def compute_tortuosity(md, inc, azi, portion_length=100.0, hz_start_md=None,
                       prominence=0.1, min_peak_distance=2):
    """
    Full Q-3D tortuosity evaluation (Jing et al. 2022) from survey-station data.

    Parameters
    ----------
    md, inc, azi    : array-like survey data (m, degrees, degrees)
    portion_length  : float, MD length of each evaluation portion (m)
    hz_start_md     : float, MD at which horizontal section starts (m).
                      If None, uses full well.
    prominence      : float, peak detection prominence threshold (m)
    min_peak_distance: int, min samples between detected peaks

    Returns
    -------
    results_df : pd.DataFrame with columns:
        portion_start_md, portion_end_md,
        T_incline, Gamma_incline, TQG_incline,
        T_azimuth, Gamma_azimuth, TQG_azimuth,
        TQG_Q3D
    survey_df  : pd.DataFrame with full survey (MD, INC, AZI, N, E, TVD, DLS)
    projections: dict with keys incline_x, incline_y, azimuth_x, azimuth_y,
                 bp_incline, bp_azimuth

    Notes
    -----
    `portion_length` sets the evaluation window, analogous to the gate length
    in a running-average DLS computation. Shorter portions (e.g. 30 m) resolve
    micro-tortuosity and individual frac-stage intervals but have fewer survey
    stations per window, making the index noisier. Longer portions (e.g. 300 m)
    smooth over local variations and give a more representative average. 100 m
    is a practical default for Montney-scale horizontal sections (~3 km lateral).
    """
    md  = np.asarray(md,  dtype=float)
    inc = np.asarray(inc, dtype=float)
    azi = np.asarray(azi, dtype=float)

    survey_df = minimum_curvature(md, inc, azi)
    return _compute_from_survey_df(survey_df, portion_length, hz_start_md,
                                   prominence, min_peak_distance)


def compute_tortuosity_from_xyz(md, x, y, z, portion_length=100.0, hz_start_md=None,
                                 prominence=0.1, min_peak_distance=2,
                                 x_axis='east', y_axis='north', z_axis='up'):
    """
    Q-3D tortuosity from cartesian trajectory.

    Use this entry point when only (md, x, y, z) is available — e.g. the ROGII
    Wellbore Geology Prediction dataset, where the horizontal-well CSVs ship
    with an XYZ trajectory but no inclination/azimuth columns.

    Parameters
    ----------
    md      : array-like, measured depth at each row (m or ft, consistent with
              x/y/z and portion_length)
    x, y, z : array-like cartesian trajectory (consistent units with md)
    x_axis, y_axis, z_axis : {'east','west','north','south','up','down'}
        Tells the function how the input columns map onto the
        (N, E, TVD)-down convention used by ``project_to_planes``.
        For ROGII (Z negative downward, X-East, Y-North) pass
        ``x_axis='east', y_axis='north', z_axis='up'``.
    portion_length, hz_start_md, prominence, min_peak_distance
        Same as ``compute_tortuosity``.

    Returns
    -------
    Same triple as ``compute_tortuosity``: (results_df, survey_df, projections).

    Notes
    -----
    INC and AZI are derived per station from successive (E, N, TVD) deltas:
      dN, dE, dTVD computed from `np.diff(..., prepend=...)`
      INC = degrees(arctan2(sqrt(dN**2 + dE**2), abs(dTVD)))
      AZI = degrees(arctan2(dE, dN)) % 360
    The first station copies the second (no preceding interval).

    The Q-3D tortuosity index is rotation-invariant by construction (peak-valley
    decomposition + RSS combination across orthogonal planes), so the
    x_axis/y_axis assignment matters only for which physical direction the
    inclined-plane vs. azimuth-plane decomposition projects onto. TQG_Q3D is
    unaffected.
    """
    md = np.asarray(md, dtype=float)
    x  = np.asarray(x,  dtype=float)
    y  = np.asarray(y,  dtype=float)
    z  = np.asarray(z,  dtype=float)

    # Map (x, y, z) input axes onto the module's (N, E, TVD-down) convention.
    axis_map = {'east':  ('E', +1.0), 'west':  ('E', -1.0),
                'north': ('N', +1.0), 'south': ('N', -1.0),
                'up':    ('TVD', -1.0), 'down': ('TVD', +1.0)}
    coords = {'N': None, 'E': None, 'TVD': None}
    for arr, label in [(x, x_axis), (y, y_axis), (z, z_axis)]:
        if label not in axis_map:
            raise ValueError(f"axis label {label!r} must be one of {list(axis_map)}")
        target, sign = axis_map[label]
        if coords[target] is not None:
            raise ValueError(f"two input axes both map to {target}; check x/y/z_axis")
        coords[target] = sign * arr
    if any(v is None for v in coords.values()):
        missing = [k for k, v in coords.items() if v is None]
        raise ValueError(f"input axes do not cover {missing}; check x/y/z_axis")

    N, E, TVD = coords['N'], coords['E'], coords['TVD']

    # Finite-difference INC and AZI at each station. First station copies second.
    dN   = np.diff(N,   prepend=N[0])
    dE   = np.diff(E,   prepend=E[0])
    dTVD = np.diff(TVD, prepend=TVD[0])
    dHoriz = np.sqrt(dN**2 + dE**2)

    inc = np.degrees(np.arctan2(dHoriz, np.abs(dTVD)))
    azi = np.degrees(np.arctan2(dE, dN)) % 360.0

    # Replace the placeholder first sample (zero deltas → INC=0, AZI=0) with
    # the second-station values so projections don't anchor on a degenerate point.
    if len(md) >= 2:
        inc[0] = inc[1]
        azi[0] = azi[1]

    # Build a survey DataFrame with the same shape as minimum_curvature's output.
    # DLS isn't used downstream by _compute_from_survey_df; populate with zeros.
    survey_df = pd.DataFrame({
        "MD":  md,
        "INC": inc,
        "AZI": azi,
        "N":   N,
        "E":   E,
        "TVD": TVD,
        "DLS": np.zeros_like(md),
    })

    return _compute_from_survey_df(survey_df, portion_length, hz_start_md,
                                   prominence, min_peak_distance)
