# Preprocessing: U-space (TVT+Z) projection denoising

**Source kernel**: pilkwang/rogii-target-free-tvt-geosteering (165 votes)

## What it does
A novel post-processing layer beyond SG-smoothing: project predictions into a per-well anchor-relative space, fit a low-degree polynomial, then back-project.

### Formula
1. For each well, define the anchor at the last known row: `A_w = TVT_known_last + Z_known_last`
2. Define `U_i = T_i^blend + Z_i - A_w` (subtract anchor to center)
3. Define normalized MD: `s_i = (MD_i - MD_last_known) / (MD_last_eval - MD_last_known)`
4. Robust polynomial fit `U_i^proj = P_d(s_i)` with degree `d=4` and `C=2.0` robust iters (4 iters)
5. Project: `T_i^projected = (1 - β) * T_i^blend + β * (A_w + U_i^proj - Z_i)` with `β = 0.75`

### Defaults
- `d = 4` (cubic-quartic flexibility)
- `RIDGE_PF_PROJECTION_ROBUST_C = 2.0` (Cauchy-like rho weight cap)
- `RIDGE_PF_PROJECTION_ROBUST_ITERS = 4`
- `β = 0.75` (75% projected, 25% raw)

## Why it matters
- TVT_i + Z_i removes the trajectory tilt — any systematic structural dip becomes a smooth low-frequency function of MD.
- Fitting a polynomial in this space and projecting back produces a denoised TVT trajectory that respects the anchor exactly while smoothing midstream wobble.
- Distinct from SG: SG is causal-symmetric local; U-projection is global per-well.
- Per the kernel notes, this trick was a notable LB lift for the ridge/PF style submissions.

## Code skeleton (conceptual)
```python
A_w = tvt_known[-1] + z_known[-1]
U = T_blend + Z - A_w
s = (MD - MD[last_known]) / (MD[last_eval] - MD[last_known])
# robust polyfit U vs s, degree d=4, ~4 iters with weight clipping
coeffs = robust_polyfit(s, U, deg=4, c=2.0, iters=4)
U_proj = polyval(coeffs, s)
T_projected = (1 - β) * T_blend + β * (A_w + U_proj - Z)
```

## Cross-refs
- preprocessing/savgol_smooth.md (alternative smoothing)
- kernels/pilkwang_rogii-target-free-tvt-geosteering.md (parent kernel)