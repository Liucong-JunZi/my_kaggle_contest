# Feature: Particle-Filter PF-ANCC + PF-Z (twin physical signals)

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp (used in dozens of forks)

## What it does
Two complementary numba-JIT particle filters that both produce per-row TVT estimates:

### PF-ANCC (anchored, GR-driven)
- State: `pos = TVT + Z` (depth in absolute frame), `rate = dpos/dMD`
- Motion: `pos_{t+1} = pos_t + rate * dMD + PN * noise`; `rate_{t+1} = ALPHA * rate_t + RN * noise`
- Likelihood: `gaussian((gr_observed - tw_gr_lookup(pos - z)) / gr_sigma)`
- Resampling: systematic when `n_eff < 0.5 * N` with rough-perturbation jitter

Constants:
- N=600, ANCC_ALPHA=0.998, ANCC_RN=0.002, ANCC_PN=0.005
- ANCC_IS=0.3 (init position spread), ANCC_RP=0.1, ANCC_RR=0.001

### PF-Z (regression-prior augmented)
- Same state, but the motion prior uses a **regression of dTVT/dMD on dZ/dMD** estimated from the known prefix:
  ```
  dTVT/dMD ≈ β * (dZ/dMD) + intercept
  ```
- The likelihood combines raw GR gaussian (weight 0.7) and rolling-5-smoothed GR gaussian (weight 0.3 = `PF_GR_WT`).
- Then a **z-prior penalty** is applied based on `(vel - β*dzd - icpt) / zsig` (encourages physically consistent motion).
- N=600, MOM=0.993, VN=0.005, PN=0.01, GR_WT=0.3, PF_ROUGH_P=0.2, PF_ROUGH_V=0.003

## Why it matters
- PF-ANCC is the dominant per-row physical signal (consistently 9-12 ft RMSE alone).
- PF-Z adds an orthogonal estimator using the geometric trajectory; the LB7.776 kernel uses both deltas and `pf_vs_z` (their difference) as features.
- Per-row stds (`pf_ancc_std`, `pf_z_std`) quantify confidence.

## Outputs
- `pf_ancc`, `pf_ancc_std`, `pf_ancc_delta`
- `pf_z`, `pf_z_delta`, `pf_vs_z`

## Cross-refs
- ensemble_weights/selector_binned.md (uses PF in heuristic branch)
- feature_engineering/anchored_gr_offsets.md (uses pf_use as anchor for `tdpf*` offsets)