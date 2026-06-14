# Feature: seg_b_well — Segment-wise formation bias fitting

**Source kernel**: [lightningv08/lb-7-776-rogii-ridge-sp](../kernels/lightningv08_lb-7-776-rogii-ridge-sp.md)
**Also in**: nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based (identical code)

## What it does
For a given formation surface column `F` imputed at known-rows (X,Y), computes the optimal "bias" `b = TVT + Z − F` across the known prefix, broken into segments to capture drift:

- `b_full = median(b)`
- `b_early = median(b[:n//3])` — early segment
- `b_mid   = median(b[n//3:2*n//3])` — mid segment  
- `b_late  = median(b[-50:])` — last-50 rows (captures recent drift)
- `b_wls   = dot(exp(0.02*arange(n)), b)` — tail-upweighted least-squares

Then each segment's is used to produce a TVT estimate: `tvt = −z_ev + F_ev + b_{segment}`.

## Why it matters
The segment breakdown captures structural dip drift nonparametrically. A formation surface near X,Y gives the "expected" TVT at that spatial position; the bias `b` absorbs the residual geometry (dip deviation, structural offset). `b_late` dominates because it captures the last-50-rows effective dip.

## Cross-refs
- `feature_engineering/plane_knn_formation.md` — the formation imputer that supplies F
- `feature_engineering/dense_ancc_imputer.md` — same segment trick applied to dense ANCC