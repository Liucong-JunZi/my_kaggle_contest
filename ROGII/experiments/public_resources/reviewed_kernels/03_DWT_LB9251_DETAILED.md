# Kernel: nihilisticneuralnet/9-251-rogii-wellbore-geology-prediction-dwt-based (LB 9.251)

**Author**: nihilisticneuralnet | **Votes**: 593 (highest-voted public kernel)
**File**: 9-251-rogii-wellbore-geology-prediction-dwt-based.ipynb

## Architecture

Identical Ridge-stacked LGB+CB backbone as LB 7.776, but **adds DTW alignment features** and uses **only the Ridge stack branch** (no heuristic selector, no 0.3/0.7 blend). The LB 9.251 score confirms the selector branch is worth ~1.5 LB.

```
LB 7.776 = Ridge stack (~9.5) + Heuristic selector (~7.8) blended 0.3/0.7
LB 9.251 = Ridge stack only (with DTW)
         ≈ Ridge stack alone (without DTW) would be ~9.5-9.8
```

## What's NEW (that we DON'T have)

### DTW Sakoe-Chiba Alignment Features

4-radius constrained DTW (Sakoe-Chiba band):
```python
DTW_RADII = (20, 50, 100, 200)
```
For each radius r:
- Aligns horizontal GR against typewell GR using DTW with band constraint r
- Extracts warp-path local slope features (how GR velocity changes)
- Computes cost-weighted ensemble TVT

Stochastic DTW (12 realizations):
```python
DTW_STOCH_K = 12
DTW_STOCH_TEMP = 3.0
```
- Adds Gumbel noise to cost matrix before DTW traceback
- Runs 12 times → per-row mean, std, coefficient of variation
- This provides uncertainty quantification for each prediction

DTW-anchored GR offsets:
```python
DTW_OFFS = [-20, -10, -5, -2, 0, 2, 5, 10, 20]
# Same pattern as anchored GR offsets, but using DTW alignment path
```

### Optuna-tuned Post-processing

Unlike LB 7.776's fixed `tau=85, w_pf=0.09, alpha=1.0, sg_w=17, sg_p=3`:
```python
# Multivariate TPE, 1000 trials, 100 warmup
search_space:
  alpha: float[0.98, 1.02]
  tau: int[35, 220]
  w_pf: float[0.03, 0.16]
  sg_w: int[5, 51]      # Savitzky-Golay window
  sg_p: int[2, 5]        # Savitzky-Golay polynomial order
```

### DTW Feature Surface

Key insight: DTW captures **nonlinear GR stretching** that PF/beam/NCC miss. Geological formations can have variable thicknesses that DTW aligns better than uniform NCC windows. The warp-path slopes indicate whether formations are thinning or thickening, which correlates with TVT changes.

## Actionable Takeaway

### Option A: Quick DTW integration (4-6h)
Add `DTW_RADII = [20, 50, 100, 200]` Sakoe-Chiba DTW features:
- 4 cost-weighted ensemble TVT offsets
- Warp-path slopes at each radius
- Anchored GR offsets from DTW path
- Expected: -0.1 to -0.3 OOF improvement beyond our current feature set

### Option B: Just do post-processing first (30min)
Skip DTW for now. The exp ramp + SG-smooth post-processing is cheaper and likely gives more per-effort improvement.

### Option C: Full DTW + LGB integration (8h)
Add DTW features, retrain all LGB/CAT candidates, re-run hill climb. Expected: -0.3 to -0.5 OOF combined with feature expansion.

## Score Comparison with LB 7.776

| Component | Score | Delta from 7.776 |
|-----------|-------|------------------|
| LB 7.776 (full) | 7.776 | — |
| This kernel (Ridge+DTW, no selector) | 9.251 | +1.475 |
| Implied: selector value | — | ~-1.5 |
| Implied: DTW value over base | — | ~-0.2 to -0.4 |
