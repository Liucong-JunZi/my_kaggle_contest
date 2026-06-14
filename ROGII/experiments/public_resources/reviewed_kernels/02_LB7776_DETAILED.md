# Kernel: lightningv08/lb-7-776-rogii-ridge-sp (LB 7.776)

**Author**: lightningv08 (fork chain through aidensong123 → ravaghi artifacts)
**Votes**: 86 | **File**: lb-7-776-rogii-ridge-sp.ipynb (33 cells, ~1353 LOC)

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         TWO-BRANCH SYSTEM                       │
├──────────────────────────┬───────────────────────────────────────┤
│  BRANCH 1: Ridge Stack   │  BRANCH 2: Heuristic Selector        │
│                          │                                       │
│  8 base signals:         │  128-seed PF likelihood ensemble      │
│    PF-ANCC, PF-Z         │    at 4 scales {3,5,8,12}            │
│    7 beam variants       │  + 14-config beam ensemble           │
│    3 multi-scale NCC     │                                       │
│    sc_ens (NCC ensem.)   │  Binned selector (6 variants):       │
│    plane-KNN per-form.   │    based on n_eval_rows & z_span      │
│    dense ANCC kNN        │    picks one of 6 PF/beam blends     │
│    segment b_well (30)   │                                       │
│    GR rolls + anchors    │                                       │
│         ↓                │          ↓                            │
│  3×LGB + 2×CAT → Ridge   │    Direct per-well PF/beam output    │
│  → PP (exp ramp + SG)    │                                       │
├──────────────────────────┴───────────────────────────────────────┤
│              FINAL: 0.3 × Ridge + 0.7 × Selector                 │
└──────────────────────────────────────────────────────────────────┘
```

## Key Feature Families (ranked by importance)

### 1. Per-formation TVT Segment Biases (30 features) — HIGHEST ROI TO ADD
```
For each formation F in {ANCC, ASTNU, ASTNL, EGFDU, EGFDL, BUDA}:
  b_full  = median(ktvt + kz - F_kn)        — all known rows
  b_late  = median(last 50 rows)             — tail-weighted
  b_early = median(first third)              — early segment
  b_mid   = median(middle third)             — middle segment
  b_wls   = exp(0.02*i) tail-upweighted WLS  — weighted least squares
  → 6 formations × 5 variants = 30 features
  + aggregates: form_mean_d, form_std_d, form_rng_d, spatial_knn_dist
```
**We have the STUB** at `experiments/public_harvest/feat_formation_segment_b_well.py` but it's NOT hooked into `data_loader.py`.

### 2. Anchored GR Offsets (4 anchors × 11 offsets = 44 features)
```
Anchor points: last_known_TVT, beam_ref, sc_ens, pf_use
For each anchor A:
  td{A}{offset} = hgr - tw_gr(anchor + offset)
  offsets: [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
  → ~44 features total
```
**We have**: 1 feature (`gr_diff_from_last = gr - last_known_gr`)
**Gap**: We replace 44 rich anchored GR features with a single difference.

### 3. Multi-scale NCC (3 windows + ensemble)
```
ncc_hws_8_d    = NCC(typewell_gr, horizontal_gr, half_window=8)
ncc_hws_15_d   = NCC(half_window=15)
ncc_hws_25_d   = NCC(half_window=25)
sc_ens_d       = softmax(T=3) weighted ensemble of above
```
**We have**: Nothing. No NCC features exist.

### 4. Dense ANCC kNN Imputation
```
Subsample 60 (X,Y,ANCC) points per well
Build cKDTree over all train wells
Predict dense ANCC + variance + nearest-neighbor distance
```
**We have**: Nothing.

### 5. GR Signal Expansion
```
Windows: 5, 21, 51, 101 (we have these for mean+std)
PLUS:
  gr_lag_{1,5,15,30}        — forward shift
  gr_lead_{1,5,15,30}       — backward shift
  gr_diff_1, gr_diff_2      — first/second discrete diff
  gr_env                    — rolling max (envelope)
  gr_nrg                    — sqrt(rolling_mean_of_squares)
```
**We have**: mean+std at 4 windows only.
**Gap**: Missing ~14 additional GR features.

### 6. Signal Aggregates
```
Across 11 base signals (PF + 7 beams + 3 NCC + sc_ens + ANCC + dense):
  sig_std      — per-row standard deviation across signals
  sig_mean_d   — per-row mean (as offset)
```
**We have**: Nothing.

## Model Architecture (Compare to Our R10)

| Aspect | LB 7.776 Kernel | Our R10 |
|--------|-----------------|---------|
| LGB models | 3: big (leaves=255) + 2× small-deep (leaves=64, seeds 0/29) | 1 LGB (leaves=127) |
| CatBoost models | 2 (depth=7, seeds 7/123, LRs 0.02/0.03) | 1 Cat (depth=8, seed=42) |
| Meta-learner | Ridge (alpha=1.66, positive=true) | Simple LGB+CAT blend (grid search) |
| Post-processing | exp ramp + SG-filter (17,3) | NONE |
| Heuristic blend | 0.7 weight on selector branch | NONE |
| Ridge stack | 5 OOF preds → Ridge → blended output | NONE |
| Feature count | ~100-150 features | 43 features (v14) |

### Detailed Hyperparams

**LightGBM #1 (big)**:
```
num_leaves=255, lr=0.030, n_estimators=5000, reg_lambda=3.0, reg_alpha=0.05
subsample=0.8 (freq=1), colsample_bytree=0.8, min_child_samples=15, max_bin=255
```

**LightGBM #2,#3 (small-deep, seeds 0/29)**:
```
num_leaves=64, lr=0.00934, n_estimators=10000, reg_lambda=95.75, reg_alpha=10.79
subsample=0.474, colsample_bytree=0.393, min_child_samples=40, min_child_weight=0.241
```
Note: much higher regularization, much lower learning rate, many more trees.

**CatBoost ×2 (seeds 7/123)**:
```
depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254
iterations=8000, lr=0.020 (seed=7) / 0.030 (seed=123), od_wait=300
```

**Ridge meta**:
```
alpha=1.66, tol=5e-4, positive=True, fit_intercept=True
```

## Post-processing Pipeline (ZERO cost for moderate gain)

We don't do ANY of this. The LB 7.776 kernel does:

### Step 1: Exponential ramp blend with PF-ANCC
```python
delta = (md * (1 - w_pf) + pf_ancc * w_pf) * (1 - exp(-md_since / tau)) * alpha
# tau=85, w_pf=0.09, alpha=1.0
```
This progressively blends the ML prediction toward the raw PF-ANCC signal as distance from the last_known point increases. The exp(-md_since/tau) term means **near the anchor** (md_since≈0), the prediction is ~all ML; **far from the anchor** (md_since large), it's ~all PF-ANCC.

### Step 2: Savitzky-Golay smoothing
```python
savgol_filter(pred, window_length=17, polyorder=3)
```
Per-well smoothing after the blend. This is test-safe because SG is purely local (no cross-well leakage).

### Why this works
The ML model learns deviations from last_known_TVT but its uncertainty increases with distance. The PF-ANCC physical model has the opposite behavior — it's rooted in GR correlation which is stable over long distances. The blend optimally merges the two.

## Heuristic Selector Branch (the 0.7 secret)

This is what pushes the score from ~9.5 (Ridge branch alone) to **7.776**.

### PF likelihood ensemble (128 seeds × 500 particles)
```
For scale in {3, 5, 8, 12}:
  p_{i,row} = softmax(log_likelihood(i, row) / scale)
  tvt = sum(p_i * tvt_i) over all 128 seeds
  → gives 4 PF ensemble TVTs
```

### Binned Selector
```python
# Key constants:
SELECTOR_N_EVAL_THRESHOLD = 4840
SELECTOR_Z_SPAN_THRESHOLDS = [136.73, 185.51]

# 6 variants + 1 default:
{
  0: 'pf_scale_5_hold_0.2',
  1: 'pf_scale_3_hold_0.15',
  2: 'pf_scale_12_beam_0.2_hold_0.15',
  3: 'pf_scale_5_hold_0.15',
  4: 'pf_scale_5_beam_0.05_hold_0.05',
  5: 'pf_scale_12_beam_0.2_hold_0.05',
}
# default: 'pf_scale_8_hold_0.2'
```
`hold` = weight on `last_known_TVT` (i.e., how much to regress toward anchor).
`beam` = weight on the 14-config beam ensemble.

The selector chooses per-well blending strategy based on `n_eval_rows` (well length) and `z_span` (vertical variation). Short wells with small z-span get different treatment than long wells with large z-span.

## What This Means for Our Score

The LB 7.776 kernel is achieving its score through **diversity of base signals** plus a **sophisticated per-well blending strategy**, not through a single better model.

Estimated contribution of each component:
| Component | Estimated LB contribution |
|-----------|--------------------------|
| Ridge stack (LGB+CB) on ~100 features | ~9-10 LB |
| Heuristic selector (128-seed PF + beam) | ~8-9 LB |
| Final blend 0.3/0.7 | **7.776 LB** |
| Post-processing (exp ramp + SG) | ~-0.3 improvement |
| Additional features (b_well, NCC, anchors) | ~-0.5 vs base features |

## "Easiest Things to Steal" Ranking

| # | What | Effort | Expected improvement (OOF perwell) | Dependencies |
|---|------|--------|-----------------------------------|-------------|
| 1 | **Post-processing (exp ramp + SG)** | 30 min | -0.3 to -0.5 | Only needs pf_ancc preds (we have them) |
| 2 | **Ridge meta-stack (3×LGB+2×CAT)** | 2h | -0.3 to -0.5 | Need to train 5 models (3 LGB seeds, 2 CAT seeds) |
| 3 | **Multi-scale NCC features** | 2h | -0.2 to -0.4 | New feature gen code needed |
| 4 | **Anchored GR offsets** | 1h | -0.2 to -0.3 | Add 44 features to data_loader |
| 5 | **Per-formation segment b_well** | 1h | -0.2 to -0.4 | Stub exists, just integrate + retrain |
| 6 | **Dense ANCC imputation** | 2h | -0.1 to -0.3 | Needs KDTree over train set |
| 7 | **GR signal expansion (lags/env/nrg)** | 30min | -0.1 to -0.2 | Add columns to data_loader |
| 8 | **Small-deep LGB with high regularization** | 30min | -0.1 to -0.2 | New candidate config |
| 9 | **128-seed PF likelihood ensemble** | 4-5h | -1.0 to -2.0 | Numba JIT needed for speed |
| 10 | **Selector branch** | 1d | -1.0 to -1.5 | Needs 128-seed PF first |
