# ACTIONABLE INSIGHTS — What to Steal First

**Date**: 2026-06-15
**Context**: Our R10 best OOF perwell = 9.182 (8 legacy + c01-c08 candidates)
**Target**: Close gap to LB 7.776 kernel

## Priority Action Plan (Ranked by Impact/Effort)

### TIER 1: DO TODAY (Zero Dependencies, < 2h total)

#### [P0] Add Post-processing to Submission Pipeline
**File**: `experiments/round_010/submit_ensemble.py`
**What**:
```python
# Step 1: Exponential ramp blend (simple formula, grid-searchable params)
def apply_pp(pred_tvt, last_known_tvt, md_since, pf_ancc, tau=85, w_pf=0.09, alpha=1.0):
    """Blend ML prediction toward PF-ANCC as distance from anchor increases."""
    ml_pred = pred_tvt - last_known_tvt  # offset form
    blend = (1 - np.exp(-md_since / tau)) * (alpha * (1-w_pf) * ml_pred + w_pf * pf_ancc)
    # Near anchor: blend ~ ml_pred*(1-0)*alpha*(1-w_pf) — mostly ML
    # Far from anchor: blend ~ 1*(alpha*(1-w_pf)*ml_pred + w_pf*pf_ancc) — mostly PF
    # Wait, re-reading the kernel: delta = (md*(1-w_pf) + pf_ancc*w_pf) * (1 - exp(-md_since/tau)) * alpha
    # Where md = ML prediction (offset), not the MD column
    delta = (ml_pred * (1 - w_pf) + pf_ancc * w_pf) * (1 - np.exp(-md_since / tau)) * alpha
    return last_known_tvt + delta

# Step 2: Savitzky-Golay smoothing (per well, test-safe)
from scipy.signal import savgol_filter
def smooth_per_well(df, pred_col, well_col='well', window=17, polyorder=3):
    """Apply SG filter separately to each well's predictions."""
    result = np.zeros(len(df))
    for w, g in df.groupby(well_col):
        idx = g.index.values
        result[idx] = savgol_filter(pred_col[idx], window, polyorder)
    return result
```
**Expected**: -0.3 to -0.5 OOF perwell improvement
**Effort**: ~30 min
**Risk**: None — test-safe, can be toggled on/off

#### [P1] Integrate Per-Formation Segment Biases
**Files**:
- Already have: `experiments/public_harvest/feat_formation_segment_b_well.py` (stub complete)
- Stub needs: `feat_formation_plane_knn.py` for the FormationPlaneKNN imputer
- Integration: add to `shared/data_loader.py`'s `_build()` → produces ~30 new features

**What changes**:
1. Hook `FormationPlaneKNN` into data_loader (or run as pre-step)
2. Call `add_formation_segment_features()` for each well
3. Add output columns to joined features
4. Retrain candidates (c01-c08)

**Expected**: -0.2 to -0.4 OOF improvement
**Effort**: ~1h (code mostly written)
**Risk**: Formation imputer needs all-train KDTree — ensure no self-well leakage

#### [P2] Add Anchored GR Offsets
**File**: `shared/data_loader.py`
**Add**: 4 anchors × 11 offsets = 44 features
```python
ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]
# For each anchor point A in {last_known_tvt, beam_ref, sc_ens, pf_use}:
#   td{A}{offset} = gr[row] - typewell_gr_interp(A + offset)
```
**Expected**: -0.2 to -0.3 OOF improvement
**Effort**: ~1h (straightforward feature engineering)

#### [P3] Upgrade to Ridge Meta-Stack with 3×LGB + 2×CAT
**File**: `orchestrator/train_one.py` + new candidates
**What**:
- Instead of single LGB + single CAT, train:
  - LGB-big (leaves=255, lr=0.03, 5000 iters, reg_lambda=3.0)
  - LGB-small-1 (leaves=64, lr=0.009, 10000 iters, seed=0, high reg)
  - LGB-small-2 (leaves=64, lr=0.009, 10000 iters, seed=29, high reg)
  - CAT-1 (depth=7, lr=0.02, seed=7)
  - CAT-2 (depth=7, lr=0.03, seed=123)
- Ridge meta-learner on 5 OOF preds (positive weights, alpha=1.66)
- Add as 5 new candidates to hill climb

**Expected**: -0.3 to -0.5 OOF (from diversity + meta-learning)
**Effort**: ~2h (mostly waiting for training)

---

### TIER 2: DO TOMORROW (< 1d total)

#### [P4] Multi-Scale NCC Features
**New file**: `src/features/ncc_features.py`
**What**: Normxcorr1d between horizontal GR and typewell GR at half-windows 8/15/25
- Score-weighted ensemble with softmax temperature T=3
- Returns per-row TVT offsets from typewell correlation

**Expected**: -0.2 to -0.4 OOF
**Effort**: ~2h

#### [P5] Dense ANCC kNN Imputation
**What**:
- Subsample 60 (X,Y,ANCC) per well
- Build cKDTree over ALL train wells
- Predict ANCC at any (X,Y) + variance + neighbor distance
- Provides spatially smooth ANCC estimate

**Expected**: -0.1 to -0.3 OOF
**Effort**: ~2h

#### [P6] GR Feature Expansion
**Add to data_loader**:
- `gr_lag_{1,5,15,30}` — forward shifts
- `gr_lead_{1,5,15,30}` — backward shifts  
- `gr_diff_1`, `gr_diff_2` — discrete differences
- `gr_env` — rolling max (envelope)
- `gr_nrg` — sqrt(rolling mean of squares)

**Expected**: -0.1 to -0.2 OOF
**Effort**: ~30min

---

### TIER 3: BIG GUNS (Multi-day, High Impact)

#### [P7] 128-Seed PF Likelihood Ensemble
**What**: Replace our 16-seed with 128-seed PF ensemble at 4 scales {3,5,8,12}
- Need Numba JIT for performance (128 × 500 particles → ~4-5h)
- This is the foundation for the selector branch

**Expected**: -1.0 to -2.0 OOF
**Effort**: 4-5h compute + 2h coding

#### [P8] Heuristic Selector Branch
**What**: Implement per-well binned selector based on n_eval_rows and z_span
- 128-seed PF ensemble + 14-config beam ensemble
- 6 blending variants with different scale/hold/beam weights
- 0.3 Ridge + 0.7 Selector final blend

**Expected**: -1.0 to -1.5 OOF (on top of ridge stack)
**Effort**: 1d (depends on P7)
**Note**: This is the single biggest gap — the 0.7 heuristic branch

---

## Estimated Cumulative Impact

| Step | Cumulative OOF perwell | Cumulative Effort |
|------|-----------------------|-------------------|
| Current (R10 hill climb) | 9.182 | — |
| + P0-P3 (Today) | ~8.2 to 8.5 | ~4-5h |
| + P4-P6 (Tomorrow) | ~7.8 to 8.2 | ~1d total |
| + P7-P8 (Big guns) | ~6.5 to 7.5 | ~2-3d total |
| **Target** | **~7.5 (≈LB 8.5)** | **2-3 days** |

## What NOT to Do (Low ROI)

| Idea | Why Skip |
|------|----------|
| DTW Sakoe-Chiba features (LB 9.251) | Too complex for ~0.2 gain; P0-P3 cheaper |
| TCN / deep learning track | Our GBDT already outperforms; cross-attention concept only |
| cdeotte's typewell lookups | Already partially covered by our anchored offsets plan |
| More candidates with same features | Hill climb already maxing out on 43-feat diversity |

## Quick Wins — Fastest Path to 8.x

### 1h Sprint (if you have 1 hour only):
1. **Add post-processing** to submission pipeline (30 min)
2. **Validate on R10 OOF**: Apply exp ramp + SG, measure delta
3. If delta ≥ 0.3 → submit immediately to LB for calibration

### 4h Afternoon Sprint (if you have an afternoon):
1. Do P0 (30 min)
2. Do P3 — train 5 new model seeds in parallel (2h)
3. Do P1 — integrate formation features (1h)
4. Re-run hill climb with all new candidates (30 min)
5. Submit to LB

## Key Constants Reference (for quick copy-paste)

```python
# Post-processing defaults (from LB 7.776)
PP_TAU = 85          # exponential ramp time constant (ft)
PP_W_PF = 0.09       # weight on pf_ancc in blend
PP_ALPHA = 1.0       # scale factor
SG_WINDOW = 17       # Savitzky-Golay window
SG_POLYORDER = 3     # Savitzky-Golay polynomial order

# Ridge meta-learner
RIDGE_ALPHA = 1.6602834637650032
RIDGE_POSITIVE = True
RIDGE_FIT_INTERCEPT = True

# Anchored GR offsets
ANCH_OFFS = [-80, -40, -20, -10, -5, 0, 5, 10, 20, 40, 80]

# Selector thresholds
SELECTOR_N_EVAL_THRESHOLD = 4840
SELECTOR_Z_SPAN_THRESHOLDS = [136.73, 185.51]

# LGB big
LGB_BIG = dict(
    num_leaves=255, learning_rate=0.030, n_estimators=5000,
    reg_lambda=3.0, reg_alpha=0.05, subsample=0.8, subsample_freq=1,
    colsample_bytree=0.8, min_child_samples=15, max_bin=255, device='gpu'
)

# LGB small-deep
LGB_SMALL = dict(
    num_leaves=64, learning_rate=0.00934, n_estimators=10000,
    reg_lambda=95.75, reg_alpha=10.79, subsample=0.474,
    colsample_bytree=0.393, min_child_samples=40,
    min_child_weight=0.241
)

# CAT params
CAT_PARAMS = dict(
    depth=7, l2_leaf_reg=2.0, min_data_in_leaf=15, border_count=254,
    iterations=8000, od_wait=300
)
```
