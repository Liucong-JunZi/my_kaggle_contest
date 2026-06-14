# Kernel: kojimar/rogii-inference-stack-with-pf-beam-and-tabicl

**Author**: kojimar
**Last run**: ~2026-05-19
**Total votes**: ~78
**Files**: rogii-inference-stack-with-pf-beam-and-tabicl.ipynb

## Architecture (one-paragraph)
**Inference-only artifact-based stack** that loads pre-trained LightGBM + CatBoost + TabICL models from `thbdh5765/rogii-v10-fresh-artifacts` (and similar v11/v50). The stack uses a saved positive-Ridge stacker where TabICL gets ~71% of the weight, with small contributions from CB and a single TabICL_B context. Builds the same 172-feature matrix (PF-ANCC + PF-Z + 7 beams + multi-scale NCC + plane-KNN + dense ANCC + anchored offsets + slopes + GR rolls) and applies the saved postprocess (alpha, tau, w_pf) and SG smoothing.

## Key Techniques

### Feature Engineering
- 172 features identical to the LB7.776 family (PF, beams, NCC, formations, anchored offsets, GR rolls, slopes)

### Model & Hyperparams
- **3× LightGBM** (seeds 7/42/123) loaded from `lgb/seed{S}_fold{F}.txt`
- **3× CatBoost** (seeds 7/42/123) loaded from `catboost/seed{S}_fold{F}.cbm`
- **2× TabICL** (variants A/B with different contexts) loaded from `tabicl_contexts/*.npz`
- Stacker: positive Ridge with hill-climbing initialization, alpha=1.0, fit_intercept=False
  - Weights: `[lgb123: 0, lgb42: 0, lgb7: 0, cb42: 0, cb7: 0.042, cb123: 0.094, tabicl_A: 0.712, tabicl_B: 0.038]`

### Particle Filter / Beam Search
- PF-ANCC + PF-Z (same JIT functions) for feature generation
- 7 beam configs for feature generation

### Ensemble / Blending
- Linear stacker (positive Ridge with hill-climbing-initialized weights from artifact)
- TabICL_A is dominant (71%) — strongly suggests TabICL is producing predictions orthogonal to the GBDT family
- Postprocess: PP blend with raw PF-ANCC + exp ramp + SG-smooth

### CV Methodology
- Inference-only — no CV required at runtime (uses pre-saved fold-averaged models)

## Anything novel vs LB-7.776 kernel?
**YES — TabICL Regressor as a major stack component**. The artifact's stacker weights show TabICL_A dominates with 71% — this is the largest novelty over the LB7.776 ridge-only stack. TabICL provides a fundamentally different learning paradigm (in-context learning via prior-fitted network) that complements GBDT.

## Score-relevant constants
| name | value |
|------|-------|
| TabICL n_estimators per context | 4 |
| TabICL chunk size | 50,000 |
| Stacker alpha | 1.0 |
| TabICL_A weight in stack | 0.7117 |
| Total feature subset for TabICL | ~50-80 |

## Cross-refs
- feature_engineering/tabicl_regressor.md
- model_params/tabicl_params.json