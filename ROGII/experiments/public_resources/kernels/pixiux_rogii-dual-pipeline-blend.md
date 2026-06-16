# Kernel: pixiux/rogii-dual-pipeline-blend (201 votes)

**Author**: pixiux
**Last run**: ~2026-05
**Total votes**: 201
**Files**: rogii-dual-pipeline-blend.ipynb

## Architecture (one-paragraph)
**Two independent pipelines blended at 0.55 / 0.45**:
- **Pipeline A "ridge-sp45"**: 128-seed × 4-scale PF + 7-config beam + plane-KNN/dense-ANCC spatial priors → LightGBM/CatBoost stack → Ridge meta → warm-up-damped PP → robust deg-4 IRLS polynomial projection on `tvt+Z` (U-space projection) → SG smoothing.
- **Pipeline B "fleongg"**: likelihood-weighted multi-scale PF + offset-well spatial priors (KNN) → GBM stack (GroupKFold by well).

Final stage: a **guarded physical override** for overlap wells (well_ids in both train and test). The override applies `tvt_from_contacts` ONLY when verified at runtime against the test well's known prefix (RMSE < 1 ft, ≥50 comparable rows, MD-aligned interpolation). By construction never worse than the blend on misaligned wells.

## Key Techniques

### Feature Engineering
- All LB7.776 base features (PF + beams + NCC + plane-KNN formations + dense ANCC + GR rolls + slopes + anchored offsets)

### Model & Hyperparams
- 3× LGB + 2× CB stacked via Ridge (Pipeline A) and 3× LGB pre-trained boosters (Pipeline B)
- All per LB7.776 params

### Particle Filter / Beam Search
- 128-seed × 500-particle PF at 4 likelihood scales (3, 5, 8, 12)
- 7-config beam search (cons/loose/vcons/sm5/vloose/mid/stiff)

### Ensemble / Blending
- **Final 2-pipeline blend**: 0.55 (A) + 0.45 (B). The author argues two independently-built pipelines have decorrelated errors → "free accuracy".
- Followed by guarded physical override.

### Postprocessing
- Robust deg-4 IRLS polynomial projection on `tvt+Z` (U-space) with `−0.09 RMSE` lift vs plain SG smoothing (CV-validated).
- Warm-up damping (PP exp-ramp).
- SG smoothing.

### CV Methodology
- GroupKFold(5) by well_id.
- Target: relative `TVT - last_known_TVT`.

## Anything novel vs LB-7.776 kernel?
**YES — three notable additions**:
1. **Two-pipeline blend** (0.55/0.45) — duplicates the entire feature pipeline using two independent implementations (ridge-sp45 + fleongg) and averages them. The author claims this is "the single biggest cheap win on this LB".
2. **Robust deg-4 IRLS U-space projection** — robust polynomial fit instead of SG smoothing, with documented CV improvement of −0.09 RMSE.
3. **Guarded physical override** — per-well runtime verification of the physical reconstruction before applying it; provably safe.

## Score-relevant constants
| name | value |
|------|-------|
| Pipeline blend | 0.55 (A) + 0.45 (B) |
| U-projection degree | 4 |
| U-projection method | IRLS (Iteratively Reweighted Least Squares) |
| Override prefix RMSE threshold | 1.0 ft |
| Override min rows | 50 |

## Cross-refs
- preprocessing/u_space_projection.md (the IRLS variant)
- preprocessing/guarded_physical_override.md
- ensemble_weights/ridge_pp_smooth.md (Pipeline A's stack)