# Kernel: pilkwang/rogii-target-free-tvt-geosteering

**Author**: pilkwang
**Last run**: ~2026-05
**Total votes**: 165
**Files**: rogii-target-free-tvt-geosteering.ipynb

## Architecture (one-paragraph)
A target-free geosteering ensemble with two/three configurable layers:
1. Ridge artifact branch: `T_ridge = w_r * T_artifact + (1−w_r) * T_heuristic_PF` where the heuristic is a 128-seed × 500-particle PF ensemble with init spread σ_0=4.5 ft, optionally projected through a per-well **U=TVT+Z anchor-relative polynomial smoothing** (degree d=4, β=0.75 blend strength).
2. Pretrained LGBM branch (artifact-loaded models).
3. Optional gated model-package correction with max weight 0.010 (very conservative regularizer).
The final prediction is `λ * projected_ridge_pf + (1−λ) * pretrained_LGBM` with λ=0.55.

## Key Techniques

### Feature Engineering
Inherits LB7.776 family. Adds U-space anchor-relative projection (see `preprocessing/u_space_projection.md`).

### Model & Hyperparams
- Pretrained LGBM models loaded from `/kaggle/input/datasets/fleongg/rogii-claude-models-pub`
- Ridge weight `w_r = 0.30` (artifact)
- PF: N_particles=500, N_seeds=128, init spread=4.5 ft
- U-space projection: degree=4, robust C=2.0, robust iters=4, blend β=0.75
- Final blend: λ=0.55 (projected ridge/PF) + (1−λ)=0.45 (pretrained LGBM)
- Gated model-package correction max weight=0.010, scale=5.7

### Particle Filter / Beam Search
- Standard 128-seed PF ensemble; initial TVT spread σ_0 = 4.5 ft (the "sp45" patch)

### Ensemble / Blending
- Three optional layers: ridge_pf, pretrained_lgbm, model_package_gated
- Multiple "submission profiles" controllable via SUBMISSION_PROFILE: blend, parameter_experiment, reference, pf_selector_only

### CV Methodology
- Inherits the GroupKFold protocol (it's an inference-time blender)
- Includes "guarded overlap override" — for test wells matching a train well exactly (TVT prefix RMSE < 0.02, GR MAD < 0.5, Z MAD < 0.02), substitute the train TVT directly. Disabled by default but a useful safety net.

## Anything novel vs LB-7.776 kernel?
**YES — U-space anchor-relative projection** (see preprocessing/u_space_projection.md). Per-well robust low-degree polynomial fit of `U=TVT+Z−A_w` vs. normalized MD, used as a denoising layer with β=0.75 blend strength. Distinct from per-well SG smoothing — projects globally rather than locally.

Also: extensive submission-profile control logic, exact-match recovery (disabled but documented), gated model-package correction.

## Score-relevant constants
| name | value |
|------|-------|
| Ridge weight w_r | 0.30 |
| PF init spread σ_0 | 4.5 |
| PF n_particles | 500 |
| PF n_seeds | 128 |
| U-projection degree d | 4 |
| U-projection robust iters | 4 |
| U-projection robust C | 2.0 |
| U-projection blend β | 0.75 |
| Final blend λ (projected vs LGBM) | 0.55 |
| Gated correction max weight | 0.010 |
| Gated correction scale | 5.7 |
| Exact-match recovery TVT-RMSE limit | 0.02 |

## Cross-refs
- preprocessing/u_space_projection.md
- ensemble_weights/ridge_pp_smooth.md (parent ridge stack)