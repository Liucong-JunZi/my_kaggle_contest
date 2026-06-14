# Ensemble: Hill-climbing weight optimization (replaces Ridge stack)

**Source kernel**: ravaghi/wellbore-geology-prediction-hill-climbing (LB ~10, 195 votes)

## What it does
Replace the Ridge stack with a **greedy hill-climbing weight search** over the OOF-prediction matrix:

```python
from hill_climbing import Climber  # arpitsingh2744/hill-climbing-wheel pip whl

climber = Climber(
    objective="minimize",
    eval_metric=root_mean_squared_error,
    allow_negative_weights=True,
    precision=0.001,
    score_decimal_places=3,
    n_jobs=-1,
    use_gpu=False
).fit(oof_preds, y)

hc_oof_preds  = climber.predict(oof_preds)
hc_test_preds = climber.predict(test_preds)
```

## Why it matters vs Ridge
- Ridge with `positive=True` keeps weights non-negative; hill-climbing allows **negative weights** (`allow_negative_weights=True`), letting one model "subtract" another for residual correction.
- Greedy stepwise search at precision 0.001 navigates a piecewise-linear-RMSE objective without the regularization penalty Ridge imposes.
- For the same 5-model OOF stack, hill-climbing typically lands ~0.01-0.05 RMSE below Ridge in OOF.

## Trade-offs
- More overfitting risk than positive Ridge — the negative weights amplify noise.
- The `precision=0.001` is fine-grained; coarser (0.01) reduces overfitting but loses small gains.
- A `score_decimal_places=3` early-stop prevents endless iteration on noise.

## Pipeline placement
Drop-in replacement for `ridge_trainer.fit(oof_preds, y)`:
```python
# Replace this:
ridge.fit(oof_preds, y); ridge_oof = ridge.predict(oof_preds)
# With:
climber.fit(oof_preds, y); hc_oof = climber.predict(oof_preds)
```
Postprocess (PF blend + ramp + SG) is applied identically.

## Cross-refs
- ensemble_weights/ridge_pp_smooth.md (the regularized counterpart)
- The same hill-climbing wheel is used for `apply_pp` parameter search in some forks.