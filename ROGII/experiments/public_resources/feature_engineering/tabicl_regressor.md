# Model: TabICL (Tabular In-Context Learning Regressor)

**Source kernels**: kojimar/rogii-inference-stack-with-pf-beam-and-tabicl;
nina2025/rogii-h-blend-v1; thbdh5765/rogii-v10-fresh-artifacts

**Source data**: `needless090/rogii-tabicl-mirror` (101.7 MB; tabicl wheel + checkpoint)

## What it is
**TabICL** is a Prior-Data Fitted Network (PFN) for tabular regression — like TabPFN but designed for arbitrary tabular tasks. Pre-trained transformer with **in-context learning**: at inference time, you provide the training set as "context" examples, and the network predicts the test set in a single forward pass — no per-task gradient updates needed.

```python
from tabicl import TabICLRegressor

reg = TabICLRegressor(
    model_path='tabicl-regressor-*.ckpt',
    device='cuda',
    random_state=seed,
    n_estimators=4,           # ensemble of 4 random sub-context picks
    n_jobs=1,
    verbose=False,
    use_amp='auto',
    batch_size=4,
)

# X_ctx is the saved per-fold training context (a sub-sample of OOF train rows)
# y_ctx is the corresponding target
reg.fit(X_ctx, y_ctx)         # no gradient steps; just stores context
preds = reg.predict(X_test)   # one forward pass per chunk; chunked at 50k
```

## In LB stacks
Per `thbdh5765/rogii-v10-fresh-artifacts/inference_config.json`, the linear stacker weights are:
```
['lgb123', 'lgb42', 'lgb7', 'cb42', 'cb7', 'cb123', 'tabicl_A', 'tabicl_B']
[0.0, 0.0, 0.0, 0.0, 0.0423, 0.0935, 0.7117, 0.0381]
```
**TabICL_A gets 71.2% of the weight** when stacked with hill-climbing and positive Ridge. This is the dominant model in this stack — implying it produces strongly orthogonal predictions to LightGBM and CatBoost.

## Notable practical bits
- **Context selection**: each fold uses a saved `X_ctx, y_ctx` (numpy `.npz`) — typically a sub-sampling of the train OOF rows. Different folds have different contexts.
- **Burn-in**: optional warm-up forward pass on extra rows (not used for prediction) to better-calibrate the network.
- **n_estimators=4**: ensemble across 4 different randomized sub-contexts (similar to TabPFN's bagging).
- **Feature subset**: TabICL uses a **subset** of the full 172 features (~50-80, listed in `tabicl_features.json`), not the full set — TabICL has a feature-count cap.

## Why it works
TabICL/TabPFN are trained on synthetic tabular tasks across diverse priors. The network has learned to do Bayesian regression on small-to-medium tabular data without explicit task-specific training. For ROGII this means:
- No per-fold model fitting required (just one forward pass at inference).
- Predictions are well-calibrated by the prior network.
- Acts as a strong "neural baseline" complementary to GBDT-style models.

## Constraints
- Requires GPU for reasonable throughput (CPU works but slow).
- `n_estimators × |context|^2 × hidden_dim` GPU memory is the bottleneck.
- Chunked prediction at `chunk = 50_000` rows.

## Cross-refs
- model_params/tabicl_params.json
- ensemble_weights/ridge_pp_smooth.md (TabICL plugs into the same stack)
- datasets/needless090_rogii-tabicl-mirror.md (the wheel + checkpoint dataset)