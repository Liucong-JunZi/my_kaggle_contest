# CV: GroupKFold by well_id (canonical ROGII protocol)

**Source**: All public ridge-stack kernels (lightningv08/lb-7-776, ravaghi/ridge, romantamrazov/super-solution, etc.)

## What it is
The mandatory CV protocol for ROGII:
```python
from sklearn.model_selection import GroupKFold
gkf = GroupKFold(n_splits=5)
for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups=well_ids)):
    ...
```

## Why
- Within-well rows are highly correlated (consecutive 1-ft samples along a single trajectory).
- Random row splits would leak across the well boundary, making CV optimistic by ~3-5 ft.
- All public kernels use `n_splits=5` on the 723-train-well corpus. Fold sizes ~145 wells each.

## Target convention (CRITICAL)
**Always use the relative target**: `y = TVT - last_known_TVT`. Inference adds `last_known_TVT` back.

Why:
- Removes the inter-well TVT scale (some wells are at 8000 ft, some at 12000 ft).
- The model only learns the **drift** from the anchor — much smaller dynamic range, much better generalization.
- LB7.776 derives nearly all its lift from this convention.

## Implementation note
Python:
```python
oof = np.zeros(len(train_df), np.float32)
test_preds = np.zeros(len(test_df), np.float32)
for fold, (tr, va) in enumerate(gkf.split(X, y, train_df['well_id'])):
    model.fit(X.iloc[tr], y.iloc[tr], eval_set=[(X.iloc[va], y.iloc[va])])
    oof[va] = model.predict(X.iloc[va])
    test_preds += model.predict(X_test) / N_SPLITS
```

## Cross-refs
- All kernels and ensemble docs assume this protocol.