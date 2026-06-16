# Target convention: PF residual learning (vs last_known residual)

**Source kernel**: ravi20076/rogii2026-public-blend-v2 (LB ~9-10, 46 votes)

## What it does
Instead of the canonical relative target `target = TVT - last_known_TVT` (LB 7.776 convention), train the GBDT on the **PF-corrected residual**:
```python
target = TVT - pf_pred
```
where `pf_pred` is the per-row PF-ANCC ensemble prediction (or PF-Z, etc.).

## Why it might matter
- Compared to `last_known`, `pf_pred` is a much sharper, per-row baseline that already absorbs most of the geometric signal.
- The GBDT only needs to learn **PF-residual corrections** — a much smaller dynamic range than `TVT - last_known`.
- Per the kernel's notes: OOF RMSE on PF-corrected predictions is **10.56 ft** with a 5-fold GBM ensemble (LGB 40% + XGB 40% + CB 20%).
- Inference: `tvt_predicted = pf_pred + ml_correction(features)`.

## Trade-off vs LB7.776 convention
- **LB7.776 (last_known residual)**: simpler baseline, larger target dynamic range, more model burden. Final blend at 0.3 ridge + 0.7 selector lifts to LB 7.776.
- **PF residual**: tighter baseline, smaller target dynamic range. But the GBDT loses access to learning `pf_vs_lastknown_diff` as a feature (it's already in the baseline). Reported OOF 10.56 ft is competitive but not state-of-the-art.

## Pipeline
```python
# Training
df_train['pf_pred'] = run_pf_ancc(...)        # per-row PF prediction
df_train['target']  = df_train['TVT'] - df_train['pf_pred']
gbm.fit(features, target)

# Inference
df_test['pf_pred']        = run_pf_ancc(...)
df_test['ml_correction']  = gbm.predict(features)
df_test['tvt_predicted']  = df_test['pf_pred'] + df_test['ml_correction']
```

## Where this is useful
- When your PF is already very strong (LB ~8-9 from PF alone) and you only want a small correction.
- When you're combining with public PF artifacts and want the GBM to focus on residual error.

## Cross-refs
- ensemble_weights/ridge_pp_smooth.md (the LB7.776 convention)
- cv_methodology/groupkfold_well.md