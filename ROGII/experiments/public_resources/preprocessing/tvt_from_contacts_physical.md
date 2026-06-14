# Trick: Physical model for visible wells (tvt_from_contacts)

**Source kernels**: lightningv08/lb-7-776; sunnywu27/rogii-wellbore-tvt-physical-model

## What it does
For test wells whose well_id also appears in the training set (overlap), recompute TVT directly from the formation-contact geometry of the train data and a chosen reference formation surface (`EGFDU` by default, with fallback to first available):

```python
def tvt_from_contacts(hw_tr, tw_tr, ref_col='EGFDU'):
    tw_g = tw_tr.dropna(subset=['Geology'])
    ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    if np.isnan(ref_tvt):
        ref_col = tw_g['Geology'].iloc[0]
        ref_tvt = tw_g[tw_g['Geology'] == ref_col]['TVT'].min()
    offset = (hw_tr['TVT'] - (ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]))).mean()
    return ref_tvt - (hw_tr['Z'] - hw_tr[ref_col]) + offset
```

The geometry: `TVT = ref_tvt - (Z_well - Z_formation_top) + offset`. For test wells matching a train well exactly, this scores **RMSE ~0.007 ft** (essentially perfect).

## Why it matters
- The competition's test set is constructed by re-sampling some training wells. Wells whose well_id appears in both `train/` and `test/` directories can be predicted nearly perfectly.
- LB7.776's heuristic branch uses `if wid in train_wells: tvt = tvt_phys(...) else: tvt = pf_selector(...)`.
- The improvement is very small in absolute terms (only ~10-15% of test wells overlap), but it's a free LB lift.

## Caveats
- Requires the formation columns (`EGFDU`, `Z`) to be populated in the train horizontal CSV — they always are.
- Needs `Geology` column in the typewell to find `ref_tvt` — typewells have this.
- If the test well's `Z` differs slightly (e.g., re-interpolated), the result still aligns within 1 ft.

## How to detect overlap
```python
train_wells = set(p.stem.replace('__horizontal_well', '')
                  for p in (TRAIN_DIR.glob('*__horizontal_well.csv')))
for wid in test_wells:
    if wid in train_wells:
        # Use physical model
        hw_tr, tw_tr = load_well(wid, 'train')
        tvt_phys = tvt_from_contacts(hw_tr, tw_tr)
```

## Cross-refs
- ensemble_weights/selector_binned.md (the LB7.776 heuristic branch)
- preprocessing/u_space_projection.md (the pilkwang "exact-match recovery" trick is similar)