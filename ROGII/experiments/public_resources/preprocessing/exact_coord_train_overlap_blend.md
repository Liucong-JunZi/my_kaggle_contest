# Trick: Exact-coordinate train/test overlap blend

**Source kernel**: thbdh5765/rogii-v10-fresh-artifact-infer

## What it does
For each test row whose (X, Y, Z) **rounded to 2 decimal places** matches a train row's (X, Y, Z), pull the train row's TVT and blend it into the test prediction at weight 0.28:

```python
def apply_exact_train_coordinate_blend(sub, data_dir, blend_weight=0.28):
    # Build train coordinate -> TVT map
    train_all = concat(train CSVs with TVT.notna())
    for col in ['X', 'Y', 'Z']:
        train_all[col + '_r'] = train_all[col].round(2)
    train_map = train_all.drop_duplicates(['X_r','Y_r','Z_r']).set_index(['X_r','Y_r','Z_r'])['TVT'].to_dict()

    # Find test rows with matching coordinate
    test_coords = test[mask].apply(lambda r: (round(r.X,2), round(r.Y,2), round(r.Z,2)))
    exact_tvt = test_coords.map(train_map)

    # Blend at 0.28
    sub.loc[exact, 'tvt'] = 0.72 * sub.loc[exact, 'tvt'] + 0.28 * exact_tvt
```

## Why it matters
- The competition's train and test wells share some physical (X,Y,Z) trajectory points — likely the test set is constructed by re-sampling some training wells.
- A 2-decimal-place coordinate match captures rows that are within ~0.01 ft of a train row.
- The blended TVT is essentially noise-free for the matched rows.
- 0.28 weight is conservative — keeps the model output as the main signal but pulls toward known-truth where available.

## Caveat: distinct from `tvt_from_contacts`
- `tvt_from_contacts` uses formation-contact geometry to recompute TVT for whole wells whose well_id matches a train well.
- This trick uses **per-row coordinate matching** without requiring well_id match — it can catch rows from test wells whose IDs differ but happen to lie along trajectory points shared with train wells.

## Score-relevant constants
| name | value |
|------|-------|
| Coordinate rounding (X,Y,Z) | 2 decimal places |
| Default blend weight | 0.28 |
| Blend formula | `(1-w)*model + w*exact_tvt` |

## Cross-refs
- preprocessing/tvt_from_contacts_physical.md (well-id matching variant)