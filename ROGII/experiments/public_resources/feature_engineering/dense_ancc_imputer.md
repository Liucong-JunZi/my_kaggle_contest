# Feature: Dense ANCC kNN imputer

**Source kernel**: [lightningv08/lb-7-776-rogii-ridge-sp](../kernels/lightningv08_lb-7-776-rogii-ridge-sp.md)

## What it does
Unlike `FormationPlaneKNN` which uses one (X,Y,F) per well, this collects **DENSE_SPW=60 evenly-spaced (X,Y,ANCC) points per well**, builds a cKDTree over the union, and for any query (X,Y):
- Fetches nearest k=20 neighbors (out of nfetch=5000 candidates)
- Inverse-distance weighted mean and variance of ANCC
- Returns (`d_ancc_pred`, `d_ancc_std`, `nearest_distance`)

## Why it matters
Captures within-well ANCC structure: a horizontal well drilled along the same formation can produce 60 spatial samples of ANCC, providing much finer-grained spatial structure than a single per-well median. Combined with `b_full = median(ktvt + kz - d_kn)` from the known prefix, gives `tvt_dense = -z + d_ancc_pred + b_d` — a separate strong TVT signal beyond the plane-KNN result.

## Code skeleton
```python
class DenseANCCImputer:
    def __init__(self, well_ids, data_dir, spw=60):
        for wid in well_ids:
            df = pd.read_csv(...).dropna()
            ix = np.linspace(0, len(df)-1, min(spw, len(df)), dtype=int)
            s = df.iloc[ix]
            xs.append(s['X'].values)
            ys.append(s['Y'].values)
            anccs.append(s['ANCC'].values)
        self.tree = cKDTree(np.column_stack([xs, ys]) / scale)

    def impute(self, xy_q, k=20, nfetch=5000):
        # IDW over k nearest from nfetch candidates (mask self-well)
        # return mean, std, dist
```

## Score-relevant constants
| name | value |
|------|-------|
| spw (samples per well) | 60 |
| k (neighbors) | 20 |
| nfetch (initial candidates) | 5000 |

## Cross-refs
- `plane_knn_formation.md` — coarser per-well plane variant