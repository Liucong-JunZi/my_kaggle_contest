# Feature: Plane-KNN formation imputer

**Source kernel**: [lightningv08/lb-7-776-rogii-ridge-sp](../kernels/lightningv08_lb-7-776-rogii-ridge-sp.md)

## What it does
For each train well, compute per-formation median surface depths at the well's median (X,Y):
```
row = {wid, x_median, y_median, ANCC_median, ASTNU_median, ..., BUDA_median}
```
Build a cKDTree over normalized (X,Y). For a query (X,Y):
1. Find k=10 nearest wells (excluding self if training).
2. Solve a weighted least-squares plane fit `F(X,Y) = a*X + b*Y + c` for each formation, using inverse-distance weights `1/(d+1e-3)`.
3. Predict each formation surface at the query (X,Y) plus the query-to-nearest-neighbor distance `knn_dist`.

## Code skeleton
```python
class FormationPlaneKNN:
    def __init__(self, well_ids, data_dir):
        rows = []
        for wid in well_ids:
            df = pd.read_csv(...).dropna()
            row = {'wid': wid, 'x': df['X'].median(), 'y': df['Y'].median()}
            for c in FORMATIONS: row[f'{c}_m'] = df[c].median()
            rows.append(row)
        self.df = pd.DataFrame(rows)
        xy = self.df[['x','y']].to_numpy()
        self.scale = np.where(xy.std(0) < 1e-3, 1., xy.std(0))
        self.tree = cKDTree(xy / self.scale)

    def impute(self, xy_q, self_wid=None, k=10):
        # nearest-neighbor; build 3x3 weighted-normal matrices for the plane fit
        # solve A @ coef = rhs ; predict F(X_q,Y_q) = X*a + Y*b + c
```

## Why it matters
This is the **dominant spatial signal** for formation columns at test time. ROGII test wells don't have ANCC/ASTNU/etc populated — the imputer reconstructs them from neighbors. Combined with `seg_b_well`, gives ~6 strong plane-fit TVT signals.

## Cross-refs
- `seg_b_well.md` — segment bias on top of the imputed formations
- `dense_ancc_imputer.md` — denser version for ANCC only