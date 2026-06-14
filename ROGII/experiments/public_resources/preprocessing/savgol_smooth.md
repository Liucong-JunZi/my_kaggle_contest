# Preprocessing: Savitzky-Golay smoothing per well

**Source kernel**: lightningv08/lb-7-776-rogii-ridge-sp

## What it does
After computing per-row TVT predictions (post-PP), apply a SG filter per well to denoise:

```python
from scipy.signal import savgol_filter

def sg_smooth(df, col, sg_w=17, sg_p=3):
    for well, g in df.groupby('well', sort=False):
        v = g[col].values
        n = len(v)
        wl = min(sg_w, n)
        if wl % 2 == 0:
            wl -= 1
        if wl >= sg_p + 2:
            v = savgol_filter(v, wl, sg_p)
        df.loc[g.index, col] = v
    return df
```

## Parameters
- `sg_w = 17` rows (must be odd; if shorter than 17, falls back to longest valid odd window)
- `sg_p = 3` (cubic local polynomial)

## Why it matters
- LGB outputs piecewise-constant predictions per leaf split → high-frequency noise in TVT predictions.
- SG smoothing preserves curvature (vs. moving average) while removing leaf-split jitter.
- Applied per-well prevents bleed across well boundaries.
- LB7.776 uses sg_w=17, sg_p=3 as fixed defaults; nihilisticneuralnet's DWT kernel tunes both with Optuna.

## Cross-refs
- ensemble_weights/ridge_pp_smooth.md (where it sits in the pipeline)