# Feature: GR linear detrend residual

**Source kernel**: romantamrazov/rogii-super-solution-lb-top-3

## What it does
Fit a linear trend `GR ~ a * MD + b` across the **entire** horizontal well (known + eval), then compute the residual `residual = GR - (a*MD + b)`. This removes the monotonic GR drift caused by the well gradually cutting through shallower/deeper strata, leaving only the local geological variability.

```python
def gr_detrend_resid(gr_arr, md_arr):
    m = np.isfinite(gr_arr)
    if m.sum() < 5:
        return np.zeros_like(gr_arr)
    a, b = np.polyfit(md_arr[m], gr_arr[m], 1)
    return gr_arr - (a * md_arr + b)
```

## Why it matters
GR in horizontal wells often drifts monotonically as the well moves structurally up or down through the stratigraphy. This trend is a well-specific nuisance and dominates the raw GR feature — the detrended residual captures the **local geological features** (formation boundaries within the lateral). The LB7.776 kernels don't use it explicitly (they rely on anchored offsets to capture the same structure indirectly), but romantamrazov's Super Solution ranks it.

## Cross-refs
- feature_engineering/anchored_gr_offsets.md (alternative way to handle the same detrinding)