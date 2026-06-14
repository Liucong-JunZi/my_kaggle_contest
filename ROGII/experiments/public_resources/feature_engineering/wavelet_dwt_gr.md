# Feature: Wavelet (DWT) GR decomposition

**Source kernel**: mitchgansemer/gr-features-outlier-detection-rogii-wellbore

## What it does
Decompose the per-well horizontal GR signal via 5-level Discrete Wavelet Transform with Daubechies-4 (db4) wavelet, periodization mode:

```python
import pywt

def _compute_dwt(gr, wavelet='db4', n_levels=5):
    coeffs = pywt.wavedec(gr_imputed, 'db4', mode='periodization', level=5)
    # Reconstruct only the approximation (level 5) — captures low-freq trend
    approx5 = pywt.waverec([coeffs[0]] + [zeros]*5, 'db4', 'periodization')
    # Reconstruct only level-3 detail — captures mid-freq features
    detail3_squared = ...
    # Rolling-16 RMS of the squared detail = local energy
    detail_energy = sqrt(rolling_mean(detail3**2, 16))
    return approx5, detail_energy
```

## Output features
- `gr_dwt_approx5`: smoothed GR (low-pass via 5-level approximation) — replaces simple rolling mean
- `gr_dwt_detail_energy`: per-row local energy of mid-frequency detail (rolling RMS of level-3 details)
- `gr_dwt_residual`: GR_imputed − approx5 (high-frequency residual)

## Why it matters
- **Multiresolution decomposition** captures GR structure at scales the linear/SG smoothers miss.
- The detail-energy is a **local complexity indicator**: high in regions with sharp formation boundaries, low in monotonic intervals.
- Gives the GBDT a richer GR description than just rolling mean+std at fixed windows.

## Score-relevant constants
| name | value |
|------|-------|
| Wavelet | db4 (Daubechies-4) |
| Mode | periodization |
| n_levels | 5 |
| Detail level for energy | 3 |
| Rolling window for RMS | 16 |

## Cross-refs
- feature_engineering/gr_fft_features.md (alternative spectral)
- feature_engineering/dtw_sakoe_chiba.md (different wavelet/DTW kernel)