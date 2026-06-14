"""
Candidate stub: Wavelet (DWT) GR features

Source: mitchgansemer/gr-features-outlier-detection-rogii-wellbore
Stage: STUB ONLY — parent agent should review before integrating.

Adds gr_dwt_approx5 (low-freq smoothed), gr_dwt_detail_energy (mid-freq RMS),
gr_dwt_residual (high-freq).

Requires `pywt` package.
"""
import numpy as np

try:
    import pywt
except ImportError:
    pywt = None


def compute_gr_dwt(gr, wavelet='db4', n_levels=5, detail_level=3, energy_window=16):
    """Decompose GR via DWT and return (approx, detail_energy, residual)."""
    if pywt is None:
        raise ImportError("pywt required for DWT features (pip install PyWavelets)")
    n = len(gr)
    gr_clean = np.where(np.isfinite(gr), gr,
                        np.nanmean(gr) if np.isfinite(gr).any() else 0.0)
    try:
        coeffs = pywt.wavedec(gr_clean, wavelet, mode='periodization', level=n_levels)
        # Approximation only (lowest freq)
        recon_approx = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
        approx = pywt.waverec(recon_approx, wavelet, mode='periodization')[:n]
        # Detail at requested level
        recon_det = [np.zeros_like(c) for c in coeffs]
        recon_det[detail_level] = coeffs[detail_level]
        det_signal = pywt.waverec(recon_det, wavelet, mode='periodization')[:n]
        # Rolling RMS of detail
        import pandas as pd
        det_energy = (pd.Series(det_signal**2).rolling(energy_window, center=True, min_periods=1).mean().values) ** 0.5
    except Exception:
        approx, det_energy = gr_clean, np.zeros(n)

    return (approx.astype(np.float32),
            det_energy.astype(np.float32),
            (gr_clean - approx).astype(np.float32))


def add_gr_dwt_features(hw_df, eval_idx, gr_imputed_col=None):
    """Per-row DWT features from the FULL well GR; sliced to eval rows."""
    if gr_imputed_col is None:
        gr = hw_df['GR'].astype(float).interpolate(limit_direction='both').values
    else:
        gr = hw_df[gr_imputed_col].values
    approx, energy, resid = compute_gr_dwt(gr)
    return {
        'gr_dwt_approx5':       approx[eval_idx],
        'gr_dwt_detail_energy': energy[eval_idx],
        'gr_dwt_residual':      resid[eval_idx],
    }
