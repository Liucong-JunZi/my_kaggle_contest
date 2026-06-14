"""
Candidate stub: GR FFT (dominant frequency) features

Source: sanidhyavijay24/9-946-rogii-geostat-softmax-ncc-hybrid
Stage: STUB ONLY — parent agent should review before integrating.

Cheap 2-feature-per-well global covariate.
"""
import numpy as np


def gr_fft_features(gr_post):
    """Returns (dom_freq_normalized, log_power) of the dominant FFT component."""
    valid = gr_post[~np.isnan(gr_post)]
    if len(valid) < 32:
        return 0.0, 0.0
    centered = valid - valid.mean()
    spec = np.abs(np.fft.rfft(centered)) ** 2
    if len(spec) < 3:
        return 0.0, 0.0
    dom = int(np.argmax(spec[1:])) + 1
    return float(dom / len(valid)), float(np.log1p(spec[dom]))


def add_gr_fft_features(hw_df, eval_idx):
    """Broadcast (gr_fft_dom, gr_fft_logpwr) to per-row features.

    Use the post-anchor (eval) GR signal — captures lateral-only periodicity.
    """
    gr_post = hw_df['GR'].values[eval_idx[0]:]
    dom, lp = gr_fft_features(gr_post)
    n = len(eval_idx)
    return {
        'gr_fft_dom':    np.full(n, dom, np.float32),
        'gr_fft_logpwr': np.full(n, lp,  np.float32),
    }
