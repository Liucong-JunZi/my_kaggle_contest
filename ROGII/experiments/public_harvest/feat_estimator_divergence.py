"""
Candidate stub: estimator-divergence features (v4 features from mitchgansemer)

Source: mitchgansemer/gr-features-outlier-detection-rogii-wellbore
Stage: STUB ONLY — parent agent should review before integrating.

Cheap, hand-crafted: 11 features per row from existing base estimators.
"""
import numpy as np


def add_estimator_divergence_features(form_drift, ncc_drift, beam_drift,
                                      pf_drift, extrap_drift,
                                      extrap_k50_drift=None):
    """Compute pairwise + aggregate divergence among 5 base estimators.

    All inputs are (n_eval,) np.float32 arrays of TVT_estimate - last_known_TVT.
    Returns dict of new feature arrays.
    """
    out = {}
    out['form_vs_ncc']  = (form_drift - ncc_drift).astype(np.float32)
    out['form_vs_pf']   = (form_drift - pf_drift).astype(np.float32)
    out['form_vs_beam'] = (form_drift - beam_drift).astype(np.float32)
    out['ncc_vs_pf']    = (ncc_drift  - pf_drift).astype(np.float32)
    out['ncc_vs_beam']  = (ncc_drift  - beam_drift).astype(np.float32)
    out['beam_vs_pf']   = (beam_drift - pf_drift).astype(np.float32)

    drifts = np.stack([form_drift, ncc_drift, beam_drift, pf_drift, extrap_drift], axis=1)
    out['estimator_drift_range'] = (drifts.max(axis=1) - drifts.min(axis=1)).astype(np.float32)
    out['estimator_drift_max']   = drifts.max(axis=1).astype(np.float32)
    out['estimator_drift_min']   = drifts.min(axis=1).astype(np.float32)

    if extrap_k50_drift is not None:
        out['extrap_k50_vs_extrap200'] = (extrap_k50_drift - extrap_drift).astype(np.float32)
        out['form_vs_extrap']          = (form_drift - extrap_drift).astype(np.float32)
    return out
