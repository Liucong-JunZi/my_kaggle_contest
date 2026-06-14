"""
Candidate stub: Formation Plane-KNN imputer

Source: lightningv08/lb-7-776-rogii-ridge-sp
Stage: STUB ONLY — parent agent should review before integrating.

Computes plane-fitted per-formation surface depths at any (X,Y) from
nearest neighbor wells. Required for the per-formation TVT + seg_b_well pipeline.

This class must be fit ONCE per CV fold or just globally (test-safe since
it only uses per-well median X/Y/F, not per-row label data).
"""
import numpy as np
from scipy.spatial import cKDTree


FORMATIONS = ['ANCC', 'ASTNU', 'ASTNL', 'EGFDU', 'EGFDL', 'BUDA']


class FormationPlaneKNN:
    """kNN plane-fit formation surface imputer.

    For each train well, stores median (X, Y, F1..F6). At query time,
    finds k nearest wells and fits a plane via 3x3 weighted normal equations.

    train_well_ids: list of well-id strings (from train directory glob).
    data_dir: Path to 'train/' directory.
    k: number of neighbors for the plane fit.
    """

    def __init__(self, train_well_ids, data_dir, k=10):
        # Build rows from per-well median X/Y + each formation.
        # For each well, read the horizontal CSV, dropna on X/Y/F columns,
        # store mp = median(X), median(Y), median(F1)... median(F6).
        # Build cKDTree over normalized (X,Y).
        # Implementation: see kernels_raw/lightningv08_.../lb-7-776-... .code.txt lines 679-719
        raise NotImplementedError("Copy from LB7.776 kernel")
        self.k = k

    def impute(self, xy_q, self_wid=None):
        """Predict all 6 formation surfaces at query (X,Y) points.

        Args:
            xy_q: (N, 2) array of query points.
            self_wid: string or None. If set, exclude that well from KNN.

        Returns:
            pred: (N, 6) float32 formation depths [ANCC..BUDA].
            min_dist: (N,) float32 distance to nearest neighbor.
        """
        raise NotImplementedError("Copy from LB7.776 kernel")