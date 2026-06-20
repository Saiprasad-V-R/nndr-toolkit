"""Observable alignment between a response plane and a scalar signal."""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .utils import corr

__all__ = ["best_plane_corr"]


def best_plane_corr(
    Q: np.ndarray, X: np.ndarray, m: np.ndarray
) -> Tuple[float, np.ndarray, np.ndarray]:
    r"""Plane-invariant correlation between a plane and a scalar observable.

    Computes ``max_{||a||=1} |corr(m(t), a^T (Q^T X(:,t)))|``: the largest
    absolute correlation between the scalar signal ``m`` and any direction
    inside the plane ``Q``.

    Parameters
    ----------
    Q:
        ``N x 2`` orthonormal plane.
    X:
        ``N x T`` state trajectory.
    m:
        Length-``T`` scalar observable (e.g. the mean field).

    Returns
    -------
    (c_best, a_best, r_best):
        The best absolute correlation, the optimal in-plane direction
        (2-vector), and that direction mapped to ambient space (``Q @ a_best``).
    """
    Z = Q.T @ X
    m = np.asarray(m).ravel()

    Zc = Z - Z.mean(axis=1, keepdims=True)
    mc = m - m.mean()

    g = Zc @ mc
    ng = np.linalg.norm(g)
    if ng < 1e-14:
        return 0.0, np.array([1.0, 0.0]), Q[:, 0]

    a = g / ng
    y = a @ Z
    c = abs(corr(m, y))
    r = Q @ a
    return float(c), a, r
