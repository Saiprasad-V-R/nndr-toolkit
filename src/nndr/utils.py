"""Low-level linear-algebra helpers used across the package."""
from __future__ import annotations

import numpy as np

__all__ = [
    "random_orthogonal",
    "corr",
    "spectral_radius",
    "principal_angle_deg",
]


def random_orthogonal(N: int, rng: np.random.Generator) -> np.ndarray:
    """Return an ``N x N`` orthogonal matrix drawn from the Haar measure.

    Built via the QR decomposition of a Gaussian matrix, with a sign fix so
    the result has determinant ``+1``.
    """
    Q, _ = np.linalg.qr(rng.standard_normal((N, N)))
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1
    return Q


def corr(a, b) -> float:
    """Pearson correlation coefficient between two 1-D arrays."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = (a - a.mean()) / (a.std() + 1e-12)
    b = (b - b.mean()) / (b.std() + 1e-12)
    return float(np.mean(a * b))


def spectral_radius(A: np.ndarray) -> float:
    """Largest absolute eigenvalue of ``A`` (its spectral radius)."""
    return float(np.max(np.abs(np.linalg.eigvals(A))))


def principal_angle_deg(Q1: np.ndarray, Q2: np.ndarray) -> float:
    """Largest principal angle (in degrees) between ``span(Q1)`` and ``span(Q2)``.

    Parameters
    ----------
    Q1, Q2:
        Arrays of shape ``(N, 2)`` with orthonormal columns.

    Returns
    -------
    float
        The largest principal angle between the two planes, in degrees. A
        value near zero means the two planes nearly coincide.
    """
    M = Q1.T @ Q2
    s = np.linalg.svd(M, compute_uv=False)
    s = np.clip(s, 0.0, 1.0)
    theta = float(np.arccos(s[-1]))
    return float(np.degrees(theta))
