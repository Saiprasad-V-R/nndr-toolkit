"""Reduced two-dimensional diagnostics.

Given a fitted operator and a response plane ``Q = [r_hat, n_hat]``, this
module computes the scalar diagnostics that the NNDR framework reports:

* ``Delta``  -- reduced eigenvalue splitting ``|(lam1 - lam2) / (lam1 + lam2)|``
* ``kappa2D`` -- reduced eigenvector non-orthogonality
* ``K``       -- reduced non-normality index
* ``Kc(Delta)`` -- the reduced transient-amplification threshold
* ``R = K / Kc(Delta)`` -- the normalized reduced non-normality ratio

The closed form of ``Kc(Delta)`` is derived in Supplementary Note 2 of the
accompanying paper.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

__all__ = [
    "Kc_from_Delta",
    "twoD_metrics_from_Gamma",
    "gamma_diagnostics",
    "compute_indices_from_Q",
]


def Kc_from_Delta(Delta: float) -> float:
    r"""Reduced threshold ``Kc(Delta)``.

    .. math::
        K_c(\Delta) = \sqrt{\frac{\sqrt{1-\Delta^2}}{1-\sqrt{1-\Delta^2}}},
        \qquad 0 \le \Delta < 1.

    As ``Delta -> 0`` the reduced spectrum is nearly degenerate and
    ``Kc -> inf``; as ``Delta -> 1`` the spectrum is strongly split and
    ``Kc -> 0``.
    """
    Delta = float(np.clip(Delta, 0.0, 1.0 - 1e-12))
    s = np.sqrt(1.0 - Delta**2)
    return float(np.sqrt(s / (1.0 - s + 1e-15)))


def twoD_metrics_from_Gamma(Gamma: np.ndarray) -> Tuple[float, float, float]:
    """Compute ``(Delta, kappa2D, K)`` from a 2x2 reduced operator ``Gamma``.

    Parameters
    ----------
    Gamma:
        A ``2 x 2`` real matrix, typically ``Q.T @ A @ Q``.

    Returns
    -------
    (Delta, kappa2D, K):
        The reduced eigenvalue splitting, the reduced eigenvector
        non-orthogonality, and the reduced non-normality index.
    """
    eigvals, eigvecs = np.linalg.eig(Gamma)
    lam1, lam2 = eigvals[0], eigvals[1]
    denom = lam1 + lam2
    Delta = 1.0 if abs(denom) < 1e-15 else abs((lam1 - lam2) / denom)

    p1 = eigvecs[:, 0] / (np.linalg.norm(eigvecs[:, 0]) + 1e-15)
    p2 = eigvecs[:, 1] / (np.linalg.norm(eigvecs[:, 1]) + 1e-15)

    c = abs(np.dot(p1.conj(), p2))
    c = float(np.clip(c, 0.0, 1.0 - 1e-12))
    kappa2D = float(np.sqrt((1.0 + c) / (1.0 - c)))
    K = float(0.5 * (kappa2D - 1.0 / kappa2D))
    return float(Delta), float(kappa2D), float(K)


def gamma_diagnostics(Gamma: np.ndarray, tol: float = 1e-12):
    """Return ``(trace, eigenvalues, ok_flag)`` for a reduced operator.

    ``ok_flag`` is ``True`` when the trace is not near zero and the two
    eigenvalues are real and share the same sign. It is used to screen out
    degenerate reduced matrices for which ``Delta`` and ``Kc(Delta)`` are not
    meaningful.
    """
    G = np.real_if_close(Gamma, tol=1000)
    tr = complex(np.trace(G))
    ev = np.linalg.eigvals(G)

    tr_ok = abs(tr) > tol
    ev_r = np.real_if_close(ev, tol=1000)
    if np.any(np.iscomplex(ev_r)):
        same_sign = False
    else:
        ev_r = np.real(ev_r)
        same_sign = (ev_r[0] * ev_r[1] > 0)

    ok = bool(tr_ok and same_sign)
    return tr, ev, ok


def compute_indices_from_Q(A: np.ndarray, Q: np.ndarray) -> Dict[str, Any]:
    """Project ``A`` onto plane ``Q`` and return all reduced diagnostics.

    Parameters
    ----------
    A:
        The (estimated or true) ``N x N`` operator.
    Q:
        An ``N x 2`` orthonormal basis ``[r_hat, n_hat]`` for the response
        plane.

    Returns
    -------
    dict
        Keys: ``Gamma`` (the 2x2 reduced operator), ``Delta``, ``K``, ``KKc``
        (the ratio ``R = K / Kc(Delta)``), ``trG``, ``evG``, ``okG``.
    """
    Gamma = Q.T @ A @ Q
    Delta, _, K = twoD_metrics_from_Gamma(Gamma)
    KKc = K / (Kc_from_Delta(Delta) + 1e-15)
    trG, evG, okG = gamma_diagnostics(Gamma)
    return {"Gamma": Gamma, "Delta": Delta, "K": K, "KKc": KKc,
            "trG": trG, "evG": evG, "okG": okG}
