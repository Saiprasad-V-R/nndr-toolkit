"""Plane-extraction methods.

Three strategies extract a two-dimensional response plane ``Q = [r_hat, n_hat]``
from a fitted operator. They differ only in how the plane is found; the reduced
operator ``Gamma = Q.T @ A @ Q`` and all downstream diagnostics are computed
identically afterwards.

* :func:`method1_eigenbasis_svd` (M1) -- Eigenbasis-SVD baseline extraction.
* :func:`method2_optimization`   (M2) -- Optimization-based directed-coupling extraction.
* :func:`method3_commutator`     (M3) -- Commutator-based plane extraction.

M2 and M3 are the recommended estimators; M1 is a transparent baseline that is
fragile when the eigenvectors of the fitted operator are unstable.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .reduce2d import gamma_diagnostics

__all__ = [
    "extract_nr_by_optimization",
    "method1_eigenbasis_svd",
    "method2_optimization",
    "method3_commutator",
    "METHODS",
]


# --------------------------------------------------------------------------
# Method 2: alternating optimization
# --------------------------------------------------------------------------

def extract_nr_by_optimization(
    A: np.ndarray, iters: int = 250, seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """Find an orthonormal pair maximizing transverse action under ``A``.

    Solves ``max <u, A v>`` subject to ``||u|| = ||v|| = 1`` and ``<u, v> = 0``
    by alternating projected power iteration. Returns ``(n, r)`` with
    ``r`` orthogonal to ``n`` and both unit norm.
    """
    rng = np.random.default_rng(seed)
    N = A.shape[0]
    v = rng.standard_normal(N)
    v /= (np.linalg.norm(v) + 1e-15)

    for _ in range(iters):
        Av = A @ v
        Av = Av - v * (v @ Av)
        nu = np.linalg.norm(Av)
        if nu < 1e-12:
            v = rng.standard_normal(N)
            v /= (np.linalg.norm(v) + 1e-15)
            continue
        u = Av / nu

        Atu = A.T @ u
        Atu = Atu - u * (u @ Atu)
        nv = np.linalg.norm(Atu)
        if nv < 1e-12:
            v = rng.standard_normal(N)
            v /= (np.linalg.norm(v) + 1e-15)
            continue
        v = Atu / nv

    n = v
    r_raw = (np.eye(N) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + 1e-15)
    return n, r


def method2_optimization(A: np.ndarray, iters: int = 250, seed: int = 0) -> Dict[str, Any]:
    """M2: optimization-based directed-coupling extraction."""
    n, r = extract_nr_by_optimization(A, iters=iters, seed=seed)
    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    r, n = Q[:, 0], Q[:, 1]
    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)
    return {"name": "M2_OPT", "n": n, "r": r, "Q": Q, "Gamma": Gamma,
            "trG": trG, "evG": evG, "okG": okG}


# --------------------------------------------------------------------------
# Method 1: eigenbasis + SVD of the eigenvector matrix
# --------------------------------------------------------------------------

def _make_real_direction(z, tol: float = 1e-12):
    """Return a real unit vector from a possibly complex eigenvector."""
    if np.iscomplexobj(z):
        zr, zi = np.real(z), np.imag(z)
        nr, ni = np.linalg.norm(zr), np.linalg.norm(zi)
        if nr > tol:
            return zr / (nr + 1e-15)
        if ni > tol:
            return zi / (ni + 1e-15)
        v = np.random.randn(z.size)
        return v / (np.linalg.norm(v) + 1e-15)
    return z / (np.linalg.norm(z) + 1e-15)


def method1_eigenbasis_svd(A: np.ndarray, eps: float = 1e-15) -> Dict[str, Any]:
    """M1: eigenbasis-SVD baseline extraction.

    Uses the eigenvector matrix ``P`` of ``A``; the input direction is taken as
    the left singular vector of ``P`` with the smallest singular value. Also
    reports ``kappa_P``, the conditioning of the eigenvector basis.
    """
    eigvals, P = np.linalg.eig(A)
    U, svals, _ = np.linalg.svd(P, full_matrices=False)
    kappa_P = float(svals[0] / (svals[-1] + eps))

    n = _make_real_direction(U[:, -1])
    r_raw = (np.eye(A.shape[0]) - np.outer(n, n)) @ (A @ n)
    r = r_raw / (np.linalg.norm(r_raw) + eps)

    Q, _ = np.linalg.qr(np.column_stack([r, n]))
    r, n = Q[:, 0], Q[:, 1]
    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)
    return {"name": "M1_BF", "n": n, "r": r, "Q": Q, "Gamma": Gamma,
            "kappa_P": kappa_P, "sigmas_P": svals,
            "trG": trG, "evG": evG, "okG": okG}


# --------------------------------------------------------------------------
# Method 3: commutator plane
# --------------------------------------------------------------------------

def method3_commutator(A: np.ndarray, tol: float = 1e-12) -> Dict[str, Any]:
    """M3: commutator-based plane extraction.

    Forms the symmetric commutator ``B = A A^T - A^T A`` and takes the plane
    spanned by its most positive and most negative eigenvectors. For a normal
    operator ``B = 0``.
    """
    B = A @ A.T - A.T @ A
    evals, evecs = np.linalg.eigh(B)
    lam_min = float(evals[0])
    lam_max = float(evals[-1])
    okB = (max(abs(lam_max), abs(lam_min)) > tol)

    w_plus = evecs[:, -1]
    w_minus = evecs[:, 0]
    Q = np.column_stack([w_plus, w_minus])

    Gamma = Q.T @ A @ Q
    trG, evG, okG = gamma_diagnostics(Gamma)
    return {"name": "M3_COMM", "n": w_plus, "r": w_minus, "Q": Q, "Gamma": Gamma,
            "lamB_max": lam_max, "lamB_min": lam_min, "okB": okB,
            "trG": trG, "evG": evG, "okG": okG}


#: Mapping from short method keys to ``(callable, human-readable name)``.
METHODS = {
    "M1": (method1_eigenbasis_svd, "Eigenbasis-SVD baseline extraction (M1)"),
    "M2": (method2_optimization, "Optimization-based directed-coupling extraction (M2)"),
    "M3": (method3_commutator, "Commutator-based plane extraction (M3)"),
}
