"""Canonical stable two-dimensional generators for NNDR validation."""
from __future__ import annotations

import numpy as np


def _positive(value, name):
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _nonnegative(value, name):
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def real_generator(gamma1: float, gamma2: float, K: float) -> np.ndarray:
    """Construct a real-spectrum generator with prescribed non-normality.

    Parameters
    ----------
    gamma1, gamma2 : float
        Decay rates, satisfying 0 < gamma1 < gamma2 (inverse time units).
    K : float
        Nonnegative eigenvector non-orthogonality index of the NNDR paper.

    Returns
    -------
    ndarray, shape (2, 2)
        Continuous-time generator, not a discrete-time map. Its eigenvalues
        are -gamma1 and -gamma2. See revised Appendix B, Eq. (B.1).
    """
    g1, g2 = _positive(gamma1, "gamma1"), _positive(gamma2, "gamma2")
    k = _nonnegative(K, "K")
    if g1 >= g2:
        raise ValueError("require gamma1 < gamma2")
    with np.errstate(over="ignore"):
        L = np.array([[-g1, k*(g2-g1)], [0., -g2]])
    if not np.all(np.isfinite(L)):
        raise ValueError("generator entries exceed floating-point range")
    return L


def complex_generator(gamma: float, omega: float, K: float) -> np.ndarray:
    """Construct a real generator with a complex pair and prescribed K.

    Parameters
    ----------
    gamma, omega : float
        Positive decay rate and angular frequency (inverse time units).
    K : float
        Nonnegative eigenvector non-orthogonality index; eta = asinh(K).

    Returns
    -------
    ndarray, shape (2, 2)
        Continuous-time generator with eigenvalues -gamma +/- i*omega.
        See revised Appendix C, Eqs. (C.1)-(C.2). To use rate diagnostics
        on its sampled map, require omega*dt < pi (principal branch).
    """
    g, w = _positive(gamma, "gamma"), _positive(omega, "omega")
    eta = np.arcsinh(_nonnegative(K, "K"))
    with np.errstate(over="ignore", under="ignore"):
        L = np.array([[-g, w*np.exp(eta)], [-w*np.exp(-eta), -g]])
    if not np.all(np.isfinite(L)) or L[1, 0] == 0:
        raise ValueError("generator entries exceed floating-point range")
    return L
