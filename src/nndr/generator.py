"""Synthetic non-normal operator families for validation benchmarks.

These constructions fix the orthogonal frames ``U, V`` and the stable spectrum
``Lambda`` and vary only an anisotropy parameter ``kappa``. This isolates
changes in the non-normal geometry from changes in the spectrum.
"""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .utils import random_orthogonal

__all__ = ["make_fixed_config", "build_F_from_config"]


def make_fixed_config(
    N: int, rho: float = 0.95, delta_sigma: float = 0.05, seed: int = 1
) -> Dict[str, Any]:
    """Fix ``U``, ``V``, ``Lambda`` and the base singular values.

    Only ``kappa`` is varied later (by setting the smallest singular value to
    ``1 / kappa``), so a family built from one config shares its spectrum and
    orientation.
    """
    rng = np.random.default_rng(seed)
    U = random_orthogonal(N, rng)
    V = random_orthogonal(N, rng)

    sigma_base = 1.0 + delta_sigma * rng.standard_normal(N - 1)
    sigma_base = np.clip(sigma_base, 0.1, 3.0)

    lam = rng.uniform(0.2 * rho, 0.95 * rho, size=N)
    lam[0] = rho
    Lambda = np.diag(lam)

    return {"N": N, "rho": rho, "delta_sigma": delta_sigma, "seed": seed,
            "U": U, "V": V, "Lambda": Lambda, "sigma_base": sigma_base}


def build_F_from_config(cfg: Dict[str, Any], kappa: float) -> np.ndarray:
    """Build a stable operator ``F`` from a fixed config at anisotropy ``kappa``.

    Larger ``kappa`` produces a more strongly non-normal operator while keeping
    the eigenvalue spectrum fixed.
    """
    U, V, Lambda = cfg["U"], cfg["V"], cfg["Lambda"]
    N = cfg["N"]

    sigma = np.ones(N)
    sigma[:-1] = cfg["sigma_base"]
    sigma[-1] = 1.0 / float(kappa)

    Sigma = np.diag(sigma)
    Sigma_inv = np.diag(1.0 / sigma)

    F = U @ Sigma @ V.T @ Lambda @ V @ Sigma_inv @ U.T
    return F
