"""NNDR -- Non-Normal Directional Response inference.

A toolkit for inferring non-normal amplification geometry from multivariate
time series. Estimate a local linear operator, extract a two-dimensional
response plane, and compute reduced diagnostics: the eigenvalue splitting
``Delta``, the non-normality index ``K``, and the normalized reduced
non-normality ratio ``R = K / Kc(Delta)``.

Quick start
-----------
>>> import numpy as np
>>> from nndr import analyze
>>> X = np.load("recording.npy")        # shape (N_channels, T_samples)   # doctest: +SKIP
>>> result = analyze(X, method="M2")                                      # doctest: +SKIP
>>> print(result.summary())                                              # doctest: +SKIP
"""
from __future__ import annotations

# High-level, inference-first API (most users start here)
from .core import (
    NNDRResult,
    analyze,
    analyze_operator,
    fit_operator,
    rolling_analyze,
)

# Plane-extraction methods
from .methods import (
    METHODS,
    method1_eigenbasis_svd,
    method2_optimization,
    method3_commutator,
)

# Reduced diagnostics
from .reduce2d import (
    Kc_from_Delta,
    compute_indices_from_Q,
    twoD_metrics_from_Gamma,
)

# Operator estimation
from .ridge import ridge_fit_F, select_lambda_ridge

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # high-level API
    "analyze",
    "rolling_analyze",
    "analyze_operator",
    "fit_operator",
    "NNDRResult",
    # methods
    "method1_eigenbasis_svd",
    "method2_optimization",
    "method3_commutator",
    "METHODS",
    # diagnostics
    "Kc_from_Delta",
    "twoD_metrics_from_Gamma",
    "compute_indices_from_Q",
    # estimation
    "ridge_fit_F",
    "select_lambda_ridge",
]
