"""NNDR -- Non-Normal Directional Response inference.

A toolkit for inferring non-normal amplification geometry from multivariate
time series. Estimate a local linear operator, extract a two-dimensional
response plane, and compute reduced diagnostics: the eigenvalue splitting
``Delta``, the non-normality index ``K``, and the revised amplification ratios ``R_A`` and ``R_sigma``.

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
    threshold_diagnostics,
)

# Operator estimation
from .ridge import ridge_fit_F, select_lambda_ridge

# Input validation
from .validation import validate_operator, validate_time_series, validate_window

__version__ = "2.0.0"

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
    "threshold_diagnostics",
    "compute_indices_from_Q",
    # estimation
    "ridge_fit_F",
    "select_lambda_ridge",
    # validation
    "validate_time_series",
    "validate_operator",
    "validate_window",
]

# Revised public scientific functions (legacy names retained).
from .scientific import (
    method1, method2, method3, order_response_plane, reaction_support,
    mean_field, response_coordinate, input_coordinate, project_operator,
    plane_correlation, eigenvalue_splitting, complex_splitting,
    non_normality_index, eigenvector_conditioning, critical_K_real,
    critical_K_complex, variance_response_ratio, one_step_ratio,
    capture_fraction, spectral_radius, sigma_max, departure_from_normality,
    commutator, commutator_gap, principal_angle, relative_operator_error,
    numerical_abscissa, condition_number,
)
from .scientific import __all__ as _scientific_exports
from .synthetic import real_generator, complex_generator
from .synthetic_recovery import embedded_map, simulate_trajectory
__all__ += _scientific_exports + [
    "real_generator", "complex_generator", "embedded_map", "simulate_trajectory",
]

# Scale-aware estimation and descriptive model adequacy.
from .estimation import adjacent_pairs, relative_ridge, shrink_operator, fit_relative_operator
from .dynamics import optimal_gain, variance_response, snapshot_gain, kreiss_constant, normal_surrogate
from .adequacy import (prediction_skill, lag_structure_error, markov_rate_consistency,
                       residual_autocorrelation, closure_error, phase_randomize, model_adequacy)
from .estimation import __all__ as _estimation_exports
from .dynamics import __all__ as _dynamics_exports
from .adequacy import __all__ as _adequacy_exports
__all__ += _estimation_exports + _dynamics_exports + _adequacy_exports

# Statistical validation and surrogate controls.
from .statistics import (portmanteau_test, squared_residual_test, variance_matched_surrogate,
                         surrogate_test, block_bootstrap, bootstrap_confidence_interval, unwrap_rates)
from .statistics import __all__ as _statistics_exports
__all__ += _statistics_exports
