"""Tests for input-validation guards."""
import numpy as np
import pytest

from nndr import analyze, rolling_analyze, fit_operator
from nndr.validation import (
    validate_time_series,
    validate_operator,
    validate_window,
)


# --------------------------------------------------------------------------
# validate_time_series
# --------------------------------------------------------------------------

def test_rejects_1d_array():
    with pytest.raises(ValueError, match="2-D"):
        validate_time_series(np.zeros(10))


def test_rejects_nan():
    X = np.random.default_rng(0).standard_normal((5, 100))
    X[2, 10] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_time_series(X)


def test_rejects_inf():
    X = np.random.default_rng(0).standard_normal((5, 100))
    X[0, 0] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        validate_time_series(X)


def test_rejects_transposed_data():
    # 200 channels, 5 samples -> looks transposed
    X = np.random.default_rng(0).standard_normal((200, 5))
    with pytest.raises(ValueError, match="transposed"):
        validate_time_series(X)


def test_rejects_too_few_samples():
    X = np.random.default_rng(0).standard_normal((5, 2))
    with pytest.raises(ValueError, match="time sample"):
        validate_time_series(X, min_samples=3)


def test_rejects_empty():
    with pytest.raises(ValueError):
        validate_time_series(np.zeros((0, 10)))


def test_accepts_valid_series():
    X = np.random.default_rng(0).standard_normal((5, 100))
    out = validate_time_series(X)
    assert out.shape == (5, 100)
    assert out.dtype == float


# --------------------------------------------------------------------------
# validate_operator
# --------------------------------------------------------------------------

def test_operator_rejects_nonsquare():
    with pytest.raises(ValueError, match="square"):
        validate_operator(np.zeros((3, 4)))


def test_operator_rejects_nan():
    A = np.eye(3)
    A[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_operator(A)


def test_operator_accepts_square():
    A = np.eye(4) * 0.5
    out = validate_operator(A)
    assert out.shape == (4, 4)


# --------------------------------------------------------------------------
# validate_window
# --------------------------------------------------------------------------

def test_window_rejects_too_large():
    with pytest.raises(ValueError, match="larger than"):
        validate_window(500, 100)


def test_window_rejects_nonpositive():
    with pytest.raises(ValueError, match="positive integer"):
        validate_window(0, 100)


def test_window_rejects_bad_step():
    with pytest.raises(ValueError, match="step"):
        validate_window(50, 100, step=0)


def test_window_accepts_valid():
    validate_window(50, 100, step=5)  # should not raise


# --------------------------------------------------------------------------
# End-to-end: guards fire through the public API
# --------------------------------------------------------------------------

def test_analyze_rejects_transposed():
    X = np.random.default_rng(0).standard_normal((300, 10))
    with pytest.raises(ValueError, match="transposed"):
        analyze(X, method="M2")


def test_analyze_rejects_nan():
    X = np.random.default_rng(0).standard_normal((10, 500))
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        analyze(X, method="M2")


def test_rolling_rejects_oversized_window():
    X = np.random.default_rng(0).standard_normal((10, 200))
    with pytest.raises(ValueError, match="larger than"):
        rolling_analyze(X, window=500, method="M2")


def test_fit_operator_rejects_1d():
    with pytest.raises(ValueError, match="2-D"):
        fit_operator(np.zeros(10))
