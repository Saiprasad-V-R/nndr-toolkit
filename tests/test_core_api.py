"""Tests for the high-level inference API."""
import numpy as np
import pytest

from nndr import analyze, analyze_operator, fit_operator, rolling_analyze
from nndr.core import NNDRResult


def _synthetic_series(N=12, T=600, q=3.0, seed=0):
    """Generate a VAR(1) series from an embedded non-normal operator."""
    rng = np.random.default_rng(seed)
    a, b = 0.6, 0.2
    block = np.array([[a, q * (a - b)], [0.0, b]])
    bg = np.diag(rng.uniform(0.1, 0.5, size=N - 2))
    full = np.zeros((N, N))
    full[:2, :2] = block
    full[2:, 2:] = bg
    Qrot, _ = np.linalg.qr(rng.standard_normal((N, N)))
    F = Qrot @ full @ Qrot.T

    X = np.zeros((N, T))
    x = np.zeros(N)
    for k in range(T):
        x = F @ x + 0.05 * rng.standard_normal(N)
        X[:, k] = x
    return X, F


def test_analyze_returns_result():
    X, _ = _synthetic_series()
    res = analyze(X, method="M2")
    assert isinstance(res, NNDRResult)
    assert isinstance(res.R, float)
    assert 0.0 <= res.Delta <= 1.0
    assert 0.0 < res.support <= 1.0 + 1e-9
    assert res.Fhat is not None
    assert res.Q.shape == (X.shape[0], 2)


def test_analyze_methods_run():
    X, _ = _synthetic_series()
    for method in ("M1", "M2", "M3"):
        res = analyze(X, method=method)
        assert res.method == method
        assert np.isfinite(res.R)


def test_analyze_unknown_method_raises():
    X, _ = _synthetic_series()
    with pytest.raises(ValueError):
        analyze(X, method="M9")


def test_fit_operator_shape_and_shrinkage():
    X, _ = _synthetic_series(N=10, T=400)
    F0 = fit_operator(X, lam=1e-3, shrinkage=0.0)
    F1 = fit_operator(X, lam=1e-3, shrinkage=0.2)
    assert F0.shape == (10, 10)
    assert F1.shape == (10, 10)
    # shrinkage pulls the operator toward a scaled identity, changing it
    assert not np.allclose(F0, F1)


def test_analyze_operator_matches_direct_diagnostics():
    _, F = _synthetic_series()
    res = analyze_operator(F, method="M2")
    assert np.isfinite(res.R)
    assert res.Fhat is None  # operator supplied directly


def test_rolling_analyze_window_count():
    X, _ = _synthetic_series(T=400)
    window, step = 100, 50
    recs = rolling_analyze(X, window=window, step=step, method="M2")
    expected = len(range(0, 400 - window + 1, step))
    assert len(recs) == expected
    for r in recs:
        assert {"t_start", "t_end", "t_center", "R", "Delta", "K", "support", "ok"} <= set(r)


def test_rolling_window_too_large_raises():
    X, _ = _synthetic_series(T=100)
    with pytest.raises(ValueError):
        rolling_analyze(X, window=500, method="M2")


def test_bad_input_shape_raises():
    with pytest.raises(ValueError):
        fit_operator(np.zeros(10))  # 1-D, not (N, T)
