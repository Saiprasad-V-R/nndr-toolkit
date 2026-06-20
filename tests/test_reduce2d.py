"""Tests for the reduced diagnostics (reduce2d)."""
import numpy as np
import pytest

from nndr.reduce2d import (
    Kc_from_Delta,
    twoD_metrics_from_Gamma,
    compute_indices_from_Q,
)


def test_Kc_limits():
    # Kc -> infinity as Delta -> 0 (large but finite at the clipped boundary)
    assert Kc_from_Delta(0.0) > 1e5
    # Kc -> 0 as Delta -> 1 (small at the boundary; the sqrt flattens the approach)
    assert Kc_from_Delta(0.999999) < 0.05
    # monotone decreasing in Delta over the interior
    grid = np.linspace(0.05, 0.95, 25)
    vals = [Kc_from_Delta(d) for d in grid]
    assert all(vals[i] > vals[i + 1] for i in range(len(vals) - 1))


def test_canonical_K_equals_q():
    """For the canonical triangular model, K should equal the coupling q."""
    a, b = 0.6, 0.2
    for q in [0.0, 0.5, 1.0, 2.0, 5.0]:
        F = np.array([[a, q * (a - b)], [0.0, b]])
        Delta, kappa2D, K = twoD_metrics_from_Gamma(F)
        assert K == pytest.approx(q, abs=1e-6)


def test_canonical_Delta():
    a, b = 0.6, 0.2
    F = np.array([[a, 0.0], [0.0, b]])
    Delta, _, K = twoD_metrics_from_Gamma(F)
    assert Delta == pytest.approx(abs((a - b) / (a + b)), abs=1e-9)
    assert K == pytest.approx(0.0, abs=1e-9)  # diagonal => normal => K=0


def test_compute_indices_identity_plane():
    a, b, q = 0.6, 0.2, 2.0
    A = np.array([[a, q * (a - b)], [0.0, b]])
    Q = np.eye(2)
    idx = compute_indices_from_Q(A, Q)
    assert idx["K"] == pytest.approx(q, abs=1e-6)
    assert idx["KKc"] == pytest.approx(q / Kc_from_Delta(idx["Delta"]), rel=1e-6)
    assert idx["okG"] in (True, False)
