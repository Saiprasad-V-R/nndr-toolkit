"""Tests for the three plane-extraction methods."""
import numpy as np
import pytest

from nndr.methods import (
    method1_eigenbasis_svd,
    method2_optimization,
    method3_commutator,
)
from nndr.reduce2d import compute_indices_from_Q
from nndr.utils import principal_angle_deg


def _embedded_operator(N=40, q=3.0, seed=0):
    """Embed a 2x2 non-normal block in a random orthogonal frame."""
    rng = np.random.default_rng(seed)
    a, b = 0.6, 0.2
    block = np.array([[a, q * (a - b)], [0.0, b]])
    bg = np.diag(rng.uniform(0.1, 0.5, size=N - 2))
    full = np.zeros((N, N))
    full[:2, :2] = block
    full[2:, 2:] = bg
    Qrot, _ = np.linalg.qr(rng.standard_normal((N, N)))
    return Qrot @ full @ Qrot.T, Qrot[:, :2]


def test_all_methods_return_orthonormal_plane():
    A, _ = _embedded_operator()
    for func in (method1_eigenbasis_svd, method2_optimization, method3_commutator):
        out = func(A)
        Q = out["Q"]
        assert Q.shape[1] == 2
        # columns orthonormal
        gram = Q.T @ Q
        assert np.allclose(gram, np.eye(2), atol=1e-8)


def test_m2_m3_agree_on_embedded_block():
    """M2 and M3 should recover nearly the same plane for a clear embedded block."""
    A, _ = _embedded_operator(q=4.0)
    q2 = method2_optimization(A)["Q"]
    q3 = method3_commutator(A)["Q"]
    angle = principal_angle_deg(q2, q3)
    assert angle < 5.0  # degrees


def test_m2_recovers_embedded_plane():
    A, Qstar = _embedded_operator(q=4.0)
    q2 = method2_optimization(A)["Q"]
    angle = principal_angle_deg(q2, Qstar)
    assert angle < 5.0


def test_normal_operator_has_zero_K():
    """A symmetric (normal) operator should have K ~ 0."""
    rng = np.random.default_rng(1)
    M = rng.standard_normal((20, 20))
    S = 0.1 * (M + M.T)  # symmetric, stable-ish scale
    out = method2_optimization(S)
    idx = compute_indices_from_Q(S, out["Q"])
    assert idx["K"] == pytest.approx(0.0, abs=1e-3)
