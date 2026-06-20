"""Simulation of stable VAR(1) trajectories with prescribed noise."""
from __future__ import annotations

from typing import Optional

import numpy as np

__all__ = ["simulate_var_with_fixed_noise"]


def simulate_var_with_fixed_noise(
    F: np.ndarray, noise: np.ndarray, x0: Optional[np.ndarray] = None
) -> np.ndarray:
    """Simulate ``x_{k+1} = F x_k + noise[k]``.

    Parameters
    ----------
    F:
        The ``N x N`` one-step operator.
    noise:
        Array of shape ``(T, N)`` of additive innovations.
    x0:
        Optional initial state (defaults to zeros).

    Returns
    -------
    np.ndarray
        Array ``X`` of shape ``(N, T)`` storing ``x_1, ..., x_T``.
    """
    T, N = noise.shape
    x = np.zeros(N) if x0 is None else np.array(x0, dtype=float).copy()
    X = np.zeros((N, T))
    for k in range(T):
        x = F @ x + noise[k]
        X[:, k] = x
    return X
