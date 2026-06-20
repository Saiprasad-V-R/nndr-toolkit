"""Ridge-regularized estimation of the one-step linear operator.

Fits ``F`` in ``x_{k+1} = F x_k`` from observed pairs by ridge regression,
with a simple cross-validation routine for selecting the regularization
strength ``lambda``.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np

__all__ = [
    "ridge_fit_F",
    "select_lambda_ridge",
    "stack_VAR_pairs_from_trajectories",
]


def ridge_fit_F(X: np.ndarray, Y: np.ndarray, lam: float) -> np.ndarray:
    r"""Closed-form ridge solution for the one-step operator.

    Solves ``min_F ||Y - F X||_F^2 + lam ||F||_F^2``, giving
    ``F = (Y X^T)(X X^T + lam I)^{-1}``.

    Parameters
    ----------
    X, Y:
        Predictor and response matrices of shape ``(N, T_eff)``.
    lam:
        Non-negative regularization strength.
    """
    N = X.shape[0]
    XXT = X @ X.T
    YXT = Y @ X.T
    A = XXT + float(lam) * np.eye(N)
    Fhat = (np.linalg.solve(A.T, YXT.T)).T
    return Fhat


def select_lambda_ridge(
    X: np.ndarray, Y: np.ndarray, lambdas: np.ndarray,
    frac_train: float = 0.8, seed: int = 0,
) -> Tuple[float, np.ndarray]:
    """Select ``lambda`` by a single random train/validation split.

    Returns the minimizing ``lambda`` and the validation MSE curve over the
    supplied ``lambdas`` grid.
    """
    rng = np.random.default_rng(seed)
    T_eff = X.shape[1]
    idx = np.arange(T_eff)
    rng.shuffle(idx)

    ntr = int(frac_train * T_eff)
    itr, iva = idx[:ntr], idx[ntr:]

    Xtr, Ytr = X[:, itr], Y[:, itr]
    Xva, Yva = X[:, iva], Y[:, iva]

    mse = []
    for lam in lambdas:
        Fhat = ridge_fit_F(Xtr, Ytr, lam)
        R = Yva - Fhat @ Xva
        mse.append(float(np.mean(R * R)))
    mse = np.array(mse)

    j = int(np.argmin(mse))
    lam_star = float(lambdas[j])
    return lam_star, mse


def stack_VAR_pairs_from_trajectories(
    X_list: List[np.ndarray], x0: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Assemble stacked ``(predictor, response)`` pairs from trajectories.

    Each entry of ``X_list`` has shape ``(N, T)`` storing ``x_1, ..., x_T``.
    The stacked predictors are ``[x0, x_1, ..., x_{T-1}]`` and the stacked
    responses are ``[x_1, ..., x_T]``, concatenated across trajectories.
    """
    Xcols, Ycols = [], []
    for X in X_list:
        N, T = X.shape
        x0_vec = np.asarray(x0).reshape(N)
        Xin = np.column_stack([x0_vec, X[:, :-1]])
        Yout = X.copy()
        Xcols.append(Xin)
        Ycols.append(Yout)
    Xstack = np.concatenate(Xcols, axis=1)
    Ystack = np.concatenate(Ycols, axis=1)
    return Xstack, Ystack
