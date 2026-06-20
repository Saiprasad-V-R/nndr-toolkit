"""High-level NNDR inference API.

This module is the main entry point for applying NNDR to your own data. The
typical workflow is::

    import numpy as np
    from nndr import analyze

    X = np.load("my_recording.npy")   # shape (N_channels, T_samples)
    result = analyze(X, method="M2")
    print(result.R, result.Delta, result.K)

For data whose local dynamics drift over time, use :func:`rolling_analyze`,
which slides a window across the recording and returns one
:class:`NNDRResult` per window position.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .methods import METHODS
from .reduce2d import Kc_from_Delta, compute_indices_from_Q
from .ridge import ridge_fit_F, select_lambda_ridge, stack_VAR_pairs_from_trajectories
from .validation import validate_operator, validate_time_series, validate_window

__all__ = [
    "NNDRResult",
    "fit_operator",
    "analyze_operator",
    "analyze",
    "rolling_analyze",
]


@dataclass
class NNDRResult:
    """Diagnostics from a single NNDR analysis.

    Attributes
    ----------
    R:
        Normalized reduced non-normality ratio ``R = K / Kc(Delta)``. Values
        below 1 lie below the reduced transient-amplification threshold;
        values above 1 lie above it.
    Delta:
        Reduced eigenvalue splitting ``|(lam1 - lam2) / (lam1 + lam2)|``.
    K:
        Reduced non-normality index.
    Kc:
        The reduced threshold ``Kc(Delta)``.
    support:
        Reaction-direction support in ``(0, 1]``; values near 1 mean the
        reaction direction is spread across many channels.
    Q:
        The ``N x 2`` response plane ``[r_hat, n_hat]``.
    r_hat, n_hat:
        The reaction and input directions (columns of ``Q``).
    method:
        Which extraction method was used (``"M1"``, ``"M2"`` or ``"M3"``).
    ok:
        ``True`` if the reduced operator is non-degenerate (``Delta`` and
        ``Kc`` are meaningful).
    Fhat:
        The fitted ``N x N`` operator (``None`` if an operator was supplied
        directly).
    extra:
        Method-specific extras (e.g. ``kappa_P`` for M1, commutator
        eigenvalues for M3).
    """

    R: float
    Delta: float
    K: float
    Kc: float
    support: float
    Q: np.ndarray
    r_hat: np.ndarray
    n_hat: np.ndarray
    method: str
    ok: bool
    Fhat: Optional[np.ndarray] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """One-line human-readable summary of the key diagnostics."""
        regime = "above" if self.R > 1 else "below"
        return (
            f"[{self.method}] R={self.R:.3f} ({regime} threshold), "
            f"Delta={self.Delta:.3f}, K={self.K:.3f}, support={self.support:.3f}"
        )


def _reaction_support(r_hat: np.ndarray) -> float:
    """Support ``S(r) = ||r||_1 / (sqrt(N) ||r||_2)`` in ``(0, 1]``."""
    r = np.asarray(r_hat).ravel()
    N = r.size
    denom = np.sqrt(N) * (np.linalg.norm(r) + 1e-15)
    return float(np.linalg.norm(r, ord=1) / denom)


def fit_operator(
    X: np.ndarray,
    *,
    lambdas: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    frac_train: float = 0.8,
    seed: int = 0,
    shrinkage: float = 0.0,
) -> np.ndarray:
    """Estimate the one-step operator ``Fhat`` from a multivariate time series.

    Parameters
    ----------
    X:
        Data of shape ``(N_channels, T_samples)``. Each column is one time
        sample.
    lambdas:
        Grid of ridge strengths to search by cross-validation. Ignored if
        ``lam`` is given. Defaults to ``np.logspace(-8, 2, 21)``.
    lam:
        Fixed ridge strength. If supplied, no cross-validation is performed.
    frac_train:
        Fraction of samples used for training during ``lambda`` selection.
    seed:
        Seed for the cross-validation split.
    shrinkage:
        Optional trace-preserving isotropic shrinkage ``alpha`` in ``[0, 1)``
        applied to the fitted operator as
        ``(1 - alpha) Fhat + alpha * mean(diag(Fhat)) * I``. Useful for short
        or noisy windows.

    Returns
    -------
    np.ndarray
        The fitted ``N x N`` operator.
    """
    X = validate_time_series(X, name="X")
    N = X.shape[0]

    Xstack, Ystack = stack_VAR_pairs_from_trajectories([X], x0=np.zeros(N))

    if lam is None:
        if lambdas is None:
            lambdas = np.logspace(-8, 2, 21)
        lam, _ = select_lambda_ridge(Xstack, Ystack, lambdas, frac_train=frac_train, seed=seed)

    Fhat = ridge_fit_F(Xstack, Ystack, lam)

    if shrinkage and shrinkage > 0.0:
        mu = float(np.trace(Fhat) / N)
        Fhat = (1.0 - shrinkage) * Fhat + shrinkage * mu * np.eye(N)

    return Fhat


def analyze_operator(A: np.ndarray, *, method: str = "M2", **method_kwargs) -> NNDRResult:
    """Run NNDR diagnostics on a known operator ``A`` (no estimation step).

    Parameters
    ----------
    A:
        An ``N x N`` operator.
    method:
        Plane-extraction method: ``"M1"``, ``"M2"`` (default) or ``"M3"``.
    **method_kwargs:
        Forwarded to the chosen method (e.g. ``iters``, ``seed`` for M2).
    """
    A = validate_operator(A, name="A")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    func, _ = METHODS[method]
    out = func(A, **method_kwargs)

    Q = out["Q"]
    idx = compute_indices_from_Q(A, Q)
    r_hat, n_hat = Q[:, 0], Q[:, 1]

    extra = {k: v for k, v in out.items()
             if k not in {"name", "n", "r", "Q", "Gamma", "trG", "evG", "okG"}}

    return NNDRResult(
        R=float(idx["KKc"]),
        Delta=float(idx["Delta"]),
        K=float(idx["K"]),
        Kc=Kc_from_Delta(float(idx["Delta"])),
        support=_reaction_support(r_hat),
        Q=Q,
        r_hat=r_hat,
        n_hat=n_hat,
        method=method,
        ok=bool(idx["okG"]),
        Fhat=None,
        extra=extra,
    )


def analyze(
    X: np.ndarray,
    *,
    method: str = "M2",
    lambdas: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    frac_train: float = 0.8,
    seed: int = 0,
    shrinkage: float = 0.0,
    iters: int = 250,
) -> NNDRResult:
    """Estimate the local operator from ``X`` and compute NNDR diagnostics.

    This is the main entry point for applying NNDR to your own data.

    Parameters
    ----------
    X:
        Data of shape ``(N_channels, T_samples)``.
    method:
        Plane-extraction method: ``"M1"``, ``"M2"`` (default) or ``"M3"``.
    lambdas, lam, frac_train, seed, shrinkage:
        Passed to :func:`fit_operator`.
    iters:
        Iteration count for the M2 optimizer (ignored by M1/M3).

    Returns
    -------
    NNDRResult
        The diagnostics, including the fitted operator in ``.Fhat``.

    Examples
    --------
    >>> import numpy as np
    >>> from nndr import analyze
    >>> rng = np.random.default_rng(0)
    >>> X = rng.standard_normal((10, 500))
    >>> res = analyze(X, method="M2")
    >>> isinstance(res.R, float)
    True
    """
    Fhat = fit_operator(
        X, lambdas=lambdas, lam=lam, frac_train=frac_train, seed=seed, shrinkage=shrinkage
    )
    method_kwargs: Dict[str, Any] = {}
    if method == "M2":
        method_kwargs = {"iters": iters, "seed": seed}
    res = analyze_operator(Fhat, method=method, **method_kwargs)
    res.Fhat = Fhat
    return res


def rolling_analyze(
    X: np.ndarray,
    *,
    window: int,
    step: int = 1,
    method: str = "M2",
    lam: Optional[float] = None,
    lambdas: Optional[np.ndarray] = None,
    shrinkage: float = 0.0,
    seed: int = 0,
    iters: int = 250,
    skip_degenerate: bool = True,
) -> List[Dict[str, Any]]:
    """Slide a window across ``X`` and run NNDR in each window.

    Mirrors the rolling-window construction used for the empirical analyses in
    the paper: at each window position a local operator is estimated from the
    samples inside the window and the reduced diagnostics are computed.

    Parameters
    ----------
    X:
        Data of shape ``(N_channels, T_samples)``.
    window:
        Window length in samples.
    step:
        Step size between consecutive windows.
    method, lam, lambdas, shrinkage, seed, iters:
        Passed through to the per-window analysis.
    skip_degenerate:
        If ``True``, windows whose reduced operator is degenerate are still
        returned but flagged with ``ok=False``; downstream code can filter on
        the ``ok`` field.

    Returns
    -------
    list of dict
        One record per window with keys ``t_start``, ``t_end``, ``t_center``
        and the diagnostics ``R``, ``Delta``, ``K``, ``support``, ``ok``.
    """
    X = validate_time_series(X, name="X")
    N, T = X.shape
    validate_window(window, T, step=step)

    records: List[Dict[str, Any]] = []
    for start in range(0, T - window + 1, step):
        end = start + window
        Xw = X[:, start:end]
        res = analyze(
            Xw, method=method, lam=lam, lambdas=lambdas,
            shrinkage=shrinkage, seed=seed, iters=iters,
        )
        records.append({
            "t_start": start,
            "t_end": end,
            "t_center": start + window // 2,
            "R": res.R,
            "Delta": res.Delta,
            "K": res.K,
            "support": res.support,
            "ok": res.ok,
        })
    return records
