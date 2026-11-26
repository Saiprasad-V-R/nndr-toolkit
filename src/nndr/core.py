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
from .estimation import adjacent_pairs, relative_ridge, shrink_operator
from .reduce2d import compute_indices_from_Q
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
        Selected ratio: corrected R_A by default, or explicitly requested legacy ratio.
        R_A and R_sigma are separately available as properties.
    Delta:
        Rate splitting for real pairs, beta for complex pairs (legacy: multiplier splitting).
    K:
        Reduced non-normality index.
    Kc:
        Threshold corresponding to the selected R convention.
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
        Admissibility of R_A by default; historical flag in legacy mode.
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
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def R_A(self) -> float:
        return float(self.diagnostics.get("R_A", np.nan))

    @property
    def R_sigma(self) -> float:
        return float(self.diagnostics.get("R_sigma", np.nan))

    def summary(self) -> str:
        """One-line human-readable summary of the key diagnostics."""
        regime = ("inadmissible" if not self.ok or np.isnan(self.R) else
                  "above" if self.R > 1 else "below" if self.R < 1 else "at")
        return (
            f"[{self.method}] R={self.R:.3f} ({regime}; {self.diagnostics.get('threshold', 'unspecified')}), "
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
    lam_rel: Optional[float] = None,
    frac_train: float = 0.8,
    seed: int = 0,
    shrinkage: float = 0.0,
    legacy_pairs: bool = False,
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
    lam_rel:
        Relative ridge penalty, scaled by predictor energy per channel.
        Cannot be combined with lam or lambdas.
    legacy_pairs:
        Include a fabricated zero-to-first-sample transition for historical
        reproduction only. Default uses observed adjacent pairs.
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

    if lam_rel is not None and (lam is not None or lambdas is not None):
        raise ValueError("lam_rel cannot be combined with lam or lambdas")
    if legacy_pairs:
        Xstack, Ystack = stack_VAR_pairs_from_trajectories([X], x0=np.zeros(N))
    else:
        Xstack, Ystack = adjacent_pairs(X)
    if lam_rel is not None:
        lam = relative_ridge(Xstack, lam_rel)
    if lam is not None and (not np.isfinite(lam) or lam < 0):
        raise ValueError("lam must be finite and nonnegative")

    if lam is None:
        if lambdas is None:
            lambdas = np.logspace(-8, 2, 21)
        lam, _ = select_lambda_ridge(Xstack, Ystack, lambdas, frac_train=frac_train, seed=seed)

    Fhat = ridge_fit_F(Xstack, Ystack, lam)

    return shrink_operator(Fhat, shrinkage)


def analyze_operator(A: np.ndarray, *, method: str = "M2", dt: float = 1.0,
                     threshold: str = "rates", **method_kwargs) -> NNDRResult:
    """Run NNDR diagnostics on a known operator ``A`` (no estimation step).

    Parameters
    ----------
    A:
        An ``N x N`` operator.
    method:
        Plane-extraction method: ``"M1"``, ``"M2"`` (default) or ``"M3"``.
    dt, threshold:
        Sample interval and ratio convention (rates or legacy).
    **method_kwargs:
        Forwarded to the chosen method (e.g. ``iters``, ``seed`` for M2).
    """
    A = validate_operator(A, name="A")
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {list(METHODS)}")
    func, _ = METHODS[method]
    if method == "M3" and threshold != "legacy":
        # Manuscript Eqs. (40)-(42): directed ordering within M3 plane.
        from .scientific import method3
        out = method3(A, dt=dt, **method_kwargs)
    else:
        out = func(A, **method_kwargs)

    Q = out["Q"]
    idx = compute_indices_from_Q(A, Q, dt=dt, threshold=threshold)
    r_hat, n_hat = Q[:, 0], Q[:, 1]

    extra = {k: v for k, v in out.items()
             if k not in {"name", "n", "r", "Q", "Gamma", "trG", "evG", "okG"}}

    return NNDRResult(
        R=float(idx["KKc"]),
        Delta=float(idx["Delta"]),
        K=float(idx["K"]),
        Kc=float(idx["Kc"]),
        support=(float("nan") if method == "M3" and threshold != "legacy"
                 and not out["direction_identifiable"] else _reaction_support(r_hat)),
        Q=Q,
        r_hat=r_hat,
        n_hat=n_hat,
        method=method,
        ok=bool(idx["okG"]),
        Fhat=None,
        extra=extra,
        diagnostics=idx,
    )


def analyze(
    X: np.ndarray,
    *,
    method: str = "M2",
    lambdas: Optional[np.ndarray] = None,
    lam: Optional[float] = None,
    lam_rel: Optional[float] = None,
    frac_train: float = 0.8,
    seed: int = 0,
    shrinkage: float = 0.0,
    iters: int = 250,
    dt: float = 1.0,
    threshold: str = "rates",
) -> NNDRResult:
    """Estimate the local operator from ``X`` and compute NNDR diagnostics.

    This is the main entry point for applying NNDR to your own data.

    Parameters
    ----------
    X:
        Data of shape ``(N_channels, T_samples)``.
    method:
        Plane-extraction method: ``"M1"``, ``"M2"`` (default) or ``"M3"``.
    lambdas, lam, lam_rel, frac_train, seed, shrinkage:
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
        X, lambdas=lambdas, lam=lam, lam_rel=lam_rel, frac_train=frac_train,
        seed=seed, shrinkage=shrinkage, legacy_pairs=(threshold == "legacy"),
    )
    method_kwargs: Dict[str, Any] = {}
    if method == "M2":
        method_kwargs = {"iters": iters, "seed": seed}
    res = analyze_operator(Fhat, method=method, dt=dt, threshold=threshold, **method_kwargs)
    res.Fhat = Fhat
    return res


def rolling_analyze(
    X: np.ndarray,
    *,
    window: int,
    step: int = 1,
    method: str = "M2",
    lam: Optional[float] = None,
    lam_rel: Optional[float] = None,
    lambdas: Optional[np.ndarray] = None,
    shrinkage: float = 0.0,
    seed: int = 0,
    iters: int = 250,
    skip_degenerate: bool = True,
    dt: float = 1.0,
    threshold: str = "rates",
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
    method, lam, lam_rel, lambdas, shrinkage, seed, iters:
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
            Xw, method=method, lam=lam, lam_rel=lam_rel, lambdas=lambdas,
            shrinkage=shrinkage, seed=seed, iters=iters, dt=dt, threshold=threshold,
        )
        records.append({
            "t_start": start,
            "t_end": end,
            "t_center": start + window // 2,
            "R": res.R,
            "Delta": res.Delta,
            "K": res.K,
            "support": res.support,
            "plane_identifiable": res.extra.get("plane_identifiable"),
            "direction_identifiable": res.extra.get("direction_identifiable"),
            "ok": res.ok,
            **{k: res.diagnostics[k] for k in (
                "R_A", "R_sigma", "Kc_A", "Kc_sigma", "Delta_rate", "beta",
                "rhoF", "sigmaF", "rhoGamma", "sigmaGamma", "dF", "spectrum_type",
                "rate_admissible", "rate_status", "stable_full", "stable_reduced",
                "stable_amplifying", "dt", "threshold")},
        })
    return records
