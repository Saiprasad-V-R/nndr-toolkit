"""Scale-aware estimation for real, regularly sampled time series."""
from __future__ import annotations
import numpy as np
from .scientific import _series, _map
from .ridge import ridge_fit_F

__all__ = ['adjacent_pairs','relative_ridge','shrink_operator','fit_relative_operator']


def adjacent_pairs(X):
    """Return observed predictor/response pairs, without a fabricated initial state."""
    X = _series(X)
    if X.shape[1] < 2:
        raise ValueError('at least two samples required')
    return X[:, :-1], X[:, 1:]


def relative_ridge(X, lam_rel):
    """Ridge penalty nu_rel * trace(X X.T)/N for the predictor matrix X."""
    X = _series(X)
    if not np.isfinite(lam_rel) or lam_rel < 0:
        raise ValueError('lam_rel must be finite and nonnegative')
    return float(lam_rel * np.sum(X**2)/X.shape[0])


def shrink_operator(F, alpha):
    """Trace-preserving shrinkage toward mean diagonal times identity."""
    F = _map(F)
    if not np.isfinite(alpha) or not 0 <= alpha < 1:
        raise ValueError('alpha must satisfy 0 <= alpha < 1')
    return (1-alpha)*F + alpha*np.trace(F)/F.shape[0]*np.eye(F.shape[0])


def fit_relative_operator(X, *, lam_rel=0.003, shrinkage=0., return_info=False):
    """Fit adjacent pairs with relative ridge. No centering or stability rescaling.

    X has shape (channels,samples). The default ridge is illustrative, not a
    universal empirically calibrated choice. Optional info records C1 conditioning.
    """
    P,Y = adjacent_pairs(X)
    lam = relative_ridge(P,lam_rel)
    gram = P@P.T + lam*np.eye(P.shape[0])
    if not np.any(P):
        raise ValueError('zero predictor energy; operator is not identifiable')
    F = shrink_operator(ridge_fit_F(P,Y,lam),shrinkage)
    info = dict(lam=lam,lam_rel=float(lam_rel),condition=float(np.linalg.cond(gram)),pairs=P.shape[1])
    return (F,info) if return_info else F
