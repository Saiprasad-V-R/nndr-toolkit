"""Revised NNDR thresholds for a real 2-by-2 discrete-time compression.

R_A: delayed variance-response criterion of revised Appendices B and C.
R_sigma: exact discrete Euclidean one-step growth criterion.
No stability rescaling, spectral clipping or eigenvector-overlap clipping.
"""
from __future__ import annotations

import numpy as np


def threshold_diagnostics(Gamma, dt=1.0):
    """Compute both revised ratios, including the repeated-eigenvalue limit.

    dt is the positive sample interval; its units determine the rates.
    R_A uses the real principal logarithm, where it exists. R_sigma requires
    only Schur stability. Undefined quantities are NaN, not zero.
    """
    G = np.asarray(Gamma)
    if G.shape != (2, 2) or not np.isrealobj(G) or not np.all(np.isfinite(G)):
        raise ValueError("Gamma must be a finite real 2 x 2 matrix")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be finite and positive")
    G = G.astype(float)
    eig = np.linalg.eigvals(G).astype(complex)
    l1, l2 = eig
    rho = float(np.max(np.abs(eig)))
    sig = np.linalg.svd(G, compute_uv=False)
    gap = float(abs(l1-l2))
    complex_pair = bool(np.any(eig.imag != 0))
    # Algebraic forms of Henrici departure avoid subtracting two nearly
    # identical squared Frobenius norms near normality.
    a,b = G[0]; c,d = G[1]
    departure = float(np.hypot(a-d, b+c) if complex_pair else abs(b-c))
    K = float(departure/gap) if gap else (np.inf if departure else 0.0)
    stable = rho < 1.0
    out = dict(R_A=np.nan, R_sigma=np.nan, Kc_A=np.nan, Kc_sigma=np.nan,
               Delta_rate=np.nan, beta=np.nan, rate_splitting=np.nan,
               rate_admissible=False, rate_status="unstable_reduced",
               stable_reduced=stable, spectrum_type=("complex_pair" if complex_pair
                    else "repeated_real" if gap == 0 else "real_pair"),
               K=K, dF=departure, rhoGamma=rho, sigmaGamma=float(sig[0]),
               sigmaGamma_min=float(sig[-1]),
               stable_amplifying=bool(stable and sig[0] > 1),
               rates=np.full(2, np.nan+0j), dt=float(dt))
    if not stable:
        return out
    mod = np.abs(eig)
    # Factor each stability margin before taking the geometric mean.
    margins = (1-mod)*(1+mod)
    margin = float(np.sqrt(margins[0])*np.sqrt(margins[1]))
    out.update(R_sigma=departure/margin,
               Kc_sigma=margin/gap if gap else np.inf)
    if np.any((eig.imag == 0) & (eig.real <= 0)):
        out['rate_status'] = 'no_real_principal_log'
        return out
    # Use unscaled log multipliers internally: ratios are time-scale invariant.
    logs = np.log(eig)
    g1,g2 = -logs.real
    if min(g1,g2) <= 0:
        out['rate_status'] = 'nonnegative_decay_rate'
        return out
    # d_F(log G) = |divided difference of log| d_F(G), with the
    # derivative at repeated positive eigenvalues (including Jordan blocks).
    if complex_pair:
        theta = float(abs(logs[0].imag))
        gamma = float((g1+g2)/2)
        beta = theta/gamma
        s = np.hypot(1.0,beta)
        dd = abs(logs[0]-logs[1])/gap
        ratio = departure*dd/(2*gamma*np.sqrt(s+1))
        kc = np.sqrt(s+1)/beta if beta else np.inf
        out.update(beta=float(beta), rate_splitting=float(beta))
    else:
        x,y = sorted(eig.real)
        dd = (np.log1p((y-x)/x)/(y-x) if y != x else 1/x)
        # For extreme eigenvalue ratios log1p can overflow; use the logs.
        if not np.isfinite(dd):
            dd = abs(logs[0]-logs[1])/gap
        total = float(g1+g2)
        delta = abs(g1-g2)/total
        s = 2*np.sqrt(g1)*np.sqrt(g2)/total
        factor = np.sqrt(s*(1+s))
        ratio = departure*dd/(total*factor)
        kc = factor/delta if delta else np.inf
        out.update(Delta_rate=float(delta), rate_splitting=float(delta))
    out.update(R_A=float(ratio), Kc_A=float(kc), rate_admissible=True,
               rate_status='ok', rates=logs/dt)
    return out
