"""Controlled high-dimensional synthetic systems and trajectories for NNDR."""
from __future__ import annotations
import numpy as np
from .simulate import simulate_var_with_fixed_noise


def embedded_map(L, *, dimension=12, dt=0.1, background_rates=None, seed=0):
    """Return (F, Q_true) for a stable 2-D generator embedded orthogonally.

    L : real (2,2) continuous-time stable generator.
    dimension : ambient dimension, >= 2.
    dt : positive sampling interval; complex frequencies must be below Nyquist.
    background_rates : positive decay rates, shape (dimension-2,).
        Default: evenly spaced from 1.2 to 1.8 in inverse time units.
    seed : seed for the orthogonal embedding.
    Requires SciPy. Q_true identifies a plane, not a unique directed basis.
    """
    from scipy.linalg import expm
    L = np.asarray(L)
    if L.shape != (2, 2) or not np.isrealobj(L) or not np.all(np.isfinite(L)):
        raise ValueError('L must be a finite real 2x2 generator')
    if not isinstance(dimension, (int, np.integer)) or dimension < 2:
        raise ValueError('dimension must be an integer >= 2')
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError('dt must be finite and positive')
    rates = np.linalg.eigvals(L)
    if np.any(rates.real >= 0) or np.any(np.abs(rates.imag)*dt >= np.pi):
        raise ValueError('require stable generator and principal-log sampling')
    bg = (np.linspace(1.2, 1.8, dimension-2) if background_rates is None
          else np.asarray(background_rates, dtype=float))
    if bg.shape != (dimension-2,) or not np.all(np.isfinite(bg)) or np.any(bg <= 0):
        raise ValueError('background_rates must contain dimension-2 positive rates')
    U, R = np.linalg.qr(np.random.default_rng(seed).standard_normal((dimension, dimension)))
    U = U * np.where(np.diag(R) < 0, -1., 1.)
    B = np.zeros((dimension, dimension))
    B[:2, :2] = expm(L*dt)
    B[2:, 2:] = np.diag(np.exp(-bg*dt))
    F = U @ B @ U.T
    if not np.all(np.isfinite(F)) or max(abs(np.linalg.eigvals(F))) >= 1:
        raise ValueError('sampled map is not numerically Schur stable')
    return F, U[:, :2].copy()


def simulate_trajectory(F, *, samples, noise_std=0.1, burn_in=500, seed=0):
    """Simulate a stable VAR(1) with isotropic Gaussian innovations.

    Return (channels, samples), after discarding burn_in samples from zero.
    All samples are observed states; no artificial transition from zero is
    included in subsequent regression. noise_std is per-step innovation SD,
    not a continuous-time diffusion coefficient or measurement noise.
    """
    F = np.asarray(F)
    if (F.ndim != 2 or F.shape[0] != F.shape[1] or F.shape[0] < 2
        or not np.isrealobj(F) or not np.all(np.isfinite(F))):
        raise ValueError('F must be a finite real square map of dimension >=2')
    if max(abs(np.linalg.eigvals(F))) >= 1:
        raise ValueError('F must be Schur stable')
    if not isinstance(samples, (int, np.integer)) or samples < 2:
        raise ValueError('samples must be an integer >=2')
    if not isinstance(burn_in, (int, np.integer)) or burn_in < 0:
        raise ValueError('burn_in must be a nonnegative integer')
    if not np.isfinite(noise_std) or noise_std <= 0:
        raise ValueError('noise_std must be finite and positive')
    noise = np.random.default_rng(seed).normal(0., noise_std, (samples+burn_in, F.shape[0]))
    X = simulate_var_with_fixed_noise(F, noise)
    if not np.all(np.isfinite(X)):
        raise ValueError('simulation produced non-finite states')
    return X[:, burn_in:].copy()
