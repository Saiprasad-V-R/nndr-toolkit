"""Input validation with clear, actionable error messages.

These helpers catch the mistakes real users make most often -- passing a 1-D
array, transposing the data, including NaNs/Infs, or asking for a window longer
than the recording -- and explain how to fix them, rather than failing later
with a cryptic NumPy error.
"""
from __future__ import annotations

import numpy as np

__all__ = ["validate_time_series", "validate_operator", "validate_window"]


def validate_time_series(X, *, name: str = "X", min_samples: int = 3) -> np.ndarray:
    """Validate and return a multivariate time series of shape ``(N, T)``.

    Parameters
    ----------
    X:
        The candidate data. Must be a 2-D array with ``N_channels`` rows and
        ``T_samples`` columns.
    name:
        Name used in error messages (e.g. ``"X"``).
    min_samples:
        Minimum number of time samples required.

    Raises
    ------
    ValueError
        If ``X`` is not 2-D, has too few samples, contains non-finite values,
        or looks transposed (more channels than samples).
    """
    arr = np.asarray(X, dtype=float)

    if arr.ndim != 2:
        raise ValueError(
            f"{name} must be a 2-D array of shape (N_channels, T_samples); "
            f"got an array with {arr.ndim} dimension(s) and shape {arr.shape}. "
            f"If you have a single channel, reshape it to (1, T)."
        )

    N, T = arr.shape

    if N == 0 or T == 0:
        raise ValueError(
            f"{name} has shape {arr.shape}; both dimensions must be non-empty."
        )

    if T < min_samples:
        raise ValueError(
            f"{name} has only {T} time sample(s) (shape {arr.shape}); at least "
            f"{min_samples} are required. NNDR expects shape "
            f"(N_channels, T_samples) -- did you pass the data transposed?"
        )

    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN or Inf). "
            f"Remove or interpolate them before calling NNDR."
        )

    if N > T:
        raise ValueError(
            f"{name} has more channels ({N}) than time samples ({T}) "
            f"(shape {arr.shape}). NNDR expects shape (N_channels, T_samples). "
            f"This usually means the array is transposed; pass X.T instead. "
            f"If you really have more channels than samples, the local operator "
            f"cannot be identified from this window."
        )

    return arr


def validate_operator(A, *, name: str = "A") -> np.ndarray:
    """Validate and return a square 2-D operator.

    Raises
    ------
    ValueError
        If ``A`` is not a finite square 2-D matrix.
    """
    arr = np.asarray(A, dtype=float)

    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError(
            f"{name} must be a square 2-D matrix (N x N); got shape {arr.shape}."
        )

    if not np.all(np.isfinite(arr)):
        n_bad = int(np.sum(~np.isfinite(arr)))
        raise ValueError(
            f"{name} contains {n_bad} non-finite value(s) (NaN or Inf)."
        )

    return arr


def validate_window(window: int, T: int, *, step: int = 1) -> None:
    """Validate rolling-window parameters against the series length ``T``.

    Raises
    ------
    ValueError
        If ``window`` or ``step`` is not a positive integer, or if ``window``
        exceeds ``T``.
    """
    if not isinstance(window, (int, np.integer)) or window <= 0:
        raise ValueError(f"window must be a positive integer; got {window!r}.")

    if not isinstance(step, (int, np.integer)) or step <= 0:
        raise ValueError(f"step must be a positive integer; got {step!r}.")

    if window > T:
        raise ValueError(
            f"window ({window}) is larger than the series length ({T}). "
            f"Choose a window no longer than the number of time samples."
        )
