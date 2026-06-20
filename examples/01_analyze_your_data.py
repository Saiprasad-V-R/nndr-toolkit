"""Example 1 — Analyze your own multivariate time series.

This is the most common use case: you have a recording with several channels
and you want the NNDR diagnostics for it.

Run with:
    python examples/01_analyze_your_data.py
"""
import numpy as np

from nndr import analyze
from nndr.utils import principal_angle_deg


def make_demo_series(N=12, T=800, q=3.5, seed=0):
    """Stand-in for your real data: a VAR(1) series with an embedded
    non-normal block. Replace this with your own (N_channels, T_samples) array.
    """
    rng = np.random.default_rng(seed)
    a, b = 0.6, 0.2
    block = np.array([[a, q * (a - b)], [0.0, b]])
    bg = np.diag(rng.uniform(0.1, 0.5, size=N - 2))
    full = np.zeros((N, N))
    full[:2, :2] = block
    full[2:, 2:] = bg
    Qrot, _ = np.linalg.qr(rng.standard_normal((N, N)))
    F = Qrot @ full @ Qrot.T

    X = np.zeros((N, T))
    x = np.zeros(N)
    for k in range(T):
        x = F @ x + 0.05 * rng.standard_normal(N)
        X[:, k] = x
    return X


def main():
    # ---- Load your data here instead -------------------------------------
    # X = np.load("my_recording.npy")   # shape (N_channels, T_samples)
    X = make_demo_series()
    print(f"Data shape: {X.shape}  (N_channels, T_samples)\n")

    # ---- Run NNDR with the recommended method (M2) -----------------------
    res = analyze(X, method="M2")
    print("Optimization-based extraction (M2):")
    print("  ", res.summary())
    print()

    # ---- Cross-check with the commutator method (M3) ---------------------
    res3 = analyze(X, method="M3")
    print("Commutator-based extraction (M3):")
    print("  ", res3.summary())
    print()

    angle = principal_angle_deg(res.Q, res3.Q)
    print(f"Principal angle between the M2 and M3 planes: {angle:.2f} degrees")
    print("(small angle => the two independent methods agree on the plane)")


if __name__ == "__main__":
    main()
