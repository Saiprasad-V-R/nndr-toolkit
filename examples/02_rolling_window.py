"""Example 2 — Rolling-window analysis of a time-varying recording.

When the local dynamics drift over time (as in most real recordings), slide a
window across the series and track how R, Delta, and K evolve. This mirrors the
rolling-window construction used for the empirical examples in the paper.

Run with:
    python examples/02_rolling_window.py
"""
import numpy as np

from nndr import rolling_analyze


def make_time_varying_series(N=10, T=2000, seed=0):
    """A series whose non-normal coupling ramps up in the middle, so R should
    rise and then fall. Replace with your own (N_channels, T_samples) array.
    """
    rng = np.random.default_rng(seed)
    a, b = 0.6, 0.2
    Qrot, _ = np.linalg.qr(rng.standard_normal((N, N)))

    X = np.zeros((N, T))
    x = np.zeros(N)
    for k in range(T):
        # coupling q rises to a peak near the middle of the recording
        frac = k / T
        q = 0.5 + 5.0 * np.exp(-((frac - 0.5) ** 2) / (2 * 0.08 ** 2))
        block = np.array([[a, q * (a - b)], [0.0, b]])
        bg = np.diag(rng.uniform(0.1, 0.5, size=N - 2))
        full = np.zeros((N, N))
        full[:2, :2] = block
        full[2:, 2:] = bg
        F = Qrot @ full @ Qrot.T
        x = F @ x + 0.05 * rng.standard_normal(N)
        X[:, k] = x
    return X


def main():
    X = make_time_varying_series()
    print(f"Data shape: {X.shape}\n")

    records = rolling_analyze(X, window=300, step=25, method="M2", shrinkage=0.02)

    print(f"{'t_center':>9}  {'R':>7}  {'Delta':>7}  {'K':>7}  ok")
    print("-" * 40)
    for r in records:
        flag = "" if r["ok"] else "  <-- degenerate"
        print(f"{r['t_center']:>9}  {r['R']:>7.3f}  {r['Delta']:>7.3f}  "
              f"{r['K']:>7.3f}  {str(r['ok']):>5}{flag}")

    Rs = [r["R"] for r in records if r["ok"]]
    if Rs:
        i_peak = int(np.argmax(Rs))
        print(f"\nPeak R = {max(Rs):.3f} (the coupling was designed to peak mid-record).")

    # To plot (requires matplotlib):
    #   import matplotlib.pyplot as plt
    #   t = [r["t_center"] for r in records]
    #   R = [r["R"] for r in records]
    #   plt.plot(t, R); plt.axhline(1.0, ls="--"); plt.xlabel("t"); plt.ylabel("R")
    #   plt.show()


if __name__ == "__main__":
    main()
