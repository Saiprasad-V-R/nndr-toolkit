"""Example 3 — Reproduce a (small) synthetic validation benchmark.

This rebuilds a miniature version of the paper's synthetic validation: known
operators are estimated from finite data and each method's recovery of the
plane and of R is measured. Requires the optional dependencies:

    pip install "nndr-toolkit[benchmark]"

Run with:
    python examples/03_reproduce_benchmark.py
"""
import numpy as np

from nndr.benchmark import BenchConfig, run_benchmark


def main():
    # A small grid so this runs in a few seconds. Widen the lists to reproduce
    # the full sweeps from the paper.
    kappas = list(np.logspace(0.0, 2.0, 12))

    df = run_benchmark(
        N_list=[60],
        M_list=[25],
        T_list=[200],
        kappas=kappas,
        bc=BenchConfig(),
        T_test=400,
        out_csv="benchmark_results_demo.csv",
    )

    print("\nMedian plane error (degrees) by method:")
    print(df.groupby("method")["angle_deg"].median().round(2).to_string())

    print("\nMedian |R_est - R_true| by method:")
    df["R_abs_err"] = (df["KKc_est"] - df["KKc_TRUE"]).abs()
    print(df.groupby("method")["R_abs_err"].median().round(4).to_string())

    print("\nWrote full results to benchmark_results_demo.csv")
    print("(M2 and M3 should show small plane error; M1 is the fragile baseline.)")


if __name__ == "__main__":
    main()
