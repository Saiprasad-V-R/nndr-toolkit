"""Validation benchmark harness.

Reproduces the synthetic validation in the paper: build a known operator,
estimate it from finite data, run all three plane-extraction methods, and
measure how well each recovers the response plane and the ratio ``R``.

The known-operator result (computed with M2 on the true ``F``) is the
*reference* against which the estimated results are compared.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from .generator import build_F_from_config, make_fixed_config
from .methods import method1_eigenbasis_svd, method2_optimization, method3_commutator
from .observables import best_plane_corr
from .reduce2d import compute_indices_from_Q
from .ridge import ridge_fit_F, select_lambda_ridge, stack_VAR_pairs_from_trajectories
from .simulate import simulate_var_with_fixed_noise
from .utils import principal_angle_deg, spectral_radius

__all__ = ["BenchConfig", "run_benchmark"]


@dataclass
class BenchConfig:
    """Configuration for :func:`run_benchmark`."""

    rho: float = 0.95
    delta_sigma: float = 0.05
    sigma_noise: float = 0.05

    lambdas: np.ndarray = field(default_factory=lambda: np.logspace(-8, 2, 21))
    frac_train: float = 0.8

    cfg_seed_base: int = 1
    seed_train_bank: int = 111
    seed_test_noise: int = 222
    seed_cv: int = 333
    seed_opt_reference: int = 0
    seed_opt_est: int = 0

    iters_opt: int = 250


def run_benchmark(
    N_list: List[int],
    M_list: List[int],
    T_list: List[int],
    kappas: List[float],
    bc: BenchConfig,
    T_test: int = 800,
    out_csv: str = "benchmark_results.csv",
) -> pd.DataFrame:
    """Run the full synthetic validation sweep and return a tidy DataFrame.

    Parameters
    ----------
    N_list, M_list, T_list:
        Grids over state dimension, number of trajectories, and training
        horizon.
    kappas:
        Anisotropy values; larger ``kappa`` gives stronger non-normality.
    bc:
        A :class:`BenchConfig`.
    T_test:
        Length of the held-out trajectory used for the observable correlation.
    out_csv:
        Path to write the results CSV.
    """
    rows: List[Dict[str, Any]] = []

    for N in N_list:
        cfg = make_fixed_config(N, rho=bc.rho, delta_sigma=bc.delta_sigma, seed=bc.cfg_seed_base)

        rng_test = np.random.default_rng(bc.seed_test_noise + 10_000 * N)
        noise_test = bc.sigma_noise * rng_test.standard_normal((T_test, N))

        for M in M_list:
            for T_train in T_list:

                rng_bank = np.random.default_rng(
                    bc.seed_train_bank + 1_000_000 * N + 10_000 * M + 100 * T_train
                )
                noise_bank = bc.sigma_noise * rng_bank.standard_normal((M, T_train, N))

                for kappa in kappas:
                    F = build_F_from_config(cfg, kappa=kappa)

                    # Reference: Method 2 on the TRUE F
                    out_ref = method2_optimization(F, iters=bc.iters_opt, seed=bc.seed_opt_reference)
                    QT = out_ref["Q"]
                    idx_true = compute_indices_from_Q(F, QT)
                    KKc_TRUE = float(idx_true["KKc"])
                    Delta_TRUE = float(idx_true["Delta"])

                    X_list = []
                    for m in range(M):
                        Xm = simulate_var_with_fixed_noise(F, noise_bank[m], x0=np.zeros(N))
                        X_list.append(Xm)
                    Xstack, Ystack = stack_VAR_pairs_from_trajectories(X_list, x0=np.zeros(N))

                    lam_star, _ = select_lambda_ridge(
                        Xstack, Ystack, bc.lambdas, frac_train=bc.frac_train, seed=bc.seed_cv
                    )
                    Fhat = ridge_fit_F(Xstack, Ystack, lam_star)

                    relErr = float(
                        np.linalg.norm(Fhat - F, ord="fro") / (np.linalg.norm(F, ord="fro") + 1e-15)
                    )
                    rho_hat = spectral_radius(Fhat)

                    Xte = simulate_var_with_fixed_noise(F, noise_test, x0=np.zeros(N))
                    mte = Xte.mean(axis=0)

                    methods = [
                        method1_eigenbasis_svd(Fhat),
                        method2_optimization(Fhat, iters=bc.iters_opt, seed=bc.seed_opt_est),
                        method3_commutator(Fhat),
                    ]

                    for outm in methods:
                        Qm = outm["Q"]
                        idx_est = compute_indices_from_Q(Fhat, Qm)

                        angle = principal_angle_deg(QT, Qm)
                        c_plane, _, _ = best_plane_corr(Qm, Xte, mte)

                        row: Dict[str, Any] = {
                            "N": int(N), "M": int(M), "T_train": int(T_train), "T_test": int(T_test),
                            "kappa": float(kappa),
                            "lam_star": float(lam_star),
                            "relErr_Fhat": float(relErr),
                            "rho_Fhat": float(rho_hat),
                            "KKc_TRUE": KKc_TRUE,
                            "Delta_TRUE": Delta_TRUE,
                            "method": outm["name"],
                            "KKc_est": float(idx_est["KKc"]),
                            "Delta_est": float(idx_est["Delta"]),
                            "angle_deg": float(angle),
                            "corr_plane": float(c_plane),
                            "tr_Gamma": float(np.real_if_close(idx_est["trG"])),
                            "ok_Gamma": bool(idx_est["okG"]),
                        }

                        if outm["name"] == "M1_BF":
                            row["kappa_P"] = float(outm["kappa_P"])
                        if outm["name"] == "M3_COMM":
                            row["lamB_max"] = float(outm["lamB_max"])
                            row["lamB_min"] = float(outm["lamB_min"])
                            row["okB"] = bool(outm["okB"])

                        rows.append(row)

                print(f"[done] N={N}, M={M}, T_train={T_train} (kappas={len(kappas)})")

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"Saved results to: {out_csv}")
    return df
