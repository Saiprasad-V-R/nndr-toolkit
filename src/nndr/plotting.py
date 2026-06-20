"""Plotting helpers for benchmark results.

These operate on the tidy DataFrame produced by
:func:`nndr.benchmark.run_benchmark` and save figures to disk.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = [
    "ensure_dir",
    "nice_symmetric_ylim",
    "plot_median_vs_x",
    "plot_zoom_vs_x",
    "add_R_error_columns",
    "plot_threshold_misclass_vs_x",
]


def ensure_dir(path) -> None:
    """Create ``path`` (and parents) if it does not already exist."""
    if path:
        os.makedirs(path, exist_ok=True)


def nice_symmetric_ylim(x: np.ndarray, pad_frac: float = 0.15, min_span: float = 1e-3):
    """Symmetric y-limits around 0 from a robust scale (median absolute value)."""
    x = np.asarray(x)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return (-1, 1)
    s = np.median(np.abs(x))
    span = max(min_span, (1.0 + pad_frac) * 3.0 * s)
    return (-span, span)


def plot_median_vs_x(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    title: str,
    ylabel: str,
    outpath: str,
    methods: Optional[Tuple[str, ...]] = None,
) -> None:
    """Plot ``median(ycol)`` vs ``xcol`` for each method."""
    plt.figure()
    meths = methods if methods is not None else tuple(sorted(df["method"].unique()))
    for method in meths:
        if method not in set(df["method"].unique()):
            continue
        dm = df[df["method"] == method].copy()
        g = dm.groupby(xcol)[ycol].median().reset_index().sort_values(xcol)
        plt.plot(g[xcol], g[ycol], marker="o", label=method)

    plt.xlabel(xcol)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def plot_zoom_vs_x(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    methods: Tuple[str, ...],
    title: str,
    ylabel: str,
    outpath: str,
    ylim=None,
) -> None:
    """Plot ``median(ycol)`` vs ``xcol`` for selected methods with optional tight ylim."""
    plt.figure()
    for method in methods:
        if method not in set(df["method"].unique()):
            continue
        dm = df[df["method"] == method].copy()
        g = dm.groupby(xcol)[ycol].median().reset_index().sort_values(xcol)
        plt.plot(g[xcol], g[ycol], marker="o", label=method)

    plt.xlabel(xcol)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    if ylim is not None:
        plt.ylim(ylim)

    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()


def add_R_error_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add absolute, signed, and relative recovery-error columns for ``R``.

    Adds ``R_abs_err``, ``R_err`` and ``R_rel_err`` based on ``KKc_est`` versus
    ``KKc_TRUE``.
    """
    dd = df.copy()
    dd["R_abs_err"] = (dd["KKc_est"] - dd["KKc_TRUE"]).abs()
    dd["R_err"] = dd["KKc_est"] - dd["KKc_TRUE"]
    dd["R_rel_err"] = dd["R_err"] / np.maximum(np.abs(dd["KKc_TRUE"]), 1e-12)
    return dd


def plot_threshold_misclass_vs_x(
    df: pd.DataFrame,
    xcol: str,
    title: str,
    outpath: str,
    methods: Optional[Tuple[str, ...]] = None,
) -> None:
    """Plot the threshold-classification error rate vs ``xcol``.

    The classification asks whether the estimated and true ratios fall on the
    same side of the threshold ``R = 1``, i.e. whether
    ``I[R_est > 1] != I[R_true > 1]``.
    """
    dd = df.copy()
    dd["y_true"] = (dd["KKc_TRUE"] > 1.0).astype(int)
    dd["y_hat"] = (dd["KKc_est"] > 1.0).astype(int)
    dd["mis"] = (dd["y_true"] != dd["y_hat"]).astype(float)

    plt.figure()
    meths = methods if methods is not None else tuple(sorted(dd["method"].unique()))
    for method in meths:
        if method not in set(dd["method"].unique()):
            continue
        dm = dd[dd["method"] == method]
        g = dm.groupby(xcol)["mis"].mean().reset_index().sort_values(xcol)
        plt.plot(g[xcol], g["mis"], marker="o", label=method)

    plt.xlabel(xcol)
    plt.ylabel("threshold misclassification rate")
    plt.title(title)
    plt.legend()
    ensure_dir(os.path.dirname(outpath))
    plt.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close()
