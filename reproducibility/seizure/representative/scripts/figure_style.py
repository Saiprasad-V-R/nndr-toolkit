from __future__ import annotations

import copy

import matplotlib as mpl
import matplotlib.ticker as mticker


_METHOD_STYLE_BASE = {
    "M1_BF": {
        "label": "Eigenbasis (M1)",
        "color": "#0072B2",
        "linestyle": ":",
        "marker": "s",
        "lw": 1.15,
        "alpha": 0.95,
    },
    "M2_OPT": {
        "label": "Optimization (M2)",
        "color": "#D55E00",
        "linestyle": "-",
        "marker": "o",
        "lw": 1.35,
        "alpha": 1.0,
    },
    "M3_COMM": {
        "label": "Commutator (M3)",
        "color": "#009E73",
        "linestyle": "--",
        "marker": "^",
        "lw": 1.35,
        "alpha": 1.0,
    },
}

METHOD_STYLE = {
    **_METHOD_STYLE_BASE,
    "M1": copy.deepcopy(_METHOD_STYLE_BASE["M1_BF"]),
    "M2": copy.deepcopy(_METHOD_STYLE_BASE["M2_OPT"]),
    "M3": copy.deepcopy(_METHOD_STYLE_BASE["M3_COMM"]),
}


def configure_style(
    *,
    font_size: float = 9.5,
    axes_labelsize: float | None = None,
    axes_titlesize: float = 10.0,
    xtick_labelsize: float = 8.0,
    ytick_labelsize: float = 8.0,
    legend_fontsize: float = 7.0,
    lines_linewidth: float = 1.4,
    axes_linewidth: float = 0.8,
    tick_major_width: float = 0.8,
    tick_minor_width: float = 0.6,
    tick_major_size: float = 4.0,
    tick_minor_size: float = 2.0,
    savefig_dpi: int = 500,
    figure_dpi: int | None = None,
) -> None:
    if axes_labelsize is None:
        axes_labelsize = font_size

    rc = {
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": font_size,
        "axes.labelsize": axes_labelsize,
        "axes.titlesize": axes_titlesize,
        "xtick.labelsize": xtick_labelsize,
        "ytick.labelsize": ytick_labelsize,
        "legend.fontsize": legend_fontsize,
        "axes.linewidth": axes_linewidth,
        "lines.linewidth": lines_linewidth,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.width": tick_major_width,
        "ytick.major.width": tick_major_width,
        "xtick.minor.width": tick_minor_width,
        "ytick.minor.width": tick_minor_width,
        "xtick.major.size": tick_major_size,
        "ytick.major.size": tick_major_size,
        "xtick.minor.size": tick_minor_size,
        "ytick.minor.size": tick_minor_size,
        "savefig.dpi": savefig_dpi,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.86,
        "legend.edgecolor": "none",
    }
    if figure_dpi is not None:
        rc["figure.dpi"] = figure_dpi
    mpl.rcParams.update(rc)


def box_axes(ax, *, line_width: float = 0.8) -> None:
    for side in ("left", "right", "bottom", "top"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(line_width)


def precision_ticks(
    ax,
    *,
    major_length: float = 4.0,
    minor_length: float = 2.0,
    top: bool = True,
    right: bool = True,
) -> None:
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=top, right=right, length=minor_length)
    ax.tick_params(which="major", direction="in", top=top, right=right, length=major_length)


def format_axis(
    ax,
    *,
    nbx: int = 5,
    nby: int = 5,
    logx: bool = False,
    logy: bool = False,
    set_x_locator: bool = True,
    set_y_locator: bool = True,
) -> None:
    box_axes(ax)
    precision_ticks(ax)
    if set_x_locator and not logx:
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=nbx))
    if set_y_locator and not logy:
        ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=nby))


def style_legend(legend) -> None:
    if legend is None:
        return
    frame = legend.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(0.86)
    frame.set_edgecolor("none")


def panel_label(
    ax,
    label: str,
    *,
    x: float = -0.18,
    y: float = 1.09,
    fontsize: float = 10.5,
    fontweight: str = "bold",
) -> None:
    text = label if label.startswith("(") else f"({label})"
    ax.text(
        x,
        y,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight=fontweight,
        clip_on=False,
    )
