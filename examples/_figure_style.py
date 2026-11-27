"""Shared manuscript-style defaults for the NNDR example gallery."""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Ellipse, Circle, FancyArrowPatch

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 9.2,
        "axes.labelsize": 9.2,
        "axes.titlesize": 10.2,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.80,
        "lines.linewidth": 1.35,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.major.size": 4.0,
        "ytick.major.size": 4.0,
        "xtick.minor.size": 2.0,
        "ytick.minor.size": 2.0,
        "savefig.dpi": 550,
        "figure.dpi": 170,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "legend.frameon": True,
        "legend.facecolor": "white",
        "legend.framealpha": 0.86,
        "legend.edgecolor": "none",
    })

def format_axis(ax, nbx=5, nby=5) -> None:
    ax.minorticks_on()
    ax.tick_params(which="minor", direction="in", top=True, right=True, length=2)
    ax.tick_params(which="major", direction="in", top=True, right=True, length=4)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nbx))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=nby))

def panel_label(ax, label: str, x=-0.18, y=1.09) -> None:
    ax.text(
        x, y, f"({label})",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=10.7,
        fontweight="bold",
        clip_on=False,
    )

# Schematic specific drawing helpers
def ell(ax, cx, cy, w, h, ang, fc, ec, al=0.80, lw=1.0, z=4):
    ax.add_patch(Ellipse((cx, cy), w, h, angle=ang, fc=fc, ec=ec, alpha=al, lw=lw, zorder=z))

def dcirc(ax, cx, cy, r, z=3):
    ax.add_patch(Circle((cx, cy), r, fill=False, ls='--', ec='#888888', lw=0.9, zorder=z))

def arr(ax, x0, y0, x1, y1, col='#222222', lw=1.2, rad=0.0, ms=10, z=6):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
        connectionstyle=f'arc3,rad={rad}', arrowstyle='-|>',
        mutation_scale=ms, color=col, lw=lw, zorder=z))

# Plot Colors
BLUE = "#0B3C6D"
RED = "#D62728"
GREEN = "#2CA02C"

# Schematic Specific Colors
C_RED='#B03A2E'; C_BLU='#1A5276'; C_PUR='#6C3483'
C_ORG='#A04000'; C_GRN='#1E8449'
F_RED='#F1948A'; F_BLU='#AED6F1'; F_PUR='#D2B4DE'


def save_figure(fig, out, stem):
    """Export vector PDF/SVG and a 550-dpi PNG."""
    for ext in ("png", "svg", "pdf"):
        fig.savefig(out / (stem + "." + ext))
    plt.close(fig)
