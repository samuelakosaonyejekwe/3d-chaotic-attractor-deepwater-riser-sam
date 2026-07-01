"""
chronos.plots
=============
Central plotting utilities for the CHRONOS post-processing suite.

Design rule (explicit project constraint): **the colour black is never used**
for any line, marker, text, axis, edge, colormap extreme or background.
A dark-navy/slate ink (#1F3A5F) replaces black for all text/axes, and a
vivid, colour-blind-aware palette drives every data series. Sequential fields
use ``turbo`` / ``viridis`` / ``plasma`` (whose extremes are deep blue/purple,
not black).

Every public function returns the saved file path, so the case-study driver
can collate them into the report.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import cm
from cycler import cycler
from mpl_toolkits.mplot3d import Axes3D  # noqa

# --------------------------------------------------------------------------- #
#  Global style -- NO BLACK anywhere
# --------------------------------------------------------------------------- #
INK = "#1F3A5F"          # dark slate-navy ink (replaces black)
GRID = "#C9D6E3"
PALETTE = [
    "#1B6CA8",  # blue
    "#E8871E",  # orange
    "#2A9D4A",  # green
    "#C42348",  # crimson
    "#6A4C93",  # purple
    "#0FA3A3",  # teal
    "#F2C14E",  # gold
    "#D45113",  # burnt orange
    "#3A7CA5",  # steel blue
    "#8E44AD",  # violet
]

plt.rcParams.update({
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.edgecolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.prop_cycle": cycler(color=PALETTE),
    "axes.titlesize": 11,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 8,
    "lines.linewidth": 1.4,
    "figure.dpi": 130,
})

SEQ = "turbo"      # sequential colormap (deep-blue..red, no black)
SEQ2 = "viridis"


def _finish(fig, path, tight=True):
    if tight:
        fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def _legend_outside(ax, ncol=1):
    """Place the legend OUTSIDE the axes (to the right) so it never overlaps
    the data curves or the title. bbox_inches='tight' keeps it in the saved
    figure."""
    leg = ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5),
                    frameon=True, framealpha=0.96, edgecolor=GRID, ncol=ncol,
                    borderaxespad=0.0)
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_color(INK)
    return leg


# --------------------------------------------------------------------------- #
#  Time series
# --------------------------------------------------------------------------- #
def time_series(t, series: dict, title, ylabel, path, xlabel="Time (s)",
                logy=False, lw=1.1, alpha=0.9):
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    for i, (lab, y) in enumerate(series.items()):
        ax.plot(t, y, label=lab, color=PALETTE[i % len(PALETTE)],
                lw=lw, alpha=alpha)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    ax.margins(x=0.01)
    if logy:
        ax.set_yscale("log")
    _legend_outside(ax, ncol=1)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
#  3-D phase portrait + 2-D projections
# --------------------------------------------------------------------------- #
def phase_portrait_3d(X, Y, Z, labels, title, path, color_by=None):
    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    if color_by is None:
        color_by = np.linspace(0, 1, len(X))
    ax.scatter(X, Y, Z, c=color_by, cmap=SEQ, s=1.2, alpha=0.7)
    ax.plot(X, Y, Z, color=PALETTE[0], lw=0.3, alpha=0.35)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.xaxis.pane.set_edgecolor(INK); ax.yaxis.pane.set_edgecolor(INK)
    ax.zaxis.pane.set_edgecolor(INK)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0.0))
    return _finish(fig, path, tight=False)


def phase_projections(X, Y, Z, labels, title, path):
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    pairs = [(X, Y, labels[0], labels[1], PALETTE[3]),
             (X, Z, labels[0], labels[2], PALETTE[2]),
             (Y, Z, labels[1], labels[2], PALETTE[0])]
    for ax, (a, b, la, lb, col) in zip(axs, pairs):
        ax.plot(a, b, color=col, lw=0.5, alpha=0.8)
        ax.set_xlabel(la); ax.set_ylabel(lb)
    fig.suptitle(title, color=INK)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
#  Poincare / return / recurrence
# --------------------------------------------------------------------------- #
def scatter_map(a, b, labels, title, path, c=None):
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    if c is None:
        c = PALETTE[4]
        ax.scatter(a, b, s=8, color=c, alpha=0.75, edgecolors="none")
    else:
        sc = ax.scatter(a, b, s=10, c=c, cmap=SEQ, alpha=0.85, edgecolors="none")
        fig.colorbar(sc, ax=ax, shrink=0.85)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_title(title)
    return _finish(fig, path)


def recurrence_plot(R, title, path):
    """Conventional recurrence plot: solid dark-navy recurrence points on a
    white field, square/equal aspect, line-of-identity along the main diagonal
    (the standard Eckmann-Kamphorst-Ruelle rendering, with navy replacing the
    usual black)."""
    from matplotlib.colors import ListedColormap
    binary = ListedColormap(["#FFFFFF", "#22336B"])   # white / dark navy
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.imshow(R, origin="lower", cmap=binary, aspect="equal",
              interpolation="none", vmin=0, vmax=1, resample=False)
    ax.set_xlabel("Time index  i"); ax.set_ylabel("Time index  j")
    ax.set_title(title)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_edgecolor(INK)
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
#  Spectral / wavelet
# --------------------------------------------------------------------------- #
def spectrum(f, amp, title, path, ylabel="Amplitude", logy=False,
             xlabel="Frequency (Hz)", mark_peak=True):
    fig, ax = plt.subplots(figsize=(6.6, 3.2))
    ax.plot(f, amp, color=PALETTE[5])
    if logy:
        ax.set_yscale("log")
    if mark_peak and len(f) > 1:
        pk = np.argmax(amp)
        ax.axvline(f[pk], color=PALETTE[1], ls="--", lw=1.0,
                   label=f"peak {f[pk]:.3f} Hz")
        _legend_outside(ax, ncol=1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return _finish(fig, path)


def scaleogram(t, freqs, power, title, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    im = ax.pcolormesh(t, freqs, power, cmap=SEQ, shading="auto")
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Frequency (Hz)"); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9, label="|CWT|")
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
#  Bifurcation / sensitivity / stability map
# --------------------------------------------------------------------------- #
def bifurcation(P, V, xlabel, ylabel, title, path):
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    ax.scatter(P, V, s=1.5, color=PALETTE[0], alpha=0.55, edgecolors="none")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    return _finish(fig, path)


def scatter_multi(x, series: dict, xlabel, ylabel, title, path, logy=False):
    """Scatter (points only, no connecting lines) for non-functional data such
    as Poincare crossings, return maps and bifurcation sweeps."""
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    for i, (lab, y) in enumerate(series.items()):
        ax.scatter(x, y, label=lab, color=PALETTE[i % len(PALETTE)],
                   s=5, alpha=0.55, edgecolors="none")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    if len(series) > 1:
        _legend_outside(ax, ncol=1)
    return _finish(fig, path)


_MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
_LINESTYLES = ["-", "--", "-.", ":"]


def line_multi(x, series: dict, xlabel, ylabel, title, path, logy=False,
               markers=False):
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    n = len(series)
    for i, (lab, y) in enumerate(series.items()):
        # distinct linestyle + marker per series so an overlapping (occluded)
        # series still shows through — e.g. a computed curve lying on top of a
        # reference curve remains visible via its dashes/markers.
        ax.plot(x, y, label=lab, color=PALETTE[i % len(PALETTE)],
                ls=_LINESTYLES[i % len(_LINESTYLES)], lw=1.5,
                alpha=0.9 if n == 1 else 0.8,
                marker=(_MARKERS[i % len(_MARKERS)] if markers else None),
                markersize=5,
                markerfacecolor="none" if i % 2 else PALETTE[i % len(PALETTE)],
                markeredgecolor=PALETTE[i % len(PALETTE)], markeredgewidth=1.1)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if logy:
        ax.set_yscale("log")
    _legend_outside(ax, ncol=1)
    return _finish(fig, path)


def bar_matrix(categories, series: dict, title, path, ncol=3):
    """Clean small-multiple bar panels for a categorical property matrix.

    Each numeric property gets its own panel with an independent y-scale (so
    quantities of very different magnitude — e.g. density ~7800 vs damping
    ~0.01 — are all readable), bars coloured per category, values annotated,
    and a single shared legend of the (possibly long) category names.
    """
    names = list(series.keys())
    n = len(names)
    ncol = min(ncol, n)
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.55 * nrow))
    axes = np.atleast_1d(axes).ravel()
    xpos = np.arange(len(categories))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(categories))]
    for a, name in enumerate(names):
        ax = axes[a]
        vals = np.asarray(series[name], dtype=float)
        bars = ax.bar(xpos, vals, color=colors, edgecolor=INK, linewidth=0.6)
        ax.set_title(name, fontsize=9.5, color=INK)
        ax.set_xticks(xpos)
        ax.set_xticklabels([])
        ax.margins(y=0.20)
        ax.grid(axis="x", visible=False)
        for b, v in zip(bars, vals):
            ax.annotate(f"{v:g}", (b.get_x() + b.get_width() / 2, v),
                        ha="center", va="bottom", fontsize=7.5, color=INK,
                        xytext=(0, 1.5), textcoords="offset points")
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    handles = [mpatches.Patch(facecolor=colors[i], edgecolor=INK,
                              label=str(categories[i]))
               for i in range(len(categories))]
    fig.legend(handles=handles, loc="lower center",
               ncol=min(len(categories), 3), fontsize=8.5, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(title, fontsize=11, color=INK)
    fig.tight_layout(rect=[0, 0.07, 1, 0.95])
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def heatmap(xv, yv, Z, xlabel, ylabel, title, path, clabel="LLE"):
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    im = ax.pcolormesh(xv, yv, Z, cmap=SEQ, shading="auto")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.9, label=clabel)
    return _finish(fig, path)


def bar_chart(labels, values, ylabel, title, path, colors=None, logy=False,
              horizontal=False):
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    colors = colors or [PALETTE[i % len(PALETTE)] for i in range(len(labels))]
    if horizontal:
        ax.barh(labels, values, color=colors, edgecolor=INK, linewidth=0.6)
        ax.set_xlabel(ylabel)
        if logy:
            ax.set_xscale("log")
    else:
        ax.bar(labels, values, color=colors, edgecolor=INK, linewidth=0.6)
        ax.set_ylabel(ylabel)
        if logy:
            ax.set_yscale("log")
    ax.set_title(title)
    fig.autofmt_xdate(rotation=20)
    return _finish(fig, path)


def rainflow_hist(centres, hist, title, path):
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    width = (centres[1]-centres[0]) if len(centres) > 1 else 1.0
    ax.bar(centres, hist, width=width*0.9, color=PALETTE[2],
           edgecolor=INK, linewidth=0.5)
    ax.set_xlabel("Stress range (MPa)"); ax.set_ylabel("Counted cycles")
    ax.set_title(title)
    return _finish(fig, path)


def scatter3d(x, y, z, labels, title, path, c=None, clabel=""):
    fig = plt.figure(figsize=(6.6, 5.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(x, y, z, c=c if c is not None else z, cmap=SEQ, s=14)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_title(title)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=clabel)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((1, 1, 1, 0.0)); pane.set_edgecolor(INK)
    return _finish(fig, path, tight=False)


def convergence(hist, title, path, names=None):
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    hist = np.asarray(hist)
    if hist.ndim == 1:
        hist = hist[:, None]
    for j in range(hist.shape[1]):
        lab = names[j] if names else f"lambda_{j+1}"
        ax.plot(hist[:, j], label=lab, color=PALETTE[j % len(PALETTE)])
    ax.axhline(0.0, color=PALETTE[3], ls=":", lw=1.0)
    ax.set_xlabel("Renormalisation block"); ax.set_ylabel("Lyapunov exponent")
    ax.set_title(title); _legend_outside(ax, ncol=1)
    return _finish(fig, path)


def current_profile_plot(z, u, title, path):
    fig, ax = plt.subplots(figsize=(3.8, 5.0))
    ax.plot(u, -z, color=PALETTE[0], lw=1.8)
    ax.fill_betweenx(-z, 0, u, color=PALETTE[0], alpha=0.15)
    ax.set_xlabel("Current speed (m/s)"); ax.set_ylabel("Depth below surface (m)")
    ax.set_title(title)
    return _finish(fig, path)


# --------------------------------------------------------------------------- #
#  LaTeX (mathtext) equation rendering  -- for the case-study equation list
# --------------------------------------------------------------------------- #
def render_math(lines, path, fontsize=13.5, width_in=6.3, color=INK,
                line_height=0.52):
    """
    Render one or more LaTeX equations to a PNG at a *standard, uniform* font
    size using matplotlib's mathtext (Computer Modern), navy ink (never black),
    on a white background. Every equation image is produced at the same canvas
    width and font size so the on-page glyph size is consistent.

    ``lines`` is a list of LaTeX strings (with or without surrounding '$').
    ``line_height`` (inches/line) is increased for blocks containing tall
    fractions / summations so nothing overlaps.
    """
    n = len(lines)
    h = line_height * n + 0.22
    fig = plt.figure(figsize=(width_in, h))
    fig.patch.set_facecolor("white")
    # evenly distribute lines across the full canvas height: line i occupies
    # row [i/n, (i+1)/n] and is vertically centred in it. Because the figure
    # height already encodes line_height inches per line, the physical spacing
    # is exactly line_height and lines never overlap.
    for i, ln in enumerate(lines):
        s = ln if ln.strip().startswith("$") else f"${ln}$"
        y = 1.0 - (i + 0.5) / n
        fig.text(0.03, y, s, fontsize=fontsize, color=color,
                 ha="left", va="center", math_fontfamily="cm")
    fig.savefig(path, dpi=150, facecolor="white")
    plt.close(fig)
    return path, width_in
