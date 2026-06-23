from __future__ import annotations

from collections.abc import Callable, Sequence

from matplotlib import rcParams
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
import numpy as np
from scipy.optimize import curve_fit

from .elements import Element1D
from .solve import split_global_vector

_LATEX_PREAMBLE = r"""
\usepackage{amsmath}
\usepackage{amssymb}
"""

rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "text.latex.preamble": _LATEX_PREAMBLE,
    }
)

TAB_COLORS = [
    "tab:blue",
    "tab:orange",
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

DEFAULT_MARKERS = ["o", "^", "s", "d", "x", "+"]
DEFAULT_LINESTYLES = ["--"]
DEFAULT_SOLUTION_LINESTYLES = ["-", "-", "-", "-", "-"]


def profile_from_elements(
    elements: Sequence[Element1D], global_u: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-element nodal profiles ``(x, u)`` in left-to-right order (not joined)."""
    sizes = [element.x.size for element in elements]
    local = split_global_vector(global_u, sizes)
    return [(element.x, u_local) for element, u_local in zip(elements, local)]


def plot_solution_profiles(
    profiles: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    legend_strings: Sequence[str],
    *,
    x_exact: np.ndarray | None = None,
    u_exact: np.ndarray | None = None,
    exact_label: str = "exact",
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    savefile: str | None = None,
    figsize: tuple[float, float] = (7, 4),
    title_size: int = 16,
    tick_size: int = 12,
    legendsize: int = 12,
    grid: bool = False,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    exact_color: str = "black",
    exact_linestyle: str = "-",
    exact_linewidth: float = 1.5,
    marker_size: float = 5,
    legendloc: str = "best",
) -> plt.Figure:
    """Overlay numerical solutions and an optional exact profile.

    Each entry in ``profiles`` is one run: a list of per-element ``(x, u)`` arrays.
    Segments are drawn separately so interface nodes are not connected across elements.
    """
    if len(profiles) != len(legend_strings):
        raise ValueError("profiles and legend_strings must have the same length")
    if not profiles:
        raise ValueError("At least one numerical profile is required")

    colors = list(colors) if colors is not None else list(TAB_COLORS)
    markers = list(markers) if markers is not None else list(DEFAULT_MARKERS)
    linestyles = (
        list(linestyles) if linestyles is not None else list(DEFAULT_SOLUTION_LINESTYLES)
    )

    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel or r"$x$", fontsize=title_size)
    ax.set_ylabel(ylabel or r"$u(x)$", fontsize=title_size)

    if x_exact is not None and u_exact is not None:
        ax.plot(
            x_exact,
            u_exact,
            color=exact_color,
            linestyle=exact_linestyle,
            linewidth=exact_linewidth,
            label=exact_label,
            zorder=1,
        )

    for i, (element_profiles, label) in enumerate(zip(profiles, legend_strings)):
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]
        for j, (x, u) in enumerate(element_profiles):
            ax.plot(
                x,
                u,
                linestyle=linestyle,
                color=color,
                marker=marker,
                markersize=marker_size,
                markevery=1,
                linewidth=1.2,
                label=label if j == 0 else None,
                zorder=2,
            )

    ax.legend(loc=legendloc, fontsize=legendsize)
    if grid:
        ax.grid(linestyle="--", color="gray", linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", labelsize=tick_size)
    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=600, bbox_inches="tight")

    return fig


def exact_profile_on_domain(
    exact_fun: Callable[[np.ndarray], np.ndarray],
    domain: tuple[float, float],
    num_points: int = 300,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample an exact solution on a uniform grid over the domain."""
    x = np.linspace(domain[0], domain[1], num_points)
    return x, np.asarray(exact_fun(x), dtype=float)


def _fit_log_log(dof: np.ndarray, err: np.ndarray) -> tuple[float, float]:
    """Least-squares log–log fit: log(err) = -slope * log(dof) + intercept."""

    def fit_func(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
        return -slope * x + intercept

    p_opt, _ = curve_fit(fit_func, np.log(dof), np.log(err), p0=(2.0, 0.0))
    return float(p_opt[0]), float(p_opt[1])


def _clean_series(dof: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    dof_out = np.copy(dof)
    err_out = np.copy(err)
    remove = 0
    for j in range(dof.size):
        k = j - remove
        if err_out[k] < 1e-16 or not np.isfinite(err_out[k]):
            dof_out = np.delete(dof_out, k)
            err_out = np.delete(err_out, k)
            remove += 1
    return dof_out, err_out


def plot_convergence(
    dof_vec: np.ndarray,
    err_vec: np.ndarray,
    legend_strings: Sequence[str],
    *,
    title: str | None = None,
    savefile: str | None = None,
    showslope: bool = True,
    skipfit_st: Sequence[int] | None = None,
    skipfit_end: Sequence[int | None] | None = None,
    skip: Sequence[int] | None = None,
    ylabel: str | None = None,
    xlabel: str | None = None,
    figsize: tuple[float, float] = (5, 4),
    title_size: int = 16,
    tick_size: int = 14,
    legendsize: int = 12,
    grid: bool = False,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    legendloc: str = "best",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Log-log convergence plot with optional slope fits in the legend.

    Parameters
    ----------
    dof_vec, err_vec
        Shape ``(n_cases, n_runs)`` or 1-D (single case). DOFs and errors at
        each mesh level.
    legend_strings
        One label per case.
    xlim, ylim
        Optional ``(min, max)`` axis limits in data coordinates (not log10).
    """
    dof_arr = np.asarray(dof_vec, dtype=float)
    err_arr = np.asarray(err_vec, dtype=float)
    if dof_arr.shape != err_arr.shape:
        raise ValueError("dof_vec and err_vec must have the same shape")
    if dof_arr.ndim == 1:
        dof_arr = dof_arr.reshape(1, -1)
        err_arr = err_arr.reshape(1, -1)

    n_cases, n_runs = dof_arr.shape
    if n_runs < 2:
        raise ValueError("Need at least two mesh levels for a convergence plot")

    if len(legend_strings) != n_cases:
        raise ValueError("legend_strings length must match number of cases")

    if skip is None:
        skip = [0] * n_cases
    if skipfit_st is None:
        skipfit_st = [0] * n_cases
    if skipfit_end is None:
        skipfit_end = [None] * n_cases
    skipfit_end = [-i if (i is not None and i != 0) else None for i in skipfit_end]

    colors = list(colors) if colors is not None else list(TAB_COLORS)
    markers = list(markers) if markers is not None else list(DEFAULT_MARKERS)
    linestyles = list(linestyles) if linestyles is not None else list(DEFAULT_LINESTYLES)

    fig = plt.figure(figsize=figsize)
    if title is not None:
        plt.title(title, fontsize=title_size)
    plt.ylabel(
        ylabel or r"$\| \boldsymbol{u}_h - u(\boldsymbol{x}) \|_\mathsf{H}$",
        fontsize=title_size,
    )
    plt.xlabel(xlabel or "DOF", fontsize=title_size)

    for i in range(n_cases):
        dof_mod, err_mod = _clean_series(dof_arr[i, skip[i] :], err_arr[i, skip[i] :])
        label = legend_strings[i]
        color = colors[i % len(colors)]
        marker = markers[i % len(markers)]
        linestyle = linestyles[i % len(linestyles)]

        fit_start = skipfit_st[i]
        fit_end = skipfit_end[i]
        dof_fit = dof_mod[fit_start:fit_end]
        err_fit = err_mod[fit_start:fit_end]

        slope_label = ""
        if len(dof_fit) > 2:
            slope, intercept = _fit_log_log(dof_fit, err_fit)
            if showslope:
                slope_label = rf" ({slope:.2f})"
            plt.loglog(
                dof_mod,
                err_mod,
                marker,
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=2,
                label=label + slope_label,
            )
            dof_line = np.linspace(dof_fit[0], dof_fit[-1], 50)
            err_line = np.exp(-slope * np.log(dof_line) + intercept)
            plt.loglog(dof_line, err_line, linewidth=1, linestyle=linestyle, color=color)
        elif len(dof_mod) == 2:
            slope = -(np.log(err_mod[1]) - np.log(err_mod[0])) / (
                np.log(dof_mod[1]) - np.log(dof_mod[0])
            )
            if showslope:
                slope_label = rf" ({slope:.2f})"
            plt.loglog(
                dof_mod,
                err_mod,
                marker,
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=2,
                label=label + slope_label,
            )
            plt.loglog(dof_mod, err_mod, linewidth=1, linestyle=linestyle, color=color)

    plt.legend(loc=legendloc, fontsize=legendsize)
    if grid:
        plt.grid(which="major", axis="both", linestyle="--", color="gray", linewidth=1)

    ax = plt.gca()
    ax.yaxis.set_major_locator(tik.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.yaxis.set_minor_locator(tik.LogLocator(base=10.0, subs="auto", numticks=10))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="both", labelsize=tick_size)
    plt.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=600, bbox_inches="tight")

    return fig
