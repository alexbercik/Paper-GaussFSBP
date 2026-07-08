from __future__ import annotations

from collections.abc import Callable, Sequence
import re
import warnings

from matplotlib import rcParams
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.ticker as tik
import numpy as np
from scipy.optimize import curve_fit

from .elements import Element1D, map_reference_to_physical
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
DEFAULT_PROFILE_POINTS_PER_ELEMENT = 50
PROFILE_PROJECTOR_COND_WARN = 1.0e12


def _split_basis_labels(labels: Sequence[str]) -> list[str]:
    """Expand legacy comma-combined labels such as ``"x^2, x^3"``."""
    terms: list[str] = []
    for label in labels:
        terms.extend(term.strip() for term in label.split(",") if term.strip())
    return terms


def _basis_power(label: str, prefix: str = "x") -> int | None:
    if label == prefix:
        return 1
    match = re.fullmatch(rf"{re.escape(prefix)}\^(?:\{{(\d+)\}}|(\d+))", label)
    if match is None:
        return None
    return int(match.group(1) or match.group(2))


def _numeric_factor(text: str) -> float:
    """Evaluate a trusted numeric factor after rejecting non-numeric tokens."""
    if text in {"", "+"}:
        return 1.0
    if text == "-":
        return -1.0
    if text.endswith("*"):
        text = text[:-1]
    if not re.fullmatch(r"[0-9eE+\-*/().]+", text):
        raise ValueError(f"Unsupported numeric factor '{text}'")
    return float(eval(text, {"__builtins__": {}}, {}))


def _linear_x_coefficient(text: str) -> float:
    """Return ``a`` for labels containing linear expressions ``a*x``."""
    compact = text.replace(" ", "")
    if not compact.endswith("x"):
        raise ValueError(f"Expected a linear-in-x expression, got '{text}'")
    return _numeric_factor(compact[:-1])


def _eval_exp_label(label: str, xi: np.ndarray) -> np.ndarray | None:
    if label == "e^x":
        return np.exp(xi)
    match = re.fullmatch(r"e\^\{(.+)\}", label)
    if match is not None:
        return np.exp(_linear_x_coefficient(match.group(1)) * xi)
    match = re.fullmatch(r"exp\((.+)\)", label)
    if match is None:
        return None
    return np.exp(_linear_x_coefficient(match.group(1)) * xi)


def _eval_sqrt_label(label: str, xi: np.ndarray) -> np.ndarray | None:
    match = re.fullmatch(r"sqrt\((.+)-x\)", label)
    if match is not None:
        return np.sqrt(_numeric_factor(match.group(1)) - xi)
    match = re.fullmatch(r"\((.+)-x\)\^\{3/2\}", label)
    if match is not None:
        return (_numeric_factor(match.group(1)) - xi) ** 1.5
    return None


def _eval_basis_label(label: str, xi: np.ndarray, interval: np.ndarray) -> np.ndarray:
    """Evaluate one supported reference-basis label at reference points ``xi``."""
    compact = label.replace(" ", "")
    if compact in {"1", "one(x)"}:
        return np.ones_like(xi)

    # Polynomial labels from hand-tabulated operators use monomials.
    power = _basis_power(compact)
    if power is not None:
        return xi**power

    # Julia-built operators label the polynomial part as Legendre modes.
    match = re.fullmatch(r"P_(?:\{(\d+)\}|(\d+))\(x\)", compact)
    if match is not None:
        degree = int(match.group(1) or match.group(2))
        coeffs = np.zeros(degree + 1, dtype=float)
        coeffs[degree] = 1.0
        eta = 2.0 * (xi - interval[0]) / (interval[1] - interval[0]) - 1.0
        return np.polynomial.legendre.legval(eta, coeffs)

    exp_value = _eval_exp_label(compact, xi)
    if exp_value is not None:
        return exp_value

    # Some enriched bases use powers multiplying an exponential or root mode.
    match = re.fullmatch(r"(x(?:\^(?:\{\d+\}|\d+))?)exp\((.+)\)", compact)
    if match is not None:
        power = _basis_power(match.group(1))
        if power is None:
            raise ValueError(f"Unsupported basis label '{label}'")
        return xi**power * np.exp(_linear_x_coefficient(match.group(2)) * xi)

    match = re.fullmatch(r"(x(?:\^(?:\{\d+\}|\d+))?)sqrt\((.+)-x\)", compact)
    if match is not None:
        power = _basis_power(match.group(1))
        if power is None:
            raise ValueError(f"Unsupported basis label '{label}'")
        return xi**power * np.sqrt(_numeric_factor(match.group(2)) - xi)

    sqrt_value = _eval_sqrt_label(compact, xi)
    if sqrt_value is not None:
        return sqrt_value

    raise ValueError(f"Unsupported basis label '{label}'")


def _basis_vandermonde(
    labels: Sequence[str], xi: np.ndarray, interval: np.ndarray
) -> np.ndarray:
    terms = _split_basis_labels(labels)
    return np.column_stack([_eval_basis_label(term, xi, interval) for term in terms])


def _project_element_solution(element: Element1D, u_local: np.ndarray) -> np.ndarray:
    """Return modal coefficients from interpolation or discrete H projection."""
    operator = element.operator
    V = _basis_vandermonde(operator.basis, operator.nodes, operator.interval)
    if V.shape[0] == V.shape[1]:
        cond = np.linalg.cond(V)
        if cond > PROFILE_PROJECTOR_COND_WARN:
            warnings.warn(
                f"Ill-conditioned interpolation Vandermonde for {operator.name}: "
                f"cond(V)={cond:.3e}",
                RuntimeWarning,
            )
        return np.linalg.solve(V, u_local)

    H = np.asarray(operator.H, dtype=float)
    M = V.T @ (H[:, None] * V)
    rhs = V.T @ (H * u_local)
    cond = np.linalg.cond(M)
    if cond > PROFILE_PROJECTOR_COND_WARN:
        warnings.warn(
            f"Ill-conditioned modal projection for {operator.name}: "
            f"cond(M)={cond:.3e}",
            RuntimeWarning,
        )
    return np.linalg.solve(M, rhs)


def interpolated_profile_from_elements(
    elements: Sequence[Element1D],
    global_u: np.ndarray,
    points_per_element: int = DEFAULT_PROFILE_POINTS_PER_ELEMENT,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Project nodal values to each element basis and sample a dense profile."""
    if points_per_element < 2:
        raise ValueError("points_per_element must be at least 2")

    sizes = [element.x.size for element in elements]
    local = split_global_vector(global_u, sizes)
    profiles: list[tuple[np.ndarray, np.ndarray]] = []

    for element, u_local in zip(elements, local):
        operator = element.operator
        xi_plot = np.linspace(
            operator.interval[0],
            operator.interval[1],
            points_per_element,
        )
        coeffs = _project_element_solution(element, u_local)
        V_plot = _basis_vandermonde(operator.basis, xi_plot, operator.interval)
        x_plot = map_reference_to_physical(
            xi_plot,
            element.x_left,
            element.x_right,
            operator.interval,
        )
        profiles.append((x_plot, V_plot @ coeffs))

    return profiles


def profile_from_elements(
    elements: Sequence[Element1D],
    global_u: np.ndarray,
    *,
    interpolate: bool = False,
    points_per_element: int = DEFAULT_PROFILE_POINTS_PER_ELEMENT,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Per-element profiles ``(x, u)`` in left-to-right order.

    By default this returns the raw nodal values. With ``interpolate=True``,
    each element is projected to its stored reference basis and sampled on a
    dense reference grid before mapping back to physical coordinates.
    """
    if interpolate:
        return interpolated_profile_from_elements(
            elements,
            global_u,
            points_per_element=points_per_element,
        )

    sizes = [element.x.size for element in elements]
    local = split_global_vector(global_u, sizes)
    return [(element.x, u_local) for element, u_local in zip(elements, local)]


def plot_solution_profiles(
    profiles: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]],
    legend_strings: Sequence[str],
    *,
    nodal_profiles: Sequence[Sequence[tuple[np.ndarray, np.ndarray]]] | None = None,
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
    markevery: int | None = 1,
    nodal_marker_size: float | None = None,
    legendloc: str = "best",
    legend_behind_data: bool = False,
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
) -> plt.Figure:
    """Overlay numerical solutions and an optional exact profile.

    Each entry in ``profiles`` is one run: a list of per-element ``(x, u)`` arrays.
    Segments are drawn separately so interface nodes are not connected across elements.
    Pass ``nodal_profiles`` when ``profiles`` contains dense interpolated curves
    and markers should show the original nodal values instead of dense samples.
    ``xlim``/``ylim`` set optional ``(min, max)`` axis limits in data coordinates.
    """
    if len(profiles) != len(legend_strings):
        raise ValueError("profiles and legend_strings must have the same length")
    if not profiles:
        raise ValueError("At least one numerical profile is required")
    if nodal_profiles is not None and len(nodal_profiles) != len(profiles):
        raise ValueError("nodal_profiles must have the same length as profiles")

    colors = list(colors) if colors is not None else list(TAB_COLORS)
    markers = list(markers) if markers is not None else list(DEFAULT_MARKERS)
    linestyles = (
        list(linestyles) if linestyles is not None else list(DEFAULT_SOLUTION_LINESTYLES)
    )
    # Grid < legend < data. Matplotlib's default grid zorder is ~2, so a legend at
    # zorder=1 ends up underneath the grid and looks transparent.
    grid_zorder = 0.5
    legend_zorder = 3.0 if legend_behind_data else None
    exact_zorder = 4.0 if legend_behind_data else 1.0
    data_zorder = 5.0 if legend_behind_data else 2.0

    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    ax.set_xlabel(xlabel or r"$x$", fontsize=title_size)
    ax.set_ylabel(ylabel or r"$u(x)$", fontsize=title_size)
    if legend_behind_data:
        ax.set_axisbelow(True)

    if x_exact is not None and u_exact is not None:
        ax.plot(
            x_exact,
            u_exact,
            color=exact_color,
            linestyle=exact_linestyle,
            linewidth=exact_linewidth,
            label=exact_label,
            zorder=exact_zorder,
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
                marker=None if nodal_profiles is not None else marker,
                markersize=marker_size,
                markevery=markevery,
                linewidth=1.2,
                label=label if j == 0 else None,
                zorder=data_zorder,
            )

        if nodal_profiles is not None:
            for x_nodes, u_nodes in nodal_profiles[i]:
                ax.plot(
                    x_nodes,
                    u_nodes,
                    linestyle="",
                    color=color,
                    marker=marker,
                    markersize=nodal_marker_size or marker_size,
                    label=None,
                    zorder=data_zorder + 0.5,
                )

    if nodal_profiles is None:
        legend = ax.legend(loc=legendloc, fontsize=legendsize)
    else:
        legend_handles = []
        if x_exact is not None and u_exact is not None:
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=exact_color,
                    linestyle=exact_linestyle,
                    linewidth=exact_linewidth,
                    label=exact_label,
                )
            )

        for i, label in enumerate(legend_strings):
            legend_handles.append(
                Line2D(
                    [],
                    [],
                    color=colors[i % len(colors)],
                    linestyle=linestyles[i % len(linestyles)],
                    marker=markers[i % len(markers)],
                    markersize=nodal_marker_size or marker_size,
                    linewidth=1.2,
                    label=label,
                )
            )
        legend = ax.legend(handles=legend_handles, loc=legendloc, fontsize=legendsize)

    if legend_zorder is not None:
        legend.set_zorder(legend_zorder)
        frame = legend.get_frame()
        frame.set_alpha(1.0)
        frame.set_facecolor("white")
    if grid:
        grid_kwargs = {
            "linestyle": "--",
            "color": "gray",
            "linewidth": 0.8,
            "alpha": 0.7,
        }
        if legend_behind_data:
            grid_kwargs["zorder"] = grid_zorder
        ax.grid(**grid_kwargs)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
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
    legend_behind_data: bool = False,
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
    legend_behind_data
        If True, draw the legend underneath the plotted curves and markers.
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
    # Grid < legend < data. Matplotlib's default grid zorder is ~2, so a legend at
    # zorder=1 ends up underneath the grid and looks transparent.
    grid_zorder = 0.5
    legend_zorder = 3.0 if legend_behind_data else None
    data_zorder = 5.0 if legend_behind_data else 2.0

    fig, ax = plt.subplots(figsize=figsize)
    if title is not None:
        ax.set_title(title, fontsize=title_size)
    ax.set_ylabel(
        ylabel or r"$\| \boldsymbol{u}_h - u(\boldsymbol{x}) \|_\mathsf{H}$",
        fontsize=title_size,
    )
    ax.set_xlabel(xlabel or "DOF", fontsize=title_size)
    if legend_behind_data:
        ax.set_axisbelow(True)

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
            ax.loglog(
                dof_mod,
                err_mod,
                marker,
                linestyle='',
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=2,
                label=label + slope_label,
                zorder=data_zorder,
            )
            dof_line = np.linspace(dof_fit[0], dof_fit[-1], 50)
            err_line = np.exp(-slope * np.log(dof_line) + intercept)
            ax.loglog(
                dof_line,
                err_line,
                linewidth=1,
                linestyle=linestyle,
                color=color,
                zorder=data_zorder,
            )
        elif len(dof_fit) == 2:
            slope = (np.log(err_mod[1]) - np.log(err_mod[0])) / (
                np.log(dof_mod[1]) - np.log(dof_mod[0])
            )
            if showslope:
                slope_label = rf" ({slope:.2f})"
            ax.loglog(
                dof_mod,
                err_mod,
                marker,
                linestyle='',
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=2,
                label=label + slope_label,
                zorder=data_zorder,
            )
            ax.loglog(
                dof_fit,
                err_fit,
                linewidth=1,
                linestyle=linestyle,
                color=color,
                zorder=data_zorder,
            )
        else:
            ax.loglog(
                dof_mod,
                err_mod,
                marker,
                linestyle='',
                markersize=8,
                color=color,
                markerfacecolor="none",
                markeredgewidth=2,
                label=label,
                zorder=data_zorder,
            )

    legend = ax.legend(loc=legendloc, fontsize=legendsize)
    if legend_zorder is not None:
        legend.set_zorder(legend_zorder)
        frame = legend.get_frame()
        frame.set_alpha(1.0)
        frame.set_facecolor("white")
    if grid:
        grid_kwargs = {
            "which": "major",
            "axis": "y",
            "linestyle": "--",
            "color": "gray",
            "linewidth": 1,
        }
        if legend_behind_data:
            grid_kwargs["zorder"] = grid_zorder
        ax.grid(**grid_kwargs)

    ax.yaxis.set_major_locator(tik.LogLocator(base=10.0, subs=[1.0], numticks=10))
    ax.yaxis.set_minor_locator(tik.LogLocator(base=10.0, subs="auto", numticks=10))
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.tick_params(axis="both", which="both", labelsize=tick_size)
    fig.tight_layout()

    if savefile is not None:
        fig.savefig(savefile, dpi=600, bbox_inches="tight")

    return fig
