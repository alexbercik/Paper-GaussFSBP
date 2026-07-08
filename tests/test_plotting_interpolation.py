import numpy as np
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from src.elements import make_uniform_elements
from src.operator_library import get_operator_by_name
from src.operators import Operator
from src.plotting import plot_solution_profiles, profile_from_elements


def test_profile_interpolation_recovers_known_polynomial_values() -> None:
    operator = get_operator_by_name("LGLp2")
    elements = make_uniform_elements(
        (2.0, 4.0),
        1,
        operator,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )

    # The interpolant is built in reference coordinates, then sampled on the
    # physical element. This quadratic is exact for the LGLp2 basis.
    u = 1.0 + 2.0 * operator.nodes + 0.5 * operator.nodes**2
    [(x_plot, u_plot)] = profile_from_elements(
        elements,
        u,
        interpolate=True,
        points_per_element=5,
    )

    expected_x = np.linspace(2.0, 4.0, 5)
    expected_xi = np.linspace(-1.0, 1.0, 5)

    np.testing.assert_allclose(x_plot, expected_x, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(
        u_plot,
        1.0 + 2.0 * expected_xi + 0.5 * expected_xi**2,
        atol=1.0e-13,
        rtol=1.0e-13,
    )


def test_profile_projection_recovers_known_exponential_values() -> None:
    nodes = np.linspace(0.0, 1.0, 5)
    operator = Operator(
        name="test_enriched",
        basis=["1", "x", "exp(2x)"],
        quad_basis=["1", "x", "x^2", "exp(2x)", "x exp(2x)"],
        op_type="open",
        selector=0,
        interval=np.array([0.0, 1.0]),
        nodes=nodes,
        D=np.zeros((nodes.size, nodes.size)),
        H=np.full(nodes.size, 1.0 / nodes.size),
        tL=np.zeros(nodes.size),
        tR=np.zeros(nodes.size),
    )
    elements = make_uniform_elements(
        (0.0, 1.0),
        1,
        operator,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )

    u = 0.25 - 0.75 * nodes + 1.5 * np.exp(2.0 * nodes)
    [(x_plot, u_plot)] = profile_from_elements(
        elements,
        u,
        interpolate=True,
        points_per_element=5,
    )

    expected_x = np.linspace(0.0, 1.0, 5)

    np.testing.assert_allclose(x_plot, expected_x, atol=1.0e-14, rtol=0.0)
    np.testing.assert_allclose(
        u_plot,
        0.25 - 0.75 * expected_x + 1.5 * np.exp(2.0 * expected_x),
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_solution_profile_markers_can_show_actual_nodal_values() -> None:
    smooth_profile = [(np.array([0.0, 0.5, 1.0]), np.array([1.0, 1.5, 2.0]))]
    nodal_profile = [(np.array([0.25, 0.75]), np.array([1.25, 1.75]))]

    fig = plot_solution_profiles(
        [smooth_profile],
        ["run"],
        nodal_profiles=[nodal_profile],
        colors=["tab:red"],
        markers=["o"],
    )
    ax = fig.axes[0]
    smooth_line, nodal_line = ax.lines[-2:]

    assert smooth_line.get_marker() in {None, "None", ""}
    np.testing.assert_allclose(nodal_line.get_xdata(), nodal_profile[0][0])
    np.testing.assert_allclose(nodal_line.get_ydata(), nodal_profile[0][1])
    assert nodal_line.get_marker() == "o"
    assert nodal_line.get_linestyle() == "None"

    legend = ax.get_legend()
    handles = getattr(legend, "legend_handles", None)
    if handles is None:
        handles = legend.legendHandles
    assert handles[0].get_marker() == "o"
    assert handles[0].get_linestyle() != "None"

    plt.close(fig)
