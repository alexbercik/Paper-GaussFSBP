import numpy as np

from src.assembly import assemble_system, calc_LHS, calc_RHS
from src.elements import (
    Element1D,
    make_elements,
    make_uniform_elements,
    trace_left,
    trace_right,
)
from src.operator_library import OperatorSpec
from src.problems import Problem
from src.sats import left_boundary_flux_state, right_boundary_flux_state
from src.solve import split_global_vector


CLOSED_SPEC = OperatorSpec(
    ["1", "x", "x^2"],
    ["1", "x", "x^2", "x^3"],
    "closed",
)
OPEN_SPEC = OperatorSpec(
    ["1", "x", "x^2"],
    ["1", "x", "x^2", "x^3", "x^4", "x^5"],
    "open",
)


def _problem_elements(problem: Problem) -> list[Element1D]:
    return make_uniform_elements(
        (0.0, 1.0),
        num_elements=3,
        operators=CLOSED_SPEC,
        a_fun=problem.a_fun,
        b_fun=problem.b_fun,
        f_fun=problem.f_fun,
        exact_fun=problem.exact_fun,
    )


def _weighted_energy_rate(
    elements: list[Element1D],
    u: np.ndarray,
    du_dt: np.ndarray,
) -> float:
    weights = np.concatenate(
        [(element.b / element.a) * element.H for element in elements]
    )
    return float(u @ (weights * du_dt))


def _expected_boundary_energy_rate(
    elements: list[Element1D],
    local_u: list[np.ndarray],
    *,
    left_bc_fun,
) -> float:
    left_flux_state = trace_left(elements[0], elements[0].b * local_u[0])
    left_inflow = left_boundary_flux_state(elements[0], left_bc_fun)
    right_outflow = right_boundary_flux_state(elements[-1], local_u[-1])
    return (
        0.5 * left_inflow * left_inflow
        - 0.5 * (left_flux_state - left_inflow) ** 2
        - 0.5 * right_outflow * right_outflow
    )


def test_calc_lhs_matches_finite_difference_rhs_jacobian() -> None:
    problem = Problem(
        a_fun=lambda x: 1.0 + 0.2 * np.cos(np.pi * x),
        b_fun=lambda x: 1.0 + 0.15 * np.sin(2.0 * np.pi * x),
        f_fun=lambda x: np.zeros_like(x),
        exact_fun=lambda x: np.zeros_like(x),
    )
    system = assemble_system(_problem_elements(problem), sat_type="upwind")
    lhs = calc_LHS(system.elements, sat_type="upwind")

    rng = np.random.default_rng(12)
    u = rng.standard_normal(lhs.shape[1])
    direction = rng.standard_normal(lhs.shape[1])
    eps = 1.0e-5

    rhs0 = calc_RHS(system.elements, u, sat_type="upwind", include_forcing=False)
    rhs1 = calc_RHS(
        system.elements,
        u + eps * direction,
        sat_type="upwind",
        include_forcing=False,
    )

    np.testing.assert_allclose(
        (rhs1 - rhs0) / eps,
        -(lhs @ direction),
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_assembled_system_matches_direct_rhs_with_boundary_data() -> None:
    left_state = 1.7
    problem = Problem(
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: 1.0 + 0.25 * x,
        f_fun=lambda x: 0.3 + x,
        exact_fun=lambda x: left_state + np.zeros_like(x),
        left_bc_fun=lambda _x: left_state,
    )
    system = assemble_system(
        _problem_elements(problem),
        left_bc_fun=problem.left_bc_fun,
        sat_type="upwind",
    )

    rng = np.random.default_rng(22)
    u = rng.standard_normal(system.matrix.shape[1])
    direct_rhs = calc_RHS(
        system.elements,
        u,
        sat_type="upwind",
        left_bc_fun=problem.left_bc_fun,
    )

    np.testing.assert_allclose(
        direct_rhs,
        system.rhs - system.matrix @ u,
        atol=1.0e-12,
        rtol=1.0e-12,
    )


def test_conservative_upwind_sat_has_expected_mass_balance() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.sin(2.0 * np.pi * x) + 2.0,
        exact_fun=lambda x: np.sin(2.0 * np.pi * x),
    )

    rng = np.random.default_rng(3)
    sizes = [element.x.size for element in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(elements, u, sat_type="upwind", include_forcing=False)
    local_u = split_global_vector(u, sizes)
    local_du = split_global_vector(du_dt, sizes)

    mass_rate = float(
        sum(np.sum(element.H * du) for element, du in zip(elements, local_du))
    )
    expected = (
        left_boundary_flux_state(elements[0], left_bc_fun=None)
        - trace_right(elements[-1], elements[-1].b * local_u[-1])
    )
    assert np.isclose(mass_rate, expected, atol=1.0e-12)


def test_symmetric_sat_matches_boundary_energy_balance() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: 1.2 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.6 + 0.3 * np.sin(2.0 * np.pi * x),
        exact_fun=lambda x: 0.7 + 0.2 * np.sin(2.0 * np.pi * x),
    )

    rng = np.random.default_rng(33)
    sizes = [element.x.size for element in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(elements, u, sat_type="symmetric", include_forcing=False)
    local_u = split_global_vector(u, sizes)

    assert np.isclose(
        _weighted_energy_rate(elements, u, du_dt),
        _expected_boundary_energy_rate(elements, local_u, left_bc_fun=None),
        atol=1.0e-12,
    )


def test_upwind_sat_is_bounded_by_boundary_energy_balance() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: 1.2 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.6 + 0.3 * np.sin(2.0 * np.pi * x),
        exact_fun=lambda x: 0.7 + 0.2 * np.sin(2.0 * np.pi * x),
    )

    rng = np.random.default_rng(34)
    sizes = [element.x.size for element in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(elements, u, sat_type="upwind", include_forcing=False)
    local_u = split_global_vector(u, sizes)

    assert _weighted_energy_rate(elements, u, du_dt) <= (
        _expected_boundary_energy_rate(elements, local_u, left_bc_fun=None)
        + 1.0e-12
    )


def test_mixed_operator_interfaces_assemble_and_trace() -> None:
    operators = [CLOSED_SPEC, OPEN_SPEC, OPEN_SPEC, CLOSED_SPEC]
    bounds = np.linspace(0.0, 1.0, len(operators) + 1)
    elements = make_elements(
        bounds,
        operators,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )

    lhs = calc_LHS(elements, sat_type="symmetric")
    assert lhs.shape[0] == sum(element.x.size for element in elements)

    for j in range(len(elements) - 1):
        rng_left = np.random.default_rng(10 + j)
        rng_right = np.random.default_rng(20 + j)
        u_left = rng_left.standard_normal(elements[j].x.size)
        u_right = rng_right.standard_normal(elements[j + 1].x.size)

        assert np.isfinite(trace_right(elements[j], elements[j].b * u_left))
        assert np.isfinite(trace_left(elements[j + 1], elements[j + 1].b * u_right))
