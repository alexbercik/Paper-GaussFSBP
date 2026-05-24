import numpy as np
import pytest

from src.assembly import assemble_system, calc_LHS, calc_RHS
from src.elements import Element1D, make_uniform_elements
from src.operator_library import OperatorSpec
from src.problems import Problem


def _elements(problem: Problem) -> list[Element1D]:
    return make_uniform_elements(
        (0.0, 1.0),
        num_elements=3,
        operators=OperatorSpec(["1", "x", "x^2"], ["1", "x", "x^2", "x^3"], "closed"),
        a_fun=problem.a_fun,
        b_fun=problem.b_fun,
        f_fun=problem.f_fun,
        exact_fun=problem.exact_fun,
    )


@pytest.mark.parametrize("sat_type", ["symmetric", "upwind", "rusanov"])
def test_calc_lhs_matches_finite_difference_rhs_jacobian(sat_type: str) -> None:
    problem = Problem(
        a_fun=lambda x: 1.0 + 0.2 * np.cos(np.pi * x),
        b_fun=lambda x: 1.0 + 0.15 * np.sin(2.0 * np.pi * x),
        f_fun=lambda x: np.zeros_like(x),
        exact_fun=lambda x: np.zeros_like(x),
    )
    system = assemble_system(_elements(problem), sat_type=sat_type)
    lhs = calc_LHS(system.elements, sat_type=sat_type)

    rng = np.random.default_rng(12)
    u = rng.standard_normal(lhs.shape[1])
    direction = rng.standard_normal(lhs.shape[1])
    eps = 1e-5

    rhs0 = calc_RHS(system.elements, u, sat_type=sat_type, include_forcing=False)
    rhs1 = calc_RHS(
        system.elements, u + eps * direction, sat_type=sat_type, include_forcing=False
    )
    finite_difference = (rhs1 - rhs0) / eps

    assert np.allclose(finite_difference, -(lhs @ direction), atol=1e-8, rtol=1e-8)


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
        _elements(problem),
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
    assembled_rhs = system.rhs - system.matrix @ u

    assert np.allclose(direct_rhs, assembled_rhs, atol=1e-12, rtol=1e-12)
