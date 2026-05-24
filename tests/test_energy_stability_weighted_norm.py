import numpy as np
import pytest

from src.assembly import calc_RHS
from src.elements import make_uniform_elements, trace_left
from src.operator_library import OperatorSpec
from src.sats import left_boundary_flux_state, right_boundary_flux_state
from src.solve import split_global_vector


OPEN_SPEC = OperatorSpec(
    ["1", "x", "x^2"],
    ["1", "x", "x^2", "x^3", "x^4", "x^5"],
    "open",
)


def _weighted_energy_rate(elements, u: np.ndarray, du_dt: np.ndarray) -> float:
    weights = np.concatenate([(el.b / el.a) * el.H for el in elements])
    return float(u @ (weights * du_dt))


def _expected_boundary_energy_rate(
    elements,
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


@pytest.mark.parametrize("sat_type", ["symmetric", "upwind"])
def test_energy_stability_weighted_norm(sat_type: str) -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=6,
        operators=OperatorSpec(["1", "x", "x^2"], ["1", "x", "x^2", "x^3"], "closed"),
        a_fun=lambda x: 1.0 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.0 + 0.1 * np.sin(2.0 * np.pi * x),
    )

    rng = np.random.default_rng(4)
    u = rng.standard_normal(sum(el.x.size for el in elements))
    du_dt = calc_RHS(
        elements,
        u,
        sat_type=sat_type,
        left_bc_fun=lambda _x: 0.0,
        include_forcing=False,
    )

    weights = np.concatenate([(el.b / el.a) * el.H for el in elements])
    energy_rate = float(u @ (weights * du_dt))

    assert energy_rate <= 1e-11


def test_symmetric_sat_matches_boundary_energy_balance() -> None:
    def u_exact(x: np.ndarray) -> np.ndarray:
        return 0.7 + 0.2 * np.sin(2.0 * np.pi * x)

    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: 1.2 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.6 + 0.3 * np.sin(2.0 * np.pi * x),
        exact_fun=u_exact,
    )

    rng = np.random.default_rng(33)
    sizes = [el.x.size for el in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(
        elements,
        u,
        sat_type="symmetric",
        include_forcing=False,
    )

    local_u = split_global_vector(u, sizes)
    energy_rate = _weighted_energy_rate(elements, u, du_dt)
    expected = _expected_boundary_energy_rate(elements, local_u, left_bc_fun=None)

    assert np.isclose(energy_rate, expected, atol=1e-12)


@pytest.mark.parametrize("sat_type", ["upwind", "rusanov"])
def test_upwind_sat_is_bounded_by_boundary_energy_balance(sat_type: str) -> None:
    def u_exact(x: np.ndarray) -> np.ndarray:
        return 0.7 + 0.2 * np.sin(2.0 * np.pi * x)

    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: 1.2 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.6 + 0.3 * np.sin(2.0 * np.pi * x),
        exact_fun=u_exact,
    )

    rng = np.random.default_rng(34)
    sizes = [el.x.size for el in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(
        elements,
        u,
        sat_type=sat_type,
        include_forcing=False,
    )

    local_u = split_global_vector(u, sizes)
    energy_rate = _weighted_energy_rate(elements, u, du_dt)
    expected = _expected_boundary_energy_rate(elements, local_u, left_bc_fun=None)

    assert energy_rate <= expected + 1e-12
