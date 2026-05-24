import numpy as np

from src.assembly import calc_RHS
from src.elements import make_uniform_elements, trace_left, trace_right
from src.operator_library import OperatorSpec
from src.sats import calc_sat, left_boundary_flux_state, right_boundary_flux_state
from src.solve import split_global_vector


OPEN_SPEC = OperatorSpec(
    ["1", "x", "x^2"],
    ["1", "x", "x^2", "x^3", "x^4", "x^5"],
    "open",
)


def _discrete_mass_rate(elements, local_du: list[np.ndarray]) -> float:
    return float(sum(np.sum(el.H * du_local) for el, du_local in zip(elements, local_du)))


def _expected_boundary_mass_rate(
    elements: list,
    local_u: list[np.ndarray],
    *,
    left_bc_fun,
) -> float:
    """Conservative identity: d/dt ∫ u_h dx ≈ inflow_left − outflow_right."""
    left_inflow = left_boundary_flux_state(elements[0], left_bc_fun)
    right_outflow = trace_right(elements[-1], elements[-1].b * local_u[-1])
    return left_inflow - right_outflow


def test_conservation_identity_conservative_form_upwind_homogeneous_left() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.sin(2.0 * np.pi * x) + 2.0,
    )

    rng = np.random.default_rng(3)
    sizes = [el.x.size for el in elements]
    u = rng.standard_normal(sum(sizes))
    zero_inflow = lambda _x: 0.0
    du_dt = calc_RHS(
        elements,
        u,
        sat_type="upwind",
        left_bc_fun=zero_inflow,
        include_forcing=False,
    )

    local_u = split_global_vector(u, sizes)
    local_du = split_global_vector(du_dt, sizes)

    mass_rate = _discrete_mass_rate(elements, local_du)
    expected = _expected_boundary_mass_rate(
        elements, local_u, left_bc_fun=zero_inflow
    )
    assert np.isclose(mass_rate, expected, atol=1e-12)


def test_conservation_identity_conservative_form_upwind_exact_left() -> None:
    def u_exact(x: np.ndarray) -> np.ndarray:
        return np.sin(2.0 * np.pi * x)

    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=5,
        operators=OPEN_SPEC,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.sin(2.0 * np.pi * x) + 2.0,
        exact_fun=u_exact,
    )

    rng = np.random.default_rng(3)
    sizes = [el.x.size for el in elements]
    u = rng.standard_normal(sum(sizes))
    du_dt = calc_RHS(
        elements,
        u,
        sat_type="upwind",
        include_forcing=False,
    )

    local_u = split_global_vector(u, sizes)
    local_du = split_global_vector(du_dt, sizes)

    mass_rate = _discrete_mass_rate(elements, local_du)
    expected = _expected_boundary_mass_rate(
        elements, local_u, left_bc_fun=None
    )
    assert np.isclose(mass_rate, expected, atol=1e-12)


def test_left_inflow_uses_exact_boundary_coefficient_not_b_trace() -> None:
    def b_fun(x: np.ndarray) -> np.ndarray:
        return np.exp(3.0 * x) + 2.0

    left_state = 1.25
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=1,
        operators=OPEN_SPEC,
        a_fun=lambda x: np.ones_like(x),
        b_fun=b_fun,
        exact_fun=lambda x: left_state + np.zeros_like(x),
    )
    element = elements[0]

    exact_boundary_flux = float(b_fun(np.array([element.x_left]))[0]) * left_state
    traced_boundary_flux = trace_left(element, element.b) * left_state

    assert not np.isclose(exact_boundary_flux, traced_boundary_flux)
    assert np.isclose(
        left_boundary_flux_state(element, left_bc_fun=None),
        exact_boundary_flux,
    )


def test_single_element_sat_has_no_right_outflow_penalty() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        num_elements=1,
        operators=OPEN_SPEC,
        a_fun=lambda x: 1.0 + x,
        b_fun=lambda x: 2.0 + x,
    )

    rng = np.random.default_rng(17)
    u_local = rng.standard_normal(elements[0].x.size)
    sat = calc_sat(
        elements,
        [u_local],
        sat_type="upwind",
        left_bc_fun=lambda _x: 0.0,
    )

    element = elements[0]
    left_flux_state = trace_left(element, element.b * u_local)
    expected = -(element.a * element.H_inv * element.tL) * left_flux_state

    assert np.isclose(
        right_boundary_flux_state(element, u_local),
        trace_right(element, element.b * u_local),
    )
    assert np.allclose(sat[0], expected)
