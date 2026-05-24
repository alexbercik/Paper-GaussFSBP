import numpy as np

from gaussfsbp.assembly import assemble_homogeneous_operator
from gaussfsbp.elements import create_elements, trace_right
from gaussfsbp.mesh import ElementSpec, Mesh1D
from gaussfsbp.operators import builtin_operator_repository
from gaussfsbp.solve import split_global_vector


def test_conservation_identity_inflow_outflow_upwind() -> None:
    repo = builtin_operator_repository()
    spec = ElementSpec(["1", "x", "x^2"], "closed", 0)
    mesh = Mesh1D.uniform((0.0, 1.0), num_elements=5, element_spec=spec)
    elements = create_elements(
        mesh,
        repo,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )

    L = assemble_homogeneous_operator(elements, sat_type="upwind")

    rng = np.random.default_rng(3)
    u = rng.standard_normal(L.shape[1])
    du_dt = -(L @ u)

    local_u = split_global_vector(u, [el.x.size for el in elements])
    local_du = split_global_vector(du_dt, [el.x.size for el in elements])

    mass_rate = 0.0
    for el, du_local in zip(elements, local_du):
        mass_rate += np.sum(el.H_x * du_local)

    expected = -trace_right(elements[-1], local_u[-1])
    assert np.isclose(mass_rate, expected, atol=1e-12)
