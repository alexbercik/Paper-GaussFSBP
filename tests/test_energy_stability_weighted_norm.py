import numpy as np
import pytest

from gaussfsbp.assembly import assemble_homogeneous_operator
from gaussfsbp.elements import create_elements
from gaussfsbp.mesh import ElementSpec, Mesh1D
from gaussfsbp.operators import builtin_operator_repository


@pytest.mark.parametrize("sat_type", ["symmetric", "upwind"])
def test_energy_stability_weighted_norm(sat_type: str) -> None:
    repo = builtin_operator_repository()
    spec = ElementSpec(["1", "x", "x^2"], "closed", 0)
    mesh = Mesh1D.uniform((0.0, 1.0), num_elements=6, element_spec=spec)

    elements = create_elements(
        mesh,
        repo,
        a_fun=lambda x: 1.0 + 0.2 * np.cos(2.0 * np.pi * x),
        b_fun=lambda x: 1.0 + 0.1 * np.sin(2.0 * np.pi * x),
    )
    L = assemble_homogeneous_operator(elements, sat_type=sat_type)

    rng = np.random.default_rng(4)
    u = rng.standard_normal(L.shape[1])
    du_dt = -(L @ u)

    weights = np.concatenate([(el.b / el.a) * el.H_x for el in elements])
    energy_rate = float(u @ (weights * du_dt))

    assert energy_rate <= 1e-11
