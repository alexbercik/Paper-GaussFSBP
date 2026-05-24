import numpy as np

from gaussfsbp.assembly import assemble_homogeneous_operator
from gaussfsbp.elements import create_elements, trace_left, trace_right
from gaussfsbp.mesh import ElementSpec, Mesh1D
from gaussfsbp.operators import builtin_operator_repository


def test_mixed_operator_interface_selection_and_traces() -> None:
    repo = builtin_operator_repository()
    specs = [
        ElementSpec(["1", "x", "x^2"], "closed", 0),
        ElementSpec(["1", "x", "x^2"], "closed", 1),
        ElementSpec(["1", "x", "x^2"], "closed", 0),
        ElementSpec(["1", "x", "x^2"], "closed", 1),
    ]
    bounds = np.linspace(0.0, 1.0, len(specs) + 1)
    mesh = Mesh1D.from_bounds((0.0, 1.0), bounds, specs)

    elements = create_elements(
        mesh,
        repo,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )
    L = assemble_homogeneous_operator(elements, sat_type="symmetric")

    assert L.shape[0] == sum(el.x.size for el in elements)

    for j in range(len(elements) - 1):
        ul = np.random.default_rng(10 + j).standard_normal(elements[j].x.size)
        ur = np.random.default_rng(20 + j).standard_normal(elements[j + 1].x.size)

        left_flux_state = trace_right(elements[j], elements[j].b * ul)
        right_flux_state = trace_left(elements[j + 1], elements[j + 1].b * ur)

        assert np.isfinite(left_flux_state)
        assert np.isfinite(right_flux_state)
