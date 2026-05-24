import numpy as np

from src.assembly import calc_LHS
from src.elements import make_elements, trace_left, trace_right
from src.operator_library import OperatorSpec


def test_mixed_operator_interface_selection_and_traces() -> None:
    closed = OperatorSpec(["1", "x", "x^2"], ["1", "x", "x^2", "x^3"], "closed")
    open_ = OperatorSpec(
        ["1", "x", "x^2"],
        ["1", "x", "x^2", "x^3", "x^4", "x^5"],
        "open",
    )
    operators = [closed, open_, open_, closed]
    bounds = np.linspace(0.0, 1.0, len(operators) + 1)
    elements = make_elements(
        bounds,
        operators,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )
    L = calc_LHS(elements, sat_type="symmetric")

    assert L.shape[0] == sum(el.x.size for el in elements)

    for j in range(len(elements) - 1):
        ul = np.random.default_rng(10 + j).standard_normal(elements[j].x.size)
        ur = np.random.default_rng(20 + j).standard_normal(elements[j + 1].x.size)

        left_flux_state = trace_right(elements[j], elements[j].b * ul)
        right_flux_state = trace_left(elements[j + 1], elements[j + 1].b * ur)

        assert np.isfinite(left_flux_state)
        assert np.isfinite(right_flux_state)
