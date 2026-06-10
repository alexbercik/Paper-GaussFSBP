from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

from .operator_library import OperatorSpec, get_operator_by_name, operator_from_spec
from .operators import Operator


OperatorChoice = Union[Operator, OperatorSpec, str]


@dataclass
class Element1D:
    x_left: float
    x_right: float
    x: np.ndarray
    D: np.ndarray
    H: np.ndarray
    H_inv: np.ndarray
    tL: np.ndarray
    tR: np.ndarray
    a: np.ndarray
    b: np.ndarray
    b_left: float
    f: np.ndarray | None = None
    exact_left: float | None = None


def map_reference_to_physical(
    xi: np.ndarray, xL: float, xR: float, interval: np.ndarray
) -> np.ndarray:
    reference_left = interval[0]
    reference_length = interval[1] - interval[0]
    h = xR - xL
    return xL + (h / reference_length) * (
        np.asarray(xi, dtype=float) - reference_left
    )


def scale_operator_to_element(
    operator: Operator, xL: float, xR: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = xR - xL
    reference_length = operator.interval[1] - operator.interval[0]
    x = map_reference_to_physical(operator.nodes, xL, xR, operator.interval)
    # Affine scaling uses the tabulated operator interval, not a fixed [-1, 1].
    D = (reference_length / h) * operator.D
    H = (h / reference_length) * operator.H
    H_inv = 1.0 / H
    return x, D, H, H_inv


def evaluate_function_at_point(
    fun: Callable[[np.ndarray], np.ndarray], x: float
) -> float:
    """Evaluate a vectorized scalar function at one physical location."""
    return float(np.asarray(fun(np.array([x])), dtype=float).reshape(()))


def _resolve_operator(operator: OperatorChoice) -> Operator:
    if isinstance(operator, Operator):
        return operator
    if isinstance(operator, OperatorSpec):
        return operator_from_spec(operator)
    if isinstance(operator, str):
        return get_operator_by_name(operator)
    raise TypeError("operators must be Operator, OperatorSpec, or operator names")


def _operators_for_elements(
    operators: OperatorChoice | Sequence[OperatorChoice],
    num_elements: int,
) -> list[Operator]:
    if isinstance(operators, (Operator, OperatorSpec, str)):
        return [_resolve_operator(operators) for _ in range(num_elements)]

    operator_list = list(operators)
    if len(operator_list) != num_elements:
        raise ValueError("Need one operator or operator spec per element")
    return [_resolve_operator(operator) for operator in operator_list]


def make_elements(
    element_bounds: np.ndarray | Sequence[float],
    operators: OperatorChoice | Sequence[OperatorChoice],
    a_fun: Callable[[np.ndarray], np.ndarray],
    b_fun: Callable[[np.ndarray], np.ndarray],
    f_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    exact_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Element1D]:
    bounds = np.asarray(element_bounds, dtype=float)
    if bounds.ndim != 1:
        raise ValueError("element_bounds must be one-dimensional")
    if bounds.size < 2:
        raise ValueError("element_bounds must contain at least two points")
    if np.any(np.diff(bounds) <= 0.0):
        raise ValueError("element_bounds must be strictly increasing")

    element_operators = _operators_for_elements(operators, bounds.size - 1)
    elements: list[Element1D] = []
    for x_left_raw, x_right_raw, operator in zip(
        bounds[:-1], bounds[1:], element_operators
    ):
        x_left = float(x_left_raw)
        x_right = float(x_right_raw)
        x, D, H, H_inv = scale_operator_to_element(operator, x_left, x_right)
        a = np.asarray(a_fun(x), dtype=float)
        b = np.asarray(b_fun(x), dtype=float)
        if a.shape != x.shape or b.shape != x.shape:
            raise ValueError("a_fun and b_fun must return arrays with element nodal shape")

        f = None if f_fun is None else np.asarray(f_fun(x), dtype=float)
        if f is not None and f.shape != x.shape:
            raise ValueError("f_fun must return arrays with element nodal shape")

        exact = None if exact_fun is None else np.asarray(exact_fun(x), dtype=float)
        if exact is not None and exact.shape != x.shape:
            raise ValueError("exact_fun must return arrays with element nodal shape")

        b_left = evaluate_function_at_point(b_fun, x_left)
        exact_left = (
            None if exact_fun is None else evaluate_function_at_point(exact_fun, x_left)
        )

        elements.append(
            Element1D(
                x_left=x_left,
                x_right=x_right,
                x=x,
                D=D,
                H=H,
                H_inv=H_inv,
                tL=operator.tL,
                tR=operator.tR,
                a=a,
                b=b,
                b_left=b_left,
                f=f,
                exact_left=exact_left,
            )
        )
    return elements


def make_uniform_elements(
    domain: tuple[float, float],
    num_elements: int,
    operators: OperatorChoice | Sequence[OperatorChoice],
    a_fun: Callable[[np.ndarray], np.ndarray],
    b_fun: Callable[[np.ndarray], np.ndarray],
    f_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    exact_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Element1D]:
    if num_elements < 1:
        raise ValueError("num_elements must be >= 1")
    bounds = np.linspace(domain[0], domain[1], num_elements + 1)
    return make_elements(
        bounds,
        operators,
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )


def trace_left(element: Element1D, u_local: np.ndarray) -> float:
    return float(element.tL @ u_local)


def trace_right(element: Element1D, u_local: np.ndarray) -> float:
    return float(element.tR @ u_local)
