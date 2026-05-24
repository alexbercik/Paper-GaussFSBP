from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .mesh import Mesh1D
from .operators import Operator, OperatorRepository


@dataclass
class Element1D:
    index: int
    x_left: float
    x_right: float
    operator: Operator
    x: np.ndarray
    D_x: np.ndarray
    H_x: np.ndarray
    H_inv_x: np.ndarray
    tL: np.ndarray
    tR: np.ndarray
    a: np.ndarray
    b: np.ndarray
    f: np.ndarray | None = None
    exact: np.ndarray | None = None


def map_reference_to_physical(xi: np.ndarray, xL: float, xR: float) -> np.ndarray:
    xc = 0.5 * (xL + xR)
    h = xR - xL
    return xc + 0.5 * h * np.asarray(xi, dtype=float)


def scale_operator_to_element(
    operator: Operator, xL: float, xR: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    h = xR - xL
    x = map_reference_to_physical(operator.nodes, xL, xR)
    D_x = (2.0 / h) * operator.D
    H_x = (0.5 * h) * operator.H
    H_inv_x = 1.0 / H_x
    return x, D_x, H_x, H_inv_x


def create_elements(
    mesh: Mesh1D,
    repository: OperatorRepository,
    a_fun: Callable[[np.ndarray], np.ndarray],
    b_fun: Callable[[np.ndarray], np.ndarray],
    f_fun: Callable[[np.ndarray], np.ndarray] | None = None,
    exact_fun: Callable[[np.ndarray], np.ndarray] | None = None,
) -> list[Element1D]:
    elements: list[Element1D] = []
    for i in range(mesh.num_elements):
        x_left = float(mesh.element_bounds[i])
        x_right = float(mesh.element_bounds[i + 1])
        spec = mesh.element_specs[i]
        operator = repository.get_operator(spec.basis, spec.op_type, spec.selector)

        x, D_x, H_x, H_inv_x = scale_operator_to_element(operator, x_left, x_right)
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

        elements.append(
            Element1D(
                index=i,
                x_left=x_left,
                x_right=x_right,
                operator=operator,
                x=x,
                D_x=D_x,
                H_x=H_x,
                H_inv_x=H_inv_x,
                tL=operator.tL,
                tR=operator.tR,
                a=a,
                b=b,
                f=f,
                exact=exact,
            )
        )
    return elements


def trace_left(element: Element1D, u_local: np.ndarray) -> float:
    return float(element.tL @ u_local)


def trace_right(element: Element1D, u_local: np.ndarray) -> float:
    return float(element.tR @ u_local)
