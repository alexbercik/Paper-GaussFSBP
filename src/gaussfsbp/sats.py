from __future__ import annotations

from typing import Callable

import numpy as np

from .elements import Element1D


BlockDict = dict[tuple[int, int], np.ndarray]


def _flux_coefficients(sat_type: str) -> tuple[float, float]:
    if sat_type == "symmetric":
        return 0.5, 0.5
    if sat_type == "upwind":
        return 1.0, 0.0
    raise ValueError(f"Unknown sat_type '{sat_type}'")


def _add_block(blocks: BlockDict, i: int, j: int, value: np.ndarray) -> None:
    key = (i, j)
    if key in blocks:
        blocks[key] += value
    else:
        blocks[key] = value.copy()


def _element_sat_vectors(element: Element1D) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ahinv = element.a * element.H_inv_x
    vL = ahinv * element.tL
    vR = ahinv * element.tR
    qL = element.tL * element.b
    qR = element.tR * element.b
    return vL, vR, qL, qR


def build_sat_contributions(
    elements: list[Element1D],
    sat_type: str,
    left_bc_fun: Callable[[float], float] | None,
    homogeneous_left_bc: bool = False,
) -> tuple[BlockDict, list[np.ndarray]]:
    num_elements = len(elements)
    blocks: BlockDict = {}
    rhs = [np.zeros_like(element.x) for element in elements]

    if num_elements == 0:
        return blocks, rhs

    cL, cR = _flux_coefficients(sat_type)
    sat_vectors = [_element_sat_vectors(element) for element in elements]

    first = elements[0]
    vL0, _, qL0, _ = sat_vectors[0]
    gL = 0.0
    if not homogeneous_left_bc:
        if left_bc_fun is None:
            raise ValueError("left_bc_fun is required for non-homogeneous assembly")
        gL = float(left_bc_fun(first.x_left))
    boundary_flux_state = float(first.b[0]) * gL

    # Left boundary contribution with outward normal convention:
    # -A H^{-1} tL (f* - left_state)
    _add_block(blocks, 0, 0, np.outer(vL0, qL0))
    rhs[0] -= vL0 * boundary_flux_state

    for j in range(num_elements - 1):
        vL_j, vR_j, qL_j, qR_j = sat_vectors[j]
        vL_k, vR_k, qL_k, qR_k = sat_vectors[j + 1]

        # Left element right face: +A H^{-1} tR (f* - right_state)
        _add_block(blocks, j, j, (cL - 1.0) * np.outer(vR_j, qR_j))
        _add_block(blocks, j, j + 1, cR * np.outer(vR_j, qL_k))

        # Right element left face: -A H^{-1} tL (f* - left_state)
        _add_block(blocks, j + 1, j, -cL * np.outer(vL_k, qR_j))
        _add_block(blocks, j + 1, j + 1, (1.0 - cR) * np.outer(vL_k, qL_k))

    return blocks, rhs
