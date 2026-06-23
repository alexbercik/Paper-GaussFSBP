from __future__ import annotations

from typing import Callable

import numpy as np

from .elements import Element1D


BlockDict = dict[tuple[int, int], np.ndarray]


def _flux_coefficients(sat_type: str) -> tuple[float, float]:
    """Coefficients in ``f* = cL * left_flux_state + cR * right_flux_state``."""
    if sat_type in {"symmetric", "central"}:
        return 0.5, 0.5
    if sat_type in {"upwind", "rusanov"}:
        return 1.0, 0.0
    raise ValueError(f"Unknown sat_type '{sat_type}'")


def _add_block(blocks: BlockDict, i: int, j: int, value: np.ndarray) -> None:
    key = (i, j)
    if key in blocks:
        blocks[key] += value
    else:
        blocks[key] = value.copy()


def _element_sat_vectors(
    element: Element1D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # vL/vR are the column vectors A H^{-1} tL/tR that distribute face penalties
    # into the element volume. qL/qR are the row vectors that evaluate
    # tL^T B u and tR^T B u from a local state vector u.
    ahinv = element.a * element.H_inv
    vL = ahinv * element.tL
    vR = ahinv * element.tR
    qL = element.tL * element.b
    qR = element.tR * element.b
    return vL, vR, qL, qR


def left_boundary_flux_state(
    element: Element1D,
    left_bc_fun: Callable[[float], float] | None = None,
) -> float:
    """Upwind left inflow flux ``b(x_L) u_bc`` with ``b`` exact at ``x_left``."""
    if left_bc_fun is None:
        if element.exact_left is None:
            raise ValueError(
                "left_bc_fun is required when elements do not store exact_left; "
                "use left_bc_fun=lambda _x: 0.0 for a homogeneous inflow"
            )
        u_left = element.exact_left
    else:
        u_left = float(left_bc_fun(element.x_left))

    return element.b_left * u_left


def right_boundary_flux_state(element: Element1D, u_local: np.ndarray) -> float:
    """Outflow flux state ``tR^T B u`` on the physical right boundary."""
    return float((element.tR * element.b) @ np.asarray(u_local, dtype=float))


def calc_sat(
    elements: list[Element1D],
    local_u: list[np.ndarray],
    sat_type: str,
    left_bc_fun: Callable[[float], float] | None = None,
) -> list[np.ndarray]:
    """Evaluate SAT terms added to ``du/dt`` on each element.

    The semidiscrete update is

        du/dt = f - A D(Bu) + SAT(u),

    with

        SAT_j = A_j H_j^{-1} [
            tL_j (f*_{j-1/2} - tL_j^T B_j u_j)
            - tR_j (f*_{j+1/2} - tR_j^T B_j u_j)
        ].

    The physical left boundary is treated as a positive-speed inflow with
    ``f* = b(x_L) u_bc``. If ``left_bc_fun`` is omitted, ``u_bc`` is the
    element's stored exact value at ``x_L``. The physical right boundary is an
    outflow with ``f* = tR^T B u``, so its SAT contribution is exactly zero.
    """
    if len(local_u) != len(elements):
        raise ValueError("local_u must contain one state vector per element")

    states = [np.asarray(u, dtype=float) for u in local_u]
    for element, u in zip(elements, states):
        if u.shape != element.x.shape:
            raise ValueError("Each local state must match its element nodal shape")

    sat = [np.zeros_like(element.x) for element in elements]
    if not elements:
        return sat

    cL, cR = _flux_coefficients(sat_type)
    sat_vectors = [_element_sat_vectors(element) for element in elements]

    vL0, _, qL0, _ = sat_vectors[0]
    boundary_flux_state = left_boundary_flux_state(elements[0], left_bc_fun)
    sat[0] += vL0 * (boundary_flux_state - float(qL0 @ states[0]))

    for j in range(len(elements) - 1):
        _, vR_j, _, qR_j = sat_vectors[j]
        vL_k, _, qL_k, _ = sat_vectors[j + 1]

        left_flux_state = float(qR_j @ states[j])
        right_flux_state = float(qL_k @ states[j + 1])
        numerical_flux = cL * left_flux_state + cR * right_flux_state

        sat[j] -= vR_j * (numerical_flux - left_flux_state)
        sat[j + 1] += vL_k * (numerical_flux - right_flux_state)

    # No right boundary SAT is added. For positive speed the outflow numerical
    # flux is the element's own right flux state, so
    # -A H^{-1} tR (f* - tR^T B u) = 0.
    return sat


def build_sat_contributions(
    elements: list[Element1D],
    sat_type: str,
    left_bc_fun: Callable[[float], float] | None,
) -> tuple[BlockDict, list[np.ndarray]]:
    """Assemble SAT contributions to the steady LHS and known-data RHS.

    ``calc_sat`` evaluates the SAT in the time update. The steady system uses
    ``A D(Bu) - SAT_linear(u) = f + SAT_known``, so these blocks are the
    negative of the SAT's unknown-state Jacobian.
    """
    num_elements = len(elements)
    blocks: BlockDict = {}
    rhs = [np.zeros_like(element.x) for element in elements]

    if num_elements == 0:
        return blocks, rhs

    cL, cR = _flux_coefficients(sat_type)
    sat_vectors = [_element_sat_vectors(element) for element in elements]

    first = elements[0]
    vL0, _, qL0, _ = sat_vectors[0]
    boundary_flux_state = left_boundary_flux_state(first, left_bc_fun)

    # Left boundary SAT in du/dt is vL * (f* - qL u). Therefore the steady LHS
    # gets +vL qL and the RHS gets the known +vL f* contribution.
    _add_block(blocks, 0, 0, np.outer(vL0, qL0))
    rhs[0] += vL0 * boundary_flux_state

    for j in range(num_elements - 1):
        _, vR_j, _, qR_j = sat_vectors[j]
        vL_k, _, qL_k, _ = sat_vectors[j + 1]

        # Left element right face in SAT: -vR_j * (f* - qR_j u_j).
        _add_block(blocks, j, j, (cL - 1.0) * np.outer(vR_j, qR_j))
        _add_block(blocks, j, j + 1, cR * np.outer(vR_j, qL_k))

        # Right element left face in SAT: +vL_k * (f* - qL_k u_k).
        _add_block(blocks, j + 1, j, -cL * np.outer(vL_k, qR_j))
        _add_block(blocks, j + 1, j + 1, (1.0 - cR) * np.outer(vL_k, qL_k))

    # No right boundary block is needed: the outflow SAT is identically zero.
    return blocks, rhs
