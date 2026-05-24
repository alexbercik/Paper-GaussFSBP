from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .elements import Element1D
from .sats import build_sat_contributions, calc_sat


@dataclass
class AssembledSystem:
    elements: list[Element1D]
    matrix: sp.csc_matrix
    rhs: np.ndarray
    offsets: np.ndarray


def element_offsets(elements: list[Element1D]) -> np.ndarray:
    # a list of the indices pointing to the start of each element in the 
    # global state vector. Used in _split_global_state
    sizes = [element.x.size for element in elements]
    offsets = np.zeros(len(sizes) + 1, dtype=int)
    offsets[1:] = np.cumsum(sizes)
    return offsets


def _base_local_operator(element: Element1D) -> np.ndarray:
    # Local strong-form derivative operator: u_j -> A_j D_j (B_j u_j).
    return (element.a[:, None] * element.D) * element.b[None, :]


def _split_global_state(elements: list[Element1D], u: np.ndarray) -> list[np.ndarray]:
    u = np.asarray(u, dtype=float)
    offsets = element_offsets(elements)
    if u.shape != (offsets[-1],):
        raise ValueError(
            "Global state length must match the total element degrees of freedom"
        )
    return [u[offsets[i] : offsets[i + 1]] for i in range(len(elements))]


def _assemble_from_elements(
    elements: list[Element1D],
    sat_type: str,
    left_bc_fun,
    include_forcing: bool,
) -> tuple[sp.csc_matrix, np.ndarray]:
    """Assemble ``L u = rhs`` for the steady SBP-SAT equation.

    The time update is ``du/dt = f - A D(Bu) + SAT``. Setting ``du/dt = 0``
    gives ``A D(Bu) - SAT_linear(u) = f + SAT_known``; that left-hand operator
    is the matrix returned here.
    """
    num_elements = len(elements)
    local_rhs = [
        np.zeros(element.x.size, dtype=float)
        if not include_forcing or element.f is None
        else element.f.astype(float).copy()
        for element in elements
    ]

    blocks: dict[tuple[int, int], np.ndarray] = {}
    for i, element in enumerate(elements):
        blocks[(i, i)] = _base_local_operator(element)

    sat_blocks, sat_rhs = build_sat_contributions(
        elements=elements,
        sat_type=sat_type,
        left_bc_fun=left_bc_fun,
    )

    for key, value in sat_blocks.items():
        if key in blocks:
            blocks[key] += value
        else:
            blocks[key] = value
    for i in range(num_elements):
        local_rhs[i] += sat_rhs[i]

    block_rows: list[list[sp.spmatrix | None]] = []
    for i in range(num_elements):
        row: list[sp.spmatrix | None] = []
        for j in range(num_elements):
            block = blocks.get((i, j))
            row.append(None if block is None else sp.csr_matrix(block))
        block_rows.append(row)

    matrix = sp.bmat(block_rows, format="csc")
    rhs = np.concatenate(local_rhs) if local_rhs else np.array([], dtype=float)
    return matrix, rhs


def calc_LHS(
    elements: list[Element1D],
    sat_type: str = "upwind",
) -> sp.csc_matrix:
    """Build the homogeneous steady left-hand side matrix.

    For a state vector ``u`` this matrix represents
    ``A D(Bu) - SAT_linear(u)``. The semidiscrete homogeneous Jacobian is
    therefore ``d(calc_RHS)/du = -calc_LHS(...)``.
    """
    matrix, _ = _assemble_from_elements(
        elements=elements,
        sat_type=sat_type,
        left_bc_fun=lambda _x: 0.0,
        include_forcing=False,
    )
    return matrix


def assemble_homogeneous_operator(
    elements: list[Element1D],
    sat_type: str = "upwind",
) -> sp.csc_matrix:
    """Backward-compatible name for ``calc_LHS``."""
    return calc_LHS(elements, sat_type=sat_type)


def calc_RHS(
    elements: list[Element1D],
    u: np.ndarray,
    sat_type: str = "upwind",
    left_bc_fun=None,
    include_forcing: bool = True,
) -> np.ndarray:
    """Evaluate the strong-form semidiscrete update ``du/dt``.

    This applies the SBP-SAT discretization
    ``du/dt = f - A D(Bu) + SAT`` directly from element-local states, without
    using the assembled sparse matrix. If ``left_bc_fun`` is omitted, the left
    inflow state defaults to the exact boundary value stored on the first
    element. Pass ``lambda _x: 0.0`` for a homogeneous inflow.
    """
    local_u = _split_global_state(elements, u)
    local_rhs: list[np.ndarray] = []

    for element, u_local in zip(elements, local_u):
        forcing = np.zeros(element.x.size, dtype=float)
        if include_forcing and element.f is not None:
            forcing = element.f.astype(float).copy()

        strong_derivative = element.a * (element.D @ (element.b * u_local))
        local_rhs.append(forcing - strong_derivative)

    local_sat = calc_sat(
        elements=elements,
        local_u=local_u,
        sat_type=sat_type,
        left_bc_fun=left_bc_fun,
    )
    for i, sat in enumerate(local_sat):
        local_rhs[i] += sat

    return np.concatenate(local_rhs) if local_rhs else np.array([], dtype=float)


def assemble_system(
    elements: list[Element1D],
    left_bc_fun=None,
    sat_type: str = "upwind",
) -> AssembledSystem:
    matrix, rhs = _assemble_from_elements(
        elements=elements,
        sat_type=sat_type,
        left_bc_fun=left_bc_fun,
        include_forcing=True,
    )

    return AssembledSystem(
        elements=elements,
        matrix=matrix,
        rhs=rhs,
        offsets=element_offsets(elements),
    )
