from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .elements import Element1D, create_elements
from .mesh import Mesh1D
from .operators import OperatorRepository
from .problems import Problem
from .sats import build_sat_contributions


@dataclass
class AssembledSystem:
    elements: list[Element1D]
    matrix: sp.csc_matrix
    rhs: np.ndarray
    offsets: np.ndarray


def element_offsets(elements: list[Element1D]) -> np.ndarray:
    sizes = [element.x.size for element in elements]
    offsets = np.zeros(len(sizes) + 1, dtype=int)
    offsets[1:] = np.cumsum(sizes)
    return offsets


def _base_local_operator(element: Element1D) -> np.ndarray:
    return (element.a[:, None] * element.D_x) * element.b[None, :]


def _assemble_from_elements(
    elements: list[Element1D],
    sat_type: str,
    left_bc_fun,
    homogeneous_left_bc: bool,
    include_forcing: bool,
) -> tuple[sp.csc_matrix, np.ndarray]:
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
        homogeneous_left_bc=homogeneous_left_bc,
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


def assemble_homogeneous_operator(
    elements: list[Element1D],
    sat_type: str = "upwind",
) -> sp.csc_matrix:
    matrix, _ = _assemble_from_elements(
        elements=elements,
        sat_type=sat_type,
        left_bc_fun=lambda _x: 0.0,
        homogeneous_left_bc=True,
        include_forcing=False,
    )
    return matrix


def assemble_system(
    mesh: Mesh1D,
    repository: OperatorRepository,
    problem: Problem,
    sat_type: str = "upwind",
) -> AssembledSystem:
    left_bc_fun = problem.left_bc_fun if problem.left_bc_fun is not None else problem.exact_fun
    if left_bc_fun is None:
        raise ValueError("Problem must provide exact_fun or left_bc_fun for left boundary flux")

    elements = create_elements(
        mesh=mesh,
        repository=repository,
        a_fun=problem.a_fun,
        b_fun=problem.b_fun,
        f_fun=problem.f_fun,
        exact_fun=problem.exact_fun,
    )

    matrix, rhs = _assemble_from_elements(
        elements=elements,
        sat_type=sat_type,
        left_bc_fun=left_bc_fun,
        homogeneous_left_bc=False,
        include_forcing=True,
    )

    return AssembledSystem(
        elements=elements,
        matrix=matrix,
        rhs=rhs,
        offsets=element_offsets(elements),
    )
