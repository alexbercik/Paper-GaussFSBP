from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def solve_steady(system_matrix: sp.spmatrix, rhs: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    matrix_csc = system_matrix.tocsc()
    lu = spla.splu(matrix_csc)
    solution = lu.solve(np.asarray(rhs, dtype=float))
    diagnostics = {"nnz": int(matrix_csc.nnz), "size": int(matrix_csc.shape[0])}
    return solution, diagnostics


def split_global_vector(global_vec: np.ndarray, sizes: list[int]) -> list[np.ndarray]:
    global_vec = np.asarray(global_vec, dtype=float)
    offsets = np.zeros(len(sizes) + 1, dtype=int)
    offsets[1:] = np.cumsum(sizes)
    return [global_vec[offsets[i] : offsets[i + 1]] for i in range(len(sizes))]


def concatenate_local_vectors(local_vectors: list[np.ndarray]) -> np.ndarray:
    if not local_vectors:
        return np.array([], dtype=float)
    return np.concatenate([np.asarray(vec, dtype=float) for vec in local_vectors])
