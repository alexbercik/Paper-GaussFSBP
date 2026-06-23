from __future__ import annotations

from typing import Literal

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

OnSingular = Literal["raise", "nan"]


def solve_steady(
    system_matrix: sp.spmatrix,
    rhs: np.ndarray,
    *,
    on_singular: OnSingular = "raise",
) -> tuple[np.ndarray, dict[str, int | bool]]:
    """Solve ``system_matrix @ u = rhs`` for the steady state.

  When the factorization fails because the matrix is singular, ``on_singular``
  controls the behavior:

  - ``"raise"``: re-raise the factorization error (default)
  - ``"nan"``: return a vector of NaNs and set ``diagnostics["singular"]``
    """
    matrix_csc = system_matrix.tocsc()
    rhs = np.asarray(rhs, dtype=float)
    diagnostics: dict[str, int | bool] = {
        "nnz": int(matrix_csc.nnz),
        "size": int(matrix_csc.shape[0]),
        "singular": False,
    }
    try:
        lu = spla.splu(matrix_csc)
        solution = lu.solve(rhs)
    except RuntimeError as exc:
        if on_singular != "nan" or "singular" not in str(exc).lower():
            raise
        diagnostics["singular"] = True
        solution = np.full(matrix_csc.shape[0], np.nan, dtype=float)
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
