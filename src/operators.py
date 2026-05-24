from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

VALID_OP_TYPES = {"open", "closed", "half-open-left", "half-open-right"}
REQUIRED_OPERATOR_KEYS = {
    "basis",
    "quad_basis",
    "op_type",
    "nodes",
    "D",
    "H",
    "tL",
    "tR",
    "selector",
}


@dataclass(frozen=True)
class Operator:
    basis: list[str]
    quad_basis: list[str]
    op_type: str
    nodes: np.ndarray
    D: np.ndarray
    H: np.ndarray
    tL: np.ndarray
    tR: np.ndarray
    selector: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", np.asarray(self.nodes, dtype=float).copy())
        object.__setattr__(self, "D", np.asarray(self.D, dtype=float).copy())
        object.__setattr__(self, "H", np.asarray(self.H, dtype=float).copy())
        object.__setattr__(self, "tL", np.asarray(self.tL, dtype=float).copy())
        object.__setattr__(self, "tR", np.asarray(self.tR, dtype=float).copy())
        validate_operator_dict(self.to_dict())

    def boundary_matrix(self) -> np.ndarray:
        return np.outer(self.tR, self.tR) - np.outer(self.tL, self.tL)

    def to_dict(self) -> dict[str, Any]:
        return {
            "basis": list(self.basis),
            "quad_basis": list(self.quad_basis),
            "op_type": self.op_type,
            "nodes": self.nodes.copy(),
            "D": self.D.copy(),
            "H": self.H.copy(),
            "tL": self.tL.copy(),
            "tR": self.tR.copy(),
            "selector": self.selector,
        }


def validate_operator_dict(operator_data: dict[str, Any]) -> None:
    if set(operator_data.keys()) != REQUIRED_OPERATOR_KEYS:
        raise ValueError(
            "Operator dictionary must contain exactly keys: "
            f"{sorted(REQUIRED_OPERATOR_KEYS)}"
        )

    basis = operator_data["basis"]
    quad_basis = operator_data["quad_basis"]
    op_type = operator_data["op_type"]
    selector = operator_data["selector"]
    nodes = np.asarray(operator_data["nodes"], dtype=float)
    D = np.asarray(operator_data["D"], dtype=float)
    H = np.asarray(operator_data["H"], dtype=float)
    tL = np.asarray(operator_data["tL"], dtype=float)
    tR = np.asarray(operator_data["tR"], dtype=float)

    if not isinstance(basis, list) or any(not isinstance(item, str) for item in basis):
        raise TypeError("basis must be a list of strings")
    if not isinstance(quad_basis, list) or any(
        not isinstance(item, str) for item in quad_basis
    ):
        raise TypeError("quad_basis must be a list of strings")
    if op_type not in VALID_OP_TYPES:
        raise ValueError(f"Invalid op_type '{op_type}'")
    if not isinstance(selector, int) or isinstance(selector, bool):
        raise TypeError("selector must be an integer")

    if nodes.ndim != 1:
        raise ValueError("nodes must be one-dimensional")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError("D must be square")

    n = nodes.size
    if D.shape != (n, n):
        raise ValueError("D must have shape (N, N), where N = len(nodes)")
    if H.ndim != 1 or H.shape[0] != n:
        raise ValueError("H must be one-dimensional with length N")
    if tL.ndim != 1 or tL.shape[0] != n:
        raise ValueError("tL must be one-dimensional with length N")
    if tR.ndim != 1 or tR.shape[0] != n:
        raise ValueError("tR must be one-dimensional with length N")
    if np.any(H <= 0.0):
        raise ValueError("H entries must be positive")


def canonical_basis_key(basis: list[str]) -> tuple[str, ...]:
    """Order-invariant key for matching basis / quad_basis lists."""
    return tuple(sorted(basis))


def check_sbp_property(operator: Operator, tol: float = 1e-12) -> bool:
    HD = operator.H[:, None] * operator.D
    residual = HD + HD.T - operator.boundary_matrix()
    return float(np.linalg.norm(residual, ord=np.inf)) <= tol

