"""Reference-element SBP operator library.

Operators are indexed by ``(basis, quad_basis, op_type, selector)`` with
``basis`` and ``quad_basis`` matched up to permutation. The ``selector``
disambiguates multiple entries with the same first three keys.

Add new operators to ``OPERATOR_ENTRIES`` below.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .operators import Operator, canonical_basis_key


@dataclass(frozen=True)
class OperatorSpec:
    """Reference operator lookup key.

    Operators are selected by basis, quadrature basis, operator type, and
    selector. Basis lists are matched up to permutation during lookup.
    """

    basis: tuple[str, ...]
    quad_basis: tuple[str, ...]
    op_type: str
    selector: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "basis", tuple(self.basis))
        object.__setattr__(self, "quad_basis", tuple(self.quad_basis))


def _entry(
    *,
    basis: list[str],
    quad_basis: list[str],
    op_type: str,
    selector: int,
    nodes: np.ndarray,
    D: np.ndarray,
    H: np.ndarray,
    tL: np.ndarray,
    tR: np.ndarray,
) -> dict[str, Any]:
    return {
        "basis": basis,
        "quad_basis": quad_basis,
        "op_type": op_type,
        "selector": selector,
        "nodes": nodes,
        "D": D,
        "H": H,
        "tL": tL,
        "tR": tR,
    }


OPERATOR_ENTRIES: tuple[dict[str, Any], ...] = (
    _entry(
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3"],
        op_type="closed",
        selector=0,
        nodes=np.array([-1.0, 0.0, 1.0]),
        D=np.array(
            [
                [-1.5, 2.0, -0.5],
                [-0.5, 0.0, 0.5],
                [0.5, -2.0, 1.5],
            ]
        ),
        H=np.array([1.0 / 3.0, 4.0 / 3.0, 1.0 / 3.0]),
        tL=np.array([1.0, 0.0, 0.0]),
        tR=np.array([0.0, 0.0, 1.0]),
    ),
    _entry(
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5"],
        op_type="open",
        selector=0,
        nodes=np.array([-np.sqrt(3.0/5.0), 0.0, np.sqrt(3.0/5.0)]),
        D=np.array(
            [
                [-np.sqrt(15.0)/2.0, 2.0*np.sqrt(15.0)/3.0, -np.sqrt(15.0)/6.0],
                [-np.sqrt(15.0)/6.0, 0.0, np.sqrt(15.0)/6.0],
                [np.sqrt(15.0)/6.0, -2.0*np.sqrt(15.0)/3.0, np.sqrt(15.0)/2.0],
            ]
        ),
        H=np.array([5.0 / 9.0, 8.0 / 9.0, 5.0 / 9.0]),
        tL=np.array([(5.0 + np.sqrt(15.0))/6.0, -2.0/3.0, (5.0 - np.sqrt(15.0))/6.0]),
        tR=np.array([(5.0 - np.sqrt(15.0))/6.0, -2.0/3.0, (5.0 + np.sqrt(15.0))/6.0]),
    ),
    _entry(
        basis=["1", "x", "x^2"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4"],
        op_type="half-open-right",
        selector=0,
        nodes=np.array([-1.0, -0.2898979485566356, 0.6898979485566356]),
        D=np.array([
                        [-2.0, 2.4288690166235205, -0.4288690166235206],
                        [-0.816496580927726, 0.3876275643042055, 0.4288690166235206],
                        [0.816496580927726, -2.4288690166235205, 1.6123724356957945],
                    ]),
        H=np.array([2.0/9.0, 1.0249716523768433, 0.7528061254009346]),
        tL=np.array([1.0, 0.0, 0.0]),
        tR=np.array([1.0/3.0, -0.8914115380582557, 1.5580782047249224]),
    ),
    _entry(
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "x^6", "x^7"],
        op_type="open",
        selector=0,
        nodes=np.array([-0.8611363115940526, -0.3399810435848563, 0.3399810435848563, 0.8611363115940526]),
        D=np.array([
                        [-3.3320002363522816, 4.8601544156851961, -2.1087823484951791, 0.5806281691622645],
                        [-0.7575576147992339, -0.3844143922232086, 1.4706702312807167, -0.3286982242582743],
                        [0.3286982242582743, -1.4706702312807167, 0.3844143922232086, 0.7575576147992339],
                        [-0.5806281691622645, 2.1087823484951791, -4.8601544156851961, 3.3320002363522816],
                    ]),
        H=np.array([0.3478548451374539, 0.6521451548625461, 0.6521451548625461, 0.3478548451374539]),
        tL=np.array([1.5267881254572668, -0.8136324494869273, 0.4007615203116504, -0.1139171962819899]),
        tR=np.array([-0.1139171962819899, 0.4007615203116504, -0.8136324494869273, 1.5267881254572668]),
    ),
    _entry(
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5"],
        op_type="closed",
        selector=0,
        nodes=np.array([-1.0, -0.4472135954999579, 0.4472135954999579, 1.0]),
        D=np.array([
                        [-3.0, 4.0450849718747373, -1.545084971874737, 0.5],
                        [-0.8090169943749475, 0.0, 1.1180339887498949, -0.3090169943749474],
                        [0.3090169943749474, -1.1180339887498949, 0.0, 0.8090169943749475],
                        [-0.5, 1.545084971874737, -4.0450849718747373, 3.0],
                    ]),
        H=np.array([1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([0.0, 0.0, 0.0, 1.0]),
    ),
    _entry(
        basis=["1", "x", "x^2, x^3"],
        quad_basis=["1", "x", "x^2", "x^3", "x^4", "x^5", "x^6"],
        op_type="half-open-right",
        selector=0,
        nodes=np.array([-1.0, -0.5753189235216941, 0.1810662711185306, 0.8228240809745921]),
        D=np.array([
                        [-3.75, 4.7935967957376917, -1.3502656444120271, 0.3066688486743356],
                        [-1.1566785437711786, 0.3173960475776094, 1.0356811085889206, -0.1963986123953511],
                        [0.5309238658498484, -1.6876714615931911, 0.6105500144473459, 0.5461975812959968],
                        [-0.9813881792215268, 2.6047041188063171, -4.4453698775598349, 2.8220539379750447],
                    ]),
        H=np.array([0.125, 0.6576886399601195, 0.7763869376863438, 0.4409244223535368]),
        tL=np.array([1.0, 0.0, 0.0, 0.0]),
        tR=np.array([-0.25, 0.6461389554268266, -0.9736765952010225, 1.5775376397741958]),
    ),
)


def operator_lookup_key(
    basis: list[str] | tuple[str, ...],
    quad_basis: list[str] | tuple[str, ...],
    op_type: str,
    selector: int,
) -> tuple[tuple[str, ...], tuple[str, ...], str, int]:
    return (
        canonical_basis_key(list(basis)),
        canonical_basis_key(list(quad_basis)),
        op_type,
        selector,
    )


_OPERATORS: list[Operator] | None = None
_OPERATOR_INDEX: dict[tuple[tuple[str, ...], tuple[str, ...], str, int], Operator] | None = None


def all_operators() -> list[Operator]:
    global _OPERATORS
    if _OPERATORS is None:
        _OPERATORS = [Operator(**entry) for entry in OPERATOR_ENTRIES]
    return list(_OPERATORS)


def _operator_index() -> dict[tuple[tuple[str, ...], tuple[str, ...], str, int], Operator]:
    global _OPERATOR_INDEX
    if _OPERATOR_INDEX is None:
        index: dict[tuple[tuple[str, ...], tuple[str, ...], str, int], Operator] = {}
        for operator in all_operators():
            key = operator_lookup_key(
                operator.basis, operator.quad_basis, operator.op_type, operator.selector
            )
            if key in index:
                raise ValueError(f"Duplicate operator entry for {key}")
            index[key] = operator
        _OPERATOR_INDEX = index
    return _OPERATOR_INDEX


def selectors_for(
    basis: list[str] | tuple[str, ...],
    quad_basis: list[str] | tuple[str, ...],
    op_type: str,
) -> list[int]:
    """Return sorted selector indices available for ``(basis, quad_basis, op_type)``."""
    key = operator_lookup_key(basis, quad_basis, op_type, 0)[:3]
    selectors = [op.selector for op_key, op in _operator_index().items() if op_key[:3] == key]
    return sorted(selectors)


def get_operator(
    basis: list[str] | tuple[str, ...],
    quad_basis: list[str] | tuple[str, ...],
    op_type: str,
    selector: int = 0,
) -> Operator:
    """Look up a built-in reference operator.

    The lookup key is exactly ``(basis, quad_basis, op_type, selector)``, with
    ``basis`` and ``quad_basis`` matched up to permutation.
    """
    key = operator_lookup_key(basis, quad_basis, op_type, selector)
    try:
        return _operator_index()[key]
    except KeyError as exc:
        available = selectors_for(basis, quad_basis, op_type)
        raise KeyError(
            f"No operator for basis={list(basis)}, quad_basis={list(quad_basis)}, "
            f"op_type={op_type}, selector={selector}. Available selectors: {available}"
        ) from exc


def operator_from_spec(spec: OperatorSpec) -> Operator:
    return get_operator(spec.basis, spec.quad_basis, spec.op_type, spec.selector)
