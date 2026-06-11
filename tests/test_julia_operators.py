from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from src.lib.julia_operators import (
    JuliaBasis,
    _infer_op_type,
    build_operator_from_julia,
)
from src.operator_library import get_operator
from src.operators import Operator, check_sbp_property


requires_juliacall = pytest.mark.skipif(
    importlib.util.find_spec("juliacall") is None,
    reason="optional dependency juliacall is not installed",
)


def polynomial_bases() -> tuple[JuliaBasis, JuliaBasis]:
    op_basis = JuliaBasis(
        labels=["1", "x", "x^2"],
        functions=["x -> one(x)", "x -> x", "x -> x^2"],
        derivatives=["x -> zero(x)", "x -> one(x)", "x -> 2 * x"],
    )
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3"],
        functions=["x -> one(x)", "x -> x", "x -> x^2", "x -> x^3"],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2 * x",
            "x -> 3 * x^2",
        ],
    )
    return op_basis, quad_basis


def test_op_type_is_inferred_from_even_quad_basis() -> None:
    _, quad_basis = polynomial_bases()

    assert _infer_op_type(quad_basis, "upper") == "closed"
    assert _infer_op_type(quad_basis, "lower") == "open"


def test_op_type_is_inferred_from_odd_quad_basis() -> None:
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2"],
        functions=["x -> one(x)", "x -> x", "x -> x^2"],
    )

    assert _infer_op_type(quad_basis, "upper") == "half-open-left"
    assert _infer_op_type(quad_basis, "lower") == "half-open-right"


def test_op_type_keyword_is_rejected_before_julia_loads() -> None:
    op_basis, quad_basis = polynomial_bases()

    with pytest.raises(TypeError, match="op_type is inferred"):
        build_operator_from_julia(op_basis, quad_basis, op_type="open")


@requires_juliacall
def test_float64_polynomial_operator_matches_builtin() -> None:
    op_basis, quad_basis = polynomial_bases()
    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        precision="float64",
        orthogonalize=True,
        principal="upper",
    )

    expected = get_operator("LGLp2")
    assert isinstance(built, Operator)
    assert built.op_type == "closed"
    np.testing.assert_allclose(built.interval, expected.interval, atol=1e-14)
    np.testing.assert_allclose(built.nodes, expected.nodes, atol=1e-12)
    np.testing.assert_allclose(built.H, expected.H, atol=1e-12)
    np.testing.assert_allclose(built.D, expected.D, atol=1e-12)
    np.testing.assert_allclose(built.tL, expected.tL, atol=1e-12)
    np.testing.assert_allclose(built.tR, expected.tR, atol=1e-12)


@requires_juliacall
def test_bigfloat_polynomial_operator_uses_32_digits() -> None:
    op_basis, quad_basis = polynomial_bases()
    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        precision="bigfloat",
        digits=32,
        orthogonalize=True,
        principal="upper",
    )

    assert isinstance(built, Operator)
    assert built.nodes.dtype == float
    assert check_sbp_property(built, tol=1e-12)


@requires_juliacall
def test_exponential_epsilon_callable_builds() -> None:
    epsilon = 0.1
    epsilon_text = format(epsilon, "g")
    eps = f'BigFloat("{epsilon_text}")'
    exp_eps = f"x -> exp(x) / {eps}"

    op_basis = JuliaBasis(
        labels=["1", "x", "exp(x)/epsilon"],
        functions=["x -> one(x)", "x -> x", exp_eps],
        derivatives=["x -> zero(x)", "x -> one(x)", exp_eps],
    )
    quad_basis = JuliaBasis(
        labels=[
            "1",
            "x",
            "x^2",
            "exp(x)/epsilon",
            "x exp(x)/epsilon",
            "exp(2x)/epsilon^2",
        ],
        functions=[
            "x -> one(x)",
            "x -> x",
            "x -> x^2",
            exp_eps,
            f"x -> x * exp(x) / {eps}",
            f"x -> exp(2 * x) / ({eps} * {eps})",
        ],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2 * x",
            exp_eps,
            f"x -> (one(x) + x) * exp(x) / {eps}",
            f"x -> 2 * exp(2 * x) / ({eps} * {eps})",
        ],
    )

    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat",
        digits=32,
        orthogonalize=True,
        principal="lower",
    )

    assert isinstance(built, Operator)
    np.testing.assert_allclose(built.interval, np.array([0.0, 1.0]))
    assert check_sbp_property(built, tol=1e-9)
