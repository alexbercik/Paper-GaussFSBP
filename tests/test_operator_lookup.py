import pytest

from src.operator_library import OperatorSpec, get_operator, operator_from_spec, selectors_for

BASIS = ["1", "x", "x^2"]
QUAD_BASIS = ["1", "x", "x^2", "x^3"]


def test_builtin_polynomial_closed_selector_is_unique() -> None:
    assert selectors_for(BASIS, QUAD_BASIS, "closed") == [0]


def test_lookup_by_basis_quad_basis_op_type_selector() -> None:
    op = get_operator(BASIS, QUAD_BASIS, "closed")
    assert op.selector == 0
    assert op.basis == BASIS
    assert op.quad_basis == QUAD_BASIS


def test_missing_selector_raises_with_available_list() -> None:
    with pytest.raises(KeyError, match="Available selectors: \\[0\\]"):
        get_operator(BASIS, QUAD_BASIS, "closed", selector=1)


def test_permuted_basis_and_quad_basis_match_same_operator() -> None:
    op_canonical = get_operator(BASIS, QUAD_BASIS, "closed")
    op_permuted = get_operator(
        ["1", "x^2", "x"], ["x^3", "1", "x", "x^2"], "closed"
    )
    assert op_permuted is op_canonical


def test_operator_spec_uses_full_lookup_key() -> None:
    spec = OperatorSpec(BASIS, QUAD_BASIS, "closed", selector=0)
    assert operator_from_spec(spec) is get_operator(BASIS, QUAD_BASIS, "closed")
