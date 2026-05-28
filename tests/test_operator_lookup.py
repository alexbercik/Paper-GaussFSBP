import pytest

from src.elements import make_uniform_elements
from src.operator_library import (
    OperatorSpec,
    get_operator,
    get_operator_by_name,
    operator_from_spec,
    operator_names,
    selectors_for,
)

BASIS = ["1", "x", "x^2"]
QUAD_BASIS = ["1", "x", "x^2", "x^3"]


def test_builtin_polynomial_closed_selector_is_unique() -> None:
    assert selectors_for(BASIS, QUAD_BASIS, "closed") == [0]


def test_lookup_by_basis_quad_basis_op_type_selector() -> None:
    op = get_operator(BASIS, QUAD_BASIS, "closed")
    assert op.selector == 0
    assert op.name == "LGLp2"
    assert op.basis == BASIS
    assert op.quad_basis == QUAD_BASIS


def test_lookup_by_name_matches_structural_lookup() -> None:
    op = get_operator_by_name("LGLp2")
    assert op is get_operator(BASIS, QUAD_BASIS, "closed")
    assert get_operator("LGLp2") is op
    assert get_operator(name="LGLp2") is op
    assert "LGLp2" in operator_names()


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


def test_operator_spec_can_use_name() -> None:
    assert operator_from_spec(OperatorSpec(name="LGLp2")) is get_operator("LGLp2")
    assert operator_from_spec(OperatorSpec("LGLp2")) is get_operator("LGLp2")


def test_named_spec_rejects_structural_fields() -> None:
    with pytest.raises(ValueError, match="either name or basis"):
        OperatorSpec(BASIS, QUAD_BASIS, "closed", name="LGLp2")


def test_unknown_name_reports_available_names() -> None:
    with pytest.raises(KeyError, match="Available names"):
        get_operator_by_name("does-not-exist")


def test_make_elements_accepts_operator_name() -> None:
    elements = make_uniform_elements(
        (0.0, 1.0),
        2,
        "LGLp2",
        a_fun=lambda x: x * 0.0 + 1.0,
        b_fun=lambda x: x * 0.0 + 1.0,
    )

    assert len(elements) == 2
    assert elements[0].x.size == get_operator("LGLp2").nodes.size
