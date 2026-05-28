import numpy as np
import pytest

from src.operator_library import get_operator
from src.operators import validate_operator_dict


def valid_operator_dict() -> dict:
    op = get_operator(["1", "x", "x^2"], ["1", "x", "x^2", "x^3"], "closed")
    return op.to_dict()


def test_invalid_non_square_D() -> None:
    data = valid_operator_dict()
    data["D"] = np.zeros((3, 2))
    with pytest.raises(ValueError):
        validate_operator_dict(data)


def test_invalid_H_length() -> None:
    data = valid_operator_dict()
    data["H"] = np.ones(2)
    with pytest.raises(ValueError):
        validate_operator_dict(data)


def test_invalid_trace_lengths() -> None:
    data = valid_operator_dict()
    data["tL"] = np.ones(2)
    with pytest.raises(ValueError):
        validate_operator_dict(data)

    data = valid_operator_dict()
    data["tR"] = np.ones(2)
    with pytest.raises(ValueError):
        validate_operator_dict(data)


def test_nonpositive_H() -> None:
    data = valid_operator_dict()
    data["H"][0] = 0.0
    with pytest.raises(ValueError):
        validate_operator_dict(data)


def test_invalid_op_type() -> None:
    data = valid_operator_dict()
    data["op_type"] = "bad-type"
    with pytest.raises(ValueError):
        validate_operator_dict(data)


def test_name_may_be_none() -> None:
    data = valid_operator_dict()
    data["name"] = None
    validate_operator_dict(data)


def test_invalid_name_type() -> None:
    data = valid_operator_dict()
    data["name"] = 42
    with pytest.raises(TypeError):
        validate_operator_dict(data)


def test_valid_builtin_operator() -> None:
    validate_operator_dict(valid_operator_dict())
