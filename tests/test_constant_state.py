import numpy as np

from src.operator_library import all_operators


def test_constant_state_derivative_zero() -> None:
    for op in all_operators():
        ones = np.ones(op.nodes.size)
        assert np.allclose(op.D @ ones, 0.0, atol=1e-14)
