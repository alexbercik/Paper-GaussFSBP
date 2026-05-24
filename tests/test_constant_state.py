import numpy as np

from gaussfsbp.operators import builtin_operator_repository


def test_constant_state_derivative_zero() -> None:
    repo = builtin_operator_repository()
    for op in repo.operators:
        ones = np.ones(op.nodes.size)
        assert np.allclose(op.D @ ones, 0.0, atol=1e-14)
