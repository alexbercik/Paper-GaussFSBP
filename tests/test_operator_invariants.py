import numpy as np

from src.operator_library import all_operators
from src.operators import check_sbp_property


def test_builtin_operator_invariants() -> None:
    for op in all_operators():
        # The checked-in operators are the trusted reference data used by the
        # drivers, so verify the two algebraic properties every run depends on.
        assert check_sbp_property(op, tol=1.0e-13)
        np.testing.assert_allclose(
            op.D @ np.ones(op.nodes.size),
            0.0,
            atol=1.0e-10,
        )
