from src.operator_library import all_operators
from src.operators import check_sbp_property


def test_reference_sbp_property_for_builtin_operators() -> None:
    for op in all_operators():
        assert check_sbp_property(op, tol=1e-13)
