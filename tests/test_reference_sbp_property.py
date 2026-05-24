from gaussfsbp.operators import builtin_operator_repository, check_sbp_property


def test_reference_sbp_property_for_builtin_operators() -> None:
    repo = builtin_operator_repository()
    for op in repo.operators:
        assert check_sbp_property(op, tol=1e-13)
