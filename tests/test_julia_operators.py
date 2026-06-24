from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

import src.lib.julia_operators as julia_operators
from src.lib.julia_operators import (
    JuliaBasis,
    JuliaOperatorError,
    _infer_op_type,
    build_operator_from_julia,
    build_operator_from_sbp_extra,
    legendre_basis_factory,
    print_fsbp_operator_python,
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


def cached_polynomial_bases() -> tuple[JuliaBasis, JuliaBasis]:
    op_basis = JuliaBasis(
        labels=["1", "x", "x^2"],
        factory=legendre_basis_factory(3),
    )
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3"],
        factory=legendre_basis_factory(4),
    )
    return op_basis, quad_basis


def sbp_extra_arrays_for_nodes(
    nodes: list[float],
) -> tuple[list[list[float]], list[float], list[float]]:
    n = len(nodes)
    H = [1.0] * n
    D = [[0.0 for _ in range(n)] for _ in range(n)]
    D[0][0] = -0.5
    D[-1][-1] = 0.5
    return D, H, nodes


def test_julia_basis_accepts_exactly_one_definition_style() -> None:
    factory = legendre_basis_factory(2)

    built = JuliaBasis(labels=["1", "x"], factory=factory)
    assert built.factory == factory
    assert built.functions is None
    assert built.derivatives is None

    with pytest.raises(ValueError, match="functions are required"):
        JuliaBasis(labels=["1"])
    with pytest.raises(ValueError, match="cannot be combined"):
        JuliaBasis(labels=["1"], functions=["x -> one(x)"], factory=factory)


def test_legendre_basis_factory_uses_one_shared_polynomial_factory_call() -> None:
    factory = legendre_basis_factory(
        3,
        additional_functions=["let a = parse(T, \"1.0\"); x -> exp(a*x); end"],
        additional_derivatives=[
            "let a = parse(T, \"1.0\"); x -> a*exp(a*x); end"
        ],
    )

    # One call returns both callable vectors, so GaussFSBP gives every
    # polynomial function and derivative the same LegendreFunctionBlock.
    assert factory.count("GaussFSBP.legendre_functions") == 1
    assert "vcat(polynomial_functions, extra_functions)" in factory
    assert "vcat(polynomial_derivatives, extra_derivatives)" in factory
    assert "parse(T" in factory


def test_legendre_basis_factory_validates_additional_pairs() -> None:
    with pytest.raises(ValueError, match="must have the same length"):
        legendre_basis_factory(
            2,
            additional_functions=["x -> exp(x)"],
        )
    with pytest.raises(ValueError, match="nonnegative integer"):
        legendre_basis_factory(-1)


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


def test_print_fsbp_operator_python_calls_julia_helper(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []
    fake_fsbp = object()

    def fake_print(fsbp: object, num_digits: int) -> None:
        calls.append((fsbp, num_digits))

    # juliacall helper names intentionally mirror the Julia functions.
    fake_julia = SimpleNamespace(__pygaussfsbp_print_operator_python=fake_print)
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    print_fsbp_operator_python(fake_fsbp, num_digits=16)

    assert calls == [(fake_fsbp, 16)]


def test_print_fsbp_operator_python_rejects_invalid_num_digits() -> None:
    with pytest.raises(TypeError, match="num_digits must be an integer"):
        print_fsbp_operator_python(object(), num_digits=True)
    with pytest.raises(ValueError, match="num_digits must be nonnegative"):
        print_fsbp_operator_python(object(), num_digits=-1)


def test_build_operator_print_options_are_rejected_before_julia_loads() -> None:
    op_basis, quad_basis = polynomial_bases()

    with pytest.raises(TypeError, match="print_operator must be True or False"):
        build_operator_from_julia(op_basis, quad_basis, print_operator=1)
    with pytest.raises(ValueError, match="print_num_digits must be nonnegative"):
        build_operator_from_julia(
            op_basis,
            quad_basis,
            print_operator=True,
            print_num_digits=-1,
        )


def test_build_operator_from_sbp_extra_converts_arrays(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    fake_operator = object()
    calls = []

    def fake_build(
        functions: list[str],
        nodes: list[float],
        source: str,
        max_iterations: int,
        g_tol: float,
    ) -> object:
        calls.append(
            (
                functions,
                nodes,
                source,
                max_iterations,
                g_tol,
            )
        )
        return fake_operator

    def fake_arrays(op: object) -> tuple[list[list[float]], list[float], list[float]]:
        assert op is fake_operator
        return (
            [[-5.0, 5.0], [-5.0, 5.0]],
            [0.1, 0.1],
            [0.0, 0.2],
        )

    fake_julia = SimpleNamespace(
        __pygaussfsbp_extra_operator=fake_build,
        __pygaussfsbp_extra_operator_arrays=fake_arrays,
        __pygaussfsbp_extra_basis_residual=lambda op, funcs: 0.0,
    )
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    built = build_operator_from_sbp_extra(
        ["x -> one(x)", "x -> x"],
        2,
        basis_labels=["1", "x"],
        quad_basis_labels=["1", "x"],
        op_type="closed",
        interval=(0.0, 0.2),
        source="basic",
        max_iterations=123,
        g_tol=1.0e-19,
        verbose=True,
    )

    assert calls == [
        (
            ["x -> one(x)", "x -> x"],
            [0.0, 0.2],
            "basic",
            123,
            1.0e-19,
        )
    ]
    assert isinstance(built, Operator)
    assert built.basis == ["1", "x"]
    assert built.quad_basis == ["1", "x"]
    np.testing.assert_allclose(built.nodes, [0.0, 0.2])
    np.testing.assert_allclose(built.tL, [1.0, 0.0])
    np.testing.assert_allclose(built.tR, [0.0, 1.0])
    assert "converged for N=2" in capsys.readouterr().out


def test_build_operator_from_sbp_extra_validates_before_julia_loads() -> None:
    with pytest.raises(ValueError, match="functions and basis_labels"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            2,
            basis_labels=["1", "x"],
            quad_basis_labels=["1"],
            op_type="closed",
        )
    with pytest.raises(ValueError, match="Invalid op_type"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            2,
            basis_labels=["1"],
            quad_basis_labels=["1"],
            op_type="bad",
        )
    with pytest.raises(ValueError, match="initial_num_nodes"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            1,
            basis_labels=["1"],
            quad_basis_labels=["1"],
            op_type="closed",
        )
    with pytest.raises(ValueError, match="source must be"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            2,
            basis_labels=["1"],
            quad_basis_labels=["1"],
            op_type="closed",
            source="regularized",
        )
    with pytest.raises(ValueError, match="g_tol"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            2,
            basis_labels=["1"],
            quad_basis_labels=["1"],
            op_type="closed",
            g_tol=0.0,
        )


def test_build_operator_from_sbp_extra_retries_after_julia_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[float]] = []

    def fake_build(
        functions: list[str],
        nodes: list[float],
        source: str,
        max_iterations: int,
        g_tol: float,
    ) -> list[float]:
        calls.append(nodes)
        if len(calls) == 1:
            raise RuntimeError("no feasible point")
        return nodes

    fake_julia = SimpleNamespace(
        __pygaussfsbp_extra_operator=fake_build,
        __pygaussfsbp_extra_operator_arrays=sbp_extra_arrays_for_nodes,
        __pygaussfsbp_extra_basis_residual=lambda op, funcs: 0.0,
    )
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    built = build_operator_from_sbp_extra(
        ["x -> one(x)"],
        2,
        basis_labels=["1"],
        quad_basis_labels=["1"],
        op_type="closed",
        max_num_nodes=3,
    )

    assert [len(nodes) for nodes in calls] == [2, 3]
    np.testing.assert_allclose(built.nodes, [0.0, 0.5, 1.0])


def test_build_operator_from_sbp_extra_retries_after_residual_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[float]] = []

    def fake_build(
        functions: list[str],
        nodes: list[float],
        source: str,
        max_iterations: int,
        g_tol: float,
    ) -> list[float]:
        calls.append(nodes)
        return nodes

    def fake_arrays(nodes: list[float]) -> tuple[list[list[float]], list[float], list[float]]:
        if len(nodes) == 2:
            return [[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0], nodes
        return sbp_extra_arrays_for_nodes(nodes)

    fake_julia = SimpleNamespace(
        __pygaussfsbp_extra_operator=fake_build,
        __pygaussfsbp_extra_operator_arrays=fake_arrays,
        __pygaussfsbp_extra_basis_residual=lambda op, funcs: 0.0,
    )
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    built = build_operator_from_sbp_extra(
        ["x -> one(x)"],
        2,
        basis_labels=["1"],
        quad_basis_labels=["1"],
        op_type="closed",
        max_num_nodes=3,
    )

    assert [len(nodes) for nodes in calls] == [2, 3]
    np.testing.assert_allclose(built.nodes, [0.0, 0.5, 1.0])


def test_build_operator_from_sbp_extra_errors_after_max_num_nodes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_build(
        functions: list[str],
        nodes: list[float],
        source: str,
        max_iterations: int,
        g_tol: float,
    ) -> list[float]:
        return nodes

    def fake_arrays(nodes: list[float]) -> tuple[list[list[float]], list[float], list[float]]:
        n = len(nodes)
        return [[0.0 for _ in range(n)] for _ in range(n)], [1.0] * n, nodes

    fake_julia = SimpleNamespace(
        __pygaussfsbp_extra_operator=fake_build,
        __pygaussfsbp_extra_operator_arrays=fake_arrays,
        __pygaussfsbp_extra_basis_residual=lambda op, funcs: 0.0,
    )
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    with pytest.raises(JuliaOperatorError, match="N=2..3"):
        build_operator_from_sbp_extra(
            ["x -> one(x)"],
            2,
            basis_labels=["1"],
            quad_basis_labels=["1"],
            op_type="closed",
            max_num_nodes=3,
        )


def test_build_operator_from_sbp_extra_verbose_reports_retry_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls = 0

    def fake_build(
        functions: list[str],
        nodes: list[float],
        source: str,
        max_iterations: int,
        g_tol: float,
    ) -> list[float]:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("not enough nodes")
        return nodes

    fake_julia = SimpleNamespace(
        __pygaussfsbp_extra_operator=fake_build,
        __pygaussfsbp_extra_operator_arrays=sbp_extra_arrays_for_nodes,
        __pygaussfsbp_extra_basis_residual=lambda op, funcs: 0.0,
    )
    monkeypatch.setattr(julia_operators, "_load_julia", lambda: fake_julia)

    build_operator_from_sbp_extra(
        ["x -> one(x)"],
        2,
        basis_labels=["1"],
        quad_basis_labels=["1"],
        op_type="closed",
        max_num_nodes=3,
        verbose=True,
    )

    output = capsys.readouterr().out
    assert "failed for N=2" in output
    assert "converged for N=3" in output


def test_interval_values_use_repr_for_bigfloat() -> None:
    values = julia_operators._interval_values_for_julia((0.0, 1.0), "bigfloat")
    assert values == ["0.0", "1.0"]

    float_values = julia_operators._interval_values_for_julia((-1.0, 1.0), "float64")
    assert float_values == [-1.0, 1.0]


    # Use an unlisted keyword to ensure conversion does not depend on a
    # manually maintained allowlist of GaussFSBP options.
    fake_julia = SimpleNamespace(
        __pygaussfsbp_make_named_tuple=lambda keys, values: dict(zip(keys, values)),
        Symbol=lambda value: ("symbol", value),
        __pygaussfsbp_true=("julia_bool", True),
        __pygaussfsbp_false=("julia_bool", False),
    )

    converted = julia_operators._make_named_tuple(
        fake_julia,
        {
            "opt_method": "sequential",
            "orthogonalize": True,
            "quad_kwargs": {
                "verbose": False,
                "future_option": ["first", "second"],
            },
        },
    )

    assert converted == {
        "opt_method": ("symbol", "sequential"),
        "orthogonalize": ("julia_bool", True),
        "quad_kwargs": {
            "verbose": ("julia_bool", False),
            "future_option": [("symbol", "first"), ("symbol", "second")],
        },
    }


def test_function_expression_kwargs_are_not_converted_to_symbols() -> None:
    calls = []

    fake_julia = SimpleNamespace(
        __pygaussfsbp_parse_function=lambda value: calls.append(("one", value)),
        __pygaussfsbp_parse_functions=lambda values: calls.append(("many", values)),
        Symbol=lambda value: ("symbol", value),
    )

    julia_operators._convert_julia_value(fake_julia, "measure", "x -> exp(-x^2)")
    vector_expr = "[x -> sin(x), x -> cos(x)]"
    julia_operators._convert_julia_value(fake_julia, "test_functions", vector_expr)
    julia_operators._convert_julia_value(
        fake_julia, "test_derivatives", ["x -> cos(x)", "x -> -sin(x)"]
    )

    assert calls == [("one", "x -> exp(-x^2)")]
    assert julia_operators._convert_julia_value(
        fake_julia, "test_functions", vector_expr
    ) == vector_expr
    assert julia_operators._convert_julia_value(
        fake_julia, "test_functions", ["x -> sin(x)", "x -> cos(x)"]
    ) == ["x -> sin(x)", "x -> cos(x)"]


def test_test_spec_string_is_not_split_into_characters() -> None:
    fake_julia = SimpleNamespace(Symbol=lambda value: ("symbol", value))
    vector_expr = "[x -> sin(x)]"

    converted = julia_operators._convert_julia_value(
        fake_julia, "test_functions", vector_expr
    )

    assert converted == vector_expr
    assert converted != list(vector_expr)


def test_objective_weight_list_converts_to_named_tuple() -> None:
    fake_julia = SimpleNamespace(
        __pygaussfsbp_make_named_tuple=lambda keys, values: dict(zip(keys, values)),
        Symbol=lambda value: ("symbol", value),
    )

    converted = julia_operators._convert_julia_value(
        fake_julia, "extrapolation_objective_weights", [0.2, 0.1]
    )

    assert converted == {"accuracy": 0.2, "norm": 0.1}


def test_objective_weight_mapping_is_unchanged() -> None:
    fake_julia = SimpleNamespace(
        __pygaussfsbp_make_named_tuple=lambda keys, values: dict(zip(keys, values)),
        Symbol=lambda value: ("symbol", value),
    )

    converted = julia_operators._convert_julia_value(
        fake_julia,
        "S_objective_weights",
        {"accuracy": 0.9, "norm": 0.1},
    )

    assert converted == {"accuracy": 0.9, "norm": 0.1}


def test_objective_weight_list_rejects_wrong_length() -> None:
    fake_julia = SimpleNamespace(
        __pygaussfsbp_make_named_tuple=lambda keys, values: dict(zip(keys, values)),
        Symbol=lambda value: ("symbol", value),
    )

    with pytest.raises(ValueError, match="exactly two weights"):
        julia_operators._convert_julia_value(
            fake_julia, "S_objective_weights", [0.9]
        )


@requires_juliacall
def test_direct_build_ignores_unparsed_test_specs() -> None:
    op_basis, quad_basis = polynomial_bases()

    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        precision="float64",
        orthogonalize=True,
        principal="upper",
        use_optimization=False,
        test_functions="this is not valid julia",
        test_derivatives="also invalid",
        test_weights=[1.0],
        extrapolation_objective_weights=[0.2, 0.1],
        S_objective_weights=[0.9, 0.1],
    )

    assert isinstance(built, Operator)
    assert check_sbp_property(built, tol=1e-12)


@requires_juliacall
def test_test_specs_resolve_with_typed_expressions() -> None:
    jl = julia_operators._load_julia()
    spec = ["let a = parse(T, \"0.5\"); x -> exp(a * x); end"]
    funcs = jl.__pygaussfsbp_resolve_test_spec(spec, jl.BigFloat)

    assert len(funcs) == 1
    assert jl.float(funcs[0](jl.BigFloat("0.25"))) == pytest.approx(
        float(np.exp(0.125)), rel=1.0e-12
    )


@requires_juliacall
def test_test_specs_accept_list_of_expressions() -> None:
    jl = julia_operators._load_julia()
    funcs = jl.__pygaussfsbp_resolve_test_spec(
        ["x -> sin(x)", "x -> cos(x)"],
        jl.Float64,
    )

    assert len(funcs) == 2
    assert jl.float(funcs[0](0.0)) == pytest.approx(0.0)
    assert jl.float(funcs[1](0.0)) == pytest.approx(1.0)


@requires_juliacall
def test_bigfloat_optimized_build_with_all_optimization_kwargs() -> None:
    op_basis, quad_basis = polynomial_bases()
    exponent = 'parse(T, "0.1")'
    test_functions = (
        f"[x -> x^3, "
        f"let a = {exponent}; x -> exp(a * x); end]"
    )
    test_derivatives = (
        f"[x -> 3 * x^2, "
        f"let a = {exponent}; x -> a * exp(a * x); end]"
    )

    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        precision="bigfloat",
        digits=32,
        orthogonalize=True,
        principal="upper",
        use_optimization=True,
        test_functions=test_functions,
        test_derivatives=test_derivatives,
        test_weights=[1.0, 1.0],
        extrapolation_objective_weights=[0.2, 0.1],
        S_objective_weights=[0.9, 0.1],
    )

    assert isinstance(built, Operator)
    assert check_sbp_property(built, tol=1e-9)


@requires_juliacall
def test_objective_weight_list_is_recognized_by_gaussfsbp() -> None:
    jl = julia_operators._load_julia()
    converted = julia_operators._convert_julia_value(
        jl, "S_objective_weights", [0.9, 0.1]
    )
    accuracy = jl.GaussFSBP._objective_weight(
        converted, jl.Symbol("accuracy"), 1, 0.5
    )
    norm = jl.GaussFSBP._objective_weight(
        converted, jl.Symbol("norm"), 2, 0.5
    )

    assert float(accuracy) == pytest.approx(0.9)
    assert float(norm) == pytest.approx(0.1)

    raw_list = [0.9, 0.1]
    defaulted = jl.GaussFSBP._objective_weight(
        raw_list, jl.Symbol("accuracy"), 1, 0.5
    )
    assert float(defaulted) == pytest.approx(0.5)


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
def test_cached_legendre_factory_operator_matches_builtin() -> None:
    op_basis, quad_basis = cached_polynomial_bases()
    built = build_operator_from_julia(
        op_basis,
        quad_basis,
        precision="float64",
        orthogonalize=True,
        principal="upper",
    )

    expected = get_operator("LGLp2")
    np.testing.assert_allclose(built.nodes, expected.nodes, atol=1e-12)
    np.testing.assert_allclose(built.H, expected.H, atol=1e-12)
    np.testing.assert_allclose(built.D, expected.D, atol=1e-12)
    assert check_sbp_property(built, tol=1e-12)


@requires_juliacall
def test_cached_legendre_factory_callables_share_one_block() -> None:
    jl = julia_operators._load_julia()
    factory = legendre_basis_factory(3)
    functions, derivatives = jl.__pygaussfsbp_parse_basis(
        [], [], False, factory, (-1.0, 1.0), 3
    )
    jl.seval(
        """
function __pygaussfsbp_test_shared_legendre_block(funcs, derivs)
    block = getfield(funcs[1], :block)
    return all(getfield(func, :block) === block for func in funcs) &&
           all(getfield(deriv, :block) === block for deriv in derivs)
end
"""
    )

    assert jl.__pygaussfsbp_test_shared_legendre_block(functions, derivatives)


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
    epsilon_text = repr(epsilon)
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
