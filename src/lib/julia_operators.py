from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np

from ..operators import Operator, VALID_OP_TYPES


_JULIA_PROJECT = Path(__file__).parent.joinpath("julia")
_JL: Any | None = None
_HELPERS_READY = False
_TEST_SPEC_KEYS = frozenset({"test_functions", "test_derivatives"})
_OBJECTIVE_WEIGHT_KEYS = frozenset(
    {"extrapolation_objective_weights", "S_objective_weights"}
)


class JuliaOperatorError(RuntimeError):
    """Raised when the optional Julia-backed operator builder cannot run."""


@dataclass(frozen=True)
class JuliaBasis:
    """String- or factory-backed definition for Julia's `FunctionBasis`.

    Pass either individual trusted Julia callable expressions in `functions`
    and `derivatives`, or one trusted `factory` expression. A factory must
    evaluate to a function that accepts the typed interval and returns
    `(functions, derivatives)`.
    """

    labels: Sequence[str]
    functions: Sequence[str] | None = None
    derivatives: Sequence[str] | None = None
    factory: str | None = None

    def __post_init__(self) -> None:
        labels = _as_string_tuple(self.labels, "labels")
        factory = self.factory
        if factory is not None:
            if not isinstance(factory, str):
                raise TypeError("factory must be a string or None")
            if not factory.strip():
                raise ValueError("factory must not be empty")
            if self.functions is not None or self.derivatives is not None:
                raise ValueError(
                    "factory cannot be combined with functions or derivatives"
                )

            object.__setattr__(self, "labels", labels)
            object.__setattr__(self, "functions", None)
            object.__setattr__(self, "derivatives", None)
            return

        if self.functions is None:
            raise ValueError("functions are required when factory is not provided")
        functions = _as_string_tuple(self.functions, "functions")
        if len(labels) != len(functions):
            raise ValueError("labels and functions must have the same length")

        derivatives = None
        if self.derivatives is not None:
            derivatives = _as_string_tuple(self.derivatives, "derivatives")
            if len(derivatives) != len(functions):
                raise ValueError("derivatives and functions must have the same length")

        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "functions", functions)
        object.__setattr__(self, "derivatives", derivatives)


def legendre_basis_factory(
    num_polynomials: int,
    *,
    additional_functions: Sequence[str] = (),
    additional_derivatives: Sequence[str] = (),
) -> str:
    """Return a Julia factory that preserves Legendre's shared cache block.

    Additional entries are trusted Julia expressions evaluated in a scope
    containing `T`, the scalar type of the interval. This allows typed
    constants such as `parse(T, "0.1")` to be captured once by each callable.
    """
    if (
        not isinstance(num_polynomials, int)
        or isinstance(num_polynomials, bool)
        or num_polynomials < 0
    ):
        raise ValueError("num_polynomials must be a nonnegative integer")

    extra_functions = _as_string_tuple(
        additional_functions, "additional_functions"
    )
    extra_derivatives = _as_string_tuple(
        additional_derivatives, "additional_derivatives"
    )
    if len(extra_functions) != len(extra_derivatives):
        raise ValueError(
            "additional_functions and additional_derivatives must have the same length"
        )

    function_entries = ",\n        ".join(extra_functions)
    derivative_entries = ",\n        ".join(extra_derivatives)
    return f"""interval -> begin
    polynomial_functions, polynomial_derivatives =
        GaussFSBP.legendre_functions({num_polynomials}, interval)
    T = typeof(interval[1])
    extra_functions = Function[
        {function_entries}
    ]
    extra_derivatives = Function[
        {derivative_entries}
    ]
    return (
        vcat(polynomial_functions, extra_functions),
        vcat(polynomial_derivatives, extra_derivatives),
    )
end"""


def build_julia_operator(
    op_basis: JuliaBasis,
    quad_basis: JuliaBasis,
    *,
    interval: tuple[float, float] = (-1.0, 1.0),
    precision: str = "float64",
    digits: int | None = None,
    **build_kwargs: Any,
) -> Any:
    """Build and return the raw Julia `FSBPOperator`.

    `build_kwargs` are converted to Julia values and forwarded directly to
    `GaussFSBP.build_fsbp_operator`. Python strings become Julia `Symbol`s,
    Python booleans become Julia `Bool`s, except for keywords that explicitly
    contain Julia function expressions.
    """

    if not isinstance(op_basis, JuliaBasis) or not isinstance(quad_basis, JuliaBasis):
        raise TypeError("op_basis and quad_basis must be JuliaBasis instances")

    precision, digits = _normalize_precision(precision, digits)
    interval_values = _interval_values_for_julia(interval, precision)
    jl = _load_julia()
    kwargs = _make_named_tuple(jl, build_kwargs)

    return jl.__pygaussfsbp_build_from_strings(
        list(op_basis.functions or ()),
        list(op_basis.derivatives or ()),
        op_basis.derivatives is not None,
        op_basis.factory or "",
        len(op_basis.labels),
        list(quad_basis.functions or ()),
        list(quad_basis.derivatives or ()),
        quad_basis.derivatives is not None,
        quad_basis.factory or "",
        len(quad_basis.labels),
        list(interval_values),
        precision,
        digits,
        kwargs,
    )


def print_fsbp_operator_python(fsbp: Any, *, num_digits: int = 5) -> None:
    """Print a Julia `FSBPOperator` using GaussFSBP's Python/NumPy format."""

    num_digits = _normalize_num_digits(num_digits)
    jl = _load_julia()

    # Keep the export formatting owned by GaussFSBP so direct Julia and Python
    # calls print the same literals.
    jl.__pygaussfsbp_print_operator_python(fsbp, num_digits)


def build_operator_from_julia(
    op_basis: JuliaBasis,
    quad_basis: JuliaBasis,
    *,
    interval: tuple[float, float] = (-1.0, 1.0),
    precision: str = "float64",
    digits: int | None = None,
    print_operator: bool = False,
    print_num_digits: int = 5,
    **build_kwargs: Any,
) -> Operator:
    """Build a Julia FSBP operator and convert it to this package's `Operator`.

    The returned data are copied directly from Julia: no interval normalization,
    rescaling, renaming, or selector selection is applied.

    String- and boolean-valued `build_kwargs` are converted to Julia `Symbol`s
    and `Bool`s, so optimization choices can be written naturally in Python,
    for example `opt_method="sequential"`.

    The ``interval`` argument sets the reference domain passed to Julia
    ``legendre_functions``, each basis factory, and ``FunctionBasis(...;
    interval=...)``. For ``precision='bigfloat'``, endpoints are forwarded with
    ``repr`` so Julia can parse them at the requested precision.

    Optimization test functions accept either one Julia vector expression string
    such as ``"[x -> sin(x), x -> cos(x)]"`` or a Python list/tuple of
    per-function expression strings. They are resolved in typed Julia scope only
    when ``use_optimization=True``.

    Objective weights accept a length-2 list/tuple ``[accuracy, norm]``, a
    mapping ``{"accuracy": ..., "norm": ...}``, or an already-formed Julia
    named tuple.
    """
    if not isinstance(print_operator, bool):
        raise TypeError("print_operator must be True or False")
    if print_operator:
        print_num_digits = _normalize_num_digits(print_num_digits, "print_num_digits")
    if "op_type" in build_kwargs:
        raise TypeError("op_type is inferred from quad_basis length and principal")
    op_type = _infer_op_type(quad_basis, build_kwargs.get("principal", "lower"))

    fsbp = build_julia_operator(
        op_basis,
        quad_basis,
        interval=interval,
        precision=precision,
        digits=digits,
        **build_kwargs,
    )
    if print_operator:
        print_fsbp_operator_python(fsbp, num_digits=print_num_digits)

    jl = _load_julia()
    interval, nodes, D, H, tL, tR = jl.__pygaussfsbp_operator_arrays(fsbp)

    return Operator(
        name=None,
        basis=list(op_basis.labels),
        quad_basis=list(quad_basis.labels),
        op_type=op_type,
        selector=0,
        interval=np.asarray(interval, dtype=float),
        nodes=np.asarray(nodes, dtype=float),
        D=np.asarray(D, dtype=float),
        H=np.asarray(H, dtype=float),
        tL=np.asarray(tL, dtype=float),
        tR=np.asarray(tR, dtype=float),
    )


def build_operator_from_sbp_extra(
    functions: Sequence[str],
    nodes: Sequence[float],
    *,
    basis_labels: Sequence[str],
    quad_basis_labels: Sequence[str],
    op_type: str,
    interval: tuple[float, float] = (0.0, 1.0),
    source: str = "regularized",
    regularization_functions: Sequence[str] = (),
    selector: int = 0,
    name: str | None = None,
    autodiff: str = "forwarddiff",
    verbose: bool = False,
) -> Operator:
    """Build an operator with `SummationByPartsOperatorsExtra.jl`.

    The function strings are trusted Julia callable expressions. `source`
    accepts the short selectors `"regularized"` and `"basic"`, or an exported
    constructor name from `SummationByPartsOperatorsExtra`.
    """

    functions = _as_string_tuple(functions, "functions")
    basis_labels = _as_string_tuple(basis_labels, "basis_labels")
    quad_basis_labels = _as_string_tuple(quad_basis_labels, "quad_basis_labels")
    regularization_functions = _as_string_tuple(
        regularization_functions, "regularization_functions"
    )
    if len(functions) != len(basis_labels):
        raise ValueError("functions and basis_labels must have the same length")
    if not quad_basis_labels:
        raise ValueError("quad_basis_labels must not be empty")
    if op_type not in VALID_OP_TYPES:
        raise ValueError(f"Invalid op_type '{op_type}'")
    if not isinstance(selector, int) or isinstance(selector, bool):
        raise TypeError("selector must be an integer")
    if name is not None and not isinstance(name, str):
        raise TypeError("name must be a string or None")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a nonempty string")
    if not isinstance(autodiff, str) or not autodiff.strip():
        raise ValueError("autodiff must be a nonempty string")
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be True or False")

    interval = _normalize_interval(interval)
    node_values = _as_float_vector(nodes, "nodes")
    jl = _load_julia()
    sbp_operator = jl.__pygaussfsbp_extra_operator(
        list(functions),
        node_values.tolist(),
        source,
        list(regularization_functions),
        autodiff,
        verbose,
    )
    D, H, tL, tR = jl.__pygaussfsbp_extra_operator_arrays(sbp_operator)

    return Operator(
        name=name,
        basis=list(basis_labels),
        quad_basis=list(quad_basis_labels),
        op_type=op_type,
        selector=selector,
        interval=np.asarray(interval, dtype=float),
        nodes=node_values,
        D=np.asarray(D, dtype=float),
        H=np.asarray(H, dtype=float),
        tL=np.asarray(tL, dtype=float),
        tR=np.asarray(tR, dtype=float),
    )


def _as_string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{field_name} must contain only strings")
    return tuple(values)


def _as_float_vector(values: Sequence[float], field_name: str) -> np.ndarray:
    if isinstance(values, str):
        raise TypeError(f"{field_name} must be a sequence of numbers")
    try:
        array = np.asarray(values, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{field_name} must be a sequence of numbers") from exc
    if array.ndim != 1:
        raise ValueError(f"{field_name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{field_name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} entries must be finite")
    return array


def _normalize_interval(interval: tuple[float, float]) -> tuple[float, float]:
    if len(interval) != 2:
        raise ValueError("interval must contain exactly two endpoints")
    a = float(interval[0])
    b = float(interval[1])
    if not np.isfinite(a) or not np.isfinite(b):
        raise ValueError("interval endpoints must be finite")
    if b <= a:
        raise ValueError("interval must be strictly increasing")
    return a, b


def _interval_values_for_julia(
    interval: tuple[float, float], precision: str
) -> list[str] | list[float]:
    """Return interval endpoints in the form expected by the Julia build helpers."""
    a, b = _normalize_interval(interval)
    if precision == "bigfloat":
        # repr preserves the decimal text of Python floats for BigFloat parsing.
        return [repr(a), repr(b)]
    return [a, b]


def _normalize_precision(precision: str, digits: int | None) -> tuple[str, int]:
    precision_key = precision.lower()
    if precision_key == "float64":
        if digits is not None:
            raise ValueError("digits is only valid with precision='bigfloat'")
        return precision_key, 0
    if precision_key == "bigfloat":
        if digits is None:
            digits = 32
        if not isinstance(digits, int) or isinstance(digits, bool) or digits <= 0:
            raise ValueError("digits must be a positive integer")
        return precision_key, digits
    raise ValueError("precision must be 'float64' or 'bigfloat'")


def _normalize_num_digits(num_digits: int, field_name: str = "num_digits") -> int:
    if not isinstance(num_digits, int) or isinstance(num_digits, bool):
        raise TypeError(f"{field_name} must be an integer")
    if num_digits < 0:
        raise ValueError(f"{field_name} must be nonnegative")
    return num_digits


def _load_julia() -> Any:
    global _JL
    if _JL is not None:
        return _JL

    project = _JULIA_PROJECT.resolve()
    if not project.exists():
        raise JuliaOperatorError(f"Julia project not found at {project}")

    # juliacall reads JULIA_PROJECT at startup. Activating again below keeps the
    # helper correct even if another environment was already selected.
    os.environ.setdefault("JULIA_PROJECT", str(project))
    try:
        from juliacall import Main as jl
    except ImportError as exc:
        raise JuliaOperatorError(
            "Julia operator construction requires the optional dependency "
            "`juliacall`. Install it with `pip install -e '.[julia]'`."
        ) from exc

    jl.seval("import Pkg")
    jl.Pkg.activate(str(project))
    jl.seval(
        """
using LinearAlgebra
using GaussFSBP
using ADTypes
using ForwardDiff
using Manifolds
using Manopt
using SummationByPartsOperators
using SummationByPartsOperatorsExtra
"""
    )
    _define_julia_helpers(jl)
    _JL = jl
    return jl


def _define_julia_helpers(jl: Any) -> None:
    global _HELPERS_READY
    if _HELPERS_READY:
        return

    jl.seval(
        r"""
function __pygaussfsbp_make_named_tuple(keys, values)
    syms = Tuple(Symbol(String(key)) for key in keys)
    return NamedTuple{syms}(Tuple(values))
end

function __pygaussfsbp_parse_functions(exprs)
    funcs = Function[]
    for expr in exprs
        value = eval(Meta.parse(String(expr)))
        value isa Function || throw(ArgumentError("Julia expression must evaluate to a function: $(expr)"))
        push!(funcs, value)
    end
    return funcs
end

function __pygaussfsbp_parse_function(expr)
    funcs = __pygaussfsbp_parse_functions([expr])
    return funcs[1]
end

function __pygaussfsbp_eval_module(::Type{T}) where {T}
    m = Module()
    Core.eval(m, :(const T = $T))
    return m
end

function __pygaussfsbp_parse_functions_in_module(exprs, m::Module)
    funcs = Function[]
    for expr in exprs
        value = Core.eval(m, Meta.parse(String(expr)))
        value isa Function || throw(ArgumentError(
            "Julia expression must evaluate to a function: $(expr)"))
        push!(funcs, value)
    end
    return funcs
end

function __pygaussfsbp_resolve_test_spec(spec, ::Type{T}) where {T}
    m = __pygaussfsbp_eval_module(T)
    if spec isa AbstractString
        result = Core.eval(m, Meta.parse(String(spec)))
        result isa AbstractVector || throw(ArgumentError(
            "test spec string must evaluate to a function vector"))
        funcs = Function[]
        for value in result
            value isa Function || throw(ArgumentError(
                "test spec vector must contain only functions"))
            push!(funcs, value)
        end
        return funcs
    end
    return __pygaussfsbp_parse_functions_in_module(collect(spec), m)
end

function __pygaussfsbp_resolve_optimization_kwargs(kwargs, ::Type{T}) where {T}
    use_opt = get(kwargs, :use_optimization, false)
    use_opt || return kwargs

    updates = Pair{Symbol, Any}[]
    if haskey(kwargs, :test_functions)
        push!(updates, :test_functions =>
            __pygaussfsbp_resolve_test_spec(kwargs.test_functions, T))
    end
    if haskey(kwargs, :test_derivatives)
        push!(updates, :test_derivatives =>
            __pygaussfsbp_resolve_test_spec(kwargs.test_derivatives, T))
    end
    isempty(updates) && return kwargs

    resolved = merge(kwargs, NamedTuple(updates))
    if haskey(resolved, :test_functions) && haskey(resolved, :test_derivatives)
        length(resolved.test_functions) == length(resolved.test_derivatives) ||
            throw(ArgumentError(
                "test_derivatives must have the same length as test_functions"))
    end
    return resolved
end

function __pygaussfsbp_parse_basis(func_exprs, deriv_exprs, has_derivs,
                                    factory_expr, interval, expected_length)
    if !isempty(String(factory_expr))
        factory = eval(Meta.parse(String(factory_expr)))
        factory isa Function || throw(ArgumentError(
            "Julia basis factory expression must evaluate to a function."))
        result = Base.invokelatest(factory, interval)
        result isa Tuple && length(result) == 2 || throw(ArgumentError(
            "Julia basis factory must return (functions, derivatives)."))
        funcs = Function[collect(result[1])...]
        derivs = Function[collect(result[2])...]
    else
        funcs = __pygaussfsbp_parse_functions(func_exprs)
        derivs = has_derivs ? __pygaussfsbp_parse_functions(deriv_exprs) : nothing
    end

    length(funcs) == expected_length || throw(ArgumentError(
        "Julia basis produced $(length(funcs)) functions; expected $(expected_length)."))
    if derivs !== nothing
        length(derivs) == expected_length || throw(ArgumentError(
            "Julia basis produced $(length(derivs)) derivatives; expected $(expected_length)."))
    end
    return funcs, derivs
end

function __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                       op_factory, op_length,
                                       quad_funcs, quad_derivs, quad_has_derivs,
                                       quad_factory, quad_length,
                                       interval_values, ::Type{T}, kwargs) where {T}
    a = T === BigFloat ? BigFloat(string(interval_values[1])) : T(interval_values[1])
    b = T === BigFloat ? BigFloat(string(interval_values[2])) : T(interval_values[2])
    interval = (a, b)
    funcs, derivs = __pygaussfsbp_parse_basis(
        op_funcs, op_derivs, op_has_derivs, op_factory, interval, op_length)
    qfuncs, qderivs = __pygaussfsbp_parse_basis(
        quad_funcs, quad_derivs, quad_has_derivs, quad_factory, interval, quad_length)
    op_basis = GaussFSBP.FunctionBasis(funcs; derivs=derivs, interval=interval)
    quad_basis = GaussFSBP.FunctionBasis(qfuncs; derivs=qderivs, interval=interval)
    kwargs = __pygaussfsbp_resolve_optimization_kwargs(kwargs, T)
    return Base.invokelatest(GaussFSBP.build_fsbp_operator, op_basis, quad_basis; kwargs...)
end

function __pygaussfsbp_build_from_strings(op_funcs, op_derivs, op_has_derivs,
                                          op_factory, op_length,
                                          quad_funcs, quad_derivs, quad_has_derivs,
                                          quad_factory, quad_length,
                                          interval_values, precision, digits, kwargs)
    precision_key = Symbol(String(precision))
    if precision_key === :float64
        return __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                             op_factory, op_length,
                                             quad_funcs, quad_derivs, quad_has_derivs,
                                             quad_factory, quad_length,
                                             interval_values, Float64, kwargs)
    elseif precision_key === :bigfloat
        return setprecision(BigFloat, Int(digits); base=10) do
            __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                          op_factory, op_length,
                                          quad_funcs, quad_derivs, quad_has_derivs,
                                          quad_factory, quad_length,
                                          interval_values, BigFloat, kwargs)
        end
    else
        throw(ArgumentError("unsupported precision: $(precision)"))
    end
end

function __pygaussfsbp_operator_arrays(op)
    return (
        Float64[op.interval[1], op.interval[2]],
        Float64.(op.x),
        Array{Float64}(op.D),
        Float64.(op.w),
        Float64.(op.tL),
        Float64.(op.tR),
    )
end

function __pygaussfsbp_print_operator_python(op, num_digits)
    return Base.invokelatest(GaussFSBP.print_fsbp_operator_python,
                             op; num_digits=Int(num_digits))
end

function __pygaussfsbp_extra_source(source_name)
    key = Symbol(String(source_name))
    if key === :regularized
        return getfield(
            SummationByPartsOperatorsExtra,
            Symbol("GlaubitzIskeLampert\u00d6ffner2026Regularized"),
        )()
    elseif key === :basic
        return getfield(
            SummationByPartsOperatorsExtra,
            Symbol("GlaubitzIskeLampert\u00d6ffner2026Basic"),
        )()
    end
    return getfield(SummationByPartsOperatorsExtra, key)()
end

function __pygaussfsbp_extra_autodiff(autodiff_name)
    key = Symbol(String(autodiff_name))
    if key === :forwarddiff
        return ADTypes.AutoForwardDiff()
    end
    throw(ArgumentError("unsupported autodiff selector: $(autodiff_name)"))
end

function __pygaussfsbp_extra_operator(func_exprs, nodes, source_name,
                                      regularization_exprs, autodiff_name,
                                      verbose)
    funcs = Tuple(__pygaussfsbp_parse_functions(func_exprs))
    source = __pygaussfsbp_extra_source(source_name)
    kwargs = Pair{Symbol, Any}[
        :autodiff => __pygaussfsbp_extra_autodiff(autodiff_name),
        :verbose => Bool(verbose),
    ]
    if !isempty(regularization_exprs)
        reg_funcs = Tuple(__pygaussfsbp_parse_functions(regularization_exprs))
        push!(kwargs, :regularization_functions => reg_funcs)
    end
    return Base.invokelatest(
        SummationByPartsOperatorsExtra.function_space_operator,
        funcs,
        Float64.(collect(nodes)),
        source;
        NamedTuple(kwargs)...,
    )
end

function __pygaussfsbp_extra_operator_arrays(op)
    return (
        Array{Float64}(Matrix(op)),
        Float64.(diag(SummationByPartsOperators.mass_matrix(op))),
        Float64.(SummationByPartsOperators.left_boundary_weight(op)),
        Float64.(SummationByPartsOperators.right_boundary_weight(op)),
    )
end

const __pygaussfsbp_true = true
const __pygaussfsbp_false = false
"""
    )
    _HELPERS_READY = True


def _julia_bool(jl: Any, value: bool) -> Any:
    if hasattr(jl, "__pygaussfsbp_true") and hasattr(jl, "__pygaussfsbp_false"):
        return jl.__pygaussfsbp_true if value else jl.__pygaussfsbp_false
    if hasattr(jl, "seval"):
        return jl.seval("true") if value else jl.seval("false")
    raise TypeError("cannot convert Python bool to Julia bool without juliacall")


def _make_named_tuple(jl: Any, values: Mapping[str, Any]) -> Any:
    if not isinstance(values, Mapping):
        raise TypeError("build kwargs must be a mapping")
    keys = [str(key) for key in values]
    converted_values = [
        _convert_julia_value(jl, key, value) for key, value in values.items()
    ]
    return jl.__pygaussfsbp_make_named_tuple(keys, converted_values)


def _normalize_test_spec(value: Any, field_name: str) -> str | list[str]:
    if isinstance(value, str):
        if not value.strip():
            raise ValueError(f"{field_name} must not be empty")
        return value
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{field_name} must not be empty")
        if any(not isinstance(item, str) for item in value):
            raise TypeError(f"{field_name} must contain only Julia expression strings")
        return list(value)
    raise TypeError(
        f"{field_name} must be a Julia vector expression string or a "
        "sequence of Julia expression strings"
    )


def _convert_objective_weights(jl: Any, key_name: str, value: Any) -> Any:
    if isinstance(value, Mapping):
        return _make_named_tuple(jl, value)
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError(
                f"{key_name} must contain exactly two weights: [accuracy, norm]"
            )
        return jl.__pygaussfsbp_make_named_tuple(
            ["accuracy", "norm"],
            [value[0], value[1]],
        )
    raise TypeError(
        f"{key_name} must be a length-2 sequence or mapping with "
        "'accuracy' and 'norm' keys"
    )


def _convert_julia_value(jl: Any, key: Any, value: Any) -> Any:
    key_name = str(key)
    if isinstance(value, Mapping):
        return _make_named_tuple(jl, value)
    if key_name in _TEST_SPEC_KEYS:
        return _normalize_test_spec(value, key_name)
    if key_name in _OBJECTIVE_WEIGHT_KEYS:
        return _convert_objective_weights(jl, key_name, value)
    if key_name == "measure":
        return jl.__pygaussfsbp_parse_function(value)
    # GaussFSBP uses symbols for named choices. Function expressions are parsed
    # above, so every remaining Python string can safely follow that convention.
    if isinstance(value, str):
        return jl.Symbol(value)
    if isinstance(value, bool):
        return _julia_bool(jl, value)
    if isinstance(value, tuple):
        return tuple(_convert_julia_value(jl, key_name, item) for item in value)
    if isinstance(value, list):
        return [_convert_julia_value(jl, key_name, item) for item in value]
    return value


def _infer_op_type(quad_basis: JuliaBasis, principal: Any) -> str:
    if not isinstance(quad_basis, JuliaBasis):
        raise TypeError("quad_basis must be a JuliaBasis instance")

    principal_key = _principal_key(principal)
    even_quad_basis = len(quad_basis.labels) % 2 == 0
    if even_quad_basis:
        if principal_key == "upper":
            return "closed"
        return "open"
    if principal_key == "upper":
        return "half-open-left"
    return "half-open-right"


def _principal_key(principal: Any) -> str:
    if not isinstance(principal, str):
        raise TypeError("principal must be passed as 'upper' or 'lower'")
    key = principal.strip().lower()
    if key.startswith(":"):
        key = key[1:]
    if key not in {"upper", "lower"}:
        raise ValueError("principal must be 'upper' or 'lower'")
    return key
