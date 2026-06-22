from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import os

import numpy as np

from ..operators import Operator


_JULIA_PROJECT = Path(__file__).parent.joinpath("GaussFSBP")
_JL: Any | None = None
_HELPERS_READY = False


class JuliaOperatorError(RuntimeError):
    """Raised when the optional Julia-backed operator builder cannot run."""


@dataclass(frozen=True)
class JuliaBasis:
    """String-backed basis definition for Julia's `FunctionBasis`.

    The callable strings are trusted Julia expressions that evaluate to unary
    functions, for example `x -> one(x)`, `x -> x^2`, or `x -> exp(x) / 0.1`.
    """

    labels: Sequence[str]
    functions: Sequence[str]
    derivatives: Sequence[str] | None = None

    def __post_init__(self) -> None:
        labels = _as_string_tuple(self.labels, "labels")
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
    except for keywords that explicitly contain Julia function expressions.
    """

    if not isinstance(op_basis, JuliaBasis) or not isinstance(quad_basis, JuliaBasis):
        raise TypeError("op_basis and quad_basis must be JuliaBasis instances")

    precision, digits = _normalize_precision(precision, digits)
    interval_values = _normalize_interval(interval)
    jl = _load_julia()
    kwargs = _make_named_tuple(jl, build_kwargs)

    return jl.__pygaussfsbp_build_from_strings(
        list(op_basis.functions),
        list(op_basis.derivatives or ()),
        op_basis.derivatives is not None,
        list(quad_basis.functions),
        list(quad_basis.derivatives or ()),
        quad_basis.derivatives is not None,
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

    String-valued `build_kwargs` are passed as Julia `Symbol`s, so optimization
    choices can be written naturally in Python, for example
    `opt_method="sequential"`.
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


def _as_string_tuple(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, str) or not isinstance(values, Sequence):
        raise TypeError(f"{field_name} must be a sequence of strings")
    if any(not isinstance(value, str) for value in values):
        raise TypeError(f"{field_name} must contain only strings")
    return tuple(values)


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
    jl.seval("using GaussFSBP")
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

function __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                       quad_funcs, quad_derivs, quad_has_derivs,
                                       interval_values, ::Type{T}, kwargs) where {T}
    a = T === BigFloat ? BigFloat(string(interval_values[1])) : T(interval_values[1])
    b = T === BigFloat ? BigFloat(string(interval_values[2])) : T(interval_values[2])
    interval = (a, b)
    funcs = __pygaussfsbp_parse_functions(op_funcs)
    derivs = op_has_derivs ? __pygaussfsbp_parse_functions(op_derivs) : nothing
    qfuncs = __pygaussfsbp_parse_functions(quad_funcs)
    qderivs = quad_has_derivs ? __pygaussfsbp_parse_functions(quad_derivs) : nothing
    op_basis = GaussFSBP.FunctionBasis(funcs; derivs=derivs, interval=interval)
    quad_basis = GaussFSBP.FunctionBasis(qfuncs; derivs=qderivs, interval=interval)
    return Base.invokelatest(GaussFSBP.build_fsbp_operator, op_basis, quad_basis; kwargs...)
end

function __pygaussfsbp_build_from_strings(op_funcs, op_derivs, op_has_derivs,
                                          quad_funcs, quad_derivs, quad_has_derivs,
                                          interval_values, precision, digits, kwargs)
    precision_key = Symbol(String(precision))
    if precision_key === :float64
        return __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                             quad_funcs, quad_derivs, quad_has_derivs,
                                             interval_values, Float64, kwargs)
    elseif precision_key === :bigfloat
        return setprecision(BigFloat, Int(digits); base=10) do
            __pygaussfsbp_build_with_type(op_funcs, op_derivs, op_has_derivs,
                                          quad_funcs, quad_derivs, quad_has_derivs,
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
"""
    )
    _HELPERS_READY = True


def _make_named_tuple(jl: Any, values: Mapping[str, Any]) -> Any:
    if not isinstance(values, Mapping):
        raise TypeError("build kwargs must be a mapping")
    keys = [str(key) for key in values]
    converted_values = [_convert_julia_value(jl, key, value) for key, value in values.items()]
    return jl.__pygaussfsbp_make_named_tuple(keys, converted_values)


def _convert_julia_value(jl: Any, key: Any, value: Any) -> Any:
    key_name = str(key)
    if isinstance(value, Mapping):
        return _make_named_tuple(jl, value)
    if key_name in {"test_functions", "test_derivatives"}:
        return jl.__pygaussfsbp_parse_functions(list(value))
    if key_name == "measure":
        return jl.__pygaussfsbp_parse_function(value)
    # GaussFSBP uses symbols for named choices. Function expressions are parsed
    # above, so every remaining Python string can safely follow that convention.
    if isinstance(value, str):
        return jl.Symbol(value)
    if isinstance(value, tuple):
        return tuple(_convert_julia_value(jl, key_name, item) for item in value)
    if isinstance(value, list):
        return [_convert_julia_value(jl, key_name, item) for item in value]
    return value


def _infer_op_type(quad_basis: JuliaBasis, principal: Any) -> str:
    if not isinstance(quad_basis, JuliaBasis):
        raise TypeError("quad_basis must be a JuliaBasis instance")

    principal_key = _principal_key(principal)
    even_quad_basis = len(quad_basis.functions) % 2 == 0
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
