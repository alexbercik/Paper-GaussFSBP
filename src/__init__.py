"""Generalized SBP-SAT solvers for 1D linear advection."""

from .assembly import (
    assemble_homogeneous_operator,
    assemble_system,
    calc_LHS,
    calc_RHS,
)
from .elements import (
    Element1D,
    make_elements,
    make_uniform_elements,
    trace_left,
    trace_right,
)
from .norms import (
    convergence_rate,
    global_H_error,
    global_H_norm,
    global_L2_error,
    global_weighted_energy,
)
from .operator_library import (
    OperatorSpec,
    all_operators,
    get_operator,
    get_operator_by_name,
    operator_names,
    selectors_for,
)
from .operators import (
    Operator,
    check_sbp_property,
    validate_operator_dict,
)
from .lib import (
    JuliaBasis,
    JuliaOperatorError,
    build_operator_from_julia,
    build_julia_operator,
    legendre_basis_factory,
    print_fsbp_operator_python,
)
from .problems import Problem
from .solve import concatenate_local_vectors, solve_steady, split_global_vector

__all__ = [
    "Operator",
    "OperatorSpec",
    "all_operators",
    "get_operator",
    "get_operator_by_name",
    "operator_names",
    "selectors_for",
    "check_sbp_property",
    "validate_operator_dict",
    "JuliaBasis",
    "JuliaOperatorError",
    "build_operator_from_julia",
    "build_julia_operator",
    "legendre_basis_factory",
    "print_fsbp_operator_python",
    "Element1D",
    "make_elements",
    "make_uniform_elements",
    "trace_left",
    "trace_right",
    "assemble_system",
    "calc_LHS",
    "calc_RHS",
    "assemble_homogeneous_operator",
    "Problem",
    "solve_steady",
    "split_global_vector",
    "concatenate_local_vectors",
    "global_H_norm",
    "global_weighted_energy",
    "global_L2_error",
    "global_H_error",
    "convergence_rate",
]
