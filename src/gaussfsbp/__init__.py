"""gaussfsbp: generalized SBP-SAT solvers for 1D linear advection."""

from .assembly import assemble_homogeneous_operator, assemble_system
from .elements import Element1D, create_elements, trace_left, trace_right
from .mesh import ElementSpec, Mesh1D
from .norms import (
    convergence_rate,
    global_H_error,
    global_H_norm,
    global_L2_error,
    global_weighted_energy,
)
from .operators import (
    Operator,
    OperatorRepository,
    builtin_operator_repository,
    check_sbp_property,
    validate_operator_dict,
)
from .problems import Problem
from .solve import concatenate_local_vectors, solve_steady, split_global_vector

__all__ = [
    "Operator",
    "OperatorRepository",
    "builtin_operator_repository",
    "check_sbp_property",
    "validate_operator_dict",
    "ElementSpec",
    "Mesh1D",
    "Element1D",
    "create_elements",
    "trace_left",
    "trace_right",
    "assemble_system",
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
