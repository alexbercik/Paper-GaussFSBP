from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import JuliaBasis, build_operator_from_julia
from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import convergence_rate, global_H_error
from src.operators import Operator
from src.plotting import (
    exact_profile_on_domain,
    plot_convergence,
    plot_solution_profiles,
    profile_from_elements,
)
from src.solve import solve_steady

def sqrt_bases(k: int, order: int, op_type: str) -> tuple[JuliaBasis, JuliaBasis, str]:
    if order == 2:
        op_labels = ["1", "x", f"sqrt({k}-x)"]
        op_funcs = ["x -> one(x)", "x -> x", f"x -> sqrt({k}.0 - x)"]
        op_derivs = ["x -> zero(x)", "x -> one(x)", f"x -> -0.5 / sqrt({k}.0 - x)"]
        
        quad_labels = ["1", "x", f"1/sqrt({k}-x)", f"x/sqrt({k}-x)"]
        quad_funcs = [
            "x -> one(x)", "x -> x", 
            f"x -> 1.0 / sqrt({k}.0 - x)", f"x -> x / sqrt({k}.0 - x)"
        ]
        quad_derivs = [
            "x -> zero(x)", "x -> one(x)", 
            f"x -> 0.5 / (({k}.0 - x) * sqrt({k}.0 - x))",
            f"x -> 1.0 / sqrt({k}.0 - x) + 0.5 * x / (({k}.0 - x) * sqrt({k}.0 - x))"
        ]
        
        if op_type == "closed":
            principal = "upper"
        elif op_type == "half-open-right":
            principal = "lower"
            quad_labels.insert(2, "x^2")
            quad_funcs.insert(2, "x -> x^2")
            quad_derivs.insert(2, "x -> 2*x")
        elif op_type == "open":
            principal = "lower"
            quad_labels.insert(2, "x^2")
            quad_funcs.insert(2, "x -> x^2")
            quad_derivs.insert(2, "x -> 2*x")
            quad_labels.insert(3, "x^3")
            quad_funcs.insert(3, "x -> x^3")
            quad_derivs.insert(3, "x -> 3*x^2")
        else:
            raise ValueError(f"Unknown op_type: {op_type}")

    elif order == 3:
        op_labels = ["1", "x", "x^2", f"sqrt({k}-x)"]
        op_funcs = ["x -> one(x)", "x -> x", "x -> x^2", f"x -> sqrt({k}.0 - x)"]
        op_derivs = ["x -> zero(x)", "x -> one(x)", "x -> 2*x", f"x -> -0.5 / sqrt({k}.0 - x)"]
        
        quad_labels = ["1", "x", "x^2", "x^3", f"1/sqrt({k}-x)", f"x/sqrt({k}-x)"]
        quad_funcs = [
            "x -> one(x)", "x -> x", "x -> x^2", "x -> x^3",
            f"x -> 1.0 / sqrt({k}.0 - x)", f"x -> x / sqrt({k}.0 - x)"
        ]
        quad_derivs = [
            "x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2",
            f"x -> 0.5 / (({k}.0 - x) * sqrt({k}.0 - x))",
            f"x -> 1.0 / sqrt({k}.0 - x) + 0.5 * x / (({k}.0 - x) * sqrt({k}.0 - x))"
        ]
        
        if op_type == "closed":
            principal = "upper"
        elif op_type == "half-open-right":
            principal = "lower"
            quad_labels.append(f"x^2/sqrt({k}-x)")
            quad_funcs.append(f"x -> x^2 / sqrt({k}.0 - x)")
            quad_derivs.append(f"x -> 2.0 * x / sqrt({k}.0 - x) + 0.5 * x^2 / (({k}.0 - x) * sqrt({k}.0 - x))")
        elif op_type == "open":
            principal = "lower"
            quad_labels.append(f"x^2/sqrt({k}-x)")
            quad_funcs.append(f"x -> x^2 / sqrt({k}.0 - x)")
            quad_derivs.append(f"x -> 2.0 * x / sqrt({k}.0 - x) + 0.5 * x^2 / (({k}.0 - x) * sqrt({k}.0 - x))")
            quad_labels.insert(4, "x^4")
            quad_funcs.insert(4, "x -> x^4")
            quad_derivs.insert(4, "x -> 4*x^3")
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
            
    else:
        raise NotImplementedError("Only order 2 and 3 supported")

    return JuliaBasis(labels=op_labels, functions=op_funcs, derivatives=op_derivs), \
           JuliaBasis(labels=quad_labels, functions=quad_funcs, derivatives=quad_derivs), \
           principal


print("p = 2, half open right: ")
op_basis, quad_basis, principal = sqrt_bases(1, 2, "half-open-right"); 
operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat", 
        digits=32,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"lost_digits": 15}, 
    )

E = np.outer(operator.tR, operator.tR) - np.outer(operator.tL, operator.tL)
Q = np.diag(operator.H) @ operator.D
x = operator.nodes

print(np.max(abs(Q + np.transpose(Q)-E)))

print(operator.D@x**2-2*x)

print("p = 2, open: ")
op_basis, quad_basis, principal = sqrt_bases(1, 2, "open"); 
operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat", 
        digits=32,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"lost_digits": 15}, 
    )

E = np.outer(operator.tR, operator.tR) - np.outer(operator.tL, operator.tL)
Q = np.diag(operator.H) @ operator.D
x = operator.nodes

print(np.max(abs(Q + np.transpose(Q)-E)))

print(operator.D@x**2-2*x)

print("p = 3, half open right: ")
op_basis, quad_basis, principal = sqrt_bases(1, 3, "half-open-right"); 
operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat", 
        digits=32,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"lost_digits": 15}, 
    )

E = np.outer(operator.tR, operator.tR) - np.outer(operator.tL, operator.tL)
Q = np.diag(operator.H) @ operator.D
x = operator.nodes

print(np.max(abs(Q + np.transpose(Q)-E)))

print(operator.D@x**3-3*x**2)
print(operator.D@x**2-2*x)

print("p = 3, open: ")
op_basis, quad_basis, principal = sqrt_bases(1, 3, "open"); 
operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat", 
        digits=32,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"lost_digits": 15}, 
    )

E = np.outer(operator.tR, operator.tR) - np.outer(operator.tL, operator.tL)
Q = np.diag(operator.H) @ operator.D
x = operator.nodes

print(np.max(abs(Q + np.transpose(Q)-E)))

print(operator.D@x**3-3*x**2)
print(operator.D@x**2-2*x)


print ("***")

k = 1
op_labels = ["1", "x", f"sqrt({k}-x)"]
op_funcs = ["x -> one(x)", "x -> x", f"x -> sqrt({k}.0 - x)"]
op_derivs = ["x -> zero(x)", "x -> one(x)", f"x -> -0.5 / sqrt({k}.0 - x)"]

quad_labels = ["1", "x", f"1/sqrt({k}-x)", f"x/sqrt({k}-x)"]
quad_funcs = [
    "x -> one(x)", "x -> x", 
    f"x -> 1.0 / sqrt({k}.0 - x)", f"x -> x / sqrt({k}.0 - x)"
]
quad_derivs = [
    "x -> zero(x)", "x -> one(x)", 
    f"x -> 0.5 / (({k}.0 - x) * sqrt({k}.0 - x))",
    f"x -> 1.0 / sqrt({k}.0 - x) + 0.5 * x / (({k}.0 - x) * sqrt({k}.0 - x))"
]




principal = "lower"
quad_labels.insert(2, "x^2")
quad_funcs.insert(2, "x -> x^2")
quad_derivs.insert(2, "x -> 2*x")
quad_labels.insert(3, "x^3")
quad_funcs.insert(3, "x -> x^3")
quad_derivs.insert(3, "x -> 3*x^2")
quad_labels.insert(4, "x^4")
quad_funcs.insert(4, "x -> x^4")
quad_derivs.insert(4, "x -> 4*x^3")

op_basis = JuliaBasis(labels=op_labels, functions=op_funcs, derivatives=op_derivs)
quad_basis = JuliaBasis(labels=quad_labels, functions=quad_funcs, derivatives=quad_derivs)


opts_funcs = ["x -> x^2", "x -> x^3"]
opts_derivs = ["x -> 2*x", "x -> 3*x^2"]

operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat", 
        digits=32,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"lost_digits": 15}, 
        use_optimization = True,
        verbose = True, 
        test_functions = opts_funcs,
        test_derivatives = opts_derivs,
        test_weights = [1,1],
        extrapolation_objective_weights = [0.5,0.2],
        S_objective_weights = [1.0, 0.2],
    )



E = np.outer(operator.tR, operator.tR) - np.outer(operator.tL, operator.tL)
Q = np.diag(operator.H) @ operator.D
x = operator.nodes

print(np.max(abs(Q + np.transpose(Q)-E)))

print(operator.D@x**3-3*x**2)
print(operator.D@x**2-2*x)


