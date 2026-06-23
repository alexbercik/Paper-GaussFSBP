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

SHOW_CONVERGENCE_PLOTS = True
SHOW_SOLUTION_PROFILES = True
SHOW_PLOTS_WINDOW = True

CACHE_FILE = Path(__file__).parent / "operator_cache_sqrt.json"

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def poly_bases(order: int, op_type: str) -> tuple[JuliaBasis, JuliaBasis, str]:
    if order == 2:
        op_labels = ["1", "x", "x^2"]
        op_funcs = ["x -> one(x)", "x -> x", "x -> x^2"]
        op_derivs = ["x -> zero(x)", "x -> one(x)", "x -> 2*x"]

        if op_type == "closed":
            q_len = 4
            principal = "upper"
        elif op_type == "half-open-right":
            q_len = 5
            principal = "lower"
        elif op_type == "open":
            q_len = 6
            principal = "lower"
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
            
    elif order == 3:
        op_labels = ["1", "x", "x^2", "x^3"]
        op_funcs = ["x -> one(x)", "x -> x", "x -> x^2", "x -> x^3"]
        op_derivs = ["x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2"]

        if op_type == "closed":
            q_len = 6
            principal = "upper"
        elif op_type == "half-open-right":
            q_len = 7
            principal = "lower"
        elif op_type == "open":
            q_len = 8
            principal = "lower"
        else:
            raise ValueError(f"Unknown op_type: {op_type}")
    else:
        raise NotImplementedError("Only order 2 and 3 supported")

    quad_labels, quad_funcs, quad_derivs = [], [], []
    for i in range(q_len):
        if i == 0:
            quad_labels.append("1")
            quad_funcs.append("x -> one(x)")
            quad_derivs.append("x -> zero(x)")
        elif i == 1:
            quad_labels.append("x")
            quad_funcs.append("x -> x")
            quad_derivs.append("x -> one(x)")
        elif i == 2:
            quad_labels.append("x^2")
            quad_funcs.append("x -> x^2")
            quad_derivs.append("x -> 2*x")
        else:
            quad_labels.append(f"x^{i}")
            quad_funcs.append(f"x -> x^{i}")
            quad_derivs.append(f"x -> {i}*x^{i-1}")

    return (
        JuliaBasis(labels=op_labels, functions=op_funcs, derivatives=op_derivs),
        JuliaBasis(labels=quad_labels, functions=quad_funcs, derivatives=quad_derivs),
        principal
    )

def get_dynamic_poly_operator(order: int, op_type: str = "closed") -> Operator:
    cache_key = f"poly_opt_p{order}_{op_type}_float64"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"POLY_{cache_key}",
            basis=data["basis"],
            quad_basis=data["quad_basis"],
            op_type=data["op_type"],
            selector=data.get("selector", 0),
            interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]),
            D=np.array(data["D"]),
            H=np.array(data["H"]),
            tL=np.array(data["tL"]),
            tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating {op_type} p{order} polynomial operator...")
    op_basis, quad_basis, principal = poly_bases(order, op_type)

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

    cache[cache_key] = {
        "node_type": "opt",
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": op_type, 
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist()
    }
    save_cache(cache)
    
    return dataclasses.replace(operator, name=f"POLY_{cache_key}", op_type=op_type)

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

def get_dynamic_sqrt_operator(k: int, order: int, op_type: str = "open") -> Operator:
    cache_key = f"sqrt_k{k}_opt_p{order}_{op_type}_float64"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"SQRT_{cache_key}",
            basis=data["basis"],
            quad_basis=data["quad_basis"],
            op_type=data["op_type"],
            selector=data.get("selector", 0),
            interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]),
            D=np.array(data["D"]),
            H=np.array(data["H"]),
            tL=np.array(data["tL"]),
            tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating {op_type} p{order} operator for sqrt(k={k}-x)...")
    op_basis, quad_basis, principal = sqrt_bases(k, order, op_type)

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

    cache[cache_key] = {
        "k": k,
        "node_type": "opt",
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": op_type, 
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist()
    }
    save_cache(cache)
    
    return dataclasses.replace(operator, name=f"SQRT_{cache_key}", op_type=op_type)

def get_optimized_right_sqrt_operator(order: int) -> Operator:
    """Fetch the custom non-linearly optimized rightmost operator."""
    if order != 2:
        raise NotImplementedError("Optimized right operator is currently only tuned for p=2.")

    # Modified cache key to reflect its structural op_type
    cache_key = f"sqrt_k1_opt_p{order}_half-open-right_optimized_float64"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"SQRT_{cache_key}",
            basis=data["basis"],
            quad_basis=data["quad_basis"],
            op_type=data["op_type"],
            selector=data.get("selector", 0),
            interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]),
            D=np.array(data["D"]),
            H=np.array(data["H"]),
            tL=np.array(data["tL"]),
            tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating explicitly optimized half-open-right operator for p{order}...")
    
    # Reuse the exact generator block for half-open-right
    op_basis, quad_basis, principal = sqrt_bases(k=1, order=order, op_type="half-open-right")

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
        use_optimization=True,
        verbose=False,
        test_functions=opts_funcs,
        test_derivatives=opts_derivs,
        test_weights=[1, 1],
        extrapolation_objective_weights=[0.5, 0.2],
        S_objective_weights=[1.0, 0.2],
    )

    cache[cache_key] = {
        "k": 1,
        "node_type": "opt",
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": "half-open-right", 
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist()
    }
    save_cache(cache)
    
    return dataclasses.replace(operator, name=f"SQRT_{cache_key}", op_type="half-open-right")

DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [4, 8, 16, 32, 64, 128, 256] 
COARSE_ELEMENTS = 32
SAT_TYPE = "upwind"

ROOT_POWER = 0.5
SHIFT_THRESHOLD = 32

K = 7.0
OMEGA = 2.0
C = -0.5
STATIC_TYPE = "root"  
T = 0.2
POLY_COEFFS = [0, 0, 0, 1]

RUNS = [
    {
        "label": "LGp2 (Open, Poly)",
        "strategy": "poly_standard",
        "order": 2,
        "interior_op_type": "open",
    },
    {
        "label": "LGp3 (Open, Poly)",
        "strategy": "poly_standard",
        "order": 3,
        "interior_op_type": "open",
    },
    {
        "label": "LGLp2 + Radau Right (Poly)",
        "strategy": "poly_mixed",
        "order": 2,
        "interior_op_type": "closed",
        "right_op_type": "half-open-right",
        "num_right_elements": 1,
    },
    {
        "label": "SQRT Static Global p2",
        "strategy": "global_1",
        "order": 2,
        "interior_op_type": "open",
        "right_op_type": "open",
    },
    {
        "label": "SQRT Static Global p3",
        "strategy": "global_1",
        "order": 3,
        "interior_op_type": "open",
        "right_op_type": "open",
    },
    {
        "label": f"SQRT Shifted p2 (Open, Thresh={SHIFT_THRESHOLD})",
        "strategy": "shifted",
        "order": 2,
        "interior_op_type": "open",
        "right_op_type": "open",
    },
    {
        "label": f"SQRT Shifted p3 (Open, Thresh={SHIFT_THRESHOLD})",
        "strategy": "shifted",
        "order": 3,
        "interior_op_type": "open",
        "right_op_type": "open",
    },
    {
        "label": f"SQRT Shifted p2 (Closed + Fully Open Right, Thresh={SHIFT_THRESHOLD})",
        "strategy": "shifted",
        "order": 2,
        "interior_op_type": "closed",
        "right_op_type": "open",
        "num_right_elements": 1,
    },
    {
        "label": f"SQRT Shifted p2 (Closed + Optimized Right, Thresh={SHIFT_THRESHOLD})",
        "strategy": "shifted_opt_right",
        "order": 2,
        "num_right_elements": 1,
    }
]

def _roughness_terms(x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poly = sum(coef * x_arr ** i for i, coef in enumerate(POLY_COEFFS))
    d_poly = sum(i * coef * x_arr ** (i - 1) for i, coef in enumerate(POLY_COEFFS) if i > 0)
    gauss = np.exp(-C * (x_arr - 0.5) ** 2)
    d_gauss = -2.0 * C * (x_arr - 0.5) * gauss
    s = np.sin(K * np.pi * x_arr - OMEGA * T)
    c = np.cos(K * np.pi * x_arr - OMEGA * T)
    return poly, d_poly, gauss, d_gauss, s, c

def static_component(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    base = np.clip(1.0 - x_arr, 0.0, None)
    return 3.0 * np.power(base, ROOT_POWER)

def singularity_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    base = np.clip(1.0 - x_arr, 1e-16, None)
    return -3.0 * ROOT_POWER * np.power(base, ROOT_POWER - 1.0)

def roughness_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    poly, _, gauss, _, s, _ = _roughness_terms(x_arr)
    return gauss * poly * s

def singularity_exact(x: np.ndarray | float) -> np.ndarray:
    return static_component(x)

def u_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)

def roughness_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    poly, d_poly, gauss, d_gauss, s, c = _roughness_terms(x_arr)
    return d_gauss * poly * s + gauss * d_poly * s + gauss * poly * (K * np.pi * c)

def mixed_f(x: np.ndarray) -> np.ndarray:
    return roughness_f(x) + singularity_f(x)

def a_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)

def b_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)

def count_right_elements(run: dict[str, object], num_elements: int) -> int:
    num_right_elements = run.get("num_right_elements")
    x_right_elements = run.get("x_right_elements")
    if num_right_elements is None and x_right_elements is None:
        return 0
    if num_right_elements is not None and x_right_elements is not None:
        raise ValueError("Specify only one of num_right_elements or x_right_elements")
    
    if num_right_elements is not None:
        return min(int(num_right_elements), num_elements)
    
    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    return int(np.count_nonzero(bounds[1:] > float(x_right_elements)))

def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[Operator]:
    ops = []
    interior_op_type = run.get("interior_op_type", "open")
    right_op_type = run.get("right_op_type", "open")
    num_right = count_right_elements(run, num_elements)
    
    for i in range(num_elements):
        k = num_elements - i
        is_right_element = (i >= num_elements - num_right)
        current_op_type = right_op_type if is_right_element else interior_op_type
        
        if run.get("strategy") == "shifted":
            if k > SHIFT_THRESHOLD:
                ops.append(get_dynamic_poly_operator(order=run.get("order", 2), op_type=current_op_type))
            else:
                ops.append(get_dynamic_sqrt_operator(k=k, order=run.get("order", 2), op_type=current_op_type))
                
        elif run.get("strategy") == "shifted_opt_right":
            if is_right_element:
                ops.append(get_optimized_right_sqrt_operator(order=run.get("order", 2)))
            else:
                if k > SHIFT_THRESHOLD:
                    ops.append(get_dynamic_poly_operator(order=run.get("order", 2), op_type="closed"))
                else:
                    ops.append(get_dynamic_sqrt_operator(k=k, order=run.get("order", 2), op_type="closed"))
            
        elif run.get("strategy") == "global_1":
            ops.append(get_dynamic_sqrt_operator(k=1, order=run.get("order", 2), op_type=current_op_type))
            
        elif run.get("strategy") in ("poly_standard", "poly_mixed"):
            ops.append(get_dynamic_poly_operator(order=run.get("order", 2), op_type=current_op_type))
            
    return ops

def solve_on_mesh(
    run: dict[str, object],
    num_elements: int,
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[list[Element1D], np.ndarray]:
    def left_bc_fun(_x: float) -> float:
        return float(exact_fun(DOMAIN[0]))

    elements = make_uniform_elements(
        domain=DOMAIN,
        num_elements=num_elements,
        operators=operators_for_mesh(run, num_elements),
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )
    system = assemble_system(elements, left_bc_fun=left_bc_fun, sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs)
    return system.elements, u

def solve_baseline_lgl(num_elements: int, exact_fun, f_fun) -> float:
    int_op = get_dynamic_poly_operator(order=3, op_type="closed")
    def left_bc_fun(_x: float) -> float:
        return float(exact_fun(DOMAIN[0]))

    elements = make_uniform_elements(
        domain=DOMAIN,
        num_elements=num_elements,
        operators=[int_op] * num_elements,
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )
    
    system = assemble_system(elements, left_bc_fun=left_bc_fun, sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs)
    return global_H_error(elements, u, exact_fun)

def run_convergence(
    run: dict[str, object],
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[np.ndarray, np.ndarray]:
    errors, dofs, hs = [], [], []

    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")

    sample_ops = operators_for_mesh(run, ELEMENT_COUNTS[0])
    int_op = sample_ops[0]
    right_op = sample_ops[-1]
    
    print("  Leftmost Operator Details:")
    print(f"    Basis Functions:      {int_op.basis}")
    print(f"    Quadrature Functions: {int_op.quad_basis}")
    print(f"    Nodes per Element:    {int_op.nodes.size} ({int_op.op_type})")
    
    if int_op.name != right_op.name:
        print("  Rightmost Operator Details:")
        print(f"    Basis Functions:      {right_op.basis}")
        print(f"    Quadrature Functions: {right_op.quad_basis}")
        print(f"    Nodes per Element:    {right_op.nodes.size} ({right_op.op_type})")
    print("-" * 60)

    print("num_elements  total_dofs  H_error         rate")

    for num_elements in ELEMENT_COUNTS:
        elements, u = solve_on_mesh(run, num_elements, exact_fun=exact_fun, f_fun=f_fun)
        errors.append(global_H_error(elements, u, exact_fun))
        dofs.append(u.size) 
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(num_elements))

    rates = convergence_rate(np.array(errors), np.array(hs))
    for num_elements, n_dof, err, rate in zip(ELEMENT_COUNTS, dofs, errors, rates):
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{num_elements:12d}  {n_dof:10d}  {err:12.4e}  {rate_str}")

    return np.array(dofs, dtype=float), np.array(errors, dtype=float)

if __name__ == "__main__":
    EXPERIMENTS = [
        {
            "label": "Smooth problem",
            "exact_fun": roughness_exact,
            "f_fun": roughness_f,
            "title": "Smooth source problem",
        },
        {
            "label": "Singularity only",
            "exact_fun": singularity_exact,
            "f_fun": singularity_f,
            "title": "Singular source problem",
        },
        {
            "label": "Mixed source",
            "exact_fun": u_exact,
            "f_fun": mixed_f,
            "title": "Mixed source problem",
        },
    ]

    for experiment in EXPERIMENTS:
        dof_rows, err_rows, profiles = [], [], []
        print(f"\n==========================================")
        print(f"Experiment: {experiment['label']}")
        print(f"==========================================")
        
        for run in RUNS:
            dofs, errors = run_convergence(run, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
            dof_rows.append(dofs)
            err_rows.append(errors)

            if SHOW_SOLUTION_PROFILES:
                coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
                profiles.append(profile_from_elements(coarse_elements, coarse_u))

        labels = [str(run["label"]) for run in RUNS]
        singularity_label = f"root (power={ROOT_POWER})"

        if SHOW_CONVERGENCE_PLOTS:
            plot_convergence(
                np.vstack(dof_rows),
                np.vstack(err_rows),
                labels,
                title=f"{experiment['title']} (singularity: {singularity_label})",
                grid=True,
                skipfit_st= None 
            )

        if SHOW_SOLUTION_PROFILES:
            x_exact, u_exact_vals = exact_profile_on_domain(experiment["exact_fun"], domain=DOMAIN)
            plot_solution_profiles(
                profiles,
                labels,
                x_exact=x_exact,
                u_exact=u_exact_vals,
                title=f"{experiment['title']}, coarsest mesh ({COARSE_ELEMENTS} elements)",
                grid=True,
            )

    if SHOW_PLOTS_WINDOW:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")