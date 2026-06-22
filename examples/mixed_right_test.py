from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

# Ensure repository root is in the path for src imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import JuliaBasis, build_operator_from_julia
from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import convergence_rate, global_H_error
from src.operator_library import OperatorSpec, operator_from_spec
from src.operators import Operator
from src.plotting import (
    exact_profile_on_domain,
    plot_convergence,
    plot_solution_profiles,
    profile_from_elements,
)
from src.solve import solve_steady

CACHE_FILE = Path(__file__).parent / "operator_cache.json"

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)


def exponential_bases_p2(beta: float, op_type: str) -> tuple[JuliaBasis, JuliaBasis]:
    """Build p2 equivalent basis, tailored to the available DOFs."""
    beta_str = f'BigFloat("{format(beta, "g")}")'
    exp_b = f"x -> exp({beta_str} * x)"
    
    op_basis = JuliaBasis(
        labels=["1", "x", f"exp({beta:g}x)"],
        functions=["x -> one(x)", "x -> x", exp_b],
        derivatives=["x -> zero(x)", "x -> one(x)", f"x -> {beta_str} * exp({beta_str} * x)"],
    )

    # For 'closed' and 'open', we utilize the fully determined 6-function space
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", f"exp({beta:g}x)", f"x exp({beta:g}x)", f"exp(2*{beta:g}x)"],
        functions=[
            "x -> one(x)", "x -> x", "x -> x^2",
            exp_b, f"x -> x * exp({beta_str} * x)", f"x -> exp(2 * {beta_str} * x)"
        ],
        derivatives=[
            "x -> zero(x)", "x -> one(x)", "x -> 2*x",
            f"x -> {beta_str} * exp({beta_str} * x)",
            f"x -> (one(x) + {beta_str} * x) * exp({beta_str} * x)",
            f"x -> 2 * {beta_str} * exp(2 * {beta_str} * x)"
        ],
    )
    return op_basis, quad_basis


def exponential_bases_p3(beta: float, op_type: str) -> tuple[JuliaBasis, JuliaBasis]:
    """Build p3 equivalent basis explicitly using the exact 8-function configuration."""
    beta_str = f'BigFloat("{format(beta, "g")}")'
    exp_b = f"x -> exp({beta_str} * x)"

    op_basis = JuliaBasis(
        labels=["1", "x", "x^2", f"exp({beta:g}x)"],
        functions=["x -> one(x)", "x -> x", "x -> x^2", exp_b],
        derivatives=["x -> zero(x)", "x -> one(x)", "x -> 2*x", f"x -> {beta_str} * exp({beta_str} * x)"],
    )

    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3", f"exp({beta:g}x)", f"x exp({beta:g}x)", f"x^2 exp({beta:g}x)", f"exp(2*{beta:g}x)"],
        functions=[
            "x -> one(x)", "x -> x", "x -> x^2", "x -> x^3",
            exp_b, f"x -> x * exp({beta_str} * x)", f"x -> x^2 * exp({beta_str} * x)", f"x -> exp(2 * {beta_str} * x)"
        ],
        derivatives=[
            "x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2",
            f"x -> {beta_str} * exp({beta_str} * x)",
            f"x -> (one(x) + {beta_str} * x) * exp({beta_str} * x)",
            f"x -> (2*x + {beta_str} * x^2) * exp({beta_str} * x)",
            f"x -> 2 * {beta_str} * exp(2 * {beta_str} * x)"
        ],
    )
    return op_basis, quad_basis


def get_dynamic_exp_operator(h: float, order: int, op_type: str) -> Operator:
    """Fetch optimal Generalized Gauss operator via Julia tailored to the specific boundary topology."""
    epsilon_0 = 0.0125
    beta = h / epsilon_0
    
    # Bumped to v11 to guarantee a fully clean slate compilation
    cache_key = f"h{h:g}_beta{beta:g}_opt_p{order}_{op_type}_v11"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"EXP_{cache_key}",
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

    print(f"  -> Cache miss. Generating {op_type} p{order} operator for h={h:g} (beta={beta:g})...")
    
    if order == 2:
        op_basis, quad_basis = exponential_bases_p2(beta, op_type)
    else:
        op_basis, quad_basis = exponential_bases_p3(beta, op_type)

    kwargs = {
        "interval": (0.0, 1.0),
        "precision": "bigfloat",
        "digits": 64,
        "orthogonalize": True,
        "quad_kwargs": {"lost_digits": 8},
    }
    
    if op_type == "closed":
        kwargs["principal"] = "upper"  # Physically closes the right edge (x=1)
    
    operator = build_operator_from_julia(op_basis, quad_basis, **kwargs)

    dataclass_op_type = "closed" if op_type == "closed" else "open"

    cache[cache_key] = {
        "h": h,
        "beta": beta,
        "node_type": "opt",
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": dataclass_op_type,
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist()
    }
    save_cache(cache)
    
    operator = dataclasses.replace(operator, name=f"EXP_{cache_key}", op_type=dataclass_op_type)
    return operator

# ==========================================
# PROBLEM DEFINITION & MESH BUILDERS
# ==========================================

DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [16, 32, 64, 128]
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = False

# Params for mixed source term
K = 7.0
OMEGA = 2.0
C = -0.5
T = 1.0
POLY_COEFFS = [0, 0, 0, 1]

# Unified runs mapping the requested strategies
RUNS = [
    {
        "label": "Global Exp p3 (Closed)",
        "strategy": "uniform_global_exp",
        "order": 3,
        "right_op_type": "closed"
    },
    {
        "label": "Global LGL p3",
        "strategy": "uniform",
        "interior_operator": OperatorSpec("LGLp3"),
        "num_right_elements": 0
    },
    {
        "label": "Fixed 1/64 Right Exp p3 (Closed), LGLp3 Int",
        "strategy": "fixed_right",
        "interior_operator": OperatorSpec("LGLp3"),
        "right_op_type": "closed",
        "order": 3,
        "h_right": 1.0/64.0
    },
    {
        "label": "Fixed 1/64 Right Exp p2 (Closed), LGLp2 Int",
        "strategy": "fixed_right",
        "interior_operator": OperatorSpec("LGLp2"),
        "right_op_type": "closed",
        "order": 2,
        "h_right": 1.0 / 64.0
    },
    {
        "label": "Uniform (x > 1-1/64 Exp p3 Closed, LGLp3 Int)",
        "strategy": "uniform",
        "interior_operator": OperatorSpec("LGLp3"),
        "right_op_type": "closed",
        "order": 3,
        "x_right_elements": 1.0 - 1.0/8.0
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
    epsilon = 0.0125
    num = np.exp((x_arr - 1.0) / epsilon) - np.exp(-1.0 / epsilon)
    den = 1.0 - np.exp(-1.0 / epsilon)
    return num / den

def singularity_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    epsilon = 0.0125
    num = (1.0 / epsilon) * np.exp((x_arr - 1.0) / epsilon)
    den = 1.0 - np.exp(-1.0 / epsilon)
    return num / den

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
    
    if num_right_elements is not None:
        return min(int(num_right_elements), num_elements)
    elif x_right_elements is not None:
        bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
        # Identify elements completely or partially to the right of the cutoff
        return int(np.count_nonzero(bounds[1:] > float(x_right_elements)))
    return 0
    
def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[Operator]:
    h = (DOMAIN[1] - DOMAIN[0]) / num_elements

    if run["strategy"] == "uniform_global_exp":
        op = get_dynamic_exp_operator(h, order=run["order"], op_type=run["right_op_type"])
        return [op] * num_elements

    # Fallback to mixed-domain uniform strategy
    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    int_op = operator_from_spec(run["interior_operator"])
    
    ops = [int_op] * num_interior
    
    if num_right > 0:
        right_op = get_dynamic_exp_operator(
            h=h, 
            order=run.get("order", 3),
            op_type=run.get("right_op_type", "closed")
        )
        ops.extend([right_op] * num_right)
        
    return ops

def solve_on_mesh(
    run: dict[str, object],
    num_elements: int,
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[list[Element1D], np.ndarray]:
    """Builds a uniformly refined mesh."""
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

def solve_on_custom_mesh(
    run: dict[str, object],
    num_interior: int,
    h_right: float,
    exact_fun: callable,
    f_fun: callable,
) -> tuple[list[Element1D], np.ndarray]:
    """Builds a tailored mesh with a strictly fixed boundary element size."""
    def left_bc_fun(_x: float) -> float:
        return float(exact_fun(DOMAIN[0]))

    int_op = operator_from_spec(run["interior_operator"])
    interior_ops = [int_op] * num_interior

    right_op = get_dynamic_exp_operator(
        h=h_right,
        order=run.get("order", 3),
        op_type=run.get("right_op_type", "closed")
    )

    boundary_val = DOMAIN[1] - h_right

    interior_elements = make_uniform_elements(
        domain=(DOMAIN[0], boundary_val),
        num_elements=num_interior,
        operators=interior_ops,
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )

    right_element = make_uniform_elements(
        domain=(boundary_val, DOMAIN[1]),
        num_elements=1,
        operators=[right_op],
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )

    elements = interior_elements + right_element
    system = assemble_system(elements, left_bc_fun=left_bc_fun, sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs)
    return system.elements, u


def run_convergence(
    run: dict[str, object],
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[np.ndarray, np.ndarray]:
    errors, dofs, hs = [], [], []

    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")

    # Fetch operators for logging purely based on the coarsest run
    if run["strategy"] == "fixed_right":
        int_op = operator_from_spec(run["interior_operator"])
        right_op = get_dynamic_exp_operator(run["h_right"], run["order"], run["right_op_type"])
    elif run["strategy"] == "uniform_global_exp":
        h = (DOMAIN[1] - DOMAIN[0]) / ELEMENT_COUNTS[0]
        int_op = get_dynamic_exp_operator(h, run["order"], run["right_op_type"])
        right_op = int_op
    else:
        sample_ops = operators_for_mesh(run, ELEMENT_COUNTS[0])
        int_op = sample_ops[0]
        right_op = sample_ops[-1]
    
    print("  Interior Operator Details:")
    print(f"    Basis: {int_op.basis}")
    if int_op.name != right_op.name:
        print("  Rightmost Operator Details:")
        print(f"    Basis: {right_op.basis}")
    print("-" * 60)

    print("num_elements  total_dofs  H_error         rate")

    for N in ELEMENT_COUNTS:
        if run["strategy"] == "fixed_right":
            # For fixed_right, subtract 1 so total elements precisely align with the uniform tests
            num_interior = N - 1
            elements, u = solve_on_custom_mesh(run, num_interior, run["h_right"], exact_fun, f_fun)
        else:
            elements, u = solve_on_mesh(run, N, exact_fun, f_fun)
            
        errors.append(global_H_error(elements, u, exact_fun))
        dofs.append(u.size) 
        
        # Nominal h for uniform comparison scaling
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(N))

    rates = convergence_rate(np.array(errors), np.array(hs))
    for N, n_dof, err, rate in zip(ELEMENT_COUNTS, dofs, errors, rates):
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{N:12d}  {n_dof:10d}  {err:12.4e}  {rate_str}")

    return np.array(dofs, dtype=float), np.array(errors, dtype=float)


if __name__ == "__main__":
    EXPERIMENTS = [
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

    print("\n" + "="*60)
    print("UNIFIED MESH REFINEMENT CONVERGENCE")
    print("="*60)

    for experiment in EXPERIMENTS:
        dof_rows, err_rows = [], []
        print(f"\nExperiment: {experiment['label']}")
        
        for run in RUNS:
            dofs, errors = run_convergence(run, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
            dof_rows.append(dofs)
            err_rows.append(errors)

        labels = [str(run["label"]) for run in RUNS]
        singularity_label = "exponential"

        plot_convergence(
            np.vstack(dof_rows),
            np.vstack(err_rows),
            labels,
            title=f"{experiment['title']} (singularity: {singularity_label})",
            grid=True,
            skipfit_st=[1] * len(RUNS),
        )

    if SHOW_PLOTS:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")