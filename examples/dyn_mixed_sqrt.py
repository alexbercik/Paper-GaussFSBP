from __future__ import annotations
import os
os.environ["JULIA_PROJECT"] = "@."
import dataclasses
import json
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (
    JuliaBasis,
    JuliaOperatorError,
    build_operator_from_julia,
    check_nullspace_consistency,
    check_sbp_property,
    legendre_basis_factory,
)
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

# Bumping cache version to v8 to lock in pure min-norm omission logic
CACHE_FILE = Path(__file__).parent / "operator_cache_sqrt_v8.json"


def load_cache() -> dict:
    if CACHE_FILE.exists() and CACHE_FILE.stat().st_size > 0:
        try:
            with open(CACHE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def save_cache(cache: dict) -> None:
    tmp_file = CACHE_FILE.with_suffix(".json.tmp")
    with open(tmp_file, "w") as f:
        json.dump(cache, f, indent=4)
    tmp_file.replace(CACHE_FILE)


def legendre_basis(num_functions: int) -> JuliaBasis:
    if num_functions < 1:
        raise ValueError("num_functions must be positive")
    return JuliaBasis(
        labels=[f"P_{degree}(x)" for degree in range(num_functions)],
        factory=legendre_basis_factory(num_functions),
    )


def polynomial_bases(degree: int, operator_type: str) -> tuple[JuliaBasis, JuliaBasis]:
    if degree < 1:
        raise ValueError("degree must be at least 1")
    op_basis = legendre_basis(degree + 1)
    if operator_type == "closed":
        quad_basis = legendre_basis(2 * degree)
    elif operator_type == "half-open-right":
        quad_basis = legendre_basis(2 * degree + 1)
    elif operator_type == "open":
        quad_basis = legendre_basis(2 * degree + 2)
    else:
        raise ValueError("op_type must be 'open', 'closed', or 'half-open-right'")
    return op_basis, quad_basis


def sqrt_bases(p: int, k: int, op_type: str) -> tuple[JuliaBasis, JuliaBasis, str]:
    """Generates degree-p trial space augmented by sqrt(k - x)."""
    if p < 1:
        raise ValueError("p must be at least 1")
    
    k_str = repr(float(k))
    op_labels = [*[f"P_{degree}(x)" for degree in range(p + 1)], f"sqrt({k}-x)"]
    
    sqrt_func = f"x -> sqrt({k_str} - x)"
    sqrt_deriv = f"x -> -0.5 / sqrt({k_str} - x)"
    
    op_basis = JuliaBasis(
        labels=op_labels,
        factory=legendre_basis_factory(p + 1, additional_functions=[sqrt_func], additional_derivatives=[sqrt_deriv]),
    )
    
    num_op = p + 2
    if op_type == "closed":
        num_quad = 2 * num_op - 2
        principal = "upper"
        num_sing = 2
    elif op_type == "half-open-right":
        num_quad = 2 * num_op - 1
        principal = "lower"
        num_sing = p + 1  
    elif op_type == "open":
        num_quad = 2 * num_op
        principal = "lower"
        num_sing = p + 1
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
        
    num_poly = num_quad - num_sing
    quad_labels = [f"P_{deg}(x)" for deg in range(num_poly)]
    add_funcs, add_derivs = [], []
    
    for s in range(num_sing):
        power = "one(x)" if s == 0 else f"x^{s}"
        quad_labels.append(f"x^{s}/sqrt({k}-x)")
        add_funcs.append(f"x -> {power} / sqrt({k_str} - x)")
        if s == 0:
            d_str = f"x -> 0.5 / (({k_str} - x) * sqrt({k_str} - x))"
        else:
            d_str = f"x -> {s}.0 * x^{s-1} / sqrt({k_str} - x) + 0.5 * x^{s} / (({k_str} - x) * sqrt({k_str} - x))"
        add_derivs.append(d_str)
        
    quad_basis = JuliaBasis(
        labels=quad_labels,
        factory=legendre_basis_factory(num_poly, additional_functions=add_funcs, additional_derivatives=add_derivs),
    )
    return op_basis, quad_basis, principal


def build_polynomial_operator(degree: int, op_type: str = "closed") -> Operator:
    cache_key = f"poly_p{degree}_{op_type}"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"POLY_{cache_key}", basis=data["basis"], quad_basis=data["quad_basis"],
            op_type=data["op_type"], selector=0, interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]), D=np.array(data["D"]), H=np.array(data["H"]),
            tL=np.array(data["tL"]), tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating {op_type.upper()} polynomial p{degree} operator...")
    op_basis, quad_basis = polynomial_bases(degree, op_type)
    principal = "upper" if op_type == "closed" else "lower"

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=42, orthogonalize=True, principal=principal
    )

    cache[cache_key] = {
        "basis": op_basis.labels, "quad_basis": quad_basis.labels, "op_type": operator.op_type,
        "interval": [0.0, 1.0], "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(), "H": operator.H.tolist(),
        "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"POLY_{cache_key}")


def build_sqrt_operator(p: int, k: int, op_type: str = "open") -> Operator:
    cache_key = f"sqrt_p{p}_k{k}_{op_type}_minnorm_v8"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"SQRT_{cache_key}", basis=data["basis"], quad_basis=data["quad_basis"],
            op_type=data["op_type"], selector=0, interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]), D=np.array(data["D"]), H=np.array(data["H"]),
            tL=np.array(data["tL"]), tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating {op_type.upper()} SQRT min-norm operator (p={p}, k={k})...")
    op_basis, quad_basis, principal = sqrt_bases(p, k, op_type)

    # Omission Strategy: Omitting 'test_functions' completely prevents PythonCall
    # from running validator checks on None, while forcing pure min-norm objective sliding.
    is_radau = (op_type == "half-open-right")
    opt_kwargs = {}
    if is_radau:
        opt_kwargs["use_optimization"] = True
        opt_kwargs["extrapolation_objective_weights"] = [1.0, 0.1]
        opt_kwargs["S_objective_weights"] = [1.0, 0.1]
    else:
        opt_kwargs["use_optimization"] = False

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=42, orthogonalize=True, principal=principal,
        **opt_kwargs
    )

    cache[cache_key] = {
        "basis": op_basis.labels, "quad_basis": quad_basis.labels, "op_type": operator.op_type,
        "interval": [0.0, 1.0], "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(), "H": operator.H.tolist(),
        "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"SQRT_{cache_key}")


DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [8, 16, 32, 64, 80, 100]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True
ROOT_POWER = 0.5

RUNS = [
    {
        "label": r"$\mathcal{P}_3$ (open)",
        "poly_order": 3,
        "op_type": "open",
    },
    {
        "label": r"$\mathcal{P}_4$ (open)",
        "poly_order": 4,
        "op_type": "open",
    },
    {
        "label": r"$\mathcal{P}_5$ (open)",
        "poly_order": 5,
        "op_type": "open",
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_4$ Radau (rightmost)",
        "poly_order": 4,
        "op_type": "closed",
        "right_poly_op_type": "half-open-right",
        "num_right_elements": 1,
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{1-x}$ ($x > 0.8$, open)",
        "poly_order": 4,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": False,
        "right_op_type": "open",
        "x_right_elements": 0.8,
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{k-x}$ ($x > 0.8$, open)",
        "poly_order": 4,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": True,
        "right_op_type": "open",
        "x_right_elements": 0.8,
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{1-x}$ Radau (rightmost)",
        "poly_order": 4,
        "op_type": "closed",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": False,
        "right_op_type": "half-open-right",
        "num_right_elements": 1,
    }
]


def _roughness_terms(x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poly = np.ones_like(x_arr)
    d_poly = np.zeros_like(x_arr)
    
    decay_rate = 3.8
    decay = np.exp(-decay_rate * x_arr**2)
    d_decay = -2.0 * decay_rate * x_arr * decay
    
    f_start = 3.0
    f_ramp = 9.0
    
    phase = 2.0 * np.pi * (f_start * x_arr + 0.5 * f_ramp * x_arr**2) + (np.pi / 3.0)
    d_phase = 2.0 * np.pi * (f_start + f_ramp * x_arr)
    
    s = np.sin(phase)
    ds_dx = d_phase * np.cos(phase)
    
    return poly, d_poly, decay, d_decay, s, ds_dx


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
    poly, _, decay, _, s, _ = _roughness_terms(x_arr)
    return 1.0 - decay * poly * s


def singularity_exact(x: np.ndarray | float) -> np.ndarray:
    return static_component(x)


def u_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)


def roughness_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    poly, d_poly, decay, d_decay, s, ds_dx = _roughness_terms(x_arr)
    return -(d_decay * poly * s + decay * d_poly * s + decay * poly * ds_dx)


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
    if (num_right_elements is None) == (x_right_elements is None):
        raise ValueError("Specify exactly one of num_right_elements or x_right_elements")
    if num_right_elements is not None:
        return min(int(num_right_elements), num_elements)
    
    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    target_x = float(repr(x_right_elements))
    return int(np.count_nonzero(np.round(bounds[1:], 12) > target_x))


def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[Operator]:
    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    
    int_deg = run.get("poly_order", 3)
    int_op_type = run.get("op_type", "closed")
    int_op = build_polynomial_operator(int_deg, int_op_type)
    
    ops = [int_op] * num_interior

    if num_right > 0:
        if run.get("right_sqrt"):
            sqrt_deg = run.get("sqrt_order", 3)
            right_op_type = run.get("right_op_type", int_op_type)
            shifted = run.get("shifted", False)
            
            for r in range(num_right):
                elem_from_right = num_right - 1 - r
                k_val = elem_from_right + 1 if shifted else 1
                ops.append(build_sqrt_operator(sqrt_deg, k_val, right_op_type))
        else:
            right_op_type = run.get("right_poly_op_type", int_op_type)
            right_op = build_polynomial_operator(int_deg, right_op_type)
            ops.extend([right_op] * num_right)

    return ops


def solve_on_mesh(
    run: dict[str, object],
    num_elements: int,
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[list[Element1D], np.ndarray]:
    ops = operators_for_mesh(run, num_elements)
    
    if not check_sbp_property(ops[0], print_report=False):
        warnings.warn(f"Operator {ops[0].name} violates SBP property on {num_elements} elements!", RuntimeWarning)

    elements = make_uniform_elements(
        domain=DOMAIN, num_elements=num_elements, operators=ops,
        a_fun=a_fun, b_fun=b_fun, f_fun=f_fun, exact_fun=exact_fun,
    )
    system = assemble_system(elements, left_bc_fun=lambda _x: float(exact_fun(DOMAIN[0])), sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs, on_singular="nan")
    return system.elements, u


def run_convergence(
    run: dict[str, object],
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[np.ndarray, np.ndarray]:
    errors, dofs, hs = [], [], []
    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")

    sample_ops = operators_for_mesh(run, ELEMENT_COUNTS[0])
    print(f"  Interior Nodes/Elem: {sample_ops[0].nodes.size} ({sample_ops[0].op_type})")
    if sample_ops[0].name != sample_ops[-1].name:
        print(f"  Right Nodes/Elem:    {sample_ops[-1].nodes.size} ({sample_ops[-1].op_type})")
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
        {"label": "Smooth problem", "exact_fun": roughness_exact, "f_fun": roughness_f, "title": "Smooth source problem"},
        {"label": "Singularity only", "exact_fun": singularity_exact, "f_fun": singularity_f, "title": "Singular source problem"},
        {"label": "Mixed source", "exact_fun": u_exact, "f_fun": mixed_f, "title": "Mixed source problem"},
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

            coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
            profiles.append(profile_from_elements(coarse_elements, coarse_u))

        labels = [str(run["label"]) for run in RUNS]
        plot_convergence(
            np.vstack(dof_rows), np.vstack(err_rows), labels,
            title=f"{experiment['title']} (singularity: $\\sqrt{{1-x}}$)",
            grid=True, skipfit_st=[1] * len(RUNS),
        )

        x_exact, u_exact_vals = exact_profile_on_domain(experiment["exact_fun"], domain=DOMAIN)
        if PLOT_SOLS: 
            plot_solution_profiles(
                profiles, labels, x_exact=x_exact, u_exact=u_exact_vals,
                title=f"{experiment['title']}, coarsest mesh ({COARSE_ELEMENTS} elements)", grid=True,
            )

    if SHOW_PLOTS:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")