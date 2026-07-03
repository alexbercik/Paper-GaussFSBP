from __future__ import annotations
import dataclasses
import json
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
plt.close("all")
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (
    JuliaBasis,
    build_operator_from_julia,
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

CACHE_FILE = Path(__file__).parent / "operator_cache_sqrt_v13.json"

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

def sqrt_bases(p: int, k: int, op_type: str, optimize: bool = True) -> tuple[JuliaBasis, JuliaBasis, str]:
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
    
    num_poly = 2 * p     
    num_sing = p + 1     

    base_constraints = num_poly + num_sing
    
    if op_type in {"open", "closed"} and base_constraints % 2 != 0:
        num_poly += 1
    elif op_type == "half-open-right" and base_constraints % 2 == 0:
        num_poly += 1

    num_quad_required = num_poly + num_sing
    
    num_op = p + 2
    if op_type == "closed":
        total_dofs = 2 * num_op - 2
        principal = "upper"
    elif op_type == "half-open-right":
        total_dofs = 2 * num_op - 1
        principal = "lower"
    elif op_type == "open":
        total_dofs = 2 * num_op
        principal = "lower"
    else:
        raise ValueError(f"Unknown op_type: {op_type}")
        
    if optimize:
        total_dofs += (num_op * (num_op - 1)) // 2
        
    if num_quad_required > total_dofs:
        raise ValueError(
            f"ERR A p={p} {op_type} operator with opt={optimize} only has {total_dofs} DOFs, "
            f"but SBP exactness requires {num_quad_required} constraints."
        )

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

def build_sqrt_operator(
    p: int, 
    k: int, 
    op_type: str = "open", 
    optimize: bool | None = None, 
    opt_method: str = "simultaneous"
) -> Operator:
    if optimize is None:
        optimize = True

    cache_key = f"sqrt_p{p}_k{k}_{op_type}_opt{optimize}_{opt_method}_v13"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"SQRT_{cache_key}", basis=data["basis"], quad_basis=data["quad_basis"],
            op_type=data["op_type"], selector=0, interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]), D=np.array(data["D"]), H=np.array(data["H"]),
            tL=np.array(data["tL"]), tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating {op_type.upper()} SQRT operator (p={p}, k={k}, opt={optimize}, method={opt_method})...")
    op_basis, quad_basis, principal = sqrt_bases(p, k, op_type, optimize)

    opt_kwargs = {}
    if optimize:
        opt_funcs = f"[x -> x^{p + 1}, x -> x^{p + 2}]"
        opt_derivs = f"[x -> {p + 1} * x^{p}, x -> {p + 2} * x^{p + 1}]"
        
        opt_kwargs["use_optimization"] = True
        opt_kwargs["opt_method"] = opt_method
        opt_kwargs["extrapolation_objective_weights"] = [0.9, 0.1]
        opt_kwargs["S_objective_weights"] = [0.9, 0.1]
        opt_kwargs["test_functions"] = opt_funcs
        opt_kwargs["test_derivatives"] = opt_derivs
        opt_kwargs["test_weights"] = [2, 1]
    else:
        opt_kwargs["use_optimization"] = False

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=56, orthogonalize=True, principal=principal,
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
ELEMENT_COUNTS = [8, 16, 32, 64, 80, 100, 160, 200]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True
ROOT_POWER = 0.5

RUNS_LO_OPEN = [
    {
        "label": r"$\mathcal{P}_4$ (open)",
        "poly_order": 4,
        "op_type": "open",
        "color": "tab:purple",
        "marker": "o",
    },
    {
        "label": r"$\mathcal{P}_5$ (open)",
        "poly_order": 5,
        "op_type": "open",
        "color": "tab:blue",
        "marker": "^",
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{1-x}$ ($x > 0.9$, open)",
        "poly_order": 4,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": False,
        "right_op_type": "open",
        "x_right_elements": 0.9,
        "color": "tab:orange",
        "marker": "+",
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{k-x}$ ($x > 0.9$, open)",
        "poly_order": 4,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": True,
        "right_op_type": "open",
        "x_right_elements": 0.9,
        "color": "tab:red",
        "marker": "d",
    }
]

RUNS_HI_OPEN = [
    {
        "label": r"$\mathcal{P}_5$ (open)",
        "poly_order": 5,
        "op_type": "open",
        "color": "tab:purple",
        "marker": "o",
    },
    {
        "label": r"$\mathcal{P}_6$ (open)",
        "poly_order": 6,
        "op_type": "open",
        "color": "tab:blue",
        "marker": "^",
    },
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_4 + \sqrt{1-x}$ ($x > 0.9$, open)",
        "poly_order": 5,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 4,
        "shifted": False,
        "right_op_type": "open",
        "x_right_elements": 0.9,
        "color": "tab:orange",
        "marker": "+",
    },
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_4 + \sqrt{k-x}$ ($x > 0.9$, open)",
        "poly_order": 5,
        "op_type": "open",
        "right_sqrt": True,
        "sqrt_order": 4,
        "shifted": True,
        "right_op_type": "open",
        "x_right_elements": 0.9,
        "color": "tab:red",
        "marker": "d",
    }
]

RUNS_LO_CLOSED = [
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_4$ Radau (rightmost)",
        "poly_order": 4,
        "op_type": "closed",
        "right_poly_op_type": "half-open-right",
        "num_right_elements": 1,
        "color": "tab:purple",
        "marker": "o",
    },
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_5$ Radau (rightmost)",
        "poly_order": 5,
        "op_type": "closed",
        "right_poly_op_type": "half-open-right",
        "num_right_elements": 1,
        "color": "tab:blue",
        "marker": "^",
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
        "color": "tab:orange",
        "marker": "+",
    },
    {
        "label": r"$\mathcal{P}_4$ / $\mathcal{P}_3 + \sqrt{k-x}$ ($x > 0.9$, closed/Radau)",
        "poly_order": 4,
        "op_type": "closed",
        "right_sqrt": True,
        "sqrt_order": 3,
        "shifted": True,
        "right_op_type": "closed",
        "rightmost_op_type": "half-open-right",
        "x_right_elements": 0.9,
        "color": "tab:red",
        "marker": "d",
    }
]

RUNS_HI_CLOSED = [
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_5$ Radau (rightmost)",
        "poly_order": 5,
        "op_type": "closed",
        "right_poly_op_type": "half-open-right",
        "num_right_elements": 1,
        "color": "tab:purple",
        "marker": "o",
    },
    {
        "label": r"$\mathcal{P}_6$ / $\mathcal{P}_6$ Radau (rightmost)",
        "poly_order": 6,
        "op_type": "closed",
        "right_poly_op_type": "half-open-right",
        "num_right_elements": 1,
        "color": "tab:blue",
        "marker": "^",
    },
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_4 + \sqrt{1-x}$ Radau (rightmost)",
        "poly_order": 5,
        "op_type": "closed",
        "right_sqrt": True,
        "sqrt_order": 4,
        "shifted": False,
        "right_op_type": "half-open-right",
        "num_right_elements": 1,
        "color": "tab:orange",
        "marker": "+",
    },
    {
        "label": r"$\mathcal{P}_5$ / $\mathcal{P}_4 + \sqrt{k-x}$ ($x > 0.9$, closed/Radau)",
        "poly_order": 5,
        "op_type": "closed",
        "right_sqrt": True,
        "sqrt_order": 4,
        "shifted": True,
        "right_op_type": "closed",
        "rightmost_op_type": "half-open-right",
        "x_right_elements": 0.9,
        "color": "tab:red",
        "marker": "d",
    }
]

RUNS = RUNS_LO_OPEN

def static_component(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    base = np.clip(1.0 - x_arr, 0.0, None)
    return 3.0 * np.power(base, ROOT_POWER)

def singularity_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    base = np.clip(1.0 - x_arr, 1e-16, None)
    return -3.0 * ROOT_POWER * np.power(base, ROOT_POWER - 1.0)

#roughness 0.5(-x^2 + x)\sin(5\pi x)
def roughness_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * (-x_arr**2 + x_arr) * np.sin(5.0 * np.pi * x_arr)

def singularity_exact(x: np.ndarray | float) -> np.ndarray:
    return static_component(x)

def u_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)

def roughness_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    term1 = 0.5 * (1.0 - 2.0 * x_arr) * np.sin(5.0 * np.pi * x_arr)
    term2 = 2.5 * np.pi * (-x_arr**2 + x_arr) * np.cos(5.0 * np.pi * x_arr)
    return term1 + term2

def mixed_f(x: np.ndarray | float) -> np.ndarray:
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
            rightmost_op_type = run.get("rightmost_op_type", right_op_type)
            shifted = run.get("shifted", False)
            
            right_optimized = run.get("right_optimized", None)
            right_opt_method = run.get("right_opt_method", "simultaneous")
            
            for r in range(num_right):
                elem_from_right = num_right - 1 - r
                k_val = elem_from_right + 1 if shifted else 1
                current_op_type = rightmost_op_type if elem_from_right == 0 else right_op_type
                
                ops.append(build_sqrt_operator(
                    sqrt_deg, 
                    k_val, 
                    current_op_type, 
                    optimize=right_optimized, 
                    opt_method=right_opt_method
                ))
        else:
            right_op_type = run.get("right_poly_op_type", int_op_type)
            rightmost_op_type = run.get("rightmost_op_type", right_op_type)
            
            for r in range(num_right):
                elem_from_right = num_right - 1 - r
                current_op_type = rightmost_op_type if elem_from_right == 0 else right_op_type
                ops.append(build_polynomial_operator(int_deg, current_op_type))

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
    
    sat_type = run.get("sat_type", SAT_TYPE)
    system = assemble_system(elements, left_bc_fun=lambda _x: float(exact_fun(DOMAIN[0])), sat_type=sat_type)
    
    u, _ = solve_steady(system.matrix, system.rhs, on_singular="nan")
    return system.elements, u

def run_convergence(
    run: dict[str, object],
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[np.ndarray, np.ndarray]:
    errors, dofs, hs = [], [], []
    sat_type = run.get("sat_type", SAT_TYPE)
    print(f"\nRun: {run['label']}, SAT: {sat_type}")

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
        {"label": "Smooth problem", "exact_fun": roughness_exact, "f_fun": roughness_f, "title": r"Smooth source problem"},
        {"label": "Singularity only", "exact_fun": singularity_exact, "f_fun": singularity_f, "title": r"Singular source problem ($\sqrt{1-x}$)"},
        {"label": "Mixed source", "exact_fun": u_exact, "f_fun": mixed_f, "title": r"Mixed source problem ($\sqrt{1-x}$)"},
    ]

    for experiment in EXPERIMENTS:
        dof_rows, err_rows, profiles, labels = [], [], [], []
        run_colors, run_markers = [], []
        
        print(f"\n==========================================")
        print(f"Experiment: {experiment['label']}")
        print(f"==========================================")
        
        for run in RUNS:
            try:
                dofs, errors = run_convergence(run, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
                coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
                
                dof_rows.append(dofs)
                err_rows.append(errors)
                profiles.append(profile_from_elements(coarse_elements, coarse_u))
                labels.append(str(run["label"]))
                run_colors.append(run.get("color", "black"))
                run_markers.append(run.get("marker", "o"))
                
            except Exception as e:
                print(f"  [Skipped] {e}")
                continue

        if dof_rows:
            plot_convergence(
                np.vstack(dof_rows), np.vstack(err_rows), labels,
                title=experiment["title"],
                grid=True, skipfit_st=[len(ELEMENT_COUNTS)-3] * len(dof_rows),
                colors=run_colors,
                markers=run_markers
            )

            x_exact, u_exact_vals = exact_profile_on_domain(experiment["exact_fun"], domain=DOMAIN)
            if PLOT_SOLS: 
                plot_solution_profiles(
                    profiles, labels, x_exact=x_exact, u_exact=u_exact_vals,
                    title=rf"{experiment['label']} solutions ({COARSE_ELEMENTS} elements)", grid=True,
                    colors=run_colors,
                    markers=run_markers
                )

    if SHOW_PLOTS:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")
