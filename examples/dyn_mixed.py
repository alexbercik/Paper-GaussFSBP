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
    build_operator_from_sbp_extra,
    check_nullspace_consistency,
    check_sbp_property,
    legendre_basis_factory,
)
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

# Bumping cache version to safely bypass older polluted records on disk
CACHE_FILE = Path(__file__).parent / "operator_cache_v2.json"


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
    elif operator_type == "open":
        quad_basis = legendre_basis(2 * degree + 2)
    else:
        raise ValueError("op_type must be 'open' or 'closed'")
    return op_basis, quad_basis


def exponential_bases(
    p: int,
    alpha: float,
    alpha_divisor: int = 1,
) -> tuple[JuliaBasis, JuliaBasis]:
    if p < 0:
        raise ValueError("p must be nonnegative")
    
    alpha_text = repr(alpha)
    scaled_alpha_text = alpha_text if alpha_divisor == 1 else f"({alpha_text}/{alpha_divisor})"
    julia_alpha = f'(parse(T, "{alpha_text}") / {alpha_divisor})'

    exp_function = f"let a = {julia_alpha}; x -> exp(a * x); end"
    exp_derivative = f"let a = {julia_alpha}; x -> a * exp(a * x); end"
    
    op_basis = JuliaBasis(
        labels=[*[f"P_{degree}(x)" for degree in range(p + 1)], f"exp({scaled_alpha_text}x)"],
        factory=legendre_basis_factory(p + 1, additional_functions=[exp_function], additional_derivatives=[exp_derivative]),
    )

    num_quad_polynomials = 2 * p
    num_quad_functions = num_quad_polynomials + (p + 1) + 1
    if num_quad_functions % 2 != 0:
        num_quad_polynomials += 1

    quad_labels = [f"P_{degree}(x)" for degree in range(num_quad_polynomials)]
    additional_functions, additional_derivatives = [], []

    for degree in range(p + 1):
        power = "one(x)" if degree == 0 else f"x^{degree}"
        quad_labels.append(f"x^{degree} exp({scaled_alpha_text}x)")
        additional_functions.append(f"let a = {julia_alpha}; x -> {power} * exp(a * x); end")
        deriv = "a * exp(a * x)" if degree == 0 else f"({degree} * x^{degree - 1} + a * x^{degree}) * exp(a * x)"
        additional_derivatives.append(f"let a = {julia_alpha}; x -> {deriv}; end")

    quad_labels.append(f"exp(2 * {scaled_alpha_text}x)")
    additional_functions.append(f"let a = {julia_alpha}; x -> exp(2 * a * x); end")
    additional_derivatives.append(f"let a = {julia_alpha}; x -> 2 * a * exp(2 * a * x); end")

    quad_basis = JuliaBasis(
        labels=quad_labels,
        factory=legendre_basis_factory(num_quad_polynomials, additional_functions=additional_functions, additional_derivatives=additional_derivatives),
    )
    return op_basis, quad_basis


def build_exponential_operator(
    p: int,
    alpha: float,
    *,
    alpha_divisor: int = 1,
    optimize: bool,
    opt_method: str = "simultaneous",
    op_type: str = "closed",
) -> Operator:
    cache_key = f"p{p}_alpha{repr(alpha)}_div{alpha_divisor}_opt{optimize}_{opt_method}_{op_type}"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"EXP_{cache_key}", basis=data["basis"], quad_basis=data["quad_basis"],
            op_type=data["op_type"], selector=data.get("selector", 0),
            interval=np.array(data["interval"]), nodes=np.array(data["nodes"]),
            D=np.array(data["D"]), H=np.array(data["H"]),
            tL=np.array(data["tL"]), tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating EXP operator (p={p}, div={alpha_divisor}, opt={optimize})...")
    op_basis, quad_basis = exponential_bases(p, alpha, alpha_divisor)
    principal = "upper" if op_type == "closed" else "lower"
    julia_alpha = f'(parse(T, "{repr(alpha)}") / {alpha_divisor})'

    opt_funcs = (
        f"[x -> x^{p + 1}, let a = {julia_alpha}; x -> x * exp(a * x); end, "
        f"let a = {julia_alpha}; x -> x^2 * exp(a * x); end, let a = {julia_alpha}; x -> exp(2 * a * x); end]"
    )
    opt_derivs = (
        f"[x -> {p + 1} * x^{p}, let a = {julia_alpha}; x -> (1 + a * x) * exp(a * x); end, "
        f"let a = {julia_alpha}; x -> (2 * x + a * x^2) * exp(a * x); end, let a = {julia_alpha}; x -> 2 * a * exp(2 * a * x); end]"
    )

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=42, orthogonalize=True, principal=principal,
        use_optimization=optimize, opt_method=opt_method,
        test_functions=opt_funcs, test_derivatives=opt_derivs,
        test_weights=[1, 1, 1, 4], extrapolation_objective_weights=[1.0, 0.1], S_objective_weights=[1.0, 0.1]
    )

    cache[cache_key] = {
        "basis": op_basis.labels, "quad_basis": quad_basis.labels, "op_type": operator.op_type,
        "selector": 0, "interval": [0.0, 1.0], "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(), "H": operator.H.tolist(),
        "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"EXP_{cache_key}")


def _bernstein_basis_on_unit_interval(p: int) -> tuple[list[str], list[str]]:
    labels, functions = [], []
    for k in range(p + 1):
        labels.append(f"B_{k}^{p}")
        if k == 0:
            functions.append("x -> one(x)" if p == 0 else f"x -> (1 - x)^{p}")
        elif k == p:
            functions.append(f"x -> x^{p}")
        else:
            functions.append(f"x -> binomial({p}, {k}) * x^{k} * (1 - x)^({p} - {k})")
    return labels, functions


def sbp_extra_exponential_basis(p: int, alpha: float, alpha_divisor: int = 1) -> tuple[list[str], list[str]]:
    alpha_text = repr(alpha)
    scaled_alpha_text = alpha_text if alpha_divisor == 1 else f"({alpha_text}/{alpha_divisor})"
    julia_alpha = f"({alpha_text} / {alpha_divisor})"

    labels, functions = _bernstein_basis_on_unit_interval(p)
    labels.append(f"exp({scaled_alpha_text}x)")
    functions.append(f"let a = {julia_alpha}; x -> exp(a * x); end")
    return labels, functions


def build_equispaced_exponential_operator(p: int, alpha: float, *, alpha_divisor: int = 1) -> Operator:
    cache_key = f"p{p}_alpha{repr(alpha)}_div{alpha_divisor}_equispaced_closed"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=f"EXP_{cache_key}", basis=data["basis"], quad_basis=data["quad_basis"],
            op_type=data["op_type"], selector=0, interval=np.array(data["interval"]),
            nodes=np.array(data["nodes"]), D=np.array(data["D"]), H=np.array(data["H"]),
            tL=np.array(data["tL"]), tR=np.array(data["tR"])
        )

    print(f"  -> Cache miss. Generating Equispaced EXP operator (p={p}, div={alpha_divisor})...")
    basis_labels, functions = sbp_extra_exponential_basis(p, alpha, alpha_divisor)
    initial_num_nodes = p + 2

    operator = build_operator_from_sbp_extra(
        functions, initial_num_nodes, basis_labels=basis_labels, quad_basis_labels=basis_labels,
        op_type="closed", interval=(0.0, 1.0), source="orig",
        max_num_nodes=initial_num_nodes + 20, max_iterations=200000,
        g_tol=1.0e-25, sbp_tolerance=1.0e-12, accuracy_tolerance=1.0e-8
    )

    cache[cache_key] = {
        "basis": basis_labels, "quad_basis": basis_labels, "op_type": operator.op_type,
        "interval": [0.0, 1.0], "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(), "H": operator.H.tolist(),
        "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"EXP_{cache_key}")


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


DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [8, 16, 32, 64, 80, 100]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True

K = 7.0
OMEGA = 2.0
C = -0.5
STATIC_TYPE = "exponential"  
T = 1.0
POLY_COEFFS = [0, 0, 0, 1]

RUNS = [
    {
        "label": r"$\mathcal{P}_3$",
        "poly_order": 3,
        "op_type": "closed",
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": r"$\mathcal{P}_4$",
        "poly_order": 4,
        "op_type": "closed",
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": r"Mixed $\mathcal{P}_4$ / $\mathcal{P}_3 + e^{\alpha x}$ ($x > 0.8$)",
        "poly_order": 4,
        "op_type": "closed",
        "right_optimized": True,
        "order": 3,
        "num_right_elements": None,
        "x_right_elements": 0.8,
    },
    {
        "label": r"$\mathcal{P}_3 + e^{\alpha x}$, equispaced",
        "sbp_extra_equispaced": True,
        "order": 3,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": r"$\mathcal{P}_3 + e^{\alpha x}$, min-norm",
        "min_norm": True,
        "order": 3,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": r"$\mathcal{P}_3 + e^{\alpha x}$, simultaneous",
        "optimized": True,
        "order": 3,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
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
    epsilon = 0.0125
    num = np.exp((x_arr - 1.0) / epsilon) - np.exp(-1.0 / epsilon)
    den = 1.0 - np.exp(-1.0 / epsilon)
    return num / den - x_arr + 1.0


def singularity_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    epsilon = 0.0125
    num = (1.0 / epsilon) * np.exp((x_arr - 1.0) / epsilon)
    den = 1.0 - np.exp(-1.0 / epsilon)
    return num / den - 1.0


def roughness_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    poly, _, decay, _, s, _ = _roughness_terms(x_arr)
    return decay * poly * s


def singularity_exact(x: np.ndarray | float) -> np.ndarray:
    return static_component(x)


def u_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)


def roughness_f(x: np.ndarray) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    poly, d_poly, decay, d_decay, s, ds_dx = _roughness_terms(x_arr)
    return d_decay * poly * s + decay * d_poly * s + decay * poly * ds_dx


def mixed_f(x: np.ndarray) -> np.ndarray:
    return roughness_f(x) + singularity_f(x)


def a_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def b_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def count_right_elements(run: dict[str, object], num_elements: int) -> int:
    num_right_elements = run.get("num_right_elements")
    x_right_elements = run.get("x_right_elements")
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
    
    domain_len = DOMAIN[1] - DOMAIN[0]
    global_alpha = domain_len / 0.0125  # Strictly 80.0

    # 1. Resolve Interior Operator
    if run.get("min_norm"):
        int_op = build_exponential_operator(run.get("order", 3), global_alpha, alpha_divisor=num_elements, optimize=False)
    elif run.get("optimized"):
        int_op = build_exponential_operator(run.get("order", 3), global_alpha, alpha_divisor=num_elements, optimize=True)
    elif run.get("sbp_extra_equispaced"):
        int_op = build_equispaced_exponential_operator(run.get("order", 3), global_alpha, alpha_divisor=num_elements)
    elif run.get("poly_order"):
        int_op = build_polynomial_operator(run["poly_order"], run.get("op_type", "closed"))
    else:
        int_op = operator_from_spec(run["interior_operator"])

    # 2. Resolve Right Boundary Operator
    if run.get("right_min_norm"):
        right_op = build_exponential_operator(run.get("order", 3), global_alpha, alpha_divisor=num_elements, optimize=False)
    elif run.get("right_optimized"):
        right_op = build_exponential_operator(run.get("order", 3), global_alpha, alpha_divisor=num_elements, optimize=True)
    else:
        right_op = int_op

    return [int_op] * num_interior + [right_op] * num_right


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
            title=f"{experiment['title']} (singularity: $e^{{\\alpha x}}$)",
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