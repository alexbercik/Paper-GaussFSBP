from __future__ import annotations
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

# Pointing directly to the shared cache architecture from dyn_mixed.py
CACHE_FILE = Path(__file__).parent / "scalesep_operator_cache_v2.json"


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

RUNS = [
    {
        "label": r"$\mathcal{P}_4$ LGL",
        "poly_order": 4,
        "op_type": "closed",
    },
    {
        "label": r"$\mathcal{P}_5$ LGL",
        "poly_order": 5,
        "op_type": "closed",
    },
    {
        "label": r"$\mathcal{P}_3 + e^{\alpha x}$ (optimized)",
        "exp_order": 3,
        "op_type": "closed",
        "optimized": True,
    },
    {
        "label": r"$\mathcal{P}_4 + e^{\alpha x}$ (optimized)",
        "exp_order": 4,
        "op_type": "closed",
        "optimized": True,
    },
]


def roughness_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * (-x_arr**2 + x_arr) * np.sin(5.0 * np.pi * x_arr)


def roughness_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    term1 = 0.5 * (1.0 - 2.0 * x_arr) * np.sin(5.0 * np.pi * x_arr)
    term2 = 2.5 * np.pi * (-x_arr**2 + x_arr) * np.cos(5.0 * np.pi * x_arr)
    return term1 + term2


def singularity_exact(x: np.ndarray | float, pe: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    num = np.exp(pe * (x_arr - 1.0)) - np.exp(-pe)
    den = 1.0 - np.exp(-pe)
    return (num / den) - x_arr + 1.0


def singularity_f(x: np.ndarray | float, pe: float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    num = pe * np.exp(pe * (x_arr - 1.0))
    den = 1.0 - np.exp(-pe)
    return (num / den) - 1.0


def u_exact(x: np.ndarray | float, pe: float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x, pe)


def mixed_f(x: np.ndarray | float, pe: float) -> np.ndarray:
    return roughness_f(x) + singularity_f(x, pe)


def operators_for_mesh(run: dict[str, object], num_elements: int, pe: float) -> list[Operator]:
    if "poly_order" in run:
        op = build_polynomial_operator(run["poly_order"], run.get("op_type", "closed"))
    elif "exp_order" in run:
        op = build_exponential_operator(
            p=run["exp_order"],
            alpha=pe,
            alpha_divisor=num_elements,
            optimize=run.get("optimized", True),
            op_type=run.get("op_type", "closed"),
        )
    else:
        raise ValueError("Unknown run specification in RUNS list.")
    
    if not check_sbp_property(op, print_report=False):
        warnings.warn(f"Operator {op.name} violates SBP property on {num_elements} elements!", RuntimeWarning)
        
    return [op] * num_elements


def solve_on_mesh(
    run: dict[str, object], num_elements: int, pe: float, exact_fun: callable, f_fun: callable
) -> tuple[list[Element1D], np.ndarray]:
    left_val = float(exact_fun(DOMAIN[0], pe))
    ops = operators_for_mesh(run, num_elements, pe)
    
    elements = make_uniform_elements(
        domain=DOMAIN, num_elements=num_elements, operators=ops,
        a_fun=lambda x: np.ones_like(x, dtype=float),
        b_fun=lambda x: np.ones_like(x, dtype=float),
        f_fun=lambda x: f_fun(x, pe),
        exact_fun=lambda x: exact_fun(x, pe),
    )
    system = assemble_system(elements, left_bc_fun=lambda _x: left_val, sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs)
    return system.elements, u


def run_convergence(
    run: dict[str, object], pe: float, exact_fun: callable, f_fun: callable
) -> tuple[np.ndarray, np.ndarray, float, float]:
    errors, dofs, hs = [], [], []
    print(f"\nRun: {run['label']} (Pe={pe:g})")
    print("num_elements  total_dofs  H_error         rate")

    err_32 = np.nan
    for num_elements in ELEMENT_COUNTS:
        elements, u = solve_on_mesh(run, num_elements, pe, exact_fun, f_fun)
        err = global_H_error(elements, u, lambda x: exact_fun(x, pe))
        errors.append(err)
        dofs.append(u.size)
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(num_elements))
        if num_elements == 32:
            err_32 = err

    errors_arr = np.array(errors, dtype=float)
    hs_arr = np.array(hs, dtype=float)
    rates = convergence_rate(errors_arr, hs_arr)

    for n_el, n_dof, err, rate in zip(ELEMENT_COUNTS, dofs, errors, rates):
        r_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{n_el:12d}  {n_dof:10d}  {err:12.4e}  {r_str}")

    valid_rates = rates[~np.isnan(rates)]
    overall_rate = float(valid_rates[-1]) if len(valid_rates) > 0 else np.nan

    return np.array(dofs, dtype=float), errors_arr, err_32, overall_rate


if __name__ == "__main__":
    STEEPNESS_CONFIGS = [
        {"label": "eps=0.0125", "pe": 80.0},
        {"label": "eps=0.0250", "pe": 40.0},
        {"label": "eps=0.0500", "pe": 20.0},
        {"label": "eps=0.1000", "pe": 10.0},
    ]

    EXPERIMENTS = [
        {"label": "Mixed problem", "exact": u_exact, "f": mixed_f},
        {"label": "Singular problem", "exact": singularity_exact, "f": singularity_f},
        {"label": "Smooth problem", "exact": lambda x, _pe: roughness_exact(x), "f": lambda x, _pe: roughness_f(x)},
    ]

    summary_records = []

    for steep_cfg in STEEPNESS_CONFIGS:
        pe_val = float(steep_cfg["pe"])
        
        for exp in EXPERIMENTS:
            dof_rows, err_rows, profiles = [], [], []
            print(f"\n==========================================")
            print(f"{steep_cfg['label']} | {exp['label']}")
            print(f"==========================================")

            for run in RUNS:
                dofs, errors, e32, r_final = run_convergence(run, pe=pe_val, exact_fun=exp["exact"], f_fun=exp["f"])
                dof_rows.append(dofs)
                err_rows.append(errors)

                summary_records.append((steep_cfg["label"], exp["label"], str(run["label"]), e32, r_final))

                coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS, pe_val, exact_fun=exp["exact"], f_fun=exp["f"])
                profiles.append(profile_from_elements(coarse_elements, coarse_u))

            labels = [str(r["label"]) for r in RUNS]
            plot_convergence(
                np.vstack(dof_rows), np.vstack(err_rows), labels,
                title=f"{exp['label']} ({steep_cfg['label']})", grid=True, skipfit_st=[1] * len(RUNS)
            )

            x_exact, u_exact_vals = exact_profile_on_domain(lambda x: exp["exact"](x, pe_val), domain=DOMAIN)
            if PLOT_SOLS:
                plot_solution_profiles(
                    profiles, labels, x_exact=x_exact, u_exact=u_exact_vals,
                    title=f"{exp['label']} Solutions ({steep_cfg['label']})", grid=True
                )

    print("\n" + "="*90)
    print(f"{'Steepness':<30s} | {'Problem':<18s} | {'Method':<14s} | {'Error (N=32)':<12s} | {'Rate':<8s}")
    print("="*90)
    for row in summary_records:
        steep, prob, meth, e32, r = row
        r_str = f"{r:.4f}" if not np.isnan(r) else "NaN"
        print(f"{steep:<30s} | {prob:<18s} | {meth:<14s} | {e32:<12.4e} | {r_str:<8s}")
    print("="*90 + "\n")

    if SHOW_PLOTS:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")