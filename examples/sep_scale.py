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
    check_sbp_property,
    legendre_basis_factory,
)
from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import convergence_rate, global_H_error
from src.operators import Operator
from src.solve import solve_steady

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

LBL_P5 = r"$\mathcal{P}_5$ LGL"
LBL_P6 = r"$\mathcal{P}_6$ LGL"
LBL_P3E = r"$\mathcal{P}_3 + e^{\alpha x}$ (optimized)"
LBL_P4E = r"$\mathcal{P}_4 + e^{\alpha x}$ (optimized)"

RUNS = [
    {"label": LBL_P5, "poly_order": 5, "op_type": "closed"},
    {"label": LBL_P6, "poly_order": 6, "op_type": "closed"},
    {"label": LBL_P3E, "exp_order": 3, "op_type": "closed", "optimized": True},
    {"label": LBL_P4E, "exp_order": 4, "op_type": "closed", "optimized": True},
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
        {"pe": 80.0},
        {"pe": 40.0},
        {"pe": 20.0},
        {"pe": 10.0},
    ]


    results = {}

    print("\n==========================================")
    print("Gathering data for the Mixed Problem...")
    print("==========================================")
    
    # Run the tests only for the Mixed Source formulation
    for steep_cfg in STEEPNESS_CONFIGS:
        pe_val = float(steep_cfg["pe"])
        results[pe_val] = {}
        
        for run in RUNS:
            dofs, errors, e32, r_final = run_convergence(run, pe_val, u_exact, mixed_f)
            results[pe_val][run["label"]] = {
                "dofs": dofs,
                "errors": errors,
                "e32": e32,
                "rate": r_final
            }


    print("\n" + "="*120)
    print(f"{'Pe (Alpha)':<10s} | {'Rate Ratio (P3+E/P5)':<22s} | {'Err Ratio (P3+E/P5)':<22s} | {'Rate Ratio (P4+E/P6)':<22s} | {'Err Ratio (P4+E/P6)':<22s}")
    print("-" * 120)
    
    for pe_val in STEEPNESS_CONFIGS:
        val = float(pe_val["pe"])
        res = results[val]
        
        # P3 + Exp vs P5 calculations
        r_P5, r_P3E = res[LBL_P5]["rate"], res[LBL_P3E]["rate"]
        e_P5, e_P3E = res[LBL_P5]["e32"], res[LBL_P3E]["e32"]
        rate_ratio_3 = (r_P3E / r_P5) if not np.isnan(r_P5) else np.nan
        err_ratio_3 = (e_P3E / e_P5) if e_P5 else np.nan
        
        # P4 + Exp vs P6 calculations
        r_P6, r_P4E = res[LBL_P6]["rate"], res[LBL_P4E]["rate"]
        e_P6, e_P4E = res[LBL_P6]["e32"], res[LBL_P4E]["e32"]
        rate_ratio_4 = (r_P4E / r_P6) if not np.isnan(r_P6) else np.nan
        err_ratio_4 = (e_P4E / e_P6) if e_P6 else np.nan

        rate_3_str = f"{rate_ratio_3:.4f}" if not np.isnan(rate_ratio_3) else "NaN"
        err_3_str = f"{err_ratio_3:.4e}" if not np.isnan(err_ratio_3) else "NaN"
        rate_4_str = f"{rate_ratio_4:.4f}" if not np.isnan(rate_ratio_4) else "NaN"
        err_4_str = f"{err_ratio_4:.4e}" if not np.isnan(err_ratio_4) else "NaN"
        
        print(f"{val:<10.1f} | {rate_3_str:<22s} | {err_3_str:<22s} | {rate_4_str:<22s} | {err_4_str:<22s}")
    
    print("="*120 + "\n")


    if SHOW_PLOTS:
        plt.figure(figsize=(10, 7))
        
        # Explicit mapping for colors (Pe) and markers (Operator type)
        colors = {80.0: 'tab:red', 40.0: 'tab:blue', 20.0: 'tab:green', 10.0: 'tab:purple'}
        markers = {
            LBL_P5: 'x',
            LBL_P3E: 'o',
            LBL_P6: '^',
            LBL_P4E: 's'
        }

        for pe_val in [80.0, 40.0, 20.0, 10.0]:
            for run_label in [LBL_P5, LBL_P3E, LBL_P6, LBL_P4E]:
                data = results[pe_val][run_label]
                
                plt.loglog(
                    data["dofs"], data["errors"],
                    marker=markers[run_label],
                    color=colors[pe_val],
                    linestyle='-',
                    markersize=6,
                    label=f"{run_label} (Pe={pe_val})"
                )

        plt.xlabel("Total Degrees of Freedom")
        plt.ylabel(r"Global $\mathbf{H}$-norm Error")
        plt.title(r"Mixed Problem Convergence: Polynomial vs Enriched ($\alpha \in \{10, 20, 40, 80\}$)")
        
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize='small')
        
        plt.grid(True, which="both", ls="--", alpha=0.6)
        plt.tight_layout()
        plt.show(block=False)
        input("Press Enter to close the plot...")
        plt.close("all")