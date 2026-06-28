from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.special

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

def get_lgl_nodes(num_nodes: int) -> np.ndarray:
    p_deriv = scipy.special.legendre(num_nodes - 1).deriv()
    interior_roots = p_deriv.roots
    roots_ref = np.concatenate([[-1.0], np.sort(interior_roots), [1.0]])
    return 0.5 * (roots_ref + 1.0)

def exponential_bases_p3(beta: float) -> tuple[JuliaBasis, JuliaBasis]:
    beta_str = f'BigFloat("{beta!r}")'
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

def exponential_bases_p4(beta: float) -> tuple[JuliaBasis, JuliaBasis]:
    beta_str = f'BigFloat("{beta!r}")'
    exp_b = f"x -> exp({beta_str} * x)"

    op_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3", f"exp({beta:g}x)"],
        functions=["x -> one(x)", "x -> x", "x -> x^2", "x -> x^3", exp_b],
        derivatives=["x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2", f"x -> {beta_str} * exp({beta_str} * x)"],
    )

    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3", "x^4", "x^5", f"exp({beta:g}x)", f"x exp({beta:g}x)", f"x^2 exp({beta:g}x)", f"x^3 exp({beta:g}x)", f"exp(2*{beta:g}x)"],
        functions=[
            "x -> one(x)", "x -> x", "x -> x^2", "x -> x^3", "x -> x^4", "x -> x^5",
            exp_b, f"x -> x * exp({beta_str} * x)", f"x -> x^2 * exp({beta_str} * x)", f"x -> x^3 * exp({beta_str} * x)", f"x -> exp(2 * {beta_str} * x)"
        ],
        derivatives=[
            "x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2", "x -> 4*x^3", "x -> 5*x^4",
            f"x -> {beta_str} * exp({beta_str} * x)",
            f"x -> (one(x) + {beta_str} * x) * exp({beta_str} * x)",
            f"x -> (2*x + {beta_str} * x^2) * exp({beta_str} * x)",
            f"x -> (3*x^2 + {beta_str} * x^3) * exp({beta_str} * x)",
            f"x -> 2 * {beta_str} * exp(2 * {beta_str} * x)"
        ],
    )
    return op_basis, quad_basis

def get_min_norm_exp_operator(h: float, order: int, pe: float) -> Operator:
    beta = h * pe
    numnodes = order + 2
    cache_key = f"h{h:g}_beta{beta:g}_min_norm_p{order}_closed_{numnodes}nodes"
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

    print(f"  -> Cache miss. Generating min-norm p{order} operator for h={h:g} (beta={beta:g})...")
    
    if order == 3:
        op_basis, quad_basis = exponential_bases_p3(beta)
    else:
        op_basis, quad_basis = exponential_bases_p4(beta)

    nodes = get_lgl_nodes(numnodes)

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=64, orthogonalize=True, principal="upper",
        quad_kwargs={"lost_digits": 8}, use_optimization=False
    )

    cache[cache_key] = {
        "h": h, "beta": beta, "node_type": "opt",
        "basis": op_basis.labels, "quad_basis": quad_basis.labels,
        "op_type": "closed", "selector": 0, "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(), "D": operator.D.tolist(),
        "H": operator.H.tolist(), "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"EXP_{cache_key}", op_type="closed")

DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [8, 16, 32, 64, 80, 100]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True

RUNS = [
    {
        "label": "LGL p3",
        "spec": OperatorSpec("LGLp3"),
        "min_norm": False,
        "order": 3,
    },
    {
        "label": "Min-Norm p3",
        "min_norm": True,
        "order": 3,
    },
    # {
    #     "label": "Min-Norm p4",
    #     "min_norm": True,
    #     "order": 4,
    # },
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
    h = (DOMAIN[1] - DOMAIN[0]) / num_elements
    if run.get("min_norm"):
        op = get_min_norm_exp_operator(h=h, order=int(run["order"]), pe=pe)
    else:
        op = operator_from_spec(run["spec"])
    return [op] * num_elements

def solve_on_mesh(
    run: dict[str, object], num_elements: int, pe: float, exact_fun: callable, f_fun: callable
) -> tuple[list[Element1D], np.ndarray]:
    left_val = float(exact_fun(DOMAIN[0], pe))
    elements = make_uniform_elements(
        domain=DOMAIN, num_elements=num_elements,
        operators=operators_for_mesh(run, num_elements, pe),
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