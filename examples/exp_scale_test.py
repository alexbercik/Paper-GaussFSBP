from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

# Ensure repository root is in the path for src imports
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import JuliaBasis, build_operator_from_julia
from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import global_H_error
from src.operator_library import OperatorSpec, operator_from_spec
from src.operators import Operator
from src.solve import solve_steady
from src.plotting import profile_from_elements

CACHE_FILE = Path(__file__).parent / "operator_cache.json"

DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [16, 32, 64]
SAT_TYPE = "upwind"

# "mixed" or "singular"
PROBLEM_TYPE = "mixed"  

# True Peclet number of the physical singularity
PE_TRUE = 500.0  

def load_cache() -> dict:
    if CACHE_FILE.exists():
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cache(cache: dict) -> None:
    with open(CACHE_FILE, "w") as f:
        json.dump(cache, f, indent=4)

def exponential_bases_p3(beta: float) -> tuple[JuliaBasis, JuliaBasis]:
    """Build p3 equivalent basis scaled by an arbitrary beta."""
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

def get_dynamic_exp_operator(h: float, beta_basis: float) -> Operator:
    """Fetch optimal Generalized Gauss-Lobatto p3 operator using a specified beta."""
    cache_key = f"h{h:g}_beta{beta_basis:g}_opt_p3_closed"
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

    print(f"  -> Cache miss. Compiling p3 operator for h={h:g} (beta={beta_basis:g})...")
    
    op_basis, quad_basis = exponential_bases_p3(beta_basis)

    operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat",
        digits=64,
        orthogonalize=True,
        principal="upper",
        quad_kwargs={"lost_digits": 8},
    )

    cache[cache_key] = {
        "h": h,
        "beta": beta_basis,
        "node_type": "opt",
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": "closed",
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist()
    }
    save_cache(cache)
    
    return dataclasses.replace(operator, name=f"EXP_{cache_key}", op_type="closed")

def roughness_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    return 0.5 * (-x_arr**2 + x_arr) * np.sin(5.0 * np.pi * x_arr)

def roughness_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    term1 = 0.5 * (1.0 - 2.0 * x_arr) * np.sin(5.0 * np.pi * x_arr)
    term2 = 2.5 * np.pi * (-x_arr**2 + x_arr) * np.cos(5.0 * np.pi * x_arr)
    return term1 + term2

def singularity_exact(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    num = np.exp(PE_TRUE * (x_arr - 1.0)) - np.exp(-PE_TRUE)
    den = 1.0 - np.exp(-PE_TRUE)
    return x_arr - (num / den)

def singularity_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    num = PE_TRUE * np.exp(PE_TRUE * (x_arr - 1.0))
    den = 1.0 - np.exp(-PE_TRUE)
    return 1.0 - (num / den)

def u_mixed_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)

def mixed_f(x: np.ndarray | float) -> np.ndarray:
    return roughness_f(x) + singularity_f(x)

def a_fun(x: np.ndarray) -> np.ndarray: return np.ones_like(x, dtype=float)
def b_fun(x: np.ndarray) -> np.ndarray: return np.ones_like(x, dtype=float)

def solve_baseline_lgl(num_elements: int, exact_fun, f_fun) -> float:
    int_op = operator_from_spec(OperatorSpec("LGLp3"))
    
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

def solve_exp_on_mesh_full(num_elements: int, beta_basis: float, exact_fun, f_fun):
    h = (DOMAIN[1] - DOMAIN[0]) / num_elements
    operator = get_dynamic_exp_operator(h, beta_basis)
    
    def left_bc_fun(_x: float) -> float:
        return float(exact_fun(DOMAIN[0]))

    elements = make_uniform_elements(
        domain=DOMAIN,
        num_elements=num_elements,
        operators=[operator] * num_elements,
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=exact_fun,
    )
    
    system = assemble_system(elements, left_bc_fun=left_bc_fun, sat_type=SAT_TYPE)
    u, _ = solve_steady(system.matrix, system.rhs)
    
    return elements, u

def solve_exp_on_mesh(num_elements: int, beta_basis: float, exact_fun, f_fun) -> float:
    elements, u = solve_exp_on_mesh_full(num_elements, beta_basis, exact_fun, f_fun)
    return global_H_error(elements, u, exact_fun)

if __name__ == "__main__":
    
    # 1. Routing based on toggle
    if PROBLEM_TYPE.lower() == "mixed":
        target_exact = u_mixed_exact
        target_f = mixed_f
        plot_label = "Mixed"
        marker_style = '.'
        line_style = '-'
        mfc_style = None
    elif PROBLEM_TYPE.lower() == "singular":
        target_exact = singularity_exact
        target_f = singularity_f
        plot_label = "Singular"
        marker_style = 'o'
        line_style = '--'
        mfc_style = 'none'
    else:
        raise ValueError("PROBLEM_TYPE must be 'mixed' or 'singular'")

    hs = np.array([(DOMAIN[1] - DOMAIN[0]) / n for n in ELEMENT_COUNTS])
    log_hs = np.log(hs)

    print(f"Computing baseline standard LGLp3 convergence for {plot_label} problem...")
    lgl_errors = {}
    
    for n in ELEMENT_COUNTS:
        lgl_errors[n] = solve_baseline_lgl(n, target_exact, target_f)
        
    lgl_rate = np.polyfit(log_hs, np.log(list(lgl_errors.values())), 1)[0]
    print(f"  LGLp3 {plot_label} Rate: {lgl_rate:.4f}\n")

    h_16 = (DOMAIN[1] - DOMAIN[0]) / 16.0
    beta_16 = h_16 * PE_TRUE
    
    elems_16, u_16 = solve_exp_on_mesh_full(16, beta_16, target_exact, target_f)
    
    # Extraction 
    prof = profile_from_elements(elems_16, u_16)
    x_plot = np.concatenate([p[0] for p in prof])
    val_plot = np.concatenate([p[1] for p in prof])
    
    x_dense = np.linspace(0, 1, 500)
    exact_dense = target_exact(x_dense)

    ratios = np.linspace(0.1, 1.0, 20)  
    
    errs = {n: [] for n in ELEMENT_COUNTS}
    rates = []
    
    print(f"Running Basis Sensitivity Sweep for {plot_label} Problem...")
    print(f"True Peclet Number: {PE_TRUE}")
    print("-" * 50)
    
    for r in ratios:
        print(f"Evaluating Ratio r = {r:.4f} (Pe_basis = {PE_TRUE * r:.2f})")
        curr_errs = []
        
        for n in ELEMENT_COUNTS:
            h = (DOMAIN[1] - DOMAIN[0]) / n
            beta_basis = h * PE_TRUE * r
            
            err = solve_exp_on_mesh(n, beta_basis, target_exact, target_f)
            errs[n].append(err)
            curr_errs.append(err)
        
        if np.max(curr_errs) < 1e-12:
            slope = np.nan
        else:
            slope, _ = np.polyfit(log_hs, np.log(curr_errs), 1)
        # -----------------------------
        
        rates.append(slope)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

    colors = {16: 'tab:blue', 32: 'tab:orange', 64: 'tab:green'}

    # --- Plot 1: Error vs Scaling Ratio ---
    for n in ELEMENT_COUNTS:
        c = colors[n]
        # EXP Opt Sweep
        ax1.plot(ratios, errs[n], marker=marker_style, mfc=mfc_style, linestyle=line_style, 
                 color=c, label=f"{plot_label} (N={n})")
        
        # LGLp3 Baseline
        lbl = f"{plot_label} LGLp3 Baseline" if n == 16 else ""
        ax1.axhline(lgl_errors[n], color=c, linestyle=':', alpha=0.4, label=lbl)

    ax1.set_yscale("log")
    ax1.set_title(f"{plot_label}: H-Norm Error vs. Basis Scaling")
    ax1.set_xlabel(r"Basis function/singularity scaling r")
    ax1.set_ylabel("Absolute H-Norm Error")
    ax1.legend(ncol=2, fontsize='small')

    # --- Plot 2: Convergence Rate vs Scaling Ratio ---
    c_rate = 'tab:red' if PROBLEM_TYPE == "mixed" else 'tab:brown'
    ax2.plot(ratios, rates, marker=marker_style, mfc=mfc_style, color=c_rate, label=f"{plot_label} (EXP Opt p3)")
    ax2.axhline(lgl_rate, color=c_rate, linestyle='--', 
                label=f"{plot_label} (LGLp3 Baseline: {lgl_rate:.2f})")

    ax2.set_title(f"{plot_label}: Order of Convergence vs. Basis Scaling")
    ax2.set_xlabel(r"Basis function/singularity scaling r")
    ax2.set_ylabel(r"Convergence Rate $\mathcal{O}(h^p)$")
    ax2.legend()
    ax2.grid(True, which="both", ls="--", alpha=0.6)

    # --- Plot 3: Solution Profiles (N=16) ---
    ax3.plot(x_dense, exact_dense, 'k-', alpha=0.6, label=f"{plot_label} (Exact)")
    ax3.plot(x_plot, val_plot, marker_style, mfc=mfc_style, color=c_rate, markersize=4, label=f"{plot_label} (N=16 Numeric)")
    
    ax3.set_title(f"{plot_label}: Solution Profile (N=16, r=1.0)")
    ax3.set_xlabel("x")
    ax3.set_ylabel("u(x)")
    ax3.legend()
    ax3.grid(True, which="both", ls="--", alpha=0.6)

    plt.tight_layout()
    plt.show()