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

from src import JuliaBasis, build_operator_from_julia, build_operator_from_sbp_extra
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

CACHE_FILE = Path(__file__).parent / "operator_cache_sqrt.json"

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

def sqrt_bases_p2(k: int) -> tuple[JuliaBasis, JuliaBasis]:
    op_basis = JuliaBasis(
        labels=["1", "x", f"sqrt({k}-x)"],
        functions=["x -> one(x)", "x -> x", f"x -> sqrt({k}.0 - x)"],
        derivatives=["x -> zero(x)", "x -> one(x)", f"x -> -0.5 / sqrt({k}.0 - x)"]
    )
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", f"sqrt({k}-x)", f"x sqrt({k}-x)"],
        functions=[
            "x -> one(x)", "x -> x", "x -> x^2",
            f"x -> sqrt({k}.0 - x)", f"x -> x * sqrt({k}.0 - x)"
        ],
        derivatives=[
            "x -> zero(x)", "x -> one(x)", "x -> 2*x",
            f"x -> -0.5 / sqrt({k}.0 - x)",
            f"x -> sqrt({k}.0 - x) - 0.5 * x / sqrt({k}.0 - x)"
        ]
    )
    return op_basis, quad_basis

def sqrt_bases_p3(k: int) -> tuple[JuliaBasis, JuliaBasis]:
    op_basis = JuliaBasis(
        labels=["1", "x", "x^2", f"sqrt({k}-x)"],
        functions=["x -> one(x)", "x -> x", "x -> x^2", f"x -> sqrt({k}.0 - x)"],
        derivatives=["x -> zero(x)", "x -> one(x)", "x -> 2*x", f"x -> -0.5 / sqrt({k}.0 - x)"]
    )
    quad_basis = JuliaBasis(
        labels=["1", "x", "x^2", "x^3", f"sqrt({k}-x)", f"x sqrt({k}-x)", f"x^2 sqrt({k}-x)"],
        functions=[
            "x -> one(x)", "x -> x", "x -> x^2", "x -> x^3",
            f"x -> sqrt({k}.0 - x)", f"x -> x * sqrt({k}.0 - x)", f"x -> x^2 * sqrt({k}.0 - x)"
        ],
        derivatives=[
            "x -> zero(x)", "x -> one(x)", "x -> 2*x", "x -> 3*x^2",
            f"x -> -0.5 / sqrt({k}.0 - x)",
            f"x -> sqrt({k}.0 - x) - 0.5 * x / sqrt({k}.0 - x)",
            f"x -> 2*x * sqrt({k}.0 - x) - 0.5 * x^2 / sqrt({k}.0 - x)"
        ]
    )
    return op_basis, quad_basis

def _bernstein_basis_on_unit_interval(p: int) -> tuple[list[str], list[str]]:
    labels = []
    functions = []
    for m in range(p + 1):
        labels.append(f"B_{m}^{p}")
        if m == 0:
            if p == 0:
                functions.append("x -> one(x)")
            else:
                functions.append(f"x -> (1 - x)^{p}")
        elif m == p:
            functions.append(f"x -> x^{p}")
        else:
            functions.append(f"x -> binomial({p}, {m}) * x^{m} * (1 - x)^({p} - {m})")
    return labels, functions

def sbp_extra_sqrt_basis(p: int, k: int) -> tuple[list[str], list[str]]:
    labels, functions = _bernstein_basis_on_unit_interval(p)
    labels.append(f"sqrt({k}-x)")
    functions.append(f"x -> sqrt({k}.0 - x)")
    return labels, functions

def get_min_norm_sqrt_operator(k: int, order: int) -> Operator:
    cache_key = f"sqrt_k{k}_min_norm_p{order}_closed"
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

    print(f"  -> Cache miss. Generating min-norm p{order} operator for sqrt({k}-x)...")
    if order == 2:
        op_basis, quad_basis = sqrt_bases_p2(k)
    else:
        op_basis, quad_basis = sqrt_bases_p3(k)

    operator = build_operator_from_julia(
        op_basis, quad_basis, interval=(0.0, 1.0), precision="bigfloat",
        digits=64, orthogonalize=True, principal="upper", quad_kwargs={"lost_digits": 8},
        use_optimization=False
    )

    cache[cache_key] = {
        "k": k, "node_type": "opt",
        "basis": op_basis.labels, "quad_basis": quad_basis.labels,
        "op_type": "closed", "selector": 0, "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(), "D": operator.D.tolist(),
        "H": operator.H.tolist(), "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"SQRT_{cache_key}", op_type="closed")

def get_sbp_extra_equispaced_sqrt_operator(k: int, order: int) -> Operator:
    cache_key = f"sqrt_k{k}_sbpextra_equi_p{order}_closed"
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

    print(f"  -> Cache miss. Generating SBP-Extra equispaced p{order} operator for sqrt({k}-x)...")
    basis_labels, functions = sbp_extra_sqrt_basis(order, k)
    initial_num_nodes = order + 2
    
    operator = build_operator_from_sbp_extra(
        functions, initial_num_nodes, basis_labels=basis_labels, quad_basis_labels=basis_labels,
        op_type="closed", interval=(0.0, 1.0), source="orig",
        max_num_nodes=initial_num_nodes + 20, max_iterations=200000,
        g_tol=1.0e-25, sbp_tolerance=1.0e-12, accuracy_tolerance=1.0e-8
    )

    cache[cache_key] = {
        "k": k, "node_type": "equispaced",
        "basis": basis_labels, "quad_basis": basis_labels,
        "op_type": "closed", "selector": 0, "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(), "D": operator.D.tolist(),
        "H": operator.H.tolist(), "tL": operator.tL.tolist(), "tR": operator.tR.tolist()
    }
    save_cache(cache)
    return dataclasses.replace(operator, name=f"SQRT_{cache_key}", op_type="closed")

def get_inexact_equispaced_5node_sqrt(k: int) -> Operator:
    cache_key = f"sqrt_k{k}_equi_5node_inexact"
    cache = load_cache()

    if cache_key in cache:
        data = cache[cache_key]
        return Operator(
            name=data["name"],
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

    print(f"  -> Cache miss. Generating inexact equispaced 5-node operator for sqrt({k}-x)...")
    x = np.linspace(0.0, 1.0, 5)
    w = np.array([7.0, 32.0, 12.0, 32.0, 7.0]) / 90.0
    
    P = np.diag(w)
    B = np.zeros((5, 5))
    B[0, 0] = -1.0
    B[4, 4] = 1.0
    
    F = np.zeros((5, 3))
    F[:, 0] = 1.0
    F[:, 1] = x
    F[:, 2] = np.sqrt(k - x)
    
    Fx = np.zeros((5, 3))
    Fx[:, 0] = 0.0
    Fx[:, 1] = 1.0
    Fx[:, 2] = -0.5 / np.sqrt(k - x)
    
    R = P @ Fx - 0.5 * B @ F
    
    A = np.zeros((15, 10))
    b = np.zeros(15)
    mapping = {}
    idx = 0
    for i in range(5):
        for j in range(i+1, 5):
            mapping[(i, j)] = idx
            idx += 1
            
    for i in range(5):
        for m in range(3):
            eq = i * 3 + m
            b[eq] = R[i, m]
            for j in range(5):
                if j > i:
                    A[eq, mapping[(i, j)]] += F[j, m]
                elif j < i:
                    A[eq, mapping[(j, i)]] -= F[j, m]
                    
    v, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
    
    S = np.zeros((5, 5))
    for (i, j), v_idx in mapping.items():
        S[i, j] = v[v_idx]
        S[j, i] = -v[v_idx]
        
    Q = 0.5 * B + S
    D = np.linalg.inv(P) @ Q
    
    op_data = {
        "name": f"SQRT_{cache_key}",
        "basis": ["1", "x", f"sqrt({k}-x)"],
        "quad_basis": ["1", "x", "x^2", "x^3", "x^4"], 
        "op_type": "closed",
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": x.tolist(),
        "D": D.tolist(),
        "H": w.tolist(),
        "tL": [1.0, 0.0, 0.0, 0.0, 0.0],
        "tR": [0.0, 0.0, 0.0, 0.0, 1.0]
    }
    cache[cache_key] = op_data
    save_cache(cache)
    return Operator(
        name=op_data["name"],
        basis=op_data["basis"],
        quad_basis=op_data["quad_basis"],
        op_type=op_data["op_type"],
        selector=op_data["selector"],
        interval=np.array(op_data["interval"]),
        nodes=np.array(op_data["nodes"]),
        D=np.array(op_data["D"]),
        H=np.array(op_data["H"]),
        tL=np.array(op_data["tL"]),
        tR=np.array(op_data["tR"])
    )

DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [8, 16, 32, 64, 80, 100]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True

ROOT_POWER = 0.5
K = 7.0
OMEGA = 2.0
C = -0.5
STATIC_TYPE = "root"  
T = 0.2
POLY_COEFFS = [0, 0, 0, 1]

RUNS = [
    {
        "label": "LGp2 (Open, 3-node)",
        "interior_operator": OperatorSpec("LGp2"),
    },
    {
        "label": "LGLp2 (Closed, 3-node)",
        "interior_operator": OperatorSpec("LGLp2"),
    },
    {
        "label": "LGLp3 (Closed, 4-node)",
        "interior_operator": OperatorSpec("LGLp3"),
    },
    {
        "label": "SQRT Inexact Equi p2 (Closed, 5-node)",
        "inexact_5node": True,
        "order": 2,
    },
    {
        "label": "SQRT Min-Norm p2 (Closed)",
        "min_norm": True,
        "order": 2,
    },
    {
        "label": "Mixed LGLp3 / SQRT Min-Norm p3 (x > 0.8)",
        "interior_operator": OperatorSpec("LGLp3"),
        "right_min_norm": True,
        "order": 3,
        "x_right_elements": 0.8,
    },
    {
        "label": "SQRT Min-Norm p3 (Closed)",
        "min_norm": True,
        "order": 3,
    },
    {
        "label": "SQRT SBP-Extra Equi p3 (Closed)",
        "sbp_extra_equispaced": True,
        "order": 3,
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
    x_right = run.get("x_right_elements")
    if x_right is None:
        return 0
    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    return int(np.count_nonzero(bounds[1:] > float(x_right)))

def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[Operator]:
    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    ops = []

    for i in range(num_elements):
        k = num_elements - i
        is_right = (i >= num_interior)

        if run.get("inexact_5node"):
            ops.append(get_inexact_equispaced_5node_sqrt(k=k))
        elif run.get("min_norm"):
            ops.append(get_min_norm_sqrt_operator(k=k, order=run.get("order", 2)))
        elif run.get("sbp_extra_equispaced"):
            ops.append(get_sbp_extra_equispaced_sqrt_operator(k=k, order=run.get("order", 3)))
        elif run.get("right_min_norm"):
            if is_right:
                ops.append(get_min_norm_sqrt_operator(k=k, order=run.get("order", 3)))
            else:
                ops.append(operator_from_spec(run["interior_operator"]))
        else:
            ops.append(operator_from_spec(run["interior_operator"]))

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

            coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"])
            profiles.append(profile_from_elements(coarse_elements, coarse_u))

        labels = [str(run["label"]) for run in RUNS]
        singularity_label = f"root (power={ROOT_POWER})"

        plot_convergence(
            np.vstack(dof_rows),
            np.vstack(err_rows),
            labels,
            title=f"{experiment['title']} (singularity: {singularity_label})",
            grid=True,
            skipfit_st=[1] * len(RUNS),
        )

        x_exact, u_exact_vals = exact_profile_on_domain(experiment["exact_fun"], domain=DOMAIN)
        if PLOT_SOLS: 
            plot_solution_profiles(
                profiles,
                labels,
                x_exact=x_exact,
                u_exact=u_exact_vals,
                title=f"{experiment['title']}, coarsest mesh ({COARSE_ELEMENTS} elements)",
                grid=True,
            )

    if SHOW_PLOTS:
        plt.show(block=False)
        input("Press Enter to close all plots...")
        plt.close("all")