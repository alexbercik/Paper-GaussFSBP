from __future__ import annotations

import dataclasses
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg

from compute_equi_FSBP import get_exact_equispaced_operator

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


def exponential_bases_p2(beta: float) -> tuple[JuliaBasis, JuliaBasis]:
    """Build p2 equivalent basis specifically scaled for the reference element."""
    beta_str = f'BigFloat("{format(beta, "g")}")'
    exp_b = f"x -> exp({beta_str} * x)"

    op_basis = JuliaBasis(
        labels=["1", "x", f"exp({beta:g}x)"],
        functions=["x -> one(x)", "x -> x", exp_b],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            f"x -> {beta_str} * exp({beta_str} * x)",
        ],
    )

    quad_basis = JuliaBasis(
        labels=[
            "1",
            "x",
            "x^2",
            f"exp({beta:g}x)",
            f"x exp({beta:g}x)",
            f"exp(2*{beta:g}x)",
        ],
        functions=[
            "x -> one(x)",
            "x -> x",
            "x -> x^2",
            exp_b,
            f"x -> x * exp({beta_str} * x)",
            f"x -> exp(2 * {beta_str} * x)",
        ],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2*x",
            f"x -> {beta_str} * exp({beta_str} * x)",
            f"x -> (one(x) + {beta_str} * x) * exp({beta_str} * x)",
            f"x -> 2 * {beta_str} * exp(2 * {beta_str} * x)",
        ],
    )
    return op_basis, quad_basis


def exponential_bases_p3(beta: float) -> tuple[JuliaBasis, JuliaBasis]:
    """Build p3 equivalent basis specifically scaled for the reference element."""
    beta_str = f'BigFloat("{format(beta, "g")}")'
    exp_b = f"x -> exp({beta_str} * x)"

    op_basis = JuliaBasis(
        labels=["1", "x", "x^2", f"exp({beta:g}x)"],
        functions=["x -> one(x)", "x -> x", "x -> x^2", exp_b],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2*x",
            f"x -> {beta_str} * exp({beta_str} * x)",
        ],
    )

    quad_basis = JuliaBasis(
        labels=[
            "1",
            "x",
            "x^2",
            "x^3",
            f"exp({beta:g}x)",
            f"x exp({beta:g}x)",
            f"x^2 exp({beta:g}x)",
            f"exp(2*{beta:g}x)",
        ],
        functions=[
            "x -> one(x)",
            "x -> x",
            "x -> x^2",
            "x -> x^3",
            exp_b,
            f"x -> x * exp({beta_str} * x)",
            f"x -> x^2 * exp({beta_str} * x)",
            f"x -> exp(2 * {beta_str} * x)",
        ],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2*x",
            "x -> 3*x^2",
            f"x -> {beta_str} * exp({beta_str} * x)",
            f"x -> (one(x) + {beta_str} * x) * exp({beta_str} * x)",
            f"x -> (2*x + {beta_str} * x^2) * exp({beta_str} * x)",
            f"x -> 2 * {beta_str} * exp(2 * {beta_str} * x)",
        ],
    )
    return op_basis, quad_basis


def get_dynamic_exp_operator(h: float, order: int) -> Operator:
    """Fetch optimal Generalized Gauss-Lobatto operator via Julia."""
    beta = h * PE_TRUE

    cache_key = f"h{h:g}_beta{beta:g}_opt_p{order}_closed"
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
            tR=np.array(data["tR"]),
        )

    print(
        f"  -> Cache miss. Generating optimal p{order} operator for h={h:g} (beta={beta:g})..."
    )

    if order == 2:
        op_basis, quad_basis = exponential_bases_p2(beta)
    else:
        op_basis, quad_basis = exponential_bases_p3(beta)

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
        "beta": beta,
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
        "tR": operator.tR.tolist(),
    }
    save_cache(cache)

    return dataclasses.replace(
        operator, name=f"EXP_{cache_key}", op_type="closed"
    )


def get_exponential_radau_operator(
    h: float, order: int = 2, use_s_opt: bool = False
) -> Operator:
    """Fetch/generate Radau half-open-right operator specifically for the rightmost boundary layer element."""
    beta = h * PE_TRUE

    opt_label = "s_opt" if use_s_opt else "radau_opt"
    cache_key = f"h{h:g}_beta{beta:g}_{opt_label}_p{order}_half_open_right"
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
            tR=np.array(data["tR"]),
        )

    print(
        f"  -> Cache miss. Generating {opt_label} p{order} Radau operator for h={h:g} (beta={beta:g})..."
    )

    if order == 2:
        op_basis, quad_basis = exponential_bases_p2(beta)
    else:
        op_basis, quad_basis = exponential_bases_p3(beta)

    julia_kwargs = {
        "interval": (0.0, 1.0),
        "precision": "bigfloat",
        "digits": 32 if use_s_opt else 64,
        "orthogonalize": True,
        "principal": "lower",
        "quad_kwargs": {"lost_digits": 15 if use_s_opt else 8},
    }

    if use_s_opt:
        opts_funcs = ["x -> x^2", "x -> x^3"]
        opts_derivs = ["x -> 2*x", "x -> 3*x^2"]

        julia_kwargs.update(
            {
                "use_optimization": True,
                "verbose": False,
                "test_functions": opts_funcs,
                "test_derivatives": opts_derivs,
                "test_weights": [1, 1],
                "extrapolation_objective_weights": [0.5, 0.2],
                "S_objective_weights": [1.0, 0.2],
            }
        )

    operator = build_operator_from_julia(op_basis, quad_basis, **julia_kwargs)

    cache[cache_key] = {
        "h": h,
        "beta": beta,
        "node_type": opt_label,
        "basis": op_basis.labels,
        "quad_basis": quad_basis.labels,
        "op_type": "half-open-right",
        "selector": 0,
        "interval": [0.0, 1.0],
        "nodes": operator.nodes.tolist(),
        "D": operator.D.tolist(),
        "H": operator.H.tolist(),
        "tL": operator.tL.tolist(),
        "tR": operator.tR.tolist(),
    }
    save_cache(cache)

    return dataclasses.replace(
        operator, name=f"EXP_{cache_key}", op_type="half-open-right"
    )


DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [32, 64, 68, 74, 80, 84, 90, 100, 160]
COARSE_ELEMENTS = 16
SAT_TYPE = "upwind"
SHOW_PLOTS = True
PLOT_SOLS = True

# True Peclet number of the physical singularity
PE_TRUE = 200.0

RUNS = [
    {
        "label": "LGp2 (Open, 3-node)",
        "interior_operator": OperatorSpec("LGp2"),
        "dynamic_interior": False,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2 (Closed, 3-node)",
        "interior_operator": OperatorSpec("LGLp2"),
        "dynamic_interior": False,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp3 (Closed, 4-node)",
        "interior_operator": OperatorSpec("LGLp3"),
        "dynamic_interior": False,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "EXP Opt p2 (Closed, 4-node)",
        "dynamic_interior": True,
        "order": 2,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "EXP Exact Equi p2 (Closed, 5-node)",
        "dynamic_interior": False,
        "inexact_5node": False,
        "exact_5node": True,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "EXP Opt p3 (Closed, 5-node)",
        "dynamic_interior": True,
        "order": 3,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2 + EXP Radau Right (opt)",
        "interior_operator": OperatorSpec("LGLp2"),
        "dynamic_right": True,
        "right_type": "radau_opt",
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2 + EXP S-Opt Right (check_ops)",
        "interior_operator": OperatorSpec("LGLp2"),
        "dynamic_right": True,
        "right_type": "s_opt",
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "EXP Opt p2 + EXP Radau Right (opt)",
        "dynamic_interior": True,  # Triggers get_dynamic_exp_operator for outer elements
        "order": 2,
        "dynamic_right": True,  # Triggers get_exponential_radau_operator for the rightmost element
        "right_type": "radau_opt",
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "EXP Opt p2 + EXP S-Opt Right (check_ops)",
        "dynamic_interior": True,
        "order": 4,
        "dynamic_right": True,
        "right_type": "s_opt",
        "num_right_elements": 1,
        "x_right_elements": None,
    },
]


# ---------------------------------------------------------
# Exact Solutions & Forcing Terms for Boundary Layer Problem
# ---------------------------------------------------------
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
    return (num / den) - x_arr + 1.0


def singularity_f(x: np.ndarray | float) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    num = PE_TRUE * np.exp(PE_TRUE * (x_arr - 1.0))
    den = 1.0 - np.exp(-PE_TRUE)
    return (num / den) - 1.0


def u_exact(x: np.ndarray | float) -> np.ndarray:
    return roughness_exact(x) + singularity_exact(x)


def mixed_f(x: np.ndarray | float) -> np.ndarray:
    return roughness_f(x) + singularity_f(x)


def a_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def b_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def count_right_elements(run: dict[str, object], num_elements: int) -> int:
    num_right_elements = run.get("num_right_elements")
    x_right_elements = run.get("x_right_elements")
    if (num_right_elements is None) == (x_right_elements is None):
        raise ValueError(
            "Specify exactly one of num_right_elements or x_right_elements"
        )
    if num_right_elements is not None:
        return min(int(num_right_elements), num_elements)

    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    return int(np.count_nonzero(bounds[1:] > float(x_right_elements)))


def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[Operator]:
    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    h = (DOMAIN[1] - DOMAIN[0]) / num_elements

    # Determine interior operator
    if run.get("exact_5node"):
        int_op = get_exact_equispaced_operator(h=h, pe=PE_TRUE, order=2)
    elif run.get("dynamic_interior"):
        int_op = get_dynamic_exp_operator(h=h, order=run.get("order", 2))
    else:
        int_op = operator_from_spec(run["interior_operator"])

    # Determine rightmost boundary layer operator
    if run.get("dynamic_right"):
        right_type = run.get("right_type")
        if right_type == "radau_opt":
            right_op = get_exponential_radau_operator(
                h=h, order=2, use_s_opt=False
            )
        elif right_type == "s_opt":
            right_op = get_exponential_radau_operator(
                h=h, order=2, use_s_opt=True
            )
        else:
            right_op = int_op
    else:
        right_op = int_op

    return [int_op] * num_interior + [right_op] * num_right


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

    print("  Interior Operator Details:")
    print(f"    Basis Functions:      {int_op.basis}")
    print(f"    Quadrature Functions: {int_op.quad_basis}")
    print(f"    Nodes per Element:    {int_op.nodes.size} ({int_op.op_type})")

    if int_op.name != right_op.name:
        print("  Right Operator Details:")
        print(f"    Basis Functions:      {right_op.basis}")
        print(f"    Quadrature Functions: {right_op.quad_basis}")
        print(
            f"    Nodes per Element:    {right_op.nodes.size} ({right_op.op_type})"
        )
    print("-" * 60)

    print("num_elements  total_dofs  H_error         rate")

    for num_elements in ELEMENT_COUNTS:
        elements, u = solve_on_mesh(
            run, num_elements, exact_fun=exact_fun, f_fun=f_fun
        )
        errors.append(global_H_error(elements, u, exact_fun))

        dofs.append(u.size)
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(num_elements))

    rates = convergence_rate(np.array(errors), np.array(hs))
    for num_elements, n_dof, err, rate in zip(
        ELEMENT_COUNTS, dofs, errors, rates
    ):
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
            dofs, errors = run_convergence(
                run, exact_fun=experiment["exact_fun"], f_fun=experiment["f_fun"]
            )
            dof_rows.append(dofs)
            err_rows.append(errors)

            coarse_elements, coarse_u = solve_on_mesh(
                run,
                COARSE_ELEMENTS,
                exact_fun=experiment["exact_fun"],
                f_fun=experiment["f_fun"],
            )
            profiles.append(profile_from_elements(coarse_elements, coarse_u))

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

        x_exact, u_exact_vals = exact_profile_on_domain(
            experiment["exact_fun"], domain=DOMAIN
        )
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