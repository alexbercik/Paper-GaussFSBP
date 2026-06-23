from __future__ import annotations
from scipy import sparse

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import convergence_rate, global_H_error
from src.operator_library import OperatorSpec
from src.plotting import (
    exact_profile_on_domain,
    plot_convergence,
    plot_solution_profiles,
    profile_from_elements,
)
from src.solve import solve_steady


# Domain and Element counts matching your mixed.py format
DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [10, 20, 40, 80, 160]
COARSE_ELEMENTS = 4
SAT_TYPE = "upwind"
SHOW_PLOTS = True

# --- Operator Definitions (From mixed.py exponential case) ---
EXP_01_OPERATOR = OperatorSpec("EXPp2_01")

RUNS_EXP = [
    {
        "label": "LGp2_01",
        "interior_operator": OperatorSpec("LGp2_01"),
        "right_operator": OperatorSpec("LGp2_01"),
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2_01 HR",
        "interior_operator": OperatorSpec("LGLp2_01"),
        "right_operator": OperatorSpec("RadauRp2_01"),
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "LGp3_01",
        "interior_operator": OperatorSpec("LGp3_01"),
        "right_operator": OperatorSpec("LGp3_01"),
        "num_right_elements": 0,
        "x_right_elements": None,
    },
]

EXP_RUNS = [
    {
        "label": "LGLp3_01",
        "interior_operator": OperatorSpec("LGLp3_01"),
        "right_operator": OperatorSpec("LGLp3_01"),
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGp2_01 LE",
        "interior_operator": OperatorSpec("LGp2_01"),
        "right_operator": EXP_01_OPERATOR,
        "num_right_elements": None,
        "x_right_elements": 0.90,
    },
]

RUNS = RUNS_EXP + EXP_RUNS

# --- Example 5.3 PDE Definition ---
def u_exact(x: np.ndarray | float) -> np.ndarray:
    """Exact steady state solution for Example 5.3: u(x) = e^(x^2)"""
    x_arr = np.asarray(x, dtype=float)
    return np.exp(x_arr**2)

def a_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)

def b_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)

def f_fun(x: np.ndarray) -> np.ndarray:
    return np.zeros_like(x, dtype=float)

def left_bc_fun(_x: float) -> float:
    # Use exact solution boundary value for inflow BC (u(0) = 1.0)
    return float(u_exact(DOMAIN[0]))

def count_right_elements(run: dict[str, object], num_elements: int) -> int:
    num_right_elements = run.get("num_right_elements")
    x_right_elements = run.get("x_right_elements")
    if (num_right_elements is None) == (x_right_elements is None):
        raise ValueError(
            "Specify exactly one of num_right_elements or x_right_elements"
        )

    if num_right_elements is not None:
        num_right = int(num_right_elements)
        if num_right < 0:
            raise ValueError("num_right_elements must be nonnegative")
        return min(num_right, num_elements)

    x_start = float(x_right_elements)
    if x_start < DOMAIN[0] or x_start > DOMAIN[1]:
        raise ValueError("x_right_elements must lie inside DOMAIN")

    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    return int(np.count_nonzero(bounds[1:] > x_start))

def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[OperatorSpec]:
    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    return (
        [run["interior_operator"] for _ in range(num_interior)]
        + [run["right_operator"] for _ in range(num_right)]
    )

def solve_on_mesh(
    run: dict[str, object],
    num_elements: int,
) -> tuple[list[Element1D], np.ndarray]:
    elements = make_uniform_elements(
        domain=DOMAIN,
        num_elements=num_elements,
        operators=operators_for_mesh(run, num_elements),
        a_fun=a_fun,
        b_fun=b_fun,
        f_fun=f_fun,
        exact_fun=u_exact,
    )
    
    # 1. Assemble the standard advection matrix (u_x) and RHS
    system = assemble_system(
        elements,
        left_bc_fun=left_bc_fun,
        sat_type=SAT_TYPE,
    )
    
    # 2. Extract the global coordinate array for all nodes
    global_x = np.concatenate([el.x for el in elements])
    
    # term (D_global - 2X) * u = 0
    
    system.matrix -= sparse.diags(2.0 * global_x)
    
    u, _ = solve_steady(system.matrix, system.rhs)
    
    return system.elements, u

def run_convergence(run: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    errors: list[float] = []
    dofs: list[int] = []
    hs: list[float] = []

    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")
    print("num_elements  total_dofs  H_error        rate")

    for num_elements in ELEMENT_COUNTS:
        elements, u = solve_on_mesh(run, num_elements)
        errors.append(global_H_error(elements, u, u_exact))
        dofs.append(sum(element.x.size for element in elements))
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(num_elements))

    rates = convergence_rate(np.array(errors), np.array(hs))
    for num_elements, n_dof, err, rate in zip(ELEMENT_COUNTS, dofs, errors, rates):
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{num_elements:12d}  {n_dof:10d}  {err:12.4e}  {rate_str}")

    return np.array(dofs, dtype=float), np.array(errors, dtype=float)

dof_rows: list[np.ndarray] = []
err_rows: list[np.ndarray] = []
profiles = []

for run in RUNS:
    dofs, errors = run_convergence(run)
    dof_rows.append(dofs)
    err_rows.append(errors)

    coarse_elements, coarse_u = solve_on_mesh(run, COARSE_ELEMENTS)
    profiles.append(profile_from_elements(coarse_elements, coarse_u))

labels = [str(run["label"]) for run in RUNS]
plot_convergence(
    np.vstack(dof_rows),
    np.vstack(err_rows),
    labels,
    title=rf"Shu Ex 5.3 ($u_x = 2xu$)",
    grid=True,
    skipfit_st=[1]*len(RUNS),
)

x_exact, u_exact_vals = exact_profile_on_domain(u_exact, domain=DOMAIN)
plot_solution_profiles(
    profiles,
    labels,
    x_exact=x_exact,
    u_exact=u_exact_vals,
    title=rf"Shu Ex 5.3, Coarsest mesh ({COARSE_ELEMENTS} elements)",
    grid=True
)

if SHOW_PLOTS:
    plt.show()