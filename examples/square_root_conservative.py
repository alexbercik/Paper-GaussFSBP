from __future__ import annotations

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


DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [4, 8, 16, 22, 32, 48, 64, 80]
COARSE_ELEMENTS = 4
SAT_TYPE = "upwind"
SHOW_PLOTS = True
RIGHT_OPEN_TYPES = {"open", "half-open-right"}


HALF_OPEN_RIGHT_dir_OPERATOR = OperatorSpec(
    ["1", "x", "x^2", "(1-x)^{3/2}"],
    ["1", "sqrt(1-x)", "x", "x^2", "x^3", "x sqrt(1-x)", "x^2 sqrt(1-x)"],
    "half-open-right",
)

# For each run, set exactly one of num_right_elements or x_right_elements.
# When x_right_elements is set, every element with right edge > x_right_elements
# uses right_operator.
RUNS = [
    {
        "label": "LGLp2/RadauRp2",
        "interior_operator": "LGLp2",
        "right_operator": "RadauRp2",
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2/dire",
        "interior_operator": "LGLp2",
        "right_operator": HALF_OPEN_RIGHT_dir_OPERATOR,
        "num_right_elements": None, #2,
        "x_right_elements": 0.9, #None,
    },
    {
        "label": "LGLp2/SQRTp1",
        "interior_operator": "LGLp2",
        "right_operator": "SQRTp1",
        "num_right_elements": None,
        "x_right_elements": 0.9,
    },
    {
        "label": "LGLp2/SQRTp1.5",
        "interior_operator": "LGLp2",
        "right_operator": "SQRTp1.5",
        "num_right_elements": None,
        "x_right_elements": 0.9,
    },
    {
        "label": "LGLp2/SQRTp2",
        "interior_operator": "LGLp2",
        "right_operator": "SQRTp2",
        "num_right_elements": None,
        "x_right_elements": 0.9,
    },
    {
        "label": "LGLp2/SQRTp2alt",
        "interior_operator": "LGLp2",
        "right_operator": "SQRTp2alt",
        "num_right_elements": None,
        "x_right_elements": 0.9,
    },
    {
        "label": "LGLp2/SQRTp2.5",
        "interior_operator": "LGLp2",
        "right_operator": "SQRTp2.5",
        "num_right_elements": None,
        "x_right_elements": 0.9,
    },
]


def _sqrt_1mx(x: np.ndarray | float) -> np.ndarray:
    return np.sqrt(np.maximum(0.0, 1.0 - np.asarray(x, dtype=float)))


def u_exact(x: np.ndarray | float) -> np.ndarray:
    return 1.0 + _sqrt_1mx(x)


def a_fun(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x, dtype=float)


def b_fun(x: np.ndarray) -> np.ndarray:
    return 1.0 - x


def f_fun(x: np.ndarray) -> np.ndarray:
    return -1.0 - 1.5 * _sqrt_1mx(x)


def left_bc_fun(_x: float) -> float:
    return 2.0


def count_right_elements(run: dict[str, object], num_elements: int) -> int:
    num_right_elements = run.get("num_right_elements")
    x_right_elements = run.get("x_right_elements")
    if (num_right_elements is None) == (x_right_elements is None):
        raise ValueError(
            "Specify exactly one of num_right_elements or x_right_elements"
        )

    if num_right_elements is not None:
        num_right = int(num_right_elements)
        if num_right < 1:
            raise ValueError("num_right_elements must be positive")
        return min(num_right, num_elements)

    x_start = float(x_right_elements)
    if x_start < DOMAIN[0] or x_start >= DOMAIN[1]:
        raise ValueError("x_right_elements must satisfy DOMAIN[0] <= x < DOMAIN[1]")

    bounds = np.linspace(DOMAIN[0], DOMAIN[1], num_elements + 1)
    num_right = int(np.count_nonzero(bounds[1:] > x_start))
    if num_right < 1:
        raise ValueError("x_right_elements selected no right-boundary elements")
    return num_right


def operators_for_mesh(run: dict[str, object], num_elements: int) -> list[OperatorSpec]:
    right_operator = run["right_operator"]
    #if not isinstance(right_operator, OperatorSpec):
    #    raise TypeError("right_operator must be an OperatorSpec")
    #if right_operator.op_type not in RIGHT_OPEN_TYPES:
    #    raise ValueError("right_operator must be open or half-open-right")

    num_right = count_right_elements(run, num_elements)
    num_interior = num_elements - num_right
    return (
        [run["interior_operator"] for _ in range(num_interior)]
        + [right_operator for _ in range(num_right)]
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
    system = assemble_system(
        elements,
        left_bc_fun=left_bc_fun,
        sat_type=SAT_TYPE,
    )
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
    title="Square-root conservative problem",
    grid=True,
    skipfit_st=[2]*len(RUNS),
    legendsize=10,
)

x_exact, u_exact_vals = exact_profile_on_domain(u_exact, domain=DOMAIN)
plot_solution_profiles(
    profiles,
    labels,
    x_exact=x_exact,
    u_exact=u_exact_vals,
    title=rf"Square-root conservative, coarsest mesh ({COARSE_ELEMENTS} elements)",
    grid=True,
)

if SHOW_PLOTS:
    plt.show()
