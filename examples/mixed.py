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
ELEMENT_COUNTS = [32, 64, 68, 74, 80, 84, 90, 100, 160]
#ELEMENT_COUNTS = [4, 8, 16, 32, 64, 80, 160, 200, 400]
COARSE_ELEMENTS = 4
SAT_TYPE = "upwind"
SHOW_PLOTS = True

# Params for mixed source term eg: sin(k*pi*x - omega*t)*exp(-C(x-0.5)^2)*x^3 + 3sqrt(1-x) 
K = 7.0
OMEGA = 2.0
C = -0.5

POW = 1/5
STATIC_TYPE = "exponential"  # power or exponential

if STATIC_TYPE == "exponential": 
    T = 1.0
elif STATIC_TYPE == "power": 
    T = 0.2


# Poly coeffs to modulate the oscillation: poly(x) = sum coeffs[i] * x**i
POLY_COEFFS = [0, 0, 0, 1]


EXP_01_OPERATOR = OperatorSpec("EXPp2_01")
SQRT_01_OPERATOR = OperatorSpec("SQRTp2_01")


# For each run, set exactly one of num_right_elements or x_right_elements.
# When x_right_elements is set, every element with right edge > x_right_elements
# uses right_operator.
COMMON_RUNS = [
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
        "label": "LGLp2_01",
        "interior_operator": OperatorSpec("LGLp2_01"),
        "right_operator": OperatorSpec("LGLp2_01"),
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp3_01",
        "interior_operator": OperatorSpec("LGLp3_01"),
        "right_operator": OperatorSpec("LGLp3_01"),
        "num_right_elements": 0,
        "x_right_elements": None,
    },
    {
        "label": "LGLp2_01 LE", #go back do open
        "interior_operator": OperatorSpec("LGLp2_01"),
        "right_operator": EXP_01_OPERATOR,
        "num_right_elements": None,
        "x_right_elements": 0.90,
    },
    {
        "label": "LGp2_01 GE",
        "interior_operator": EXP_01_OPERATOR,
        "right_operator": EXP_01_OPERATOR,
        "num_right_elements": 0,
        "x_right_elements": None,
    },
]
SQRT_RUNS = [
    {
        "label": "LGp2_01 SQRT_01",
        "interior_operator": OperatorSpec("LGp2_01"),
        "right_operator": SQRT_01_OPERATOR,
        "num_right_elements": 1,
        "x_right_elements": None,
    },
    {
        "label": "SQRT_01 Full",
        "interior_operator": SQRT_01_OPERATOR,
        "right_operator": SQRT_01_OPERATOR,
        "num_right_elements": 0,
        "x_right_elements": None,
    }
]


if STATIC_TYPE == "exponential":
    RUNS = COMMON_RUNS + EXP_RUNS
elif STATIC_TYPE == "power" or STATIC_TYPE == "root":
    RUNS = COMMON_RUNS + SQRT_RUNS
else:
    raise ValueError(f"Unknown STATIC_TYPE '{STATIC_TYPE}' for RUNS selection")


def _roughness_terms(x_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    poly = sum(coef * x_arr ** i for i, coef in enumerate(POLY_COEFFS))
    d_poly = sum(i * coef * x_arr ** (i - 1) for i, coef in enumerate(POLY_COEFFS) if i > 0)
    gauss = np.exp(-C * (x_arr - 0.5) ** 2)
    d_gauss = -2.0 * C * (x_arr - 0.5) * gauss
    s = np.sin(K * np.pi * x_arr - OMEGA * T)
    c = np.cos(K * np.pi * x_arr - OMEGA * T)
    return poly, d_poly, gauss, d_gauss, s, c

'''
def static_component(x: np.ndarray | float) -> np.ndarray:
    """Exact steady state contribution of the singularity component."""
    x_arr = np.asarray(x, dtype=float)
    if STATIC_TYPE == "power" or STATIC_TYPE == "root":
        return np.clip(3.0 * np.sqrt(1.0 - x_arr), 0.0, None)
    if STATIC_TYPE == "exponential":
        return np.exp(x_arr)
    raise ValueError(f"Unknown STATIC_TYPE '{STATIC_TYPE}'")

def singularity_f(x: np.ndarray) -> np.ndarray:
    """Analytical spatial derivative for advection u_x matching static_component."""
    x_arr = np.asarray(x, dtype=float)
    if STATIC_TYPE == "power" or STATIC_TYPE == "root":
        base = np.clip(1.0 - x_arr, 1e-16, None)
        # d/dx of 3*sqrt(1-x) = -1.5 / sqrt(1-x)
        return -1.5 / np.sqrt(base)
    if STATIC_TYPE == "exponential":
        return np.exp(x_arr)
    raise ValueError(f"Unknown STATIC_TYPE '{STATIC_TYPE}'")
'''

def static_component(x: np.ndarray | float) -> np.ndarray:
    """Exact steady state contribution of the boundary layer singularity."""
    x_arr = np.asarray(x, dtype=float)
    if STATIC_TYPE == "power" or STATIC_TYPE == "root":
        return np.clip(3.0 * np.sqrt(1.0 - x_arr), 0.0, None)
        
    if STATIC_TYPE == "exponential":
        epsilon = 0.0125
        
        num = np.exp((x_arr - 1.0) / epsilon) - np.exp(-1.0 / epsilon)
        den = 1.0 - np.exp(-1.0 / epsilon)
        return num / den
        
    raise ValueError(f"Unknown STATIC_TYPE '{STATIC_TYPE}'")


def singularity_f(x: np.ndarray) -> np.ndarray:
    """Analytical spatial derivative (u_x) for the advection equation."""
    x_arr = np.asarray(x, dtype=float)
    if STATIC_TYPE == "power" or STATIC_TYPE == "root":
        base = np.clip(1.0 - x_arr, 1e-16, None)
        return -1.5 / np.sqrt(base)
        
    if STATIC_TYPE == "exponential":
        epsilon = 0.0125
       
        num = (1.0 / epsilon) * np.exp((x_arr - 1.0) / epsilon)
        den = 1.0 - np.exp(-1.0 / epsilon)
        return num / den
        
    raise ValueError(f"Unknown STATIC_TYPE '{STATIC_TYPE}'")

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


f_fun = mixed_f


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
    system = assemble_system(
        elements,
        left_bc_fun=left_bc_fun,
        sat_type=SAT_TYPE,
    )
    u, _ = solve_steady(system.matrix, system.rhs)
    return system.elements, u


def run_convergence(
    run: dict[str, object],
    exact_fun: callable = u_exact,
    f_fun: callable = mixed_f,
) -> tuple[np.ndarray, np.ndarray]:
    errors: list[float] = []
    dofs: list[int] = []
    hs: list[float] = []

    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")
    print("num_elements  total_dofs  H_error         rate")

    for num_elements in ELEMENT_COUNTS:
        elements, u = solve_on_mesh(run, num_elements, exact_fun=exact_fun, f_fun=f_fun)
        errors.append(global_H_error(elements, u, exact_fun))
        dofs.append(sum(element.x.size for element in elements))
        hs.append((DOMAIN[1] - DOMAIN[0]) / float(num_elements))

    rates = convergence_rate(np.array(errors), np.array(hs))
    for num_elements, n_dof, err, rate in zip(ELEMENT_COUNTS, dofs, errors, rates):
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{num_elements:12d}  {n_dof:10d}  {err:12.4e}  {rate_str}")

    return np.array(dofs, dtype=float), np.array(errors, dtype=float)


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
    dof_rows: list[np.ndarray] = []
    err_rows: list[np.ndarray] = []
    profiles = []

    print(f"\nExperiment: {experiment['label']}")
    for run in RUNS:
        dofs, errors = run_convergence(
            run,
            exact_fun=experiment["exact_fun"],
            f_fun=experiment["f_fun"],
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

    singularity_label = "exponential" if STATIC_TYPE == "exponential" else "root"

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
    plot_solution_profiles(
        profiles,
        labels,
        x_exact=x_exact,
        u_exact=u_exact_vals,
        title=f"{experiment['title']}, coarsest mesh ({COARSE_ELEMENTS} elements) - singularity: {singularity_label}",
        grid=True,
    )

if SHOW_PLOTS:
    plt.show(block=False)
    input("Press Enter to close all plots...")
    plt.close("all")