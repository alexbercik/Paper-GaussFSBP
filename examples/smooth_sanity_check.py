from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt

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
from src.problems import Problem
from src.solve import solve_steady


def u_exact(x: np.ndarray) -> np.ndarray:
    return np.sin(2.0 * np.pi * x)


def build_problem(case: str) -> Problem:
    if case == "conservative":
        a_fun = lambda x: np.ones_like(x)
        b_fun = lambda x: 1.0 + 0.25 * np.sin(2.0 * np.pi * x)
        f_fun = (
            lambda x: 2.0 * np.pi * np.cos(2.0 * np.pi * x)
            + 0.5 * np.pi * np.sin(4.0 * np.pi * x)
        )
    elif case == "non-conservative":
        a_fun = lambda x: 1.0 + 0.25 * np.sin(2.0 * np.pi * x)
        b_fun = lambda x: np.ones_like(x)
        f_fun = lambda x: (1.0 + 0.25 * np.sin(2.0 * np.pi * x)) * (2.0 * np.pi * np.cos(2.0 * np.pi * x))
    else:
        raise ValueError(f"Unknown case '{case}'")

    return Problem(a_fun=a_fun, b_fun=b_fun, f_fun=f_fun, exact_fun=u_exact)


def solve_on_mesh(case: str, num_elements: int, sat_type: str = "upwind") -> tuple[list[Element1D], np.ndarray]:
    problem = build_problem(case)
    operator = OperatorSpec(
        ["1", "x", "x^2"],
        ["1", "x", "x^2", "x^3", "x^4", "x^5"],
        "open",
    )
    #operator = OperatorSpec("LGp2") # alternatively, can just use the name
    elements = make_uniform_elements(
        domain=(0.0, 1.0),
        num_elements=num_elements,
        operators=operator,
        a_fun=problem.a_fun,
        b_fun=problem.b_fun,
        f_fun=problem.f_fun,
        exact_fun=problem.exact_fun,
    )
    system = assemble_system(
        elements,
        left_bc_fun=problem.left_bc_fun or problem.exact_fun,
        sat_type=sat_type,
    )
    u, _ = solve_steady(system.matrix, system.rhs)
    return system.elements, u


def run_case(case: str, sat_type: str = "upwind") -> tuple[np.ndarray, np.ndarray]:
    element_counts = [4, 8, 16, 32]

    errors: list[float] = []
    dofs: list[float] = []
    hs: list[float] = []

    print(f"\nCase: {case}, SAT: {sat_type}")
    print("num_elements  total_dofs  H_error        rate")

    for nelem in element_counts:
        elements, u = solve_on_mesh(case, nelem, sat_type=sat_type)
        err = global_H_error(elements, u, u_exact)

        n_dof = nelem * 3
        errors.append(err)
        dofs.append(n_dof)
        hs.append(1.0 / nelem)

    rates = convergence_rate(np.array(errors), np.array(hs))
    for nelem, n_dof, err, rate in zip(element_counts, dofs, errors, rates):
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{nelem:12d}  {n_dof:10d}  {err:12.4e}  {rate_str}")

    return np.array(dofs), np.array(errors)


if __name__ == "__main__":
    sat_type = "upwind"
    coarse_elements = 4

    dof_cons, err_cons = run_case("conservative", sat_type=sat_type)
    dof_noncons, err_noncons = run_case("non-conservative", sat_type=sat_type)

    plot_convergence(
        np.vstack([dof_cons, dof_noncons]),
        np.vstack([err_cons, err_noncons]),
        [f"conservative ({sat_type})", f"non-conservative ({sat_type})"],
        title="Smooth problem: $H$ error vs. degrees of freedom",
        grid=True
    )

    elements_cons, u_cons = solve_on_mesh("conservative", coarse_elements, sat_type=sat_type)
    elements_noncons, u_noncons = solve_on_mesh("non-conservative", coarse_elements, sat_type=sat_type)
    x_exact, u_exact_vals = exact_profile_on_domain(u_exact, domain=(0.0, 1.0))

    plot_solution_profiles(
        [
            profile_from_elements(elements_cons, u_cons),
            profile_from_elements(elements_noncons, u_noncons),
        ],
        [f"conservative ({sat_type})", f"non-conservative ({sat_type})"],
        x_exact=x_exact,
        u_exact=u_exact_vals,
        title=rf"Coarsest mesh ({coarse_elements} elements)",
        grid=True,
    )

    plt.show()
