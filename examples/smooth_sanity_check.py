from __future__ import annotations

import numpy as np

from gaussfsbp.assembly import assemble_system
from gaussfsbp.mesh import ElementSpec, Mesh1D
from gaussfsbp.norms import convergence_rate, global_H_error
from gaussfsbp.operators import builtin_operator_repository
from gaussfsbp.problems import Problem
from gaussfsbp.solve import solve_steady


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


def run_case(case: str, sat_type: str = "upwind") -> None:
    repository = builtin_operator_repository()
    element_counts = [4, 8, 16, 32]

    errors: list[float] = []
    hs: list[float] = []

    print(f"\nCase: {case}, SAT: {sat_type}")
    print("num_elements  total_dofs  H_error        rate")

    for nelem in element_counts:
        specs = [
            ElementSpec(["1", "x", "x^2"], "closed", selector=(i % 2))
            for i in range(nelem)
        ]
        mesh = Mesh1D.uniform(domain=(0.0, 1.0), num_elements=nelem, element_spec=specs[0])
        mesh = Mesh1D.from_bounds(mesh.domain, mesh.element_bounds, specs)

        system = assemble_system(mesh, repository, build_problem(case), sat_type=sat_type)
        u, _ = solve_steady(system.matrix, system.rhs)
        err = global_H_error(system.elements, u, u_exact)

        errors.append(err)
        hs.append(1.0 / nelem)

    rates = convergence_rate(np.array(errors), np.array(hs))
    for nelem, err, rate in zip(element_counts, errors, rates):
        dofs = nelem * 3
        rate_str = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{nelem:12d}  {dofs:10d}  {err:12.4e}  {rate_str}")


if __name__ == "__main__":
    run_case("conservative", sat_type="upwind")
    run_case("non-conservative", sat_type="upwind")
