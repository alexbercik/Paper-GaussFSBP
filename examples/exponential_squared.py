from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from sympy import false

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import (
    JuliaBasis,
    build_operator_from_julia,
    check_nullspace_consistency,
    check_sbp_property,
    legendre_basis_factory,
)
from src.assembly import assemble_system
from src.elements import Element1D, make_uniform_elements
from src.norms import convergence_rate, global_H_error
from src.operators import Operator
from src.plotting import (
    exact_profile_on_domain,
    plot_convergence,
    plot_solution_profiles,
    profile_from_elements,
)
from src.solve import solve_steady


DOMAIN = (0.0, 1.0)
REF_DOMAIN = (0.0, 1.0)
ELEMENT_COUNTS = [4, 8, 16, 32]
COARSE_ELEMENTS = 4
SAT_TYPE = "upwind"
SHOW_PLOTS = True
VERBOSE = False
QUAD_VERBOSE = False

# Select "closed" for an LGL-type rule or "open" for an LG-type rule.
op_type = "closed"

# The enriched basis contains P_0, ..., P_p and exp(BASIS_EXPONENT * x).  The
# polynomial comparison operator has degree p + 1 and therefore has the same
# number of approximation functions as the enriched basis.
P = 3
BASIS_EXPONENT = 11 # use 10.1 for p2, 10.9 for p3, 11.4 for p4

# Scale the reference-element enrichment to represent the same physical-space
# exponential on every uniform mesh. If True, the actual basis exponent is
# e^{(BASIS_EXPONENT / N_elem)*x}. If False, it is e^{BASIS_EXPONENT*x}.
SCALE_BASIS_EXPONENT = True

# Keep the model exponent independent of the exponential basis enrichment.
MODEL_EXPONENT_a = 4.0
MODEL_EXPONENT_b = 0.
MODEL_EXPONENT_k = 4
PRECISION_DIGITS = 42

# Optimization test-function, extrapolation, and S weights.  For p=3 the best
# four-function simultaneous operator in the sweep uses x^(p+1), the first two
# mixed exponential modes, and the exponential-product mode.
opt_func_weights = [1, 1, 1, 4]
opt_extrap_weights = [1.0, 0.1]
opt_S_weights = [1.0, 0.1]

if not isinstance(MODEL_EXPONENT_k, int) or isinstance(MODEL_EXPONENT_k, bool):
    raise ValueError("MODEL_EXPONENT_k must be an integer")


def legendre_basis(num_functions: int) -> JuliaBasis:
    """Create a polynomial basis whose Julia callables share one cache block."""
    if num_functions < 1:
        raise ValueError("num_functions must be positive")

    return JuliaBasis(
        labels=[f"P_{degree}(x)" for degree in range(num_functions)],
        factory=legendre_basis_factory(num_functions),
    )


def polynomial_bases(degree: int, operator_type: str) -> tuple[JuliaBasis, JuliaBasis]:
    """Create standard polynomial bases for an LGL- or LG-type operator."""
    if degree < 1:
        raise ValueError("degree must be at least 1")

    op_basis = legendre_basis(degree + 1)

    # LGL degree-r quadrature is exact through degree 2r-1.  LG uses two
    # additional moments so that both rules have r+1 nodes.
    if operator_type == "closed":
        quad_basis = legendre_basis(2 * degree)
    elif operator_type == "open":
        quad_basis = legendre_basis(2 * degree + 2)
    else:
        raise ValueError("op_type must be 'open' or 'closed'")
    return op_basis, quad_basis


def exponential_bases(
    p: int,
    exponent: float,
    exponent_divisor: int = 1,
) -> tuple[JuliaBasis, JuliaBasis]:
    """Create degree-p polynomial bases augmented by a scaled exponential."""
    if p < 0:
        raise ValueError("p must be nonnegative")
    if not np.isfinite(exponent):
        raise ValueError("exponent must be finite")
    if exponent == 0.0:
        # exp(0x) duplicates the constant polynomial and makes both bases
        # linearly dependent.
        raise ValueError("exponent must be nonzero")
    if (
        not isinstance(exponent_divisor, int)
        or isinstance(exponent_divisor, bool)
        or exponent_divisor < 1
    ):
        raise ValueError("exponent_divisor must be a positive integer")

    exponent_text = repr(exponent)
    scaled_exponent_text = exponent_text
    if exponent_divisor != 1:
        scaled_exponent_text = f"({exponent_text}/{exponent_divisor})"

    # Parse the unscaled decimal directly as T, then divide in Julia.  In
    # particular, a non-dyadic quotient never passes through Python Float64
    # before Julia evaluates it at the active BigFloat precision.
    julia_exponent = f'(parse(T, "{exponent_text}") / {exponent_divisor})'

    exp_function = f"let a = {julia_exponent}; x -> exp(a * x); end"
    exp_derivative = f"let a = {julia_exponent}; x -> a * exp(a * x); end"
    op_basis = JuliaBasis(
        labels=[
            *[f"P_{degree}(x)" for degree in range(p + 1)],
            f"exp({scaled_exponent_text}x)",
        ],
        factory=legendre_basis_factory(
            p + 1,
            additional_functions=[exp_function],
            additional_derivatives=[exp_derivative],
        ),
    )

    # Start with P_0, ..., P_(2p-1), so the polynomial part is exact through
    # degree 2p-1.  The other moments come from derivatives of
    # polynomial-exponential and exponential-exponential products.
    num_quad_polynomials = 2 * p
    num_quad_functions = num_quad_polynomials + (p + 1) + 1
    if num_quad_functions % 2:
        # Odd basis lengths produce Radau rules, so add P_(2p) to retain an
        # LG/LGL pair selected solely through the principal representation.
        num_quad_polynomials += 1

    quad_labels = [
        f"P_{degree}(x)" for degree in range(num_quad_polynomials)
    ]
    additional_functions: list[str] = []
    additional_derivatives: list[str] = []

    for degree in range(p + 1):
        power = "one(x)" if degree == 0 else f"x^{degree}"
        quad_labels.append(f"x^{degree} exp({scaled_exponent_text}x)")
        additional_functions.append(
            f"let a = {julia_exponent}; x -> {power} * exp(a * x); end"
        )
        if degree == 0:
            derivative = f"a * exp(a * x)"
        else:
            derivative = f"({degree} * x^{degree - 1} + a * x^{degree}) * exp(a * x)"
        additional_derivatives.append(
            f"let a = {julia_exponent}; x -> {derivative}; end"
        )

    quad_labels.append(f"exp(2 * {scaled_exponent_text}x)")
    additional_functions.append(
        f"let a = {julia_exponent}; x -> exp(2 * a * x); end"
    )
    additional_derivatives.append(
        f"let a = {julia_exponent}; x -> 2 * a * exp(2 * a * x); end"
    )
    quad_basis = JuliaBasis(
        labels=quad_labels,
        factory=legendre_basis_factory(
            num_quad_polynomials,
            additional_functions=additional_functions,
            additional_derivatives=additional_derivatives,
        ),
    )
    return op_basis, quad_basis


def build_exponential_operator(
    p: int,
    exponent: float,
    *,
    exponent_divisor: int = 1,
    optimize: bool,
    opt_method: str = "simultaneous",
) -> Operator:
    """Build one enriched operator with exponent/exponent_divisor in Julia."""
    if opt_method not in {"simultaneous", "sequential"}:
        raise ValueError("opt_method must be 'simultaneous' or 'sequential'")

    op_basis, quad_basis = exponential_bases(p, exponent, exponent_divisor)
    principal = "upper" if op_type == "closed" else "lower"
    julia_exponent = f'(parse(T, "{repr(exponent)}") / {exponent_divisor})'

    # The optimization targets should use the same scaled exponent as the
    # approximation and quadrature bases on this mesh.
    opt_funcs = (
        f"[x -> x^{p + 1}, "
        f"let a = {julia_exponent}; x -> x * exp(a * x); end, "
        f"let a = {julia_exponent}; x -> x^2 * exp(a * x); end, "
        f"let a = {julia_exponent}; x -> exp(2 * a * x); end]"
    )
    opt_derivs = (
        f"[x -> {p + 1} * x^{p}, "
        f"let a = {julia_exponent}; "
        f"x -> (1 + a * x) * exp(a * x); end, "
        f"let a = {julia_exponent}; "
        f"x -> (2 * x + a * x^2) * exp(a * x); end, "
        f"let a = {julia_exponent}; x -> 2 * a * exp(2 * a * x); end]"
    )
    if VERBOSE:
        construction = opt_method if optimize else "min-norm"
        print(
            f"\nBuilding {construction} exponential operator ({op_type})",
            flush=True,
        )
    return build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=REF_DOMAIN,
        precision="bigfloat",
        digits=PRECISION_DIGITS,
        orthogonalize=True,
        principal=principal,
        quad_kwargs={"verbose": QUAD_VERBOSE},
        #quad_kwargs={"intermediate_tolerance": "strict"},
        use_optimization=optimize,
        opt_method=opt_method,
        verbose=VERBOSE,
        print_operator=VERBOSE,
        print_num_digits=16,
        test_functions=opt_funcs,
        test_derivatives=opt_derivs,
        test_weights=opt_func_weights,
        extrapolation_objective_weights=opt_extrap_weights,
        S_objective_weights=opt_S_weights,
    )


def build_runs() -> list[dict[str, object]]:
    """Build the polynomial and enriched comparison operators in Julia."""
    if op_type not in {"open", "closed"}:
        raise ValueError("op_type must be 'open' or 'closed'")

    principal = "upper" if op_type == "closed" else "lower"
    # Include the profile mesh so its operator can be reused if it is also one
    # of the convergence meshes.
    mesh_counts = list(dict.fromkeys([*ELEMENT_COUNTS, COARSE_ELEMENTS]))
    #rule_name = "LGL" if op_type == "closed" else "LG"
    runs: list[dict[str, object]] = []

    # Keep the existing same-size polynomial comparison and add the requested
    # lower-order P-1 polynomial comparison.
    for polynomial_degree in (P, P + 1):
        poly_op_basis, poly_quad_basis = polynomial_bases(
            polynomial_degree,
            op_type,
        )
        polynomial_operator = build_operator_from_julia(
            poly_op_basis,
            poly_quad_basis,
            interval=REF_DOMAIN,
            precision="bigfloat",
            digits=PRECISION_DIGITS,
            orthogonalize=True,
            principal=principal,
        )
        runs.append(
            {
                "label": f"$p={polynomial_degree}$",
                "operators": {
                    num_elements: polynomial_operator
                    for num_elements in mesh_counts
                },
            }
        )

    if op_type == "closed":
        # Both optimization paths produce the same closed operator, so only
        # construct and plot the default simultaneous path.
        constructions = (
            (False, "simultaneous", "min-norm"),
            (True, "simultaneous", "optimized"),
        )
    else:
        constructions = (
            (False, "simultaneous", "min-norm"),
            (True, "simultaneous", "simultaneous"),
            (True, "sequential", "sequential"),
        )
    for optimize, opt_method, construction in constructions:
        if SCALE_BASIS_EXPONENT:
            # Each uniform mesh gets an operator whose reference exponent is
            # BASIS_EXPONENT / num_elements.
            operators = {
                num_elements: build_exponential_operator(
                    P,
                    BASIS_EXPONENT,
                    exponent_divisor=num_elements,
                    optimize=optimize,
                    opt_method=opt_method,
                )
                for num_elements in mesh_counts
            }
        else:
            operator = build_exponential_operator(
                P,
                BASIS_EXPONENT,
                optimize=optimize,
                opt_method=opt_method,
            )
            operators = {num_elements: operator for num_elements in mesh_counts}

        runs.append(
            {
                "label": (
                    f"$p={P}$ + exp, {construction}"
                ),
                "operators": operators,
            }
        )

    return runs


def u_exact(x: np.ndarray | float) -> np.ndarray:
    x_array = np.asarray(x, dtype=float)
    exponent = (
        MODEL_EXPONENT_a * np.square(x_array)
        + MODEL_EXPONENT_b
        * np.sin(MODEL_EXPONENT_k * np.pi * x_array)
    )
    return np.exp(exponent)


def reaction_coefficient(x: np.ndarray | float) -> np.ndarray:
    """Return c(x) in the steady equation u_x = c(x) u."""
    x_array = np.asarray(x, dtype=float)
    return (
        2.0 * MODEL_EXPONENT_a * x_array
        + MODEL_EXPONENT_b
        * MODEL_EXPONENT_k
        * np.pi
        * np.cos(MODEL_EXPONENT_k * np.pi * x_array)
    )


def solve_on_mesh(
    operator: Operator,
    num_elements: int,
) -> tuple[list[Element1D], np.ndarray]:
    elements = make_uniform_elements(
        domain=DOMAIN,
        num_elements=num_elements,
        operators=operator,
        a_fun=lambda x: np.ones_like(x, dtype=float),
        b_fun=lambda x: np.ones_like(x, dtype=float),
        f_fun=lambda x: np.zeros_like(x, dtype=float),
        exact_fun=u_exact,
    )
    system = assemble_system(
        elements,
        left_bc_fun=lambda _x: 1.0,
        sat_type=SAT_TYPE,
    )

    # At steady state, u_x - c(x)u = 0.  The generic assembler supplies the
    # derivative and SAT terms; add the reaction term to its left-hand side.
    reaction = np.concatenate(
        [reaction_coefficient(element.x) for element in elements]
    )
    steady_matrix = system.matrix - sp.diags(reaction, format="csc")
    u, _ = solve_steady(steady_matrix, system.rhs, on_singular="nan")
    return system.elements, u


def operator_for_mesh(run: dict[str, object], num_elements: int) -> Operator:
    """Return the operator assigned to one mesh in a convergence run."""
    operators = run.get("operators")
    if not isinstance(operators, Mapping):
        raise TypeError("run operators must be a mapping keyed by mesh size")

    operator = operators.get(num_elements)
    if not isinstance(operator, Operator):
        raise TypeError(
            f"run has no valid Operator for {num_elements} elements"
        )
    return operator


def run_convergence(
    run: dict[str, object],
) -> tuple[np.ndarray, np.ndarray]:
    errors: list[float] = []
    dofs: list[int] = []
    hs: list[float] = []

    print(f"\nRun: {run['label']}, SAT: {SAT_TYPE}")
    print("num_elements  total_dofs  H_error        rate")
    for num_elements in ELEMENT_COUNTS:
        operator = operator_for_mesh(run, num_elements)
        if not check_sbp_property(operator, print_report=True):
            warnings.warn(
                f"SBP check failed for {run['label']} on the "
                f"{num_elements}-element mesh",
                RuntimeWarning,
            )
        if not check_nullspace_consistency(operator, print_report=True):
            warnings.warn(
                f"Operator D is not nullspace consistent for {run['label']} on "
                f"the {num_elements}-element mesh",
                RuntimeWarning,
            )
        elements, u = solve_on_mesh(operator, num_elements)
        errors.append(global_H_error(elements, u, u_exact))
        dofs.append(sum(element.x.size for element in elements))
        hs.append((DOMAIN[1] - DOMAIN[0]) / num_elements)

    rates = convergence_rate(np.asarray(errors), np.asarray(hs))
    for num_elements, num_dofs, error, rate in zip(
        ELEMENT_COUNTS, dofs, errors, rates
    ):
        rate_text = "-" if np.isnan(rate) else f"{rate:8.4f}"
        print(f"{num_elements:12d}  {num_dofs:10d}  {error:12.4e}  {rate_text}")

    return np.asarray(dofs, dtype=float), np.asarray(errors)


def main() -> None:
    runs = build_runs()
    dof_rows: list[np.ndarray] = []
    error_rows: list[np.ndarray] = []
    profiles: list[list[tuple[np.ndarray, np.ndarray]]] = []

    for run in runs:
        dofs, errors = run_convergence(run)
        dof_rows.append(dofs)
        error_rows.append(errors)

        operator = operator_for_mesh(run, COARSE_ELEMENTS)
        coarse_elements, coarse_u = solve_on_mesh(operator, COARSE_ELEMENTS)
        profiles.append(profile_from_elements(coarse_elements, coarse_u))

    labels = [str(run["label"]) for run in runs]
    if MODEL_EXPONENT_b == 0.0:
        convergence_title = rf"$u_x=(2a x)u$, {op_type} operators"
        solution_title = (
            rf"$u=e^{{{MODEL_EXPONENT_a:g}x^2}}$ "
            f"({COARSE_ELEMENTS} elements)"
        )
    else:
        convergence_title = (
            r"$u_x=(2a x + b k\pi\cos(k\pi x))u$, "
            f"{op_type} operators"
        )
        solution_title = (
            rf"$u=e^{{{MODEL_EXPONENT_a:g}x^2 + "
            rf"{MODEL_EXPONENT_b:g}\sin({MODEL_EXPONENT_k}\pi x)}}$ "
            f"({COARSE_ELEMENTS} elements)"
        )

    plot_convergence(
        np.vstack(dof_rows),
        np.vstack(error_rows),
        labels,
        title=None, #convergence_title,
        grid=True,
        ylim=(1e-10, 1e-1),
    )

    x_exact, exact_values = exact_profile_on_domain(u_exact, domain=DOMAIN)
    plot_solution_profiles(
        profiles,
        labels,
        x_exact=x_exact,
        u_exact=exact_values,
        title=solution_title,
        grid=True,
    )

    if SHOW_PLOTS:
        plt.show()


if __name__ == "__main__":
    main()
