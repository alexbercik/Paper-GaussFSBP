from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import JuliaBasis, build_operator_from_julia, check_sbp_property
from src.elements import make_uniform_elements


def exponential_bases(epsilon: float) -> tuple[JuliaBasis, JuliaBasis]:
    """Build FSBP bases containing exp(x) / epsilon."""

    # Keep epsilon in Python, then insert its decimal text into trusted Julia
    # callable strings. BigFloat(...) keeps the constant type-friendly when the
    # Julia operator is built in BigFloat precision.
    eps = f'BigFloat("{repr(epsilon)}")'
    exp_eps = f"x -> exp(x) / {eps}"

    op_basis = JuliaBasis(
        labels=["1", "x", "exp(x)/epsilon"],
        functions=[
            "x -> one(x)",
            "x -> x",
            exp_eps,
        ],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            exp_eps,
        ],
    )

    quad_basis = JuliaBasis(
        labels=[
            "1",
            "x",
            "x^2",
            "exp(x)/epsilon",
            "x exp(x)/epsilon",
            "exp(2x)/epsilon^2",
        ],
        functions=[
            "x -> one(x)",
            "x -> x",
            "x -> x^2",
            exp_eps,
            f"x -> x * exp(x) / {eps}",
            f"x -> exp(2 * x) / ({eps} * {eps})",
        ],
        derivatives=[
            "x -> zero(x)",
            "x -> one(x)",
            "x -> 2 * x",
            exp_eps,
            f"x -> (one(x) + x) * exp(x) / {eps}",
            f"x -> 2 * exp(2 * x) / ({eps} * {eps})",
        ],
    )
    return op_basis, quad_basis


def main() -> None:
    epsilon = 0.1
    op_basis, quad_basis = exponential_bases(epsilon)

    operator = build_operator_from_julia(
        op_basis,
        quad_basis,
        interval=(0.0, 1.0),
        precision="bigfloat",
        digits=32,
        orthogonalize=True,
        principal="upper",
        print_operator=True,
        print_num_digits=16,
    )

    print(f"SBP check passed: {check_sbp_property(operator, tol=1e-9)}")

    elements = make_uniform_elements(
        (0.0, 1.0),
        1,
        operator,
        a_fun=lambda x: np.ones_like(x),
        b_fun=lambda x: np.ones_like(x),
    )
    print(f"built {len(elements)} element with {elements[0].x.size} nodes")


if __name__ == "__main__":
    main()
