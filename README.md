# Paper-GaussFSBP

`gaussfsbp` is an initial 1D generalized SBP-SAT solver for forced linear advection.

## Model and sign convention

The semidiscrete equation is

u_t + A D(Bu) = f + SAT(u),

so

u_t = f - A D(Bu) + SAT(u).

The steady system is assembled as

A D(Bu) - SAT_linear(u) = f + SAT_known,

and solved as a sparse linear system `L u = rhs`.

`calc_RHS(elements, u, ...)` evaluates the semidiscrete update directly. If no
left inflow function is passed, it uses the exact value stored on the leftmost
element; pass `left_bc_fun=lambda _x: 0.0` for a homogeneous inflow.
`calc_LHS(elements, ...)` builds the homogeneous steady matrix `L`, so the
homogeneous Jacobian satisfies `d(calc_RHS)/du = -L`.

The strong-form hook is always `A D(Bu)`, with
- conservative form: `a(x)=1`, variable `b(x)`
- non-conservative form: variable `a(x)`, `b(x)=1`
- general form: arbitrary positive `a(x)`, `b(x)`.

No split forms are used.

## Operators and reference element

Operators live on reference element `[-1, 1]` and use dictionary fields:

- `basis`, `quad_basis`
- `op_type` in `{open, closed, half-open-left, half-open-right}`
- `nodes`, `D`, `H`, `tL`, `tR`, `selector`

Reference operators live in `src/operator_library.py`. Use `OperatorSpec` or
`get_operator(...)` to choose entries by `(basis, quad_basis, op_type,
selector)`; ``basis`` and ``quad_basis`` are matched up to permutation.
``selector`` defaults to ``0``.

Affine element scaling for `[x_L, x_R]` with `h = x_R - x_L`:
- `D = (2/h) D_ref`
- `H = (h/2) H_ref`.

`H` is diagonal and stored as a vector.

## Elements and SAT coupling

Build elements directly from physical element bounds and one reference operator
choice, or a list with one choice per element:

```python
operator = OperatorSpec(["1", "x", "x^2"], ["1", "x", "x^2", "x^3"], "closed")
elements = make_uniform_elements((0.0, 1.0), 8, operator, a_fun, b_fun, f_fun)
```

For mixed reference operators, pass `operators=[op0, op1, ...]` to
`make_elements(bounds, operators, ...)`.

Each element owns local nodal state values. Interface nodal values are duplicated across neighboring elements and coupled weakly through SATs.

For element `j`, the SAT added to `du/dt` is

```
SAT_j = A_j H_j^{-1} [
    tL_j (f*_{j-1/2} - tL_j^T B_j u_j)
    - tR_j (f*_{j+1/2} - tR_j^T B_j u_j)
]
```

At an interior interface, the left and right flux states are
`tR_j^T B_j u_j` and `tL_{j+1}^T B_{j+1} u_{j+1}`. The symmetric flux averages
these states; the positive-speed upwind/Rusanov flux uses the left state.

At the physical left boundary, positive-speed inflow is imposed with the upwind
flux `f* = b(x_L) u_bc(x_L)`, giving
`SAT^L = A H^{-1} tL (b(x_L) u_bc(x_L) - tL^T B u)`. At the physical right
boundary, `f* = tR^T B u`, so the outflow SAT is zero.

All boundary and interface traces are computed using `tL` and `tR`. No hard-coded endpoint indexing is used.

This gives one code path for open/closed/half-open operators.

## Solver and sparse format

The initial solver assembles dense local element blocks and global sparse matrix with `scipy.sparse.bmat`, then converts to CSC and solves with `scipy.sparse.linalg.splu`.

BSR is not the default because local enrichment can produce different element node counts.

## Layout

Source modules live under the `src` package. Import from the repo root, e.g.
`from src.assembly import assemble_system, calc_LHS, calc_RHS`.

```
Paper-GaussFSBP/
  pyproject.toml
  src/
    __init__.py
    operator_library.py
    assembly.py
    elements.py
    norms.py
    operators.py
    problems.py
    sats.py
    solve.py
  examples/smooth_sanity_check.py
  tests/
```

## Quick start

Use Python 3.10 or newer. From the repository root, create a virtual environment,
install the project in editable mode (this reads `pyproject.toml` and installs
`numpy` and `scipy`), then run tests or examples:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
pytest -q
python examples/smooth_sanity_check.py
```
