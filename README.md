# Paper-GaussFSBP

This repository contains the reproduction code for the paper

> **Construction and Optimization of Summation-by-Parts Operators for General
> Function Spaces Using an Improved Generalized Gaussian Quadrature Algorithm**
>
> Alex Bercik, Lisa Patrascu, and David Zingg, 2026.

ArXiv link: https://arxiv.org/abs/2607.08934.

The scripts reproduce the one-dimensional SBP-SAT experiments used to compare
polynomial operators with operators built for enriched function spaces, including
exponential and endpoint-singular bases. The Julia-backed operator construction
is provided by `GaussFSBP` and `GeneralizedGauss`; this repository supplies the
Python assembly, solve, plotting, and driver code used for the paper figures and
tables.

## Repository Layout

```text
Paper-GaussFSBP/
  driver/                 paper reproduction scripts
  lib/julia/              Julia project used by the Python-Julia bridge
  lib/julia_operators.py  bridge from Python to GaussFSBP
  src/                    1D SBP-SAT assembly, solve, norms, and plotting
  tests/                  small Python smoke-test suite
```

`driver/operator_cache.json` stores precomputed operators used by some driver
scripts. If an entry is missing, the drivers rebuild it through Julia.

## Dependencies

Python dependencies are listed in `pyproject.toml`.

- Python 3.10 or newer
- `numpy`
- `scipy`
- `matplotlib`
- `pytest`, only for the small smoke-test suite
- `juliacall`, for Julia-backed operator construction

Julia dependencies are listed in `lib/julia/Project.toml` and pinned by
`lib/julia/Manifest.toml`. The Julia project expects:

- Julia 1.12 or newer
- `GaussFSBP`
- `GeneralizedGauss`, available at `GaussFSBP/lib/GeneralizedGauss.jl`
- the other Julia packages declared in `lib/julia/Project.toml`

Set up the Python environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,julia]"
```

Set up the local Julia dependency pointer:

```bash
ln -s /path/to/GaussFSBP lib/GaussFSBP  # only if lib/GaussFSBP is missing
julia --project=lib/julia -e 'import Pkg; Pkg.instantiate()'
julia --project=lib/julia -e 'using GaussFSBP, GeneralizedGauss'
```

If Python should embed a specific Julia executable, set
`PYTHON_JULIACALL_EXE=/path/to/julia` before running the Python scripts.

## Running the Code

Run the smoke tests:

```bash
python -m pytest -q
```

Run the pure-Python smooth sanity check:

```bash
python driver/smooth_sanity_check.py
```

Run the Julia-backed operator construction example:

```bash
python driver/build_exponential_operator_example.py
```

Run the main paper drivers:

```bash
python driver/Exponential_Problem.py
python driver/Mixed_Exponential_Problem.py
python driver/Mixed_Endpoint_Singularity.py
```


## Discrete System

The code solves steady one-dimensional variable-coefficient advection problems
using element-local SBP differentiation and SAT coupling. The semidiscrete form
is

```text
u_t = f - A D(Bu) + SAT(u),
```

where `A` and `B` are diagonal coefficient matrices evaluated at element nodes.
For steady problems the code assembles the sparse linear system

```text
A D(Bu) - SAT_linear(u) = f + SAT_known
```

and solves it with SciPy sparse linear algebra. Neighboring elements keep
separate nodal states and are coupled weakly through SAT fluxes, so open,
closed, and mixed reference operators can be used in the same mesh.
