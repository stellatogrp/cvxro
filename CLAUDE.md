# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CVXRO (Convex Optimization under Robust Optimization) is a Python package for decision-making under uncertainty, built on CVXPY. It solves robust optimization problems of the form:

```
minimize f(x) subject to g(x,u) ≤ 0 for all u ∈ U(θ)
```

**Dual mode operation**: Users can either define uncertainty sets explicitly OR pass historical data and let CVXRO learn the optimal uncertainty set parameters through gradient-based training.

## Commands

All commands use `uv run` to ensure the virtual environment is active:

```bash
# Run all tests
uv run pytest tests

# Run specific test file or test
uv run pytest tests/test_simple_opt.py
uv run pytest tests/test_simple_opt.py::test_name -v

# Lint check / auto-fix
uv run ruff check cvxro/
uv run ruff check --fix cvxro/

# Install in development mode
uv pip install -e ".[dev]"
```

## Architecture

### Solution Pipeline

```
RobustProblem(objective, constraints)
    ↓ [UncertainCanonicalization]
Canonicalized Problem (separates certain/uncertain parts)
    ↓ [RemoveUncertainty - applies duality]
Standard CVXPY Problem (no uncertain parameters)
    ↓
Solver (Clarabel, SCS, etc.)
```

The key mechanism is **duality reformulation**: for each uncertain constraint `g(x,u) ≤ 0`, dual variables are introduced and the uncertainty set's conjugate function transforms it into a tractable deterministic constraint.

### Learning Pipeline

When data is provided, CVXRO learns optimal uncertainty set parameters:

```
RobustProblem + UncertaintySet.data
    ↓
Trainer.train()
    ├─ Canonicalize to standard CVXPY
    ├─ TorchExpressionGenerator (CVXPY → PyTorch)
    ├─ CVXPyLayer (differentiable optimization)
    └─ Gradient descent on A, b, rho parameters
    ↓
Updated uncertainty set (shape A, center b, size rho)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `robust_problem.py` | `RobustProblem` class extending CVXPY Problem |
| `uncertain_parameter.py` | `UncertainParameter` (CVXPY Parameter + uncertainty_set) |
| `train/trainer.py` | `Trainer` orchestrating learning with CVXPyLayers |
| `train/settings.py` | `TrainerSettings` (45+ hyperparameters) |
| `uncertainty_sets/` | Set implementations with `conjugate()` methods |
| `uncertain_canon/` | Duality-based reformulation reductions |
| `torch_expression_generator.py` | CVXPY → PyTorch expression conversion |

### Uncertainty Set Hierarchy

```
UncertaintySet (base)
├─ Norm (p-norm family)
│  ├─ Ellipsoidal (‖z‖₂ ≤ ρ)
│  └─ Box (‖z‖∞ ≤ ρ)
├─ Polyhedral (Cz ≤ d)
├─ Budget (dual norm constraints)
├─ MRO (Wasserstein/data-driven)
└─ Scenario (satisfy all scenarios)
```

Each set defines: `rho` (size), `a` (shape matrix), `b` (center), and optional support constraints (`c`, `d`, `ub`, `lb`).

### Public API

**Core**: `RobustProblem`, `UncertainParameter`, `Trainer`, `TrainerSettings`, `ContextParameter`, `Parameter`

**Uncertainty sets**: `Ellipsoidal`, `Box`, `Norm`, `Polyhedral`, `Budget`, `MRO`, `Scenario`

**Predictors** (contextual learning): `LinearPredictor`, `NNPredictor`, `CovPredictor`, `DeepNormalModel`

**Utilities**: `max_of_uncertain`, `sum_of_max_of_uncertain`, `Simulator`

## Typical Usage Patterns

**Explicit uncertainty set:**
```python
u = cvxro.UncertainParameter(dim, uncertainty_set=cvxro.Ellipsoidal(rho=2.0))
prob = cvxro.RobustProblem(objective, constraints)
prob.solve()
```

**Learning from data:**
```python
u = cvxro.UncertainParameter(dim, uncertainty_set=cvxro.Ellipsoidal())
u.uncertainty_set.data = historical_data  # triggers learning
prob = cvxro.RobustProblem(objective, constraints)
prob.solve()  # auto-trains via Trainer
```

**Contextual learning** (uncertainty depends on state):
```python
x = cvxro.ContextParameter(data=X_train)
settings = cvxro.TrainerSettings(contextual=True, predictor=cvxro.NNPredictor())
```

## Code Style

From CONVENTIONS.md:
- Keep functions small and focused on one task
- Use meaningful names that reveal intent
- Prefer fewer arguments (2-3 max)
- Code should be self-explanatory; avoid comments unless necessary
- Use exceptions rather than error codes

Ruff config: line-length 100, Python 3.12+ target, rules E/F/I enabled.

## Dependencies

Core: cvxpy, torch, scipy, scikit-learn, pandas, diffcp, cvxpylayers, cvxtorch, tqdm

Dev: pytest, ruff, sphinx, jupyterlab, marimo, pyscipopt, hydra-core
