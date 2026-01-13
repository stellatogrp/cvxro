# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

LROPT (Learning for Robust Optimization) is a Python package for decision-making under uncertainty, built on CVXPY. It solves robust optimization problems where decision-makers must protect against uncertain parameters: `minimize f(x) subject to g(x,u) ≤ 0 for all u ∈ U(θ)`.

The key differentiator is that users can either define uncertainty sets explicitly OR pass historical data and let LROPT learn the optimal uncertainty set through training.

## Commands

All commands should be prefixed with `uv run` to use the virtual environment:

```bash
# Run all tests
uv run pytest tests

# Run a specific test file
uv run pytest tests/test_simple_opt.py

# Run a specific test
uv run pytest tests/test_simple_opt.py::test_name -v

# Lint check
uv run ruff check lropt/

# Auto-fix lint issues
uv run ruff check --fix lropt/

# Install in development mode
uv pip install -e ".[dev]"
```

## Architecture

### Core Flow

```
RobustProblem(objective, constraints)
    ↓
RemoveUncertainty (canonicalization)
    ↓
Standard CVXPY Problem
    ↓
Solver (SCS, Clarabel, etc.)
```

### Learning Flow

```
RobustProblem with uncertainty set + data
    ↓
Trainer → TorchExpressionGenerator (CVXPY→PyTorch)
    ↓
Gradient-based optimization
    ↓
Updated uncertainty set parameters
```

### Key Modules

- **`lropt/robust_problem.py`**: Main `RobustProblem` class extending CVXPY Problem
- **`lropt/uncertain_parameter.py`**: `UncertainParameter` class (CVXPY Parameter subclass with uncertainty_set)
- **`lropt/train/trainer.py`**: `Trainer` class orchestrating uncertainty set learning
- **`lropt/train/settings.py`**: `TrainerSettings` with training hyperparameters
- **`lropt/uncertainty_sets/`**: Uncertainty set implementations (Ellipsoidal, Box, Polyhedral, Budget, MRO, Scenario, Norm)
- **`lropt/uncertain_canon/`**: Canonicalization/reformulation of uncertain constraints

### Public API (from `__init__.py`)

Main classes: `RobustProblem`, `UncertainParameter`, `Trainer`, `TrainerSettings`, `ContextParameter`, `Parameter`

Uncertainty sets: `Ellipsoidal`, `Box`, `Polyhedral`, `Budget`, `MRO`, `Scenario`, `Norm`

Predictors: `LinearPredictor`, `NNPredictor`, `CovPredictor`, `DeepNormalModel`

Utilities: `max_of_uncertain`, `sum_of_max_of_uncertain`, `Simulator`

## Code Style

From CONVENTIONS.md:
- Keep functions small and focused on one task
- Use meaningful names that reveal intent
- Prefer fewer arguments (2-3 max)
- Code should be self-explanatory; avoid comments unless necessary
- Use exceptions rather than error codes

Ruff config: line-length 100, Python 3.10+ target, rules E/F/I enabled.

## Dependencies

Core: cvxpy, torch, scipy, scikit-learn, pandas, diffcp, cvxpylayers, cvxtorch

Dev: pytest, ruff, sphinx, jupyterlab, marimo
