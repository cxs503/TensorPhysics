# Development guide

## Install and test

Use Python 3.10 or newer.

    python -m pip install -e ".[test]"
    python -m pytest
    python examples/minimal_run.py

## Adding a solver

Implement a stable unique name and the four lifecycle methods in the Solver protocol. Keep solver-specific numerical state and kernels in the specialist project. Add unit tests for lifecycle behavior and a runnable example before proposing an adapter.

## Branch and commit policy

The initial project is developed on main as requested by the maintainer. Keep commits focused and do not report tests as passing unless they were actually executed. The GitHub Actions workflow runs the test suite and minimal example on pushes and pull requests to main.

## Compatibility

The core has no runtime dependencies. PyTorch remains an optional extra because not every orchestration task requires tensor operations. Backend-specific adapters may declare their own optional dependencies.
