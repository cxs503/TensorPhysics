# TensorPhysics

TensorPhysics is the shared multiphysics simulation framework for the Tensor scientific-computing ecosystem.

Its purpose is to coordinate specialized solvers such as TensorLBM, TensorFEM, and TensorDEM without replacing or duplicating their numerical kernels.

## Project status

The repository is at the initial architecture stage. The first milestone establishes project boundaries, the solver protocol, simulation lifecycle, configuration, diagnostics, checkpoint contracts, and verification strategy. Features will be implemented incrementally with executable tests and benchmark cases.

## Design principles

- **Solver independence:** TensorLBM, TensorFEM, and TensorDEM remain independently usable.
- **Thin orchestration:** TensorPhysics coordinates solvers and data exchange; it does not reimplement LBM/FEM/DEM algorithms.
- **Explicit contracts:** units, time advancement, field exchange, errors, and checkpoint behavior are documented.
- **Testable increments:** every feature includes automated tests or a runnable example.
- **Optional dependencies:** the orchestration core should not require PyTorch unless a selected backend needs it.

## Initial roadmap

1. Architecture decision records and module boundaries.
2. Solver protocol and deterministic simulation scheduler.
3. Configuration, logging, diagnostics, and checkpoint interfaces.
4. Coupling contracts and a minimal two-component demonstration.
5. Adapters for existing Tensor solvers after repository-level audits.
6. Reproducible benchmarks and CI.

See [docs/architecture.md](docs/architecture.md) and [docs/roadmap.md](docs/roadmap.md).
