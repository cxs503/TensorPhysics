# TensorPhysics

TensorPhysics is the shared multiphysics simulation framework for the Tensor scientific-computing ecosystem.

Its purpose is to coordinate specialized solvers such as TensorLBM, TensorFEM, TensorDEM, and TensorFVM without replacing or duplicating their numerical kernels.

## Current implementation

The main branch contains:
- A deterministic solver lifecycle scheduler with explicit physical-completion and step-limit states.
- Backend-neutral field descriptors and an in-memory field store with producer ownership, shape checks, timestamps, freshness limits, bounded history, and bracketed linear time interpolation (no extrapolation).
- Explicit PRE_STEP, AFTER_SOLVER, and POST_STEP coupling phases with stable callback ordering.
- A two-reservoir coupling test that checks total-mass conservation and same-step field delivery.
- A versioned JSON checkpoint envelope with atomic local-file writes and schema/clock validation.
- Structured diagnostics, a minimal runnable example, and a Python 3.10–3.12 GitHub Actions test matrix.

The coupling example is deliberately a contract test, not a validated CFD/DEM/FEM physical model. It demonstrates data flow and a conservation invariant only. The field store supports temporal linear interpolation for data types that implement arithmetic, but does not perform unit conversion or conservative mesh/particle mapping. Its history is bounded to two distinct-time samples by default; set `history_limit` to retain more samples for subcycling.

Checkpoint files persist orchestration metadata and JSON-compatible solver snapshots. They do not yet provide full restart/resume: each specialist solver still needs a defined restore API and compatibility checks.

## Design principles

- **Solver independence:** TensorLBM, TensorFEM, TensorDEM, and TensorFVM remain independently usable.
- **Thin orchestration:** TensorPhysics coordinates solvers and data exchange; it does not reimplement numerical kernels.
- **Explicit contracts:** units, time advancement, field exchange, errors, and checkpoint behavior are documented.
- **Testable increments:** features include automated tests or a runnable example.
- **Optional dependencies:** the orchestration core does not require PyTorch unless a selected backend needs it.

## Next milestones

1. Run and verify the complete Python 3.10–3.12 CI matrix.
2. Define subcycling and time interpolation contracts, including rollback/partial-update semantics.
3. Add solver restore contracts and end-to-end checkpoint/restart tests.
4. Audit current TensorLBM, TensorFEM, TensorDEM, and TensorFVM APIs before writing adapters.
5. Add engineering reference cases, quantitative tolerances, and reproducible benchmark manifests.

See [docs/architecture.md](docs/architecture.md), [docs/roadmap.md](docs/roadmap.md), and [docs/development.md](docs/development.md).
