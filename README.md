# TensorPhysics

TensorPhysics is the HPC + AI scientific-computing and engineering workflow platform for the Tensor ecosystem.

**System-wide entry point:** [Tensor Ecosystem System Overview](docs/SYSTEM_OVERVIEW.md) — architecture, repository boundaries, shared contracts, end-to-end lifecycle, icebreaker application roadmap, and evidence/maturity rules.

Its goal is to let authorized agents drive the complete lifecycle—from engineering problem definition and solver selection, through reproducible HPC execution and data production, to dataset creation, AI/ROM training and evaluation, and delivery of engineering applications. TensorLBM, TensorFEM, TensorDEM, and TensorFVM remain independent specialist solvers; TensorPhysics provides the contracts, execution control, provenance, and validation workflow around them.

## Current implementation

The main branch contains:
- A deterministic solver lifecycle scheduler with explicit physical-completion and step-limit states.
- Backend-neutral field descriptors and an in-memory field store with producer ownership, shape checks, timestamps, freshness limits, bounded history, and bracketed linear time interpolation (no extrapolation).
- Explicit PRE_STEP, AFTER_SOLVER, and POST_STEP coupling phases with stable callback ordering.
- A two-reservoir coupling test that checks total-mass conservation and same-step field delivery.
- A versioned JSON checkpoint envelope with atomic local-file writes and schema/clock validation.
- Initial RunManifest and ArtifactRef contracts with canonical config hashing, code-revision provenance, artifact checksums, resource/environment metadata, and explicit validation status.
- An optional TensorDEM 2-D adapter with integer-ratio subcycling, per-substep load callbacks, and strict clock checks.
- Structured diagnostics, a minimal runnable example, and a Python 3.10–3.12 GitHub Actions test matrix.

The coupling example is deliberately a contract test, not a validated CFD/DEM/FEM physical model. It demonstrates data flow and a conservation invariant only. The field store supports temporal linear interpolation for data types that implement arithmetic, but does not perform unit conversion or conservative mesh/particle mapping. Its history is bounded to two distinct-time samples by default; set `history_limit` to retain more samples for subcycling.

Checkpoint files persist orchestration metadata and JSON-compatible solver snapshots. They do not yet provide full restart/resume: each specialist solver still needs a defined restore API and compatibility checks.

## Platform architecture

The long-term architecture has seven cooperating planes: agent/application tools; workflow and experiment control; HPC execution; specialist solvers and multiphysics; scientific data and provenance; AI model lifecycle; and verification/governance/observability. Agents operate through typed, permissioned tools and durable job/artifact IDs—not unrestricted shell access. The core remains backend-neutral and does not require one scheduler, storage provider, or ML framework.

The complete platform architecture, development phases, and acceptance gates are described in [docs/hpc-ai-platform-architecture.md](docs/hpc-ai-platform-architecture.md).

## Design principles

- **Solver independence:** TensorLBM, TensorFEM, TensorDEM, and TensorFVM remain independently usable.
- **Thin orchestration:** TensorPhysics coordinates solvers and data exchange; it does not reimplement numerical kernels.
- **Explicit contracts:** units, time advancement, field exchange, errors, and checkpoint behavior are documented.
- **Testable increments:** features include automated tests or a runnable example.
- **Optional dependencies:** the orchestration core does not require PyTorch unless a selected backend needs it.

## Next milestones

1. Extend and validate RunManifest/ArtifactRef into versioned workflow and solver-capability contracts.
2. Establish local end-to-end execution with structured logs, checksums, validation reports, and durable run IDs.
3. Finish the first real solver adapters and quantitative engineering benchmarks.
4. Add an execution-backend interface (local first, HPC scheduler later) with resource limits and cancellation.
5. Build scientific artifact lineage and dataset generation before adding training/model-registry workflows.
6. Expose validated capabilities to agents with permissions, approval gates, quotas, and complete audit trails.

See [docs/hpc-ai-platform-architecture.md](docs/hpc-ai-platform-architecture.md), [docs/architecture.md](docs/architecture.md), [docs/adapter-contracts.md](docs/adapter-contracts.md), [docs/roadmap.md](docs/roadmap.md), and [docs/development.md](docs/development.md).

## Tensor ecosystem integration

This repository's role, interface boundaries, data-exchange requirements and recommended integration path are documented in [Ecosystem Integration](docs/ecosystem-integration.md).
