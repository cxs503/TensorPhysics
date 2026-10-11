# Roadmap

## M0 — Repository foundation
- [x] Package metadata and project boundaries.
- [x] Solver protocol and deterministic lifecycle scheduler.
- [x] Structured in-memory diagnostics.
- [x] Minimal executable example and scheduler tests.
- [ ] Run tests in CI and record supported Python versions.

## M1 — Reproducibility and operations
- [ ] TOML/YAML configuration loading and schema validation.
- [ ] Checkpoint reader/writer contract and atomic local-file implementation.
- [ ] Logging sinks and performance timers.
- [ ] CI matrix, linting, typing, coverage, package build.
- [ ] Edge cases: zero end time, max-step truncation, cleanup errors, retries.

## M2 — Coupling foundations
- [ ] Field descriptors with units, location, shape, dtype, and ownership.
- [ ] Data-exchange protocol and explicit coupling phases.
- [ ] Deterministic coupled toy problem with conservation assertions.
- [ ] Subcycling and time interpolation contracts.

## M3 — Ecosystem adapters
- [ ] Audit current TensorLBM, TensorFEM, TensorDEM, and TensorFVM APIs.
- [ ] Implement adapters against real, versioned solver APIs.
- [ ] Keep each solver independently installable.
- [ ] Cross-repository integration tests with optional dependencies.

## M4 — Engineering validation
- [ ] Reference cases and quantitative tolerances.
- [ ] Reproducible benchmark inputs, metrics, and result manifests.
- [ ] Compatibility matrix and release policy.
