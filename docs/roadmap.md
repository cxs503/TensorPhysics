# Roadmap

## M0 — Repository foundation
- [x] Package metadata and project boundaries.
- [x] Solver protocol and deterministic lifecycle scheduler.
- [x] Structured in-memory diagnostics.
- [x] Minimal executable example and scheduler tests.
- [ ] Run tests in CI and record supported Python versions.

## M1 — Reproducibility and operations
- [ ] TOML/YAML configuration loading and schema validation.
- [x] Versioned checkpoint envelope with atomic local JSON writes and reader validation.
- [ ] Full simulation resume API with solver restore/compatibility contracts.
- [ ] Logging sinks and performance timers.
- [x] CI matrix and automated tests configured; confirm actual CI results.
- [x] Edge cases: zero end time, max-step truncation, basic run argument validation.
- [ ] Cleanup-error behavior, retry policy, linting, typing, coverage, package build.

## M2 — Coupling foundations
- [x] Field descriptors with units, location, shape, dtype, and producer ownership.
- [x] In-memory field publication/read API with timestamps and freshness checks.
- [x] Explicit PRE_STEP, AFTER_SOLVER, and POST_STEP callbacks.
- [x] Deterministic two-reservoir toy case with total-mass conservation assertion.
- [x] Time interpolation contract: bracketed linear interpolation, no extrapolation, optional maximum bracket gap, and tests.
- [ ] Solver subcycling contract.
- [ ] Define and test rollback/partial-update policy.

## M3 — Ecosystem adapters
- [ ] Audit current TensorLBM, TensorFEM, TensorDEM, and TensorFVM APIs.
- [ ] Implement adapters against real, versioned solver APIs.
- [ ] Keep each solver independently installable.
- [ ] Cross-repository integration tests with optional dependencies.

## M4 — Engineering validation
- [ ] Reference cases and quantitative tolerances.
- [ ] Reproducible benchmark inputs, metrics, and result manifests.
- [ ] Compatibility matrix and release policy.
