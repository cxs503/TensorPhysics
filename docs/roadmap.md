# Roadmap: HPC + AI scientific-computing platform

The roadmap is ordered by platform dependencies, not by how easy a feature is to
demo. The critical path is **reproducible solver execution → traceable data artifacts
→ verified benchmarks → HPC campaigns → datasets → model lifecycle → agent-operated
applications**. Agents and AI training must not be built on untraceable numerical runs.

## P0 — Product contracts and boundaries

- [x] Establish TensorPhysics as a solver-neutral orchestration repository.
- [x] Document the HPC + AI platform planes, security boundaries, and acceptance gates.
- [x] Keep specialist numerical kernels in TensorLBM/TensorDEM/TensorFEM/TensorFVM.
- [ ] Define versioned public contracts for SolverAdapter, WorkflowSpec, ExecutionBackend,
  DatasetSpec, TrainingJob, ValidationSuite, and AgentTool.
- [x] Add initial RunManifest and ArtifactRef contracts with config digest, code revisions,
  artifact checksums, resource/environment metadata, and validation status.
- [ ] Add API/schema versioning and compatibility tests.
- [ ] Publish a capability manifest for each adapter: dimensions, steady/transient,
  native clock, supported load/field interfaces, device support, restart support,
  and known limitations.

## P1 — Reproducible single-node execution and observability

- [x] Deterministic lifecycle scheduler, coupling phases, diagnostics, checkpoint envelope.
- [x] Initial field exchange contract, bounded temporal history, no-extrapolation interpolation.
- [x] First optional TensorDEM 2-D adapter with integer-ratio subcycling.
- [ ] Run the full test matrix in CI and record verified results.
- [ ] Define solver snapshot/restore, partial-update/failure, cancellation, and cleanup
  semantics; add restart equivalence tests.
- [ ] Integrate RunManifest creation and lifecycle updates into Simulation so each run captures its ID, immutable normalized config, Git revisions, environment identity, random seeds, hardware/device details, and exit/failure reason.
- [ ] Add structured console/file logging, per-stage timings, memory/step metrics,
  and machine-readable run reports.
- [ ] Add config loading and schema validation; fail preflight before expensive work.
- [ ] Add local process execution backend with submit/status/cancel/log/artifact staging.
- [ ] Add artifact checksums and atomic metadata publication; do not serialize large
  tensors or arrays inside checkpoint JSON.
- [ ] Add package build, lint/type checks, coverage, and CI result visibility.

**Exit gate:** a user can rerun a small solver case from a versioned manifest and trace
every output back to exact inputs and code revisions. Invalid inputs fail before launch.

## P2 — Real solver adapters and verified engineering benchmarks

- [x] Audit current public interfaces of TensorLBM, TensorDEM, TensorFEM, and TensorFVM.
- [x] Implement the first TensorDEM 2-D adapter against the native stepping API.
- [ ] Validate the TensorDEM adapter with the real package, not only fake models.
- [ ] Implement a named TensorLBM adapter for one real transient case, with explicit
  state ownership and field/load hooks.
- [ ] Add dedicated 3-D DEM adapter only after the 2-D contract and tests are stable.
- [ ] Reuse TensorFEM surface exchange where its geometry, time, units, and force/moment
  assumptions match; avoid falsely presenting a case-specific receiver as universal.
- [ ] Treat TensorFVM steady solve and transient time-step execution as distinct
  capabilities; do not wrap a whole steady solve as one global time step.
- [ ] Add benchmark manifests with inputs, expected observables, tolerances, solver
  revisions, run environment, and validation status.
- [ ] Add force/moment/work transfer checks and relevant mass/momentum/energy balances.
- [ ] Separate adapter contract tests, numerical verification, and physical validation.
- [ ] Only then deliver a two-way or one-way LBM–DEM engineering reference workflow.

**Exit gate:** at least one solver run and one coupling operator have reproducible
quantitative tests. No interface test alone is labeled physical validation.

## P3 — HPC execution plane

- [ ] Define resource requirements: CPU/GPU, memory, MPI ranks, walltime, scratch,
  accelerator constraints, and queue/accounting metadata.
- [ ] Implement a backend-neutral execution protocol: submit, status, cancel, logs,
  metrics, stage-in/out, and resource accounting.
- [ ] Finish local execution first; then implement a Slurm backend without changing
  workflow or solver contracts.
- [ ] Track external scheduler job IDs and reconcile pending/running/completed/failed/
  cancelled/preempted states after controller restarts.
- [ ] Add quotas, walltime, retry limits, concurrency limits, and cancellation.
- [ ] Add environment/container digest and hardware/backend identity to every run.
- [ ] Add large-output streaming and restart/checkpoint staging.

**Exit gate:** a campaign can run locally and on a supported HPC backend with the same
versioned workflow semantics, and failed/interrupted jobs reconcile without losing
provenance.

## P4 — Scientific artifact and data-production factory

- [x] Define initial ArtifactRef and RunManifest contracts, checksums, provenance fields,
  and validation status.
- [ ] Add the metadata catalog, storage backend, and immutable cross-run lineage.
- [ ] Separate metadata/catalog from bulk field storage; make storage pluggable.
- [ ] Add streaming snapshots, probes, regional extracts, reductions, and derived
  quantities so agents need not load whole simulations into memory.
- [ ] Add parameter sweeps and campaign-level aggregation with explicit parameter grids.
- [ ] Store units, coordinate frame, topology/geometry identity, sampling/time semantics,
  solver revisions, and validation labels with every artifact.
- [ ] Add quality gates: convergence, finite-state, balance residual, benchmark status,
  and explicit failure/invalid labels.
- [ ] Make any result traceable to exact config, solver versions, workflow and parent runs.

**Exit gate:** any selected dataset row/field can be traced to the exact run and source
artifacts, and invalid results cannot silently enter a high-quality dataset.

## P5 — AI, surrogate, and model lifecycle

- [ ] Define DatasetSpec with source selectors, transforms, feature/label schema, filters,
  units, splits, and quality requirements.
- [ ] Implement deterministic dataset builds, split/leakage checks, and dataset versioning.
- [ ] Define pluggable TrainingBackend and EvaluationBackend protocols; do not require a
  single ML framework in the TensorPhysics core.
- [ ] Record training code revision, dataset version, hyperparameters, random seeds,
  compute resources, metrics, checkpoints and model hashes.
- [ ] Add model registry states: exploratory, evaluated, validated, production-approved.
- [ ] Add held-out engineering benchmark evaluation, uncertainty reporting where supported,
  and out-of-domain rejection/fallback to the numerical solver.
- [ ] Support parameter inference, surrogate/ROM, anomaly detection, optimization, and
  control as separate registered capabilities rather than opaque agent behavior.

**Exit gate:** a model artifact is reproducible, has known evaluation scope, traces back to
its data, and cannot be promoted to production without explicit validation.

## P6 — Agent-operated engineering applications

- [ ] Define AgentTool schemas with typed inputs/outputs, permissions, side effects,
  resource estimate, timeout, and durable job/artifact references.
- [ ] Provide tools for solver/capability discovery, configuration, preflight, run,
  monitor, diagnosis, artifact inspection, dataset build, train/evaluate, and report.
- [ ] Store plans as inspectable workflow DAGs; support resume and approval checkpoints.
- [ ] Keep credentials out of prompts/logs; use least-privilege identities and audit trails.
- [ ] Require approval for expensive campaigns, destructive actions, or publishing
  production artifacts; enforce quotas and hard time/resource limits.
- [ ] Build one end-to-end demonstration from an engineering objective to a validated
  report and reusable application package.
- [ ] Allow agents to generate application glue/UI/API code only against tested platform
  contracts, with generated code versioned and subjected to tests and validation.

**Exit gate:** a natural-language engineering request produces an auditable workflow,
validated numerical/data artifacts, and a repeatable application package with no hidden
manual state.

## Deferred until justified

- Multi-node distributed field exchange and MPI-wide orchestration.
- Microservice decomposition of the control plane.
- Automatic solver/plugin installation and unrestricted code execution.
- Universal mesh/particle representation and automatic conservative remapping.
- Autonomous production approval without explicit engineering validation.

These are not forbidden forever; they are deferred until contract stability, security,
and demonstrated scaling requirements justify the added complexity.
