# Tensor ecosystem integration

TensorPhysics is the scientific execution and coordination layer between the shared TensorPlatform services and independent specialist solvers. It should coordinate numerical tools, not reimplement their numerical kernels.

## Responsibilities

- Versioned solver capability descriptors and adapter contracts.
- Workflow/run lifecycle, reproducible configuration, execution backends and cancellation.
- Multiphysics scheduling, clock/unit checks, field exchange and coupling diagnostics.
- Run manifests, artifact references, provenance, validation reports and acceptance gates.
- Agent-facing tools that expose bounded, permissioned scientific actions.

## Boundaries with sibling projects

- **TensorPlatform** owns shared identity, data catalog, dataset/model lifecycle, audit and reusable AI services.
- **TensorLBM**, **TensorFVM**, **TensorFEM** and **TensorDEM** own solver kernels, native inputs/outputs and solver-specific verification.
- Do not copy large solver outputs into orchestration metadata. Register immutable artifact references and store data in configured artifact storage.
- A coupling demonstration is not automatically a validated physical model; every engineering claim must link to a reproducible benchmark and declared applicability limits.

## Shared contracts

Keep contracts explicit and versioned: SolverDescriptor, RunManifest, ArtifactRef, WorkflowSpec, DatasetSpec and ModelVersion. Contracts should declare units, dimensions, clock semantics, schema versions, provenance, errors and compatibility rules.

## First end-to-end application: icebreaker and broken ice

1. TensorDEM models bonded ice/fragment contact; TensorFEM models hull/structure response.
2. TensorLBM and/or TensorFVM supplies fluid loads when the selected model and validation support them.
3. TensorPhysics schedules solver steps and transfers fields through typed adapters, checking time alignment, units and conservation invariants.
4. TensorPlatform registers outputs, lineage and validation results, then enables reproducible dataset generation and AI/ROM experiments.

## Minimum integration gates

- Unit/time-step compatibility is checked before coupled execution.
- Each run captures solver versions, config hash, environment/resource metadata and artifact checksums.
- Failed, incomplete and unvalidated runs remain distinguishable from accepted results.
- A runnable integration test checks data delivery, conservation where applicable, restart semantics and deterministic manifest generation.
