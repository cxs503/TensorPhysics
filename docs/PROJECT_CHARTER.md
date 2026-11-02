# TensorPhysics — scientific domain charter

**Status:** repository responsibility and agent-development contract  
**Audience:** maintainers, contributors, autonomous coding agents  
**Canonical ecosystem vision:** [Tensor Ecosystem: Vision for the AI Era](https://github.com/cxs503/TensorPlatform/blob/main/docs/architecture/TENSOR_ECOSYSTEM_AI_ERA_VISION.md)

## Mission

Own the scientific-research domain layer: typed scientific workflows, solver coordination, run manifests, provenance semantics, numerical/physical validation, and evidence-backed scientific acceptance.

## This repository owns

- Scientific workflow semantics and experiment/campaign policies
- Coordination contracts across specialist solvers; field/time exchange and run lifecycle
- Scientific validation plans, reference provenance, uncertainty/error analysis, and acceptance gates
- Domain-specific run manifests and evidence interpretation

## Explicit non-goals

- Generic multi-domain agent runtime, identity, authorization, durable generic task primitives, or platform-wide audit infrastructure
- Reimplementing solver kernels owned by TensorLBM/FEM/DEM/BEM/SPH/DG/FVM
- Treating a successful process, passing software test, or generated artifact as scientifically validated by itself

## Integration contract

Consume TensorPlatform's domain-neutral runtime, governance, artifact/catalog, and registration services through versioned interfaces. Keep science-specific rules here or in specialist solvers; publish results with explicit evidence level and provenance.

## Quality and evidence gates

Separate software integrity, integration correctness, numerical verification, validation against credible references/experiments, uncertainty, and engineering applicability. Every claim must name which evidence level was actually achieved.

## Agent instructions

Before implementing a substantial change, an agent must:

1. Read this charter, the README, the relevant code/tests, and recent PRs.
2. Explain why the change belongs in this repository and identify affected cross-repository contracts.
3. Prefer a stable, versioned public interface over private imports or copied logic.
4. Include executable tests and a reproducible example or benchmark appropriate to the change.
5. Report exactly what was run, what passed/failed, and what was not tested.
6. Distinguish implementation, unit-test, integration, benchmark, validation, and production evidence.
7. Never claim a result is scientifically or engineering-validated unless the relevant domain evidence supports that claim.
8. Avoid adding general platform features here when they belong in TensorPlatform; avoid adding specialist domain rules to TensorPlatform.
9. Keep changes scoped, reviewable, and reversible. Do not equate PR count or workflow completion with quality.

## Current maturity

This charter states intended ownership. Actual feature maturity must be checked against current source, tests, benchmark outputs, and deployment evidence; existing demos may be contract or software-integrity tests rather than validated physics.

This charter describes responsibility and acceptance expectations; it does not override the source code or claim that planned functionality is implemented. Update the README and this document when ownership changes, and resolve conflicting documentation explicitly.
