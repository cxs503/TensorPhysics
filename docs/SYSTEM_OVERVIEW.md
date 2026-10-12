# Tensor Ecosystem — System Overview

**Status:** architecture and integration map. This document describes the intended system and explicitly distinguishes implemented contracts from capabilities that still require end-to-end validation.

## 1. Mission

The Tensor ecosystem is an engineering simulation and AI workflow system. It should let an authorized user or agent define an engineering objective, configure and execute one or more specialist solvers, validate numerical results, preserve provenance, build datasets, train/evaluate AI or reduced-order models, and package repeatable engineering applications.

It is not one monolithic solver. The specialist solvers remain independently usable; the platform and orchestration layers connect them through versioned contracts and reproducible artifacts.

## 2. Canonical repository map

| Repository | Owns | Must not become |
|---|---|---|
| [TensorPlatform](https://github.com/cxs503/TensorPlatform) | Shared identity/authorization, data catalog, datasets, experiments, model lifecycle, audit and reusable platform services | A second solver scheduler or duplicate numerical kernel |
| [TensorPhysics](https://github.com/cxs503/TensorPhysics) | Scientific workflow, solver lifecycle, execution adapters, multiphysics coordination, field exchange, run provenance and numerical acceptance gates | A replacement for each solver's numerical implementation |
| [TensorLBM](https://github.com/cxs503/TensorLBM) | Lattice Boltzmann fluid solver and its validation cases | A cross-ecosystem workflow authority |
| [TensorFVM](https://github.com/cxs503/TensorFVM) | Finite-volume fluid solver and its validation cases | A general-purpose data/AI platform |
| [TensorFEM](https://github.com/cxs503/TensorFEM) | Finite-element structural/solid mechanics solver and its validation cases | A duplicate orchestration layer |
| [TensorDEM](https://github.com/cxs503/TensorDEM) | Discrete-element particles, contacts, bonded materials and fracture | A substitute for fluid or continuum structural solvers |

## 3. Logical architecture

```text
Engineering application / authorized AI agent
                 |
                 v
TensorPhysics — workflow DAG, lifecycle, coupling, validation, provenance
       |                 |                 |
       v                 v                 v
   TensorLBM         TensorFVM         TensorFEM / TensorDEM
       \                 |                 /
        \________ typed adapters ________/
                         |
                         v
TensorPlatform — catalog, artifacts, datasets, experiments, model lifecycle
                         |
                         v
              Reports / visualization / APIs
```

This is a responsibility diagram, not a claim that every adapter or production deployment is already complete. TensorPhysics coordinates execution and references outputs; large fields and trajectories should be stored as artifacts rather than copied into workflow metadata. TensorPlatform manages their catalog, access policy, lineage and downstream dataset/model lifecycle.

## 4. End-to-end lifecycle

1. **Specify:** capture physical objective, quantities of interest, units, constraints, geometry/materials, boundary conditions and acceptance criteria.
2. **Plan:** select solver(s), compatible versions, coupling strategy, resources and a versioned workflow graph.
3. **Preflight:** validate schemas, dimensions/units, solver capabilities, mesh/particle inputs, clock compatibility and a small verification case.
4. **Execute:** submit through a local or HPC execution backend; track durable run/job IDs, progress, logs, resources and cancellation.
5. **Couple:** exchange typed fields/loads through adapters; check time alignment, mapping assumptions, conservation/balance and solver-specific stability.
6. **Verify:** run benchmark/regression checks and mesh/time-step sensitivity where applicable. A successful process exit alone is not physical validation.
7. **Publish:** write immutable artifacts with schemas, units, checksums and producer revisions; record configuration hash, environment, resources, failure reason and validation status in a run manifest.
8. **Learn:** construct versioned datasets with transformation lineage and leakage checks; train/evaluate AI or ROM models against held-out simulation or experimental evidence.
9. **Deliver:** package an engineering workflow with reproducible configuration, report, API and visualization, including applicability limits and uncertainty.

## 5. Shared contracts

The cross-repository contract surface should be versioned and tested. Names below describe the intended shared model; each implementation must document its current maturity and compatibility policy.

- **SolverDescriptor:** solver identity/version, capabilities, supported dimensions/physics, input/output schemas, units, clock semantics and validation status.
- **SolverAdapter:** initialize, run/advance, diagnostics, finalize, checkpoint/restore capability and compatibility checks.
- **WorkflowSpec:** typed DAG, dependencies, inputs/outputs, conditions, retries, resource requests and approval gates.
- **RunManifest:** immutable configuration/hash, source revisions, environment, resources, solver statuses, timestamps, exit/failure reason and validation outcome.
- **ArtifactRef:** durable URI/ID, media type, schema version, units/frame/time convention, checksum, producer and quality label.
- **DatasetSpec / ModelVersion:** source artifacts, transformations, split policy, feature/label schema, training code/config, metrics, model hash and approval status.

No interface should silently assume all solvers advance in the same way: steady SIMPLE iterations, transient LBM steps, DEM subcycling and FEM equilibrium/dynamics need explicit clock/lifecycle semantics. Unit conversion, mesh/particle remapping and conservative transfer must be explicit operations, not implied by a matching field name.

## 6. First system-level application: ship–water–ice interaction

A staged engineering path is recommended:

1. **Baseline and verification:** independently validate fluid, structure and bonded-particle components with documented benchmark cases.
2. **DEM ice + prescribed hull motion:** measure reaction force, fracture evolution, fragment connectivity and energy accounting; quantify particle-spacing/time-step sensitivity.
3. **Hull FEM + DEM contact/fracture:** validate action/reaction, contact work, structural response and mesh/contact sensitivity.
4. **Fluid loads:** add LBM and/or FVM only after the selected flow model, boundary conditions and load-transfer path are verified.
5. **Closed-loop coupling:** synchronize physical clocks, exchange loads/displacements, handle subcycling, and test conservation and restart/recovery.
6. **Campaign and AI:** generate parameter sweeps, register validated artifacts, build traceable datasets, train surrogates/ROMs and compare them against held-out high-fidelity results.

The target application is continuous icebreaker navigation and icebreaking resistance, but an individual coupling fixture or visualization is not evidence that this complete capability has been achieved.

## 7. Maturity and evidence rules

Use these labels consistently across all repositories:

- **Implemented:** code and a reproducible usage path exist.
- **Tested:** automated tests exercise the declared behavior.
- **Benchmark-verified:** quantitative results meet a named reference and tolerance.
- **Coupled prototype:** components exchange data, but full physical accuracy is not established.
- **Engineering-qualified:** applicable validation evidence, sensitivity analysis and documented operating limits support the stated use.

Do not promote a feature to a higher maturity label based only on a passing unit test, successful execution, a plotted animation or an unvalidated demonstration. Every engineering claim should link to its reproducible case, parameters, outputs and limitations.

## 8. Operational and governance boundaries

Agents should use typed, permissioned operations rather than unrestricted shell access. Credentials must stay outside prompts and manifests. Expensive jobs, destructive actions and publishing require explicit policy/approval gates. Runs must be cancellable, attributable and auditable; failures and incomplete results must remain visible rather than being silently registered as accepted data.

## 9. Canonical documentation

- [TensorPhysics architecture and development phases](docs/hpc-ai-platform-architecture.md)
- [Evidence-based integration status and next acceptance gates](docs/INTEGRATION_STATUS.md)
- [TensorPhysics ecosystem integration](docs/ecosystem-integration.md)
- [TensorPlatform integration boundary](https://github.com/cxs503/TensorPlatform/blob/main/docs/ecosystem-integration.md)
- [TensorLBM integration boundary](https://github.com/cxs503/TensorLBM/blob/main/docs/ecosystem-integration.md)
- [TensorFVM integration boundary](https://github.com/cxs503/TensorFVM/blob/main/docs/ecosystem-integration.rst)
- [TensorFEM integration boundary](https://github.com/cxs503/TensorFEM/blob/main/docs/ecosystem-integration.md)
- [TensorDEM integration boundary](https://github.com/cxs503/TensorDEM/blob/main/docs/ecosystem-integration.md)

This overview is the cross-repository entry point. The specialist README files remain authoritative for each solver's actual supported features and runnable examples.
