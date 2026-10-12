# Tensor Science Factory — Agent-First Research Infrastructure

**Status:** target architecture and implementation contract. This document distinguishes strategic direction from capabilities already implemented.

## 1. Product definition

Build a persistent, permissioned research environment in which AI Agents can act as scientific operators: formulate hypotheses, design computational experiments, execute them on controlled scientific tools, assess evidence, and plan the next experiment. Humans set research goals, constraints, budgets, risk policies, and approval requirements.

“Lights-out” means autonomous execution within explicit boundaries, not unbounded code execution or the removal of scientific accountability.

## 2. Design principles

- **Agent-first, not model-first:** support multiple replaceable Agent/model implementations through stable tool contracts.
- **Execution is real:** an experiment must invoke a solver, data pipeline, or instrument—not merely produce a plan or narrative.
- **Verification is independent:** the actor that generates a result should not be the sole authority that accepts it.
- **Every run is traceable:** preserve inputs, code revisions, environment, parameters, resource use, outputs, logs, checksums, failure reason, and validation status.
- **Failure is a first-class outcome:** retain failed/incomplete runs and their evidence; never silently convert them into accepted data.
- **Safe autonomy:** typed tools, scoped credentials, isolated execution, quotas, cancellation, policy checks, and approval gates for costly/destructive/high-risk actions.
- **Solver plurality:** Tensor solvers and external tools coexist behind adapters. Reuse open source where it is economically sensible; own the numerical core where it creates measurable advantage.
- **Evidence before claims:** distinguish implemented, tested, benchmark-verified, coupled prototype, and engineering-qualified capabilities.
- **Research memory is curated:** only attach a claim to a run with provenance and an evidence level; do not treat every Agent output as fact.

## 3. Logical architecture

```text
Human research owner: goals, constraints, budget, risk policy, approval boundaries
                                  |
                                  v
Agent research actors: hypotheses | experiment design | interpretation | next-step choice
                                  |
                                  v
TensorPhysics: typed tools | workflow DAG | experiment lifecycle | solver adapters
               coupling | run provenance | validation gates | experiment campaign
                      |                                  |
                      v                                  v
        Scientific instruments                  TensorPlatform
  TensorLBM / TensorFVM / TensorFEM / DEM     identity / policy / catalog /
  external OpenFOAM / Python tools / HPC      datasets / artifacts / model lifecycle
                      |                                  |
                      +------------------+---------------+
                                         v
                      Independent evidence and verification
             benchmarks | conservation | sensitivity | uncertainty |
                   experimental/reference comparison | audit
                                         |
                                         v
                       Curated research memory and datasets
                                         |
                                         +----> next Agent experiment
```

This is a responsibility map, not a claim that every arrow is already production-ready.

## 4. Repository responsibility and non-goals

| Repository | Owns | Does not own |
|---|---|---|
| [TensorPhysics](https://github.com/cxs503/TensorPhysics) | Agent-facing scientific tool contracts, workflow and experiment orchestration, solver adapters, multiphysics coordination, run manifests, validation gates, research-loop state | The numerical kernels of specialist solvers; an unrestricted agent shell |
| [TensorPlatform](https://github.com/cxs503/TensorPlatform) | Identity, authorization, audit, scientific data catalog, artifacts, datasets, lineage, experiment/model lifecycle, reusable platform services | Solver numerics or a duplicate scientific workflow engine |
| [TensorLBM](https://github.com/cxs503/TensorLBM) | Lattice Boltzmann fluid numerics and solver-specific validation | Global research orchestration authority |
| [TensorFVM](https://github.com/cxs503/TensorFVM) | Finite-volume numerics and solver-specific validation | Generic Agent runtime or data platform |
| [TensorFEM](https://github.com/cxs503/TensorFEM) | Finite-element solid/structural mechanics and validation | Global workflow and data governance |
| [TensorDEM](https://github.com/cxs503/TensorDEM) | Discrete-element particles, contacts, bonds, fracture and validation | Fluid/continuum solver or global orchestration |
| External tools | Mature specialist capabilities (e.g. OpenFOAM), meshing, visualization, experiment devices | Ownership of the Tensor cross-system contracts |

## 5. Canonical research lifecycle

1. **Specify:** record question, hypothesis, quantities of interest, constraints, units, applicable domain, cost budget, safety boundary, and acceptance criteria.
2. **Plan:** produce a versioned experiment plan with rationale, required tools, dependencies, expected evidence, resource estimate, and stopping criteria.
3. **Preflight:** validate schemas, dimensions/units, solver capability, input geometry/mesh/particles, software versions, and a cheap verification case.
4. **Authorize:** apply permissions, resource quotas, data policy, and human approval gates where required.
5. **Execute:** run in an isolated environment through a local or HPC backend; record durable run/job IDs, logs, resources, progress, and cancellation state.
6. **Observe:** expose typed diagnostics and intermediate states suitable for Agent inspection; preserve checkpoints only when the solver supports them.
7. **Verify:** run independent tests, benchmark comparisons, conservation checks, mesh/time-step/particle sensitivity, and reference/experimental comparisons as appropriate.
8. **Interpret:** require claims to reference specific artifacts and checks; record uncertainty, limitations, contradictory evidence, and unresolved questions.
9. **Curate:** publish immutable artifacts and manifests; promote only qualified data into versioned datasets with lineage and leakage checks.
10. **Choose next experiment:** select a new hypothesis/parameter range based on evidence and budget; stop or request human review when acceptance or risk policy requires it.
11. **Deliver:** package reproducible configs, reports, artifacts, environment, validation evidence, and applicable limits.

## 6. Shared contracts

The initial contract family should be versioned, schema-validated, and tested across repositories:

- **ResearchGoal:** question/hypothesis, scope, constraints, quantities of interest, acceptance policy, budget and approvals.
- **ExperimentPlan / WorkflowSpec:** typed DAG, rationale, dependencies, inputs/outputs, branching conditions, retries, resource requests, stopping conditions and approval gates.
- **SolverDescriptor:** solver identity/version, supported physics/dimensions, capability declarations, input/output schemas, units, clock semantics and validation evidence.
- **SolverAdapter:** capability check, preflight, initialize, run/advance, diagnostics, finalize, and explicitly declared checkpoint/restore support.
- **RunManifest:** immutable configuration and hash, code revisions, environment, resource usage, timestamps, solver states, exit/failure reason and validation outcome.
- **ArtifactRef:** durable URI/ID, media type, schema version, units/frame/time convention, checksum, producer, and quality/evidence label.
- **ValidationReport:** check identity/version, inputs, measured metrics, reference, tolerance, pass/fail/indeterminate outcome, and evidence artifact links.
- **ResearchMemoryRecord:** evidence-backed claim, linked runs/artifacts, scope, confidence/limitations, and promotion status.
- **DatasetSpec / ModelVersion:** source artifacts, transformations, split policy, feature/label schema, training code/config, metrics, model hash and approval status.

No shared contract should assume all solvers advance identically: steady SIMPLE iterations, transient LBM steps, DEM subcycling and FEM equilibrium/dynamics require explicit lifecycle and clock semantics. Unit conversion, field remapping, and conservative transfer are explicit validated operations.

## 7. Safety, integrity, and autonomy levels

Use progressively enabled autonomy rather than one unrestricted mode:

- **A0 — Suggest:** Agent proposes plans; a human runs them.
- **A1 — Supervised run:** Agent executes low-risk tasks after approval.
- **A2 — Bounded campaign:** Agent runs parameter sweeps within quotas and pre-approved experiment templates.
- **A3 — Evidence-driven iteration:** Agent selects the next experiment based on verified results, within a fixed budget and policy.
- **A4 — Multi-agent factory:** multiple specialized Agents coordinate campaigns; high-impact operations remain gated and independently audited.

All levels require attributable runs, scoped permissions, secret isolation, resource caps, cancellation, and visible failure states. Destructive actions, external publication, expensive campaigns, and physical-actuator commands require explicit policies and approvals.

## 8. Scientific quality gates

Use consistent maturity labels:

- **Implemented:** code and a reproducible usage path exist.
- **Tested:** automated tests exercise the declared behavior.
- **Benchmark-verified:** quantitative results meet a named reference and tolerance.
- **Coupled prototype:** components exchange data, but full physical accuracy is not established.
- **Engineering-qualified:** validation evidence, sensitivity analysis, uncertainty and operating limits support the stated use.

Successful execution is not validation. A coupled fixture is not automatically a validated multiphysics model. Agent-written explanations are not evidence unless linked to run artifacts and independent checks.

## 9. Application strategy: ship–water–ice

The first cross-solver research campaign should be staged rather than advertised as a completed full-physics predictor:

1. Independently verify each fluid, structural, and bonded-particle component.
2. Run DEM ice fracture with prescribed hull motion; record force, fracture, energy and sensitivity.
3. Add FEM hull/structure response and validate contact balance, energy and mesh/contact sensitivity.
4. Add LBM/FVM fluid loads only after flow models and load-transfer paths are verified.
5. Test time synchronization, subcycling, conservative exchange, checkpoint/recovery, and coupling sensitivity.
6. Run bounded campaigns; register validated artifacts; build traceable datasets; evaluate surrogates/ROMs against held-out high-fidelity and experimental evidence.

## 10. Implementation status and first acceptance gates

This architecture is a target, not a statement that every capability exists. The current TensorPhysics README and repository tests describe implemented orchestration primitives; the platform run registry and artifact stores cover initial metadata/artifact workflows; individual solvers have different maturity levels. Full production HPC scheduling, universal Agent execution, autonomous hypothesis selection, full restart/resume across all solvers, and a physically qualified icebreaking model remain separate acceptance milestones unless explicitly demonstrated by reproducible evidence.

The first system-level acceptance test should prove one bounded research loop end to end: Agent-readable goal → typed plan → solver run → immutable artifact and RunManifest → independent validation report → evidence-linked next-step recommendation. The first test can be deterministic and narrow; it should not pretend to prove general autonomous science.

## 11. Canonical links

- [System overview](SYSTEM_OVERVIEW.md)
- [HPC + AI platform architecture](hpc-ai-platform-architecture.md)
- [Integration status](INTEGRATION_STATUS.md)
- [TensorPlatform integration boundary](https://github.com/cxs503/TensorPlatform/blob/main/docs/ecosystem-integration.md)
- [TensorLBM integration boundary](https://github.com/cxs503/TensorLBM/blob/main/docs/ecosystem-integration.md)
- [TensorFVM integration boundary](https://github.com/cxs503/TensorFVM/blob/main/docs/ecosystem-integration.rst)
- [TensorFEM integration boundary](https://github.com/cxs503/TensorFEM/blob/main/docs/ecosystem-integration.md)
- [TensorDEM integration boundary](https://github.com/cxs503/TensorDEM/blob/main/docs/ecosystem-integration.md)
