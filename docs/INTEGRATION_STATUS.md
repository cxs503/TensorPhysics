# Tensor ecosystem integration status

**Audit date:** 2026-10-12  
**Purpose:** distinguish code-backed integration surfaces from target architecture. This is a source/documentation audit, not a claim that all integration tests were run in this session.

## Status vocabulary

- **Present:** a concrete interface or implementation is identified in the repository.
- **Case-specific:** works for a named model or workflow and must not be generalized to every solver.
- **Contract-only / pending:** the desired cross-repository contract is documented, but an end-to-end implementation or evidence is not established by this audit.
- **Not qualified:** a working interface or test does not establish physical engineering accuracy.

## Current integration map

| Component | Identified implementation surface | Current boundary | Next acceptance gate |
|---|---|---|---|
| TensorPhysics core | Solver lifecycle scheduler; field descriptors/store; PRE_STEP, AFTER_SOLVER, POST_STEP callbacks; RunManifest and ArtifactRef; versioned checkpoint envelope | Core orchestration and metadata are present. Checkpoint envelope is not a full solver restart/resume API; field store does not convert units or conservatively map meshes/particles. | Add a reproducible local end-to-end run with durable run ID, artifact checksums, diagnostics and validation report. |
| TensorDEM | `TensorDEMAdapter` for native 2-D `tensordem.IceDEM`; native `dt` and clock checks; integer-ratio subcycling; per-substep load provider | Identified adapter is 2-D and accepts an already constructed model. Snapshot metadata is not a full restart image. 3-D requires a separate adapter. | Automated adapter tests for clock mismatch, subcycling, force shape/units, failure propagation and deterministic manifest; then benchmark and sensitivity checks. |
| TensorLBM | Native kernels and multiple solver APIs; `LBMSimulation.step()`; lower-level `LBMStepExecutor` for named D3Q19/D3Q27 configurations | Public kernels are not a universal stateful solver protocol. The inspected minimal periodic D2Q9 API is suitable for smoke tests, not an engineering boundary/force adapter. No generic adapter is established by this audit. | Select one named, validated solver case; implement an adapter preserving solver state and avoiding full distribution-field copies; verify clocks, force hooks and artifacts. |
| TensorFEM | `QuadSurfaceExchange`, `PlanarEmbedding`, and case-specific `LinearHullReceiver` | Useful surface-transfer/receiver boundary with mapping tests, but not a universal FEM lifecycle adapter. Geometry mapping and load-hold semantics must remain explicit. | Define the first named FEM use case, test units/frame/time, resultant force, moment and virtual work, and document when mapping must be rebuilt. |
| TensorFVM | `SolverConfig`, `SimpleSolver`, `SolverResult`; separate `Cylinder3DSolver` | The standard SIMPLE API is a run-level `solve()`, not a one-global-step API. Do not call a full steady solve once per global coupling step. 3-D API has its own constraints. | Implement a run-level adapter first (configure → solve → diagnostics/artifacts); add transient stepping only for a specific supported mode. |
| TensorPlatform | Shared identity, catalog/dataset/experiment/model lifecycle and governance responsibilities documented | Architectural boundary is clear; this audit does not establish that TensorPhysics automatically registers every run/artifact or that the entire data→training→model-publication loop is end-to-end operational. | Add a minimal integration test registering a manifest and artifact, then a dataset lineage record; verify permission checks and failure/incomplete status preservation. |

## Shared contract requirements

Before a coupled workflow is called integrated, its adapter and artifacts must declare:

1. **Identity and provenance:** solver name/version/revision, configuration hash, run ID, producer and artifact checksum.
2. **Physical meaning:** units, coordinate frame, sign convention, mesh/lattice/particle location, dimensions/shape and quantity type (force versus traction).
3. **Time semantics:** steady run versus transient step, physical timestamp, native time step, subcycling ratio, interpolation/hold policy and no silent extrapolation.
4. **Failure and recovery:** initialization/run/finalization errors, cancellation behavior, partial-output status and explicit checkpoint/restore capability.
5. **Acceptance evidence:** named benchmark, quantitative tolerance, conservation/work/energy checks where applicable, sensitivity evidence and applicability limits.

Matching field names are not sufficient to imply matching units, discretization, time semantics or conservation behavior.

## Recommended implementation order

1. Finish deterministic TensorDEM 2-D adapter tests and a local manifest/artifact example.
2. Add TensorPlatform artifact registration and lineage for that exact run; keep large fields/trajectories in artifact storage.
3. Add one named TensorFEM surface/receiver workflow with force, moment and virtual-work tests.
4. Add a narrowly scoped TensorLBM or TensorFVM adapter selected by the required physics and benchmark evidence; do not force unlike solver lifecycles into one generic step API.
5. Only then run the first coupled icebreaker campaign; promote maturity labels only when quantitative verification and sensitivity criteria pass.

## Related documentation

- [System overview](SYSTEM_OVERVIEW.md)
- [Adapter contract audit](adapter-contracts.md)
- [Ecosystem integration](ecosystem-integration.md)
- [Core architecture](architecture.md)


## Update: TensorPlatform remote run registration

The branch `feat/platform-registered-dem-research-loop` adds an executable remote-registration path in `examples/research_loop_dem_platform.py` and a manifest-v1 adapter in `src/tensorphysics/platform_manifest.py`. It runs the existing bounded TensorDEM research-loop demo, uploads real outputs and the validation report to S3-compatible storage, registers checksum-backed remote URIs with TensorPlatform, then retrieves the record and compares the stored manifest to the submitted manifest.

This is implemented code plus mocked/unit-test coverage; it is not yet evidence of a live cloud run. A live acceptance run still requires configured S3 credentials and a deployed TensorPlatform API token with `datasets:upload` and `datasets:read`. The output checks are software integrity checks, not a physical icebreaking validation.
