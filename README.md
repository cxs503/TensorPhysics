# TensorPhysics

TensorPhysics is the HPC + AI scientific-computing and engineering workflow platform for the Tensor ecosystem.

**System-wide entry point:** [Tensor Ecosystem System Overview](docs/SYSTEM_OVERVIEW.md) — architecture, repository boundaries, shared contracts, end-to-end lifecycle, icebreaker application roadmap, and evidence/maturity rules.

Its goal is to let authorized agents drive the complete lifecycle—from engineering problem definition and solver selection, through reproducible HPC execution and data production, to dataset creation, AI/ROM training and evaluation, and delivery of engineering applications. TensorLBM, TensorFEM, TensorDEM, and TensorFVM remain independent specialist solvers; TensorPhysics provides the contracts, execution control, provenance, and validation workflow around them.

## Mission: Agent-first lights-out science factory

TensorPhysics is the **scientific execution and control plane** for a persistent research environment in which authorized Agents can plan experiments, invoke scientific tools, inspect diagnostics, trigger independent validation, and choose evidence-backed next steps. Humans define goals, budgets, constraints, and approval boundaries; the system makes each permitted experiment reproducible, observable, and auditable.

It is not merely a chatbot wrapper around solvers. Its job is to turn an Agent's proposed research plan into a typed, bounded, traceable workflow and return evidence—not just a success message or a plausible explanation. Agent-generated results must pass independent checks before being promoted as validated evidence.


**Runnable remote research loop:** [TensorDEM → S3 artifacts → TensorPlatform registration and retrieval](docs/RESEARCH_LOOP_PLATFORM_REGISTRATION.md).

**Canonical architecture:** [Science Factory Architecture](docs/SCIENCE_FACTORY_ARCHITECTURE.md) · [System Overview](docs/SYSTEM_OVERVIEW.md)

## Role in the Tensor ecosystem

TensorPhysics is the **scientific workflow and multiphysics coordination layer** between an Agent/application and the specialist solvers. It turns approved plans into traceable execution: solver selection and lifecycle, dependency ordering, field exchange, checkpoints, run manifests, artifacts, diagnostics, and numerical acceptance gates.

- **TensorPlatform** supplies shared identity, data/experiment/model lifecycle, and governance services.
- **TensorLBM / TensorFVM / TensorFEM / TensorDEM** remain independent specialist solvers; TensorPhysics must not reimplement their numerical kernels.
- **Agent tools** should call typed, permissioned workflow operations and durable job/artifact identifiers rather than rely on unrestricted shell execution.
- **HPC backends** are execution targets behind adapters; orchestration contracts should remain backend-neutral.

The intended loop is goal and constraints → workflow plan → solver execution and field exchange → validation and provenance → dataset curation → model training/evaluation → engineering decision. Physical correctness must be established by solver-specific verification, conservation checks, sensitivity studies, and appropriate experimental or reference data—not by successful task completion alone.

**Maturity note:** the current implementation below lists concrete code-level capabilities. Coupling fixtures are contract tests unless explicitly qualified otherwise; checkpoint envelopes do not imply full solver restart/resume; and production cluster execution, complete adapters, and a validated icebreaking model remain separate milestones.

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

## Optional TensorPlatform run registration

The end-to-end smoke example can register its manifest after generating the local artifact. Registration is opt-in and requires a shared-storage URI for the artifact that has already been uploaded by your storage workflow; the example does not upload artifact bytes.

Set `TENSORPLATFORM_URL`, `TENSORPLATFORM_API_TOKEN`, and `TENSORPLATFORM_ORGANIZATION_ID`, then run:

```bash
python examples/end_to_end_artifact_run.py \
  --output outputs/end_to_end_smoke \
  --register-platform \
  --artifact-uri s3://your-bucket/path/to/toy_history.csv
```

Replace the example URI with the actual shared URI of the exact file generated by the example. The SHA-256 in the manifest is computed locally; this client does not independently verify that the remote object contains identical bytes. A local `file://` URI is rejected for remote registration. Without `--register-platform`, the example runs locally and makes no network request.

This remains a software provenance smoke test, not a validated physical simulation.


## First executable research-loop demo

The first bounded TensorDEM integration slice is documented in
[Research Loop Demo](docs/RESEARCH_LOOP_DEMO.md) and implemented in
`examples/research_loop_dem.py`. It runs a real solver process, independently
checks history/trajectory readability and finite values, stores outputs and
validation evidence through `LocalArtifactStore`, emits a typed `RunManifest`,
and recommends a next action based on the checks. Versioned goal, plan,
validation-report, and research-memory contracts live in
`tensorphysics.research_contracts`. The deterministic, fail-closed next-action
policy is implemented in `tensorphysics.research_policy` and documented in
[Executable Research Methodology](docs/EXECUTABLE_RESEARCH_METHODOLOGY.md).

This is a software-integrity gate, not physical validation; it does not yet
register the run in TensorPlatform or perform an AI-driven parameter search.


## Bounded multi-run campaign

[Research Campaign Demo](docs/RESEARCH_CAMPAIGN_DEMO.md) adds a two-run, fail-closed campaign controller around the single-run demo. It varies one input parameter, independently validates each real TensorDEM run, records per-run manifest references, and stops on a failed gate. CI executes both runs. This remains a software-integrity integration exercise, not a physical sensitivity study or AI-driven research policy.

- **Platform-backed campaign:** [TensorDEM → S3 artifacts → TensorPlatform registration, read-back, and parent-run lineage](docs/RESEARCH_LOOP_PLATFORM_REGISTRATION.md).


## Persistent worker deployment

The durable task ledger is now paired with a standalone polling worker that can
run independently of an interactive Agent terminal. It supports task submission
and status inspection, restart recovery, bounded retries, structured event
history, and a separate fail-closed validation command. Deploy and configure it
using [Persistent Research Worker](docs/PERSISTENT_RESEARCH_WORKER.md). This
is execution infrastructure: production use still requires a real authenticated
Agent CLI, an independent domain validator, persistent storage, and operating
system isolation. A running worker alone does not prove scientific correctness.
