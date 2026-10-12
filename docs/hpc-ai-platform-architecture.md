# HPC + AI scientific-computing platform architecture

# HPC + AI scientific-computing platform architecture

> **Position in the system:** this document details execution, HPC, data, and model lifecycle design for the [Agent-first Science Factory](SCIENCE_FACTORY_ARCHITECTURE.md). The strategic product is a lights-out research environment where authorized Agents plan and iterate on experiments under policy; this document focuses on the engineering planes that make those experiments executable, scalable, and auditable.

## 1. Product objective

TensorPhysics is the control and contract layer for an Agent-first HPC + AI science factory. It enables authorized Agents to act as research operators: propose a testable experiment plan, invoke typed scientific capabilities, inspect diagnostics, and recommend evidence-backed next steps. Human owners set the research goals, constraints, budget, and approval boundaries.

Its end-to-end product is not just a coupled solver or a chatbot. It should turn a bounded research objective into a reproducible, auditable workflow:

1. clarify the physical question, hypotheses, outputs, constraints, and acceptance criteria;
2. select a suitable numerical method and compatible solver versions;
3. build or configure a model, mesh/particle set, materials, and boundary conditions;
4. preflight inputs and run a cheap verification case;
5. execute experiments on local workstations, GPUs, or an HPC scheduler;
6. monitor progress, diagnose numerical failure, and make bounded recovery decisions;
7. independently verify results and preserve versioned artifacts with provenance;
8. curate datasets and train/evaluate AI, ROM, or surrogate models against held-out evidence;
9. use verified evidence to select the next experiment or stop when criteria are met;
10. package reproducible research results, reports, APIs, and visualization.

Agents act through typed, auditable capabilities, not unrestricted shell access. Numerical correctness remains the responsibility of solvers, coupling operators, independent verification tests, and domain acceptance criteria. A successful process exit or plausible plot is not scientific acceptance.

The agent is an operator of typed, auditable capabilities, not an unrestricted shell
script generator. Numerical correctness remains the responsibility of solvers,
coupling operators, verification tests, and domain acceptance criteria.

## 2. Architectural planes

### A. Agent and application plane

Provides typed tools for discovery, planning, configuration, validation, execution,
monitoring, dataset creation, training, evaluation, and report generation. Tool
schemas must expose parameter types, units, allowed ranges, side effects, resource
costs, and expected artifacts. Long operations return durable job IDs and status,
not a blocked conversational request. Agent plans are stored as inspectable workflow
graphs with human approval gates for expensive, destructive, or externally visible
actions.

### B. Workflow and experiment control plane

Owns a versioned workflow specification, dependency graph, parameter sweeps,
retries, cancellation, artifact lineage, and run state. It distinguishes:
- a solver step (one numerical update),
- a simulation run (many steps),
- a campaign (many parameterized runs),
- a dataset build (selection/transformation of run outputs),
- a training/evaluation job (an ML workload).

Workflow orchestration is not the HPC scheduler. It submits jobs through a backend
adapter (for example, local process, Slurm, or another site scheduler), records
scheduler IDs, and reconciles actual job state with platform state.

### C. HPC execution plane

Owns resource-aware execution: CPU/GPU counts, memory, walltime, accelerator type,
MPI ranks, affinity, scratch storage, container/environment identity, queue selection,
and cost/accounting metadata. It must support local development and remote HPC without
forcing the same launch mechanism on both. MPI/distributed execution belongs behind
an execution backend and solver-specific distributed contracts, not in a hidden
assumption inside the core scheduler.

### D. Solver and multiphysics plane

TensorLBM, TensorDEM, TensorFEM, and TensorFVM retain their numerical kernels,
native data structures, discretization choices, and solver-specific validation.
TensorPhysics supplies stable adapter contracts, clocks, field/mesh references,
coupling operators, diagnostics, checkpoints, and cross-solver orchestration. A solver
may be used standalone or via an adapter. A common interface must not erase meaningful
differences between steady SIMPLE solves, transient LBM steps, DEM subcycling, and
FEM equilibrium/dynamics analyses.

### E. Scientific data plane

Treat data as versioned artifacts, not just files or transient in-memory tensors.
Every run manifest should capture:
- immutable input/configuration and parameter values;
- Git revisions for TensorPhysics and every selected solver/adapter;
- software environment/container digest and hardware/backend information;
- mesh/particle/model identity, units, coordinate frame, and time convention;
- random seeds, solver tolerances, convergence/failure reason, and exit status;
- output artifact URIs, schemas, checksums, and quality/validation labels;
- links to parent runs, restart checkpoints, and the workflow that created them.

Use metadata/catalog interfaces separately from bulk array storage. Large fields should
be written in chunked/streamed formats by the owning backend; the orchestration core
must not copy multi-gigabyte tensors into JSON or keep all time history in RAM. Support
selective probes, reductions, regions, snapshots, and derived quantities so AI agents
can inspect results without loading every field.

### F. AI and model lifecycle plane

A dataset build is a versioned transformation graph with explicit filtering, unit
normalization, feature/label definitions, train/validation/test splits, and leakage
checks. Training jobs record code revision, hyperparameters, seeds, dataset version,
compute resources, metrics, checkpoints, and model artifact hashes. A model registry
must distinguish exploratory, validated, and production-approved models.

AI components may assist with meshing, parameter selection, reduced-order modeling,
surrogate inference, anomaly detection, control, and workflow planning. They must report
the model domain and uncertainty where supported. A surrogate cannot silently replace a
validated high-fidelity solver: define an applicability check, uncertainty threshold,
fallback route, and validation gate.

### G. Verification, governance, and observability plane

Each workflow stage emits structured events and metrics. Verification gates include
schema checks, dimensional/unit checks, conservation and balance checks where relevant,
mesh/time-step sensitivity, benchmark tolerances, finite-state checks, and regression
comparisons. Failed checks are artifacts with explicit reasons, not silently accepted
outputs.

All agent actions are attributable to a run/user/service identity. Credentials are
managed outside prompts and manifests. Enforce least privilege, resource quotas, walltime
limits, filesystem boundaries, cancellation, and human approval for costly jobs or
publishing production artifacts. Keep a full audit trail of inputs, decisions, tool
calls, and outputs.

## 3. Stable contracts to implement

The core should evolve toward these explicit contracts; do not implement every backend
at once.

- **SolverAdapter**: discoverable metadata, initialize, advance/run, finalize,
  capabilities, native clock semantics, snapshot/restore, and version compatibility.
- **WorkflowSpec**: versioned DAG of typed steps, inputs, outputs, conditions, retry
  policy, and resource requirements.
- **ExecutionBackend**: submit, status, cancel, logs, metrics, artifact staging, and
  resource accounting.
- **ArtifactStore / RunManifest**: initial RunManifest and ArtifactRef value contracts now exist in tensorphysics.provenance; next integrate lifecycle updates and storage backends. The target contract covers immutable artifacts, content hashes, provenance,
  schema/version, and validation state.
- **Field/Geometry contracts**: units, coordinate frame, location, topology/geometry
  identity, ownership, timestamp, sampling semantics, and optional conservative
  transfer operator. Temporal interpolation alone is not spatial mapping.
- **DatasetSpec**: source run selectors, transforms, features/labels, splits, filters,
  and quality requirements.
- **TrainingJob / ModelArtifact**: reproducible train/evaluate lifecycle and registry
  metadata.
- **ValidationSuite**: benchmark ID/version, inputs, metrics, tolerances, and pass/fail
  report.
- **AgentTool**: typed input/output schema, permissions, resource estimate, timeout,
  side-effect class, and durable job/artifact references.

Prefer JSON Schema or another explicit schema format at system boundaries, while keeping
Python dataclasses/protocols convenient inside the core. Add API versioning and migration
tests before external agents depend on these contracts.

## 4. Deployment and scaling

Start with a single-node execution backend and local artifact directory so contracts can
be tested cheaply. Add a Slurm backend and shared artifact store only after the local
workflow is reproducible. Distributed field movement, MPI communicators, GPU placement,
and multi-node checkpoints require explicit capability negotiation; never infer these
from a solver name.

Avoid prematurely splitting the repository into microservices. Keep a modular Python
control plane until independent scaling, access boundaries, or team ownership justify
service separation. The control plane may remain light even when solver jobs consume
large HPC allocations.

## 5. Development phases and acceptance gates

### Phase 1 — Contract-first, reproducible single-node workflow
- Define versioned run manifest, workflow spec, capability metadata, artifact references,
  and typed agent tool boundary.
- Add configuration validation, resource estimates, structured logs, run IDs, and
  artifact checksums.
- Make a fake solver and one real adapter run end-to-end, with a machine-readable report.
- Acceptance: same inputs/revisions produce a fully traceable run; invalid configurations
  fail before resource allocation.

### Phase 2 — Solver adapters and engineering benchmark pipeline
- Complete the TensorDEM adapter contract and tests; add a named TensorLBM adapter for
  one real transient case; use existing TensorFEM surface exchange only where geometry
  and force contracts are satisfied.
- Create benchmark manifests and quantitative regression gates before claiming physical
  coupling. Separate interface correctness from physical accuracy.
- Acceptance: independently reproducible reference cases and cross-solver balance reports.

### Phase 3 — HPC execution backend
- Add backend-neutral resource requirements, job submission/status/cancel, logs, staging,
  environment identity, and queue/resource accounting.
- Implement local process execution first, then Slurm as a separate plugin/backend.
- Acceptance: interrupted and failed jobs reconcile correctly; resource usage and artifacts
  are associated with durable run IDs.

### Phase 4 — Scientific data and dataset factory
- Stream field/probe/derived outputs to artifact storage; index metadata and lineage.
- Add parameter campaigns, dataset versioning, quality filters, and deterministic splits.
- Acceptance: any dataset sample traces back to exact input configuration and solver run.

### Phase 5 — AI training, evaluation, and model registry
- Provide pluggable training backends rather than coupling core to one ML framework.
- Record data, code, parameters, metrics, checkpoints, hardware, and uncertainty/evaluation.
- Add held-out benchmark and out-of-domain rejection gates.
- Acceptance: model results are reproducible and production promotion requires validation.

### Phase 6 — Agent-operated engineering applications
- Expose the above capabilities through permissioned typed tools and inspectable plans.
- Agents can configure, run, monitor, diagnose, build datasets, train/evaluate, and package
  an application while respecting resource quotas and approval gates.
- Acceptance: a complete demo workflow executes from a natural-language engineering goal
  to a validated, traceable application artifact with no hidden manual steps.

## 6. Non-goals and anti-patterns

- Do not move solver kernels into TensorPhysics just to create a common API.
- Do not treat a successful process exit as physical validation.
- Do not call a field exchange test a validated multiphysics simulation.
- Do not store bulk tensor fields in checkpoint JSON or unbounded in-memory histories.
- Do not let agents execute unrestricted shell commands or launch unlimited expensive jobs.
- Do not make TensorPhysics depend mandatorily on PyTorch, one HPC scheduler, one object
  store, or one model-training framework.
- Do not build a generic AI-agent framework before the scientific execution contracts
  and artifacts are reliable.
