# Architecture and module boundaries

## Purpose
TensorPhysics coordinates specialist solvers. It does not replace TensorLBM, TensorFEM, TensorDEM, or TensorFVM and must not duplicate their numerical kernels.

## Layers
- **Specialist solvers:** own discretization, numerical kernels, constitutive/contact models, solver state, and performance-critical operations.
- **TensorPhysics core:** owns lifecycle, global time coordination, registration, diagnostics, run metadata, and orchestration contracts.
- **Adapters:** translate framework contracts to each solver's native API. They remain optional dependencies.
- **Coupling manager:** executes callbacks at PRE_STEP, AFTER_SOLVER, and POST_STEP. AFTER_SOLVER callbacks run immediately after the named solver advances, allowing a later solver to consume newly published data in the same global step. Callbacks run in priority order, with registration order breaking ties.
- **Field exchange:** FieldDescriptor records producer, units, location, optional shape and dtype. FieldStore holds references to published data and timestamps without requiring NumPy or PyTorch.
- **Checkpoint store:** writes a versioned JSON envelope atomically. Solver payloads must be JSON-compatible; large arrays/tensors should be serialized by solver-specific code and referenced from the payload.

## Time semantics
At each global step, solvers receive the same start-of-step time and increment, in registration order. The global clock advances only after all solvers complete successfully. POST_STEP callbacks receive the updated global step/time. Runtime failure marks the simulation FAILED; callers should close it to release resources.

The final increment is shortened when end_time is not divisible by dt. max_steps limits the planned number of increments. If the step budget is exhausted before end_time, state is LIMIT_REACHED, not FINISHED. Solver-specific subcycling must be explicit.

## Field exchange semantics
Field publication is explicit and producer-owned. Re-registering identical metadata is idempotent; conflicting metadata fails. When the data object exposes a shape attribute, its shape is checked against the descriptor. Reads can enforce a maximum age in global steps. The store does not copy data, convert units, validate arbitrary nested Python containers, or guarantee thread safety; these remain explicit adapter/operator responsibilities.

## Checkpoint semantics
Checkpoints include the schema version, simulation clock, time-step configuration, and per-solver SolverState payloads. Writes use a temporary file in the destination directory followed by atomic replacement. Unknown schema versions and clock mismatches are rejected. This is a persistence envelope, not yet a full simulation-resume API: solver-specific restore methods and compatibility checks must be designed before resuming a coupled run.

## Error handling
Initialization failure finalizes previously initialized solvers in reverse order. Finalization attempts every initialized solver even if one finalizer fails. If cleanup also fails during initialization failure, both the initialization and cleanup errors are surfaced.

Coupling callback failures mark the simulation FAILED and are surfaced as SimulationError. Solver state is not automatically rolled back; callbacks and solver adapters must document any partial updates.

## Non-goals for v0.1
- Universal mesh or particle data model.
- Automatic unit conversion.
- MPI/distributed execution.
- Automatic solver discovery.
- Claims of validated multiphysics coupling without benchmark evidence.

## Coupling requirements
A physically meaningful coupling operator must define exchange phases, data ownership, units, interpolation/conservation properties, convergence criteria, and failure/rollback semantics before implementation.
