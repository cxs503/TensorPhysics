# Architecture and module boundaries

## Purpose
TensorPhysics coordinates specialist solvers. It does not replace TensorLBM, TensorFEM, TensorDEM, or TensorFVM and must not duplicate their numerical kernels.

## Layers
- **Specialist solvers:** own discretization, numerical kernels, constitutive/contact models, solver state, and performance-critical operations.
- **TensorPhysics core:** owns lifecycle, global time coordination, registration, diagnostics, run metadata, and orchestration contracts.
- **Adapters:** translate framework contracts to each solver's native API. They remain optional dependencies.
- **Coupling operators:** own transfer maps and exchange policies. The scheduler only defines when components advance; it does not guarantee physically correct coupling.

## Time semantics
At each global step, solvers receive the same start-of-step time and increment, in registration order. The global clock advances only after all solvers complete successfully. Runtime failure marks the simulation FAILED; callers should close it to release resources.

The final increment is shortened when end_time is not divisible by dt. max_steps limits the planned number of increments. Solver-specific subcycling must be explicit.

## Error handling
Initialization failure finalizes previously initialized solvers in reverse order. Finalization attempts every initialized solver even if one finalizer fails.

## Non-goals for v0.1
- Universal mesh or particle data model.
- Automatic unit conversion.
- MPI/distributed execution.
- Automatic solver discovery.
- Claims of validated multiphysics coupling without benchmark evidence.

## Coupling requirements
A future coupling protocol must define exchange phases, data ownership, units, interpolation/conservation properties, convergence criteria, and failure/rollback semantics before implementation.
