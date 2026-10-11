# Specialist solver adapter audit

This document records the public interfaces inspected before wiring the solver
repositories into TensorPhysics. It is an integration contract, not a claim that
all four solvers are already interchangeable or physically coupled.

## Current adapter in TensorPhysics

The first adapter is `tensorphysics.adapters.TensorDEMAdapter`, for the native
2-D `tensordem.IceDEM` API. It accepts an already-constructed model instance,
so importing the TensorPhysics core does not import PyTorch or TensorDEM.

- Requires native `dt` and `time`, and checks the initial/global clock.
- Requires each global increment to be an integer multiple of the native DEM
  time step. A load provider is called at every DEM substep start.
- Passes the instantaneous nodal load and its physical `load_time` to the
  native `IceDEM.step(external_forces=..., load_time=...)` call.
- The provider is responsible for mapping forces into the DEM's global x/y
  particle ordering, shape `(N, 2)`, and SI newtons.
- Its snapshot contains scalar orchestration metadata only. It is explicitly
  not a full restart image; the current Solver protocol has no restore method.

The adapter does not silently interpolate, scale, or remap loads. A production
LBM-to-particle mapping must separately prove resultant force, moment, and work
consistency, and must document whether a load is held or recomputed at each
substep.

## Repository findings

### TensorDEM

- Package: `tensordem`; `DEMConfig` and `IceDEM` are public exports.
- The 2-D solver exposes `dt`, `time`, `positions`, `velocities`,
  `step_count`, `step(external_forces=None, load_time=None)`, and diagnostics.
- External forces are instantaneous nodal forces in SI newtons with shape
  `(N, 2)`; the solver does not accumulate them between calls.
- A separate `IceDEM3D` API uses three-component state/loads. It is not silently
  routed through the 2-D adapter; a dedicated adapter must validate its own
  clock, shape, checkpoint, and force contracts.
- The current 2-D DEM uses a fixed native time step and explicit stepping, so
  the scheduler must subcycle rather than change the native integrator's dt.

### TensorLBM

- The package exports a large, lazily resolved public API. Individual
  `equilibrium`, `macroscopic`, collision and streaming functions are kernels,
  not a complete stateful solver protocol.
- `LBMSimulation` provides `step()`, but the inspected implementation is a
  minimal periodic D2Q9 BGK demonstration. It is useful for adapter smoke tests,
  not a replacement for an engineering boundary-condition/force solver.
- `LBMStepExecutor` is a lower-level preallocated step executor documented for
  D3Q19/D3Q27. Its constructor requires explicit lattice, collision, dimensions,
  device and optional boundary/force callbacks; the adapter must preserve the
  selected solver's state tensor and avoid copying the full distribution field
  each step.
- The initial production adapter should target a named, tested case/solver and
  expose its native macros and load/force hooks. It must not pretend that every
  one of the hundreds of public exports shares one generic interface.

### TensorFEM

- The existing surface interface provides `QuadSurfaceExchange` and an
  explicit `PlanarEmbedding` for 2-D-to-3-D transfer. It carries geometry,
  units, time and geometry identity; mapping tests cover resultant force,
  moment, and virtual work.
- `LinearHullReceiver` is a case-specific linear hull response, CPU-side, and
  its documented load interface holds loads constant over an explicit step.
- This is a useful verified load/kinematics boundary, not a general-purpose
  Solver adapter for every FEM analysis. Geometry changes require rebuilding
  the mapping; no unverified extrapolation or implicit coordinate transform is
  allowed.

### TensorFVM

- Public exports include `SolverConfig`, `SimpleSolver`, and `SolverResult`.
  `SimpleSolver(config).solve()` returns the result of a complete SIMPLE solve;
  the standard API is not a one-global-step lifecycle like TensorDEM.
- `SolverConfig.time_step` enables a specific implicit Euler URANS mode with
  inner iterations, but the user-facing `solve()` remains a run-level entry
  point. A generic adapter should not call the entire run once per global step.
- `Cylinder3DSolver` is a separate API and currently documents Cartesian mesh
  support only. Mesh/configuration validation must remain in TensorFVM.

## Environment and dependency boundary

The inspected project metadata currently differs:

| Repository | Python requirement | Runtime dependency declaration |
|---|---|---|
| TensorLBM | >=3.11 | torch >=2.0 plus numerical/utility packages |
| TensorDEM | >=3.10 | torch pinned to ==2.14.1 |
| TensorFEM | >=3.11 | torch >=2.1 |
| TensorFVM | >=3.10 | torch >=2.14.1 |

TensorPhysics core remains dependency-free and Python >=3.10. For a combined
environment, Python >=3.11 is the immediate common floor, but dependency
resolution should be verified against the exact selected releases. TensorDEM's
exact PyTorch pin is a packaging constraint worth revisiting in that repository;
TensorPhysics must not force-install all solver extras by default.

## Integration gates before a physical coupled run

1. Agree on one physical clock and explicit solver subcycling.
2. Specify field shape, coordinate frame, units, geometry version, and data
   ownership for every exchanged quantity.
3. Demonstrate force and moment resultants across each interface.
4. Check action/reaction signs and distinguish hull-fluid, hull-ice and
   fluid-ice force ledgers so the same force is not counted twice.
5. Demonstrate mass/momentum or energy balance appropriate to the operator.
6. Run zero-load, no-ice, rigid-ice, and open-water controls before interpreting
   combined resistance.
7. Record exact solver revisions, parameters, raw outputs and quantitative
   tolerances. Interface tests alone do not establish physical accuracy.
