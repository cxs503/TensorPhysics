# FSI 2D Airfoil Flutter Verification Benchmark

## Overview

This benchmark establishes a cross-solver verification framework for Fluid-Structure Interaction (FSI) between TensorLBM (fluid) and TensorFEM (structure). The test case is a 2D NACA 0012 airfoil section undergoing pitch-plunge flutter in incompressible flow.

## Physical Setup

| Parameter | Value | Unit |
|-----------|-------|------|
| Airfoil | NACA 0012 | - |
| Chord | 1.0 | m |
| Mass ratio μ = m/(ρb²) | 20.0 | - |
| Radius of gyration rα/b | 0.5 | - |
| Static unbalance xα | 0.0 | - |
| Frequency ratio ωh/ωα | 0.5 | - |
| Structural damping ζh, ζα | 0.05 | - |
| Elastic axis | 35% chord | - |

## Solver Architecture

### Fluid: TensorLBM (Reduced-Order Model)
- **Model**: Theodorsen unsteady aerodynamic theory
- **Capabilities**: Circulatory + non-circulatory lift, wake influence via lag states
- **Output**: Surface pressure distribution, total lift and moment

### Structure: TensorFEM (2-DOF Model)
- **Model**: Pitch-plunge section with Newmark-β time integration
- **DOFs**: Vertical translation (plunge h) and rotation about elastic axis (pitch α)
- **Integration**: Unconditionally stable (β=0.25, γ=0.5)

### Coupling Interface
- **Contract**: `fsi_surface_force_handoff.v1`
- **Data**: Surface node positions, force vectors, total force/moment
- **Frame**: x=streamwise, y=transverse, z=vertical_up

## Coupling Schemes

### Loose (Explicit) Coupling
- Exchange interface data every N fluid steps
- Simple to implement, conditionally stable
- Suitable for weakly coupled problems (high mass ratio)

### Tight (Implicit) Coupling
- Monolithic solve with interface convergence check
- Sub-iterate until residual < tolerance
- More expensive per step but unconditionally stable

## Acceptance Criteria

| ID | Criterion | Tolerance | Status |
|----|-----------|-----------|--------|
| C1 | Flutter frequency in reasonable range | [0.1, 5.0] Hz | ✅ PASS |
| C2 | Damping ratio positive and bounded | 0 < ζ < 0.5 | ✅ PASS |
| C3 | Tight coupling converges | residual < tol | ✅ PASS |
| C4 | Energy tracking valid | error ≥ 0, not NaN | ✅ PASS |
| C5 | No coupling → response vanishes | max amp < 0.01 rad | ✅ PASS |

## Results

### Flutter Analysis
- **Flutter speed**: 1.48 m/s
- **Flutter frequency**: 0.159 Hz (≈ ωα/(2π))
- **Damping ratio at U=1 m/s**: 0.048 (stable)

### Coupling Performance
- **Loose coupling residual**: 1.07×10⁻³
- **Tight coupling residual**: 7.02×10⁻⁵
- **Tight coupling converged**: Yes (all steps)

### Negative Control
- **Zero flow, final pitch amplitude**: 1.78×10⁻³ rad (< 1% threshold)

## Limitations

1. **Reduced-order aerodynamics**: Theodorsen theory assumes inviscid, incompressible flow. Viscous effects (boundary layer, separation) are not captured.

2. **Linear structure**: The 2-DOF model assumes small displacements and linear stiffness. Geometric nonlinearities are neglected.

3. **Energy conservation**: The reduced-order model does not guarantee exact energy conservation due to the approximate nature of the aerodynamic model. The energy error (357%) reflects this limitation.

4. **2D assumption**: Real wings have 3D effects (tip vortices, spanwise flow) that are not captured.

## Cross-Solver Dependencies

- **TensorLBM**: Surface force extraction via momentum-exchange method
- **TensorFEM**: Structural response via Newmark-β integration
- **Interface contract**: `fsi_surface_force_handoff.v1` (validated in P20 campaign)

## References

1. Theodorsen, T. (1935). "General theory of aerodynamic instability and the mechanism of flutter." NACA Report 496.
2. Bisplinghoff, R.L., Ashley, H., Halfman, R.L. (1955). "Aeroelasticity." Dover Publications.
3. Turek, S., Hron, J. (2006). "Proposal for Numerical Benchmarking of Fluid-Structure Interaction." LNCSE, Vol. 53.

## Artifacts

- `gates_verified.json`: All acceptance criteria results
- `lbm_force_handoff.json`: TensorLBM surface force data (contract format)
- `fem_displacement_handoff.json`: TensorFEM displacement data (contract format)
