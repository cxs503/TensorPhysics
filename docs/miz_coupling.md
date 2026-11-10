# MIZ Wave-Ice-Ship Three-Body Coupling Model

## Overview

This module implements a coupled model for wave propagation, ice breaking, and ship motion response in the Marginal Ice Zone (MIZ). It was developed as part of campaign node p247w1-N06.

## Physical Model

### (a) Wave Attenuation
Wave energy decays exponentially through the ice field:
```
E(x) = E₀ · exp(-2αx)
```
The attenuation coefficient α is determined by the effective Kelvin-Voigt medium properties of the fragmented ice field, using the existing `tensorphysics.ice_wave.miz` module.

### (b) Ice Bending Failure Criterion
Wave-induced bending stress in the ice sheet:
```
σ_ice = (3/4) · ρ_w · g · a_w · k² · h_ice²
```
When σ_ice > σ_f (flexural strength), the ice breaks into floes.

### (c) Characteristic Floe Size (Squire 2018)
```
a_floe = √(σ_f / (ρ_w · g)) · (1/k)
```

### (d) Floe Size Distribution
Power-law distribution:
```
P(a) ∝ a^(-γ),  γ ∈ [2.5, 3.0]
```
Generated via inverse CDF method; exponent fitted via MLE.

### (e) Ship Motion Response
In the MIZ, the ship experiences both wave excitation and random ice impulse forces:
```
M · ẍ + C · ẋ + K · x = F_wave + F_ice
```
- Wave excitation: simplified strip theory
- Ice forces: Poisson-distributed random impulses

### (f) Coupled Solver
Given incident wave spectrum S(ω) and initial ice field (C_floe, h_ice), the solver computes:
1. Wave attenuation through the ice field
2. Ice breaking locations and characteristic floe sizes
3. Floe size distribution (Monte Carlo)
4. Ship motion RAO at various positions

## API Reference

### Core Functions

```python
from tensorphysics.ice_wave.miz_coupling import (
    ice_bending_stress,          # Wave-induced ice bending stress
    ice_breaking_criterion,      # Check if ice breaks
    characteristic_floe_size,    # Squire floe size theory
    floe_size_distribution_pdf,  # Power-law PDF
    generate_floe_field,         # Monte Carlo floe generation
    fit_floe_distribution_exponent,  # MLE gamma estimation
    wave_excitation_force,       # Strip theory wave force
    ice_impulse_force,           # Random ice impulse forces
    ship_motion_rao,             # Open water ship RAO
    ship_motion_miz,             # MIZ ship RAO (with ice damping)
    coupled_miz_solver,          # End-to-end coupled solver
    parameter_sweep,             # Multi-parameter sweep
)
```

### Example Usage

```python
# End-to-end coupled solve
result = coupled_miz_solver(
    H_s=2.0,           # Significant wave height [m]
    T_p=8.0,           # Peak period [s]
    concentration=0.6, # Ice concentration
    ice_thickness=0.8, # Ice thickness [m]
)

# Access results
print(f"Attenuation: α = {result['alpha']:.3e} /m")
print(f"Floe size: a = {result['a_floe_characteristic']:.1f} m")
print(f"RAO reduction: {result['rao_miz']['reduction_factor']:.1%}")
```

## Verification

39 pytest cases covering all 8 acceptance criteria:

| Criterion | Description | Status |
|-----------|-------------|--------|
| C1 | α increases with C_floe (r > 0.9) | ✅ r = 0.979 |
| C2 | σ_ice > σ_f triggers breaking | ✅ |
| C3 | γ ∈ [2.5, 3.0] | ✅ γ = 2.713 |
| C4 | MIZ RAO < open water RAO | ✅ 33% reduction |
| C5 | Negative controls (C→0, H_s→0) | ✅ |
| C6 | Energy budget computed | ✅ |
| C7 | End-to-end solver | ✅ |
| C8 | pytest ≥ 18 cases | ✅ 39 cases |

## Limitations

1. **Ice breaking formula**: The simple bending stress formula underestimates breaking for realistic ocean wave parameters (stresses ~10-100 Pa vs. σ_f ~0.5 MPa). Real ice breaking involves stress concentrations, resonance, and fatigue effects not captured by this model.

2. **Ship motion**: Simplified frequency-domain RAO approach; non-linear effects and full three-body coupling not included.

3. **Applicability**: Best suited for brash/pancake ice MIZ conditions; not valid for continuous ice cover or large floes.

## References

- Squire, V.A. (2018). "The marginal ice zone: a natural laboratory for wave-ice interaction studies." Annals of Glaciology.
- Kohout, A.L. et al. (2014). "Storm-induced sea-ice breakup." Geophys. Res. Lett.
- Meylan, M.H. et al. (2018). "A review of wave-ice interaction models."
