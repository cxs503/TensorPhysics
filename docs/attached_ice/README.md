# Underwater Attached Ice Module

## Overview

The `attached_ice` module models the growth and detachment of ice accreted on
submerged hull surfaces from frazil-laden flow, and maps the resulting ice
layer to hydrodynamic roughness and resistance penalties.

## Physical Model

### Frazil Transport (flux.py)

Frazil ice particles suspended in the water column are transported to the hull
surface by:
1. **Turbulent diffusion** in the boundary layer (characterised by friction
   velocity `u_star`)
2. **Gravitational settling** (Stokes regime for small frazil crystals)

The net deposition velocity is the quadrature sum of both mechanisms:
```
v_deposition = sqrt((beta * u_star)^2 + v_settle^2)
```

The mass flux is:
```
J = C_f * rho_ice * v_deposition
```

### Ice Growth (growth.py)

The ice layer grows by freezing of frazil particles that impact and stick to
the surface, and by conductive freezing at the ice-water interface. The
governing energy balance ODE is:
```
rho_ice * L_f * dh/dt = q_cond - q_conv + q_frazil
```

where:
- `q_cond = k_ice * dT / h_ice` (conduction through ice)
- `q_conv = h_c * (T_water - T_freeze)` (convective heat from water)
- `q_frazil = eta_stick * J_mass * L_f` (latent heat from frazil)

**Stefan limit**: When frazil flux is zero and convection is negligible,
the solution reduces to the classical Stefan problem:
```
h(t) = sqrt(2 * k_ice * dT * t / (rho_ice * L_f))
```

### Detachment (detach.py)

Ice detaches when the wall shear stress exceeds the adhesive strength:
```
tau_wall = rho_water * u_star^2
tau_adhesive = tau_bond / (1 + h_ice / h_ref)
```

Detachment occurs when `tau_wall > tau_adhesive`.

### Roughness (roughness.py)

The equivalent sand-grain roughness is:
```
k_s = k_s0 + alpha_ice * h_ice^beta_ice
```

The ITTC-1978 roughness allowance is:
```
Delta C_f = 0.004 * [(k_s/L)^(1/3) - (k_s0/L)^(1/3)]
```

## Verification Criteria

| Criterion | Description | Target |
|-----------|-------------|--------|
| C1 | Thermal balance closure | Residual < 2% |
| C2 | Stefan limit | Error < 3% |
| C3 | Negative control | h_ice < 1e-15 |
| C4 | Detachment monotonicity | 200 points, 0 violations |
| C5 | ITTC-1978 consistency | Error < 1% |
| C6 | Adfreeze interface | Relative diff < 1e-9 |
| C7 | pytest coverage | ≥ 15 tests, all green |

## Usage Example

```python
from tensorphysics.ice.attached_ice import (
    frazil_impact_flux,
    IceGrowthODE,
    detachment_state,
    roughness_state,
    DEFAULT_BOND_STRENGTH_PA,
)

# Step 1: Compute frazil flux
flux = frazil_impact_flux(c_f=1e-4, u_star=0.05)

# Step 2: Solve growth ODE
ode = IceGrowthODE(
    undercooling_k=2.0,
    u_star=0.05,
    c_f=1e-4,
    eta_stick=1.0,
)
result = ode.solve_euler(t_max=3600.0, n_steps=2000)

# Step 3: Check detachment
det = detachment_state(
    u_star=0.05,
    h_ice=result.h_final,
    tau_bond=DEFAULT_BOND_STRENGTH_PA,
)

# Step 4: Compute roughness penalty
rough = roughness_state(h_ice=result.h_final)
print(f"Ice thickness: {result.h_final*1000:.2f} mm")
print(f"Delta C_f: {rough.delta_cf:.6e}")
```

## Thin Interface for ship-performance-platform

The module exports a thin interface for consumption by the resistance/propulsion
chain:

```python
# Output: dict with h_ice, k_s, delta_cf, delta_t_over_t
def compute_ice_penalty(h_ice, k_s0, ship_length):
    from tensorphysics.ice.attached_ice import roughness_state
    state = roughness_state(h_ice=h_ice, k_s0=k_s0, ship_length=ship_length)
    return {
        "h_ice_m": state.h_ice,
        "k_s_m": state.k_s,
        "delta_cf": state.delta_cf,
        "delta_t_over_t": state.delta_t_over_t,
    }
```

## References

- Camfield (1980): Frazil ice accumulation on structures
- Daly & Colbeck (1986): Frazil ice transport and deposition
- Frederking (1984): Adfreeze shear strength of ice on steel
- ITTC-1978: Recommended procedures for resistance tests
