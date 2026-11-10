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

The ITTC-1978 roughness allowance of one surface is
```
Delta C_f(k_s) = (105 * (k_s/L)^(1/3) - 0.64) * 1e-3
```
and the penalty relative to the clean hull is its increment
```
Delta C_f = 0.105 * [(k_s/L)^(1/3) - (k_s0/L)^(1/3)]
```
(the constant -0.64e-3 cancels; `ittc_1978_delta_cf` is the canonical reference
used by the tests).  Until 2026-11-10 this file quoted a coefficient of 0.004,
which under-predicted the penalty by a factor 26 — see
`docs/attached_ice/maturity_p302w2_n09.md`.

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

`attached_ice.handoff` is the single bridge to the resistance / propulsion
chain.  It emits a versioned, JSON-serialisable payload and contains no
ship-domain resistance formula of its own (the consumer supplies its total
resistance coefficient):

```python
from tensorphysics.ice.attached_ice import build_handoff, to_json

payload = build_handoff(
    undercooling_k=2.0,          # K, T_freeze - T_surface
    duration_s=7200.0,           # accretion window
    speed_m_s=3.0,               # or u_star=...
    c_f=1e-4,                    # frazil volume concentration
    water_temperature_k=273.35,  # bulk water temperature
    ice_temperature_c=-5.0,      # -> bond strength from the adfreeze model
    cf_total=0.004,              # consumer's total resistance coefficient
    n_mc=200,                    # optional P50/P90 envelope
)
open("/tmp/attached_ice.json", "w").write(to_json(payload))
```

Payload sections: `inputs` (value+unit echo), `time_series` (h_ice, k_s,
Delta C_f), `scalars` (end-of-window values, closed-form equilibrium,
detachment regime, linearisation flag), `envelope` (P50/P90 in time) and
`limitations`.

## Maturity status (p302w2-n09, 2026-11-10)

Criterion-by-criterion acceptance, the three defects found and fixed, and the
honest limitations are documented in
[`maturity_p302w2_n09.md`](maturity_p302w2_n09.md); the machine-readable verdicts
live in `artifacts/attached_ice_maturity/criteria.json`, produced by
`benchmarks/attached_ice_maturity/run_maturity.py`.

API added in the maturity pass:

| function | purpose |
|---|---|
| `stefan_with_initial` | exact similarity solution from a finite seed (C2 truth) |
| `IceGrowthODE.solve_euler(..., method="stefan_split")` | exact advance of the singular conduction term |
| `IceGrowthODE.energy_closure` | integral latent-heat balance diagnostic (C1) |
| `equilibrium_thickness` | closed-form steady state `min(h_growth, h_detach)` (C4) |
| `detachment_limited_thickness`, `detachment_regime` | stripping limit + reachability diagnostic |
| `bond_strength_from_adfreeze_pa` | MPa->Pa nerve point to `ice.adfreeze` (C6) |
| `friction_velocity_from_speed` / `speed_from_friction_velocity` | u_* <-> ship speed |
| `ittc_1978_delta_cf`, `ittc_1957_cf` | canonical ITTC references (C5) |
| `monte_carlo_envelope` | time-resolved P50/P90 envelope over the uncertain forcing |
| `build_handoff`, `to_json` | thin JSON interface for the resistance/propulsion chain |

## References

- Camfield (1980): Frazil ice accumulation on structures
- Daly & Colbeck (1986): Frazil ice transport and deposition
- Frederking (1984): Adfreeze shear strength of ice on steel
- ITTC-1978: Recommended procedures for resistance tests
