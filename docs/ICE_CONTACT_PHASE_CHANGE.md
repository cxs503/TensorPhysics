# Ice–structure contact phase change and melt-film evolution

Domain module: `src/tensorphysics/ice_contact/`.
Verification artefacts: `artifacts/ice_contact_m1212/`.
Full report: [`docs/reports/ice_contact_phase_change_report.md`](reports/ice_contact_phase_change_report.md).

## What this module is

A self-contained, auditable **physics closure plus verification fixture** for the
three-phase interface ice / meltwater / structure. It is not a new specialist
solver kernel: it provides the relationships TensorPhysics needs to reason about
sliding ice–structure contact (friction, melt-film growth, refreezing, adhesion)
and it carries its own analytic references so that every claim is checkable.

| module | content | analytic reference |
|---|---|---|
| `properties.py` | ice Ih / water / steel property basis, Vogel water viscosity | tabulated values, 0 °C |
| `pressure_melting.py` | Clausius–Clapeyron melting line, threshold pressure | exact integral `T0 exp(Δv p / Lf)`, RK4 cross-check |
| `stefan.py` | Neumann similarity solution; 1-D enthalpy solver | `X = 2λ√(αt)`, `λ e^{λ²}erf λ = Ste/√π` |
| `film.py` | melt-film ODE (friction + Stefan sinks + drainage), steady states, adaptive integration | closed-form quadratic steady state |
| `adhesion.py` | interfacial premelting (QLL) + adhesion strength vs temperature | vdW bridging `A_H/(6πh³)`, measured range 0.2–1.5 MPa |
| `tribology.py` | Stribeck curve extended with the film parameter, load-sharing mixture | `μ = A_r μ_b + (1-A_r) ηV/(ph)` |
| `competition.py` | mechanism decomposition, melting-onset speed, parameter scan, direction gates | `V_c = (-(Δ_i+Δ_s)/(μp))²` |
| `verification.py` | acceptance harness → JSON | criteria (a)–(e) |

## Governed equations

```
dT_melt/dp = T_melt (v_water - v_ice) / L_f                     (−0.0742 K/MPa)

dh/dt = (q_fric + q_ice + q_struct)/(rho_ice L_f) − c_d h V / L
q_fric   = mu(h,p,V,T) p V
q_ice    = k_ice (T_ice − T_melt(p)) / sqrt(alpha_ice L / V)
q_struct = (T_struct − T_melt(p)) / (h/k_water + sqrt(alpha_s L/V)/k_s)

mu = A_r(Lambda) mu_b + (1 − A_r) eta(T) V / (p h),  Lambda = h/sigma_rms
```

Positive fluxes melt ice and feed the film; negative fluxes refreeze it. Pressure
melting enters only through `T_melt(p)`, which is why it matters at
asperity-scale pressures (137 MPa to melt ice at −10 °C) and is negligible for
nominal hull contact pressures.

## Physics conclusions supported by the artefacts

* Pressure melting is negligible at nominal contact pressure (0.074 K/MPa) and
  becomes the dominant mechanism only above ~137 MPa at −10 °C, or at moderate
  pressure when the ice is within a few tenths of a kelvin of melting.
* A steady melt film needs the frictional power to beat both Stefan sinks; for
  ice at −10 °C on an equilibrated slider at 1 MPa the melting-onset speed is
  0.40 m/s, so 0.1 m/s sits in the boundary-friction branch (velocity weakening).
* A cold structural counterface adds a Stefan sink that is comparable to or
  larger than the ice-side sink, so a slider that has not yet been heated by the
  contact suppresses the film entirely — the film's survival is a competition,
  not a property of ice alone.
* Steady film thickness rises with load and with speed, falls as the ice cools,
  and the mixed-lubrication feedback makes the friction coefficient rise as the
  film thins (self-regulation at low temperature).

## Reproducing

```bash
PYTHONPATH=src:. python3 benchmarks/ice_contact/run_verification.py   # criteria + figures
PYTHONPATH=src:. python3 benchmarks/ice_contact/render_report.py      # report from artefacts
PYTHONPATH=src:. python3 -m pytest tests/test_ice_contact_*.py -q     # 53 tests
```

`numpy` and `matplotlib` are only needed for the 1-D Stefan reference solver, the
verification runner and the figures (`pip install -e .[ice-contact]`).

## Limits

0-D single-node contact (no lateral flow, no Reynolds solution, no asperity
temperature distribution), empirical adhesion calibration, plug-flow drainage with
an O(1) coefficient (absolute film thickness is order-of-magnitude only), and
constant `Δv`, `L_f` in the Clapeyron relation (curvature matters above ~50 MPa).
The model is a closure for the orchestration layer, not a replacement for a
specialist contact/CFD solver.