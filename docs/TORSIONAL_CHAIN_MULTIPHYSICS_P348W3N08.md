# Ice-class drivetrain torsional chain with multiphysics time-step coordination

**Node** p348w3-n08-physics-torsional-timestep · **Repository** TensorPhysics · **Module** `tensorphysics.torsional_chain`

## 1. Case and physical background

A Polar-Class ship's propulsion train is a chain of rotary inertias coupled by torsional springs: propeller – shaft 1 – reduction gear – shaft 2 – electric motor. When the propeller mills ice, the torque is not smooth: blade impacts produce a narrow-band ripple at the blade-passing frequency whose harmonics can sit on the torsional natural frequencies. Two extra mechanisms make the problem genuinely multiphysics rather than a forced-vibration exercise:

1. **Parametric gear excitation.** Tooth-mesh stiffness pulsates once per tooth pair engagement, `k_m(theta) = k_m0 (1 + eps cos(Z1 theta))`. Because the stiffness is a function of the pinion *angle*, a constant-speed operation turns the modal equation into a Mathieu equation with a principal instability tongue at tooth-passing frequency `= 2 omega_n`.
2. **Back-EMF feedback.** The motor torque is not a prescribed load: `L di_q/dt = V - R i_q - k_e omega_m`, `T_e = k_t i_q`. The shaft speed therefore feeds back into the electrical circuit, and the electrical time constant `L/R` sets a third, much faster, stability constraint.

Parameter table (code inputs, `tensorphysics.torsional_chain.acceptance` defaults; repeated here for reproducibility):

| quantity | value |
|---|---|
| propeller inertia / motor inertia | 1200 / 800 kg m² |
| shaft 1: L, D_o, D_i, G | 6.0 m, 0.42 m, 0.18 m, 79.3 GPa |
| shaft 1 torsional stiffness (derived) | `k = G J_p / L` = 39013491 N m/rad |
| shaft 1 polar section modulus (derived) | `W_p` = 0.0140564 m³ |
| gear ratio i = Z_wheel/Z_pinion, pinion teeth | 4.0, 21 |
| mean mesh stiffness k_m0 / modulation eps | 1e9 N/m, 0.2 |
| PMSM R, L, k_t, k_e, V | 0.05 ohm, 5 mH, 1.2 N m/A, 1.15 V s/rad, 400 V |
| structural damping ratio (reference chain) | 0.05 |

## 2. Method

| layer | implementation | integrator |
|---|---|---|
| drivetrain | lumped inertias, torsional spring-dampers, gear pair with conjugate mesh torques, optional backlash dead zone, per-segment polar section modulus | classical RK4 on `[theta, omega]` |
| gear | `MeshStiffness` (periodic), `GearPair` (ratio, backlash), `ParametricMeshOscillator` (Mathieu test bed) | RK4 |
| motor | `PMSM` q-axis model + `simulate_motor_drivetrain` (torque injected at the motor node, node speed closes the back-EMF loop) | RK4 on `[theta, omega, i_q]` |
| excitation | ice-milling torque = mean + harmonic comb at blade passing frequency, `1/n` decay, thickness/speed scaling | analytic |
| coordination | `SolverRate` stability limits (CFL, structural accuracy, circuit, mesh, interface bandwidth), `TimeStepCoordinator` (synchronised / sub-cycling / super-cycling), `AdaptiveTimeStep` (growth cap + shrink on stability alert) | closed-form plans, verified by invariants |
| IACS | Polar Class alternating shear-stress limits (model parameters), S-N curve with a knee, Miner damage | analytic |

Verification anchors: every acceptance criterion compares a *simulation* against an independent closed form (eigenvalue algebra, SDOF compliance, Mathieu two-timing, analytic steady-state speed, high-resolution quadrature), never against another simulation of the same model.

## 3. Results

### C1 — torsional natural frequencies vs closed form

| configuration | simulated [Hz] | reference [Hz] | relative error |
|---|---|---|---|
| two_inertia_grounded | 28.697028 | 28.697028 | 0.000 % |
| two_inertia_free_free | 45.373985 | 45.373985 | 0.000 % |
| three_inertia_free_free_mode1 | 21.146291 | 21.146291 | 0.000 % |
| three_inertia_free_free_mode2 | 60.338554 | 60.338554 | 0.000 % |
| reflected_inertia_low_speed_mode | 29.837029 | 30.006068 | 0.563 % |

Worst error **0.563 %** against a 3 % gate → **PASS**.

### C2 — periodic mesh stiffness / parametric response

- quasi-static ripple amplitude: simulation 0.0005293 rad vs closed form 0.00052083 rad → error **1.63 %** (gate 5 %)
- peak position inside the mesh cycle: 0.5045 of a period vs theory 0.5 → error **0.45 %** of a mesh period
- point-wise agreement with the quasi-static solution: L2 error 1.24 % of the amplitude
- Mathieu growth rate at tooth passing `= 2 omega_n`: simulation 8.471 1/s vs two-timing formula 8.4823 1/s → error **0.13 %**

→ **PASS** (amplitude and peak-position gates both 5 %).

### C3 — PMSM steady speed with back-EMF feedback

| load torque [N m] | simulated omega [rad/s] | analytic omega_ss [rad/s] | relative error |
|---|---|---|---|
| 0 | 347.82843 | 347.82609 | 0.001 % |
| 20 | 347.10371 | 347.10145 | 0.001 % |
| 40 | 346.379 | 346.37681 | 0.001 % |

Worst error **0.001 %** against a 2 % gate → **PASS**. Electrical time constant `L/R` = 0.1 s, mechanical time constant `J R /(k_e k_t)` = 0.01812 s; locked-rotor current fraction at `t = L/R` = 0.6322 (theory `1 - 1/e` = 0.6321).

### C4 — ice torque → shaft stress transfer function

| frequency [Hz] | simulated stress amplitude [Pa] | closed form [Pa] | relative error |
|---|---|---|---|
| 15.783 | 1.0168e+07 | 1.0168e+07 | 0.000 % |
| 17.936 | 1.16136e+07 | 1.16137e+07 | 0.000 % |
| 20.088 | 1.38198e+07 | 1.38199e+07 | 0.000 % |
| 22.24 | 1.74872e+07 | 1.74871e+07 | 0.000 % |
| 24.392 | 2.45127e+07 | 2.45126e+07 | 0.000 % |
| 26.545 | 4.14907e+07 | 4.14906e+07 | 0.000 % |
| 28.697 | 7.11411e+07 | 7.11421e+07 | 0.001 % |
| 30.849 | 3.76128e+07 | 3.76127e+07 | 0.000 % |
| 33.002 | 2.07781e+07 | 2.0778e+07 | 0.000 % |
| 35.154 | 1.38034e+07 | 1.38034e+07 | 0.000 % |
| 37.306 | 1.01322e+07 | 1.01322e+07 | 0.000 % |
| 39.458 | 7.89434e+06 | 7.89435e+06 | 0.000 % |
| 41.611 | 6.3977e+06 | 6.3977e+06 | 0.000 % |

- worst point of the identified transfer function: **0.001 %** (gate 3 %)
- resonance peak: identified 28.697 Hz vs analytic `omega_n sqrt(1-2 zeta^2)` 28.6252 Hz → error **0.251 %**
- ice-milling line spectrum (lines follow `|H|`): worst line error **0.045 %**

→ **PASS**.

### C5 — time-step coordination and interface conservation

| sub-cycle ratio N | dt_sub [s] | interface work [J] | work residual | impulse residual |
|---|---|---|---|---|
| 2 | 0.0009524 | 84.9435 | 1.21 % | 0.35 % |
| 4 | 0.0004762 | 85.2206 | 0.89 % | 0.16 % |
| 8 | 0.0002381 | 85.3628 | 0.73 % | 0.07 % |

Macro step `dt_macro` = 0.001905 s, chosen by the interface-bandwidth limit `1/(25 f_max)` for the highest ice harmonic at 21 Hz; the reference is the same integral evaluated at `dt_macro/64`. Worst residual **1.21 %** against the 2 % gate → **PASS**.

Convergence of the interface residual with the macro step (sub-cycle ratio held at 8):

| dt_macro [s] | work residual |
|---|---|
| 0.00381 | 2.58 % |
| 0.002857 | 1.52 % |
| 0.001905 | 0.73 % |
| 0.0009524 | 0.22 % |

Observed order **1.81** (≈2, the order of a linearly reconstructed interface signal). A zero-order hold instead of linear reconstruction gives 7.26 % against 0.24 % for the same step: reconstruction order, not the sub-cycle ratio, is the first-order knob.

Coordinator plans for a five-solver situation (flow CFL, structural accuracy, gear-mesh resolution, electrical circuit, ice-load bandwidth), stability limits in seconds:

| solver | dt_max | synchronised dt | sub-cycled dt (N) | super-cycled update every |
|---|---|---|---|---|
| lbm_flow | 0.002083 | 0.001069 | 0.0018 (N=10) | 1 macro steps |
| torsion_fem | 0.001742 | 0.001069 | 0.0015 (N=12) | 1 macro steps |
| gear_mesh | 0.001187 | 0.001069 | 0.001059 (N=17) | 1 macro steps |
| motor_circuit | 0.02 | 0.001069 | 0.018 (N=1) | 16 macro steps |
| ice_load | 0.001905 | 0.001069 | 0.001636 (N=11) | 1 macro steps |

Every planned step satisfies its own stability limit (checked by `CoordinationPlan.check`), and the adaptive controller (growth cap 1.5, shrink 0.5 on a stability alert) stays below the strictest limit: `adaptive_ok` = True.

### C6 — IACS Polar Class screening and fatigue life

| polar class | alternating stress [MPa] | allowable [MPa] | utilisation | life [years] |
|---|---|---|---|---|
| PC2 | 8.903 | 32 | 0.278 | inf |
| PC4 | 8.903 | 36 | 0.247 | inf |
| PC6 | 8.903 | 44 | 0.202 | inf |

Design point: mean torque 5e+05 N m (mean shear 35.57 MPa, 0.174 of the shear yield), torque ripple 1e+05 N m, 4.5e+04 cycles/hour at the blade-passing frequency. Alternating stress stays below the S-N knee, so the predicted life is unbounded (`inf`) and the design-life gate passes with margin → **PASS**.

Discrimination check (the gate must not be vacuous): the same ripple moved onto the first torsional mode (hypothetical, 4× amplitude) gives 284.6 MPa alternating, utilisation 8.89, life 1.276e-05 years → rejected. `severe_discriminates` = True.

**Parameter status.** The Polar Class allowable stresses and the S-N reference point are `model_parameter`: the *framework* (PC1..PC7, torsional-vibration and fatigue checks) is IACS UR I, but the numbers are assembled from common classification practice and must be replaced by the applicable class society's rule values before certification use. The ice-milling mean/amplitude coefficients are likewise uncalibrated model parameters.

## 4. Figures

See the attached figures (generated by `run_torsional_chain_acceptance.py`, sha256 listed in the report payload): C1 modal comparison, C2 periodic mesh response + Mathieu growth, C3 back-EMF transient + steady speed vs load, C4 transfer function with ice spectral lines, C5 interface residual convergence + coordinated steps, C6 stress utilisation including the hypothetical resonant case.

## 5. Pitfalls, limitations and applicability

- **Internal-torque sign convention.** The first implementation subtracted the internal torque from the state derivative, which turns the spring anti-restoring. The response grew to `1e62` within one test window. The regression guard is an energy-conservation test on an undamped two-inertia chain (drift < 1e-6 over 20 periods), not a tolerance tweak.
- **Interface residual is a property of the (macro step, bandwidth) pair.** The first C5 run measured 3.7 % at `dt_macro` = 4 ms with 21 Hz content — over the 2 % gate. The fix was not to loosen the gate but to add the interface-bandwidth limit `dt <= 1/(25 f_max)` to the coordinator, which selects 1.9 ms and brings the residual to 1.21 %, with the O(2) convergence table as evidence.
- **Reflected-inertia direction.** Reflecting a low-speed-side inertia to the high-speed side divides by `i**2`; the model now places the propeller on the wheel (low-speed) side and the motor on the pinion side, and the test pins the resulting frequency to the classical formula.
- **Model scope.** Linear torsional springs, diagonal mass matrix, a single gear pair, backlash as a dead zone only, no gyroscopic/coupling effects, no shaft misalignment, no gear tooth micro-geometry; the motor is a linear q-axis model with an ideal voltage source (no PWM, no saturation, no current loop dynamics).
- **What the coordination demo is and is not.** The coordinator, stability limits, sub/super-cycle bookkeeping and interface residual are general and tested; the *transport* side is represented by the ice-torque signal plus a first-order upwind CFL surrogate in the tests, **not** by a full LBM solve. A real LBM–FEM run would exercise the same coordinator but its residual budget must be re-measured.
- **Excitation and IACS numbers are engineering assumptions** (see the parameter-status note); the ice-milling spectrum shape `1/n` and the linear thickness/speed scaling are not measured propeller data.

## 6. Error judgement against the criteria

| criterion | quantity | error | gate | `< gate` |
|---|---|---|---|---|
| C1 | first torsional natural frequency (3 configurations) | 0.563 % | 3 % | yes |
| C2 | ripple amplitude / peak position | 1.63 % / 0.45 % | 5 % | yes |
| C3 | steady speed (3 loads) | 0.001 % | 2 % | yes |
| C4 | transfer function / resonance peak | 0.001 % / 0.251 % | 3 % | yes |
| C5 | interface work residual (N=2/4/8) | 1.21 % | 2 % | yes |
| C6 | stress utilisation / life (PC2, PC4, PC6) | utilisation 0.278, life `inf` | ≤ 1, ≥ 20 yr | yes |

**Explicit answer to “are the errors below 3 %?”** Yes for the three quantitative metrics that carry a 3 % gate: C1 has 0.563 %, C4 has 0.001 % on the transfer function and 0.251 % on the resonance peak. C2 (5 % gate) reaches 1.63 %, C3 (2 % gate) reaches 0.001 %, C5 (2 % gate) reaches 1.21 %; C6 is an inequality gate, satisfied with utilisation ≤ 0.278 and infinite predicted life. No criterion is claimed on a reference that the implementation also produced.

## 7. Reproduction

```bash
python -m pytest tests/test_torsional_chain_modal.py tests/test_torsional_chain_gear.py \
                 tests/test_torsional_chain_motor.py tests/test_torsional_chain_timestep.py \
                 tests/test_torsional_chain_iacs.py -q
python run_torsional_chain_acceptance.py   # regenerates acceptance.json + figures
```

Aggregate: `all_pass` = True.

