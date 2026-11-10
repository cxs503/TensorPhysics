# Cross-solver field-exchange conservation audit (FEM -> DEM -> LBM)

**Campaign node** `p270w1-N10` · **repo** TensorPhysics · **module**
`tensorphysics.cross_solver_conservation` · **evidence**
`artifacts/cross_solver_conservation/audit.json` · **recompute**
`python scripts/recompute_cross_solver_conservation.py` · **re-gate**
`python scripts/recompute_cross_solver_conservation.py --check`

Verdict of the frozen criteria (digest `632b23e36e628b01...`): **PASS**

| criterion | status | value | threshold |
|---|---|---|---|
| C1_fem_dem_mapping | PASS | 2.998e-03 | 1.0e-02 |
| C2_fem_dem_energy_closure | PASS | 3.867e-05 | 1.0e-02 |
| C3_cohesive_energy_bookkeeping | PASS | 5.528e-05 | 1.0e-02 |
| C4_crack_arrest_resolution | PASS | 4.861e-02 | 1.0e-01 |
| C5_dem_lbm_momentum_closure | PASS | 5.064e-03 | 2.0e-02 |
| C6_lbm_wall_flux_consistency | PASS | 1.271e-02 | 2.0e-02 |
| C7_lbm_momentum_identity | PASS | 7.958e-12 | 1.0e-08 |
| C8_lbm_poiseuille_profile | PASS | 7.542e-04 | 3.0e-02 |
| C9_lbm_wall_shear_vs_analytic | PASS | 1.603e-04 | 2.0e-02 |
| C10_multirate_work_error | PASS | 3.519e-06 | 1.0e-02 |
| C11_multirate_impulse_error | PASS | 7.466e-10 | 2.0e-02 |
| C12_chain_energy_closure | PASS | 5.064e-03 | 1.0e-02 |

---

## 1. Case (physics background, geometry, parameters)

A 1D synthetic fixture joins the three solvers **sequentially**: an ice plate
strip is bent in the FEM until a fracture face forms, the face is handed to a
bonded-particle (DEM) chain that propagates the crack dynamically and gives a
fragment velocity, and that velocity is imposed as a moving wall on an LBM
channel whose wall momentum flux is measured three independent ways.

| quantity | symbol | value |
|---|---|---|
| Young's modulus | E | 9e+09 Pa |
| density | rho | 917 kg/m^3 |
| plate width / thickness / span | b / h / L | 0.100 / 0.005 / 0.500 m |
| flexural strength | sigma_f | 7e+05 Pa |
| fracture energy | G_f | 0.08 J/m^2 |
| strength load | P_f | 0.5833 N |
| handoff load | lambda P_f | 1.5 P_f = 0.8750 N |
| LEFM release rate at handoff | G | 0.1021 J/m^2 (G/G_f = 1.276) |
| Irwin cohesive length | l_pz = E G_f/sigma_f^2 | 1.469 mm |
| LBM channel | nx x ny, tau, nu | 8 x 32, 0.8, 0.100 |
| exchange rates | FEM / DEM / LBM | 100 / 1000 / 10000 Hz |

## 2. Method

### 2.1 FEM stage -- bending stress field and failure face

Displacement-based Euler-Bernoulli FE (cubic Hermite, 400 elements, nodally
exact for a tip load).  Tip deflection error against `P L^3/(3 E I)`:
5.19e-06 %; strain-energy error against
`P^2 L^3/(6 E I)`: 1.04e-05 %.  The failure face is the critical
section: `sigma_max = 1.050 MPa`, the traction profile
`sigma_xx(0,y)` is antisymmetric with peak `M/Z`, and the fixed-grip LEFM
release rate `G = M^2/(2 E I b)` is `0.1021 J/m^2`.

Two consistency checks are imposed *before* the handoff: the strength criterion
(`sigma_max > sigma_f`) and the energy criterion (`G > G_f`).  The bilinear
cohesive law additionally requires `G_f > sigma_f^2 h/(6E) = 0.04537 J/m^2`,
otherwise the softening branch would not exist (`theta_c < theta_0`); this is
the discrete resolution constraint that fixes the parameter window.

### 2.2 DEM stage -- bonded-particle chain with cohesive hinges

A lumped-mass chain (nodes at `i l_b`) with curvature hinges
`theta_j = (w_{j+1}-2w_j+w_{j-1})/l_b`, `k_theta = E I/l_b`, so the chain
elastic energy *is* the beam bending energy.  Each hinge follows a bilinear
cohesive law with peak moment `M_f = sigma_f I/(h/2)` and **total** area
`0.5 M_f theta_c = G_f b l_b` (exact Griffith).  The secant damage
`D = 1 - (theta_0/theta)(theta_c-theta)/(theta_c-theta_0)` is used (the naive
linear-in-`theta` damage leaves a 1.2x inconsistency between stored, dissipated
and work -- caught by the self-audit C3).  Integration is velocity Verlet with
nodal damping and a per-step damage rate limit (viscous regularisation of the
snap-back mode).  Ledger:

```
U_ref = KE + U_el + E_diss + E_damp + E_handoff
```

### 2.3 LBM stage -- D2Q9 moving wall with exact momentum bookkeeping

D2Q9 BGK, periodic in x, half-way bounce-back walls, Guo forcing.  The ice
fragment slides with `v_ice(t)`, which is imposed as the wall velocity; the ice
equation of motion is `m dv/dt = tau_wall A` with `tau_wall` measured from the
LBM velocity field, i.e. a genuine two-way weak coupling.  Three independent
momentum measurements: the discrete Ladd exchange, the wall shear integral
`integral tau dt`, and the ice momentum deficit `Delta P_ice`.  The discrete
identity `P(t)-P(0) = force - wall impulse` is checked every step
(C7, relative error 8.0e-12).

Dimensional note: the specification writes `int rho v_ice dA = int tau_wall dt`,
but the two sides differ by a length (`kg/s` vs `kg/(m s)`).  The
dimensionally-consistent pairing used here is *impulse per unit wall area*
`int tau dt` against the momentum deficit of the ice, and `int rho v_ice A dt`
for the mass flux.  The spec's literal equality cannot hold as written; this is
recorded rather than silently "fixed".

### 2.4 Multirate exchange audit

The measured `v_ice(t)` / `tau(t)` history from the coupled run (LBM clock
10 kHz) is down-sampled to 1 kHz (DEM) and 100 Hz (FEM) and reconstructed with
linear and natural-cubic spline.  The audited functionals are the exchanged
impulse and the exchanged work `int tau v dt`, not the pointwise L2 error.  An
analytic broadband signal (400/1200/2600 Hz) provides the order-of-accuracy
measurement in the resolved regime and the aliasing regime separately.

## 3. Results

### 3.1 FEM -> DEM fracture handoff

| n_bonds | l_b [mm] | l_b/l_pz | dt [s] | steps | mapping err | ledger closure | path-vs-analytic | broken bonds | crack [mm] |
|---|---|---|---|---|---|---|---|---|---|
| 500 | 1.000 | 0.68 | 1.38e-09 | 14480 | 3.00e-03 | 3.87e-05 | 5.53e-05 | 125 | 125.0 |
| 1000 | 0.500 | 0.34 | 3.45e-10 | 40956 | 1.50e-03 | 3.09e-05 | 4.25e-05 | 259 | 129.5 |
| 2000 | 0.250 | 0.17 | 8.63e-11 | 115840 | 7.50e-04 | 2.62e-06 | 4.56e-06 | 525 | 131.2 |

* mapping error of the conservative field projection: 0.3 % (mesh-refined, half-order in `l_b`)
* ledger closure over all meshes: **0.00387 %** (gate 1 %)
* cohesive bookkeeping (analytic vs path-integrated dissipation): 0.00553 %
* energy partition at the finest mesh: kinetic 5.88 %, residual elastic
  10.06 %, fracture dissipation 75.52 %,
  damping 0.00 %, handoff deficit 8.54 %
* crack length: 131.2 mm (finest mesh), mesh spread
  4.86 %, static LEFM arrest prediction
  57.4 mm -> the dynamic crack **overshoots** the static
  arrest length by 2.29x, which is the
  expected dynamic crack-propagation behaviour (the released strain energy must be
  absorbed as fracture energy plus fragment kinetic energy, not as a quasi-static
  energy balance).

### 3.2 DEM -> LBM momentum flux

Steady validation of the channel (C7-C9): profile error 0.0754 %,
discrete wall shear vs analytic `rho g h/2`: 0.016 %,
u_max 0.01278 vs 0.01280.

| m_ice | v_ice(0) | v_ice(T) | Delta P_ice | J_Ladd | J_shear | ice-vs-Ladd | shear-vs-Ladd | discrete balance |
|---|---|---|---|---|---|---|---|---|
| 200 | 0.020 | 0.0147 | -1.0611 | -1.0478 | -1.0611 | 1.26 % | 1.27 % | 8.0e-12 |
| 500 | 0.020 | 0.0173 | -1.3539 | -1.3470 | -1.3539 | 0.506 % | 0.509 % | 6.8e-12 |
| 1000 | 0.020 | 0.0185 | -1.4860 | -1.4822 | -1.4860 | 0.257 % | 0.257 % | 6.5e-12 |

The closure grows with the deceleration rate (largest for the lightest fragment),
as expected: the residual is the per-step difference between the macroscopic
gradient estimator and the discrete exchange integrated coherently.  Worst case
1.26 %, nominal 0.506 % (gate 2 %).

### 3.3 Multirate exchange error

| exchange | rate [Hz] | scheme | L2 rel | impulse rel | work rel |
|---|---|---|---|---|---|
| fem | 100 | linear | 4.26e-04 | 7.45e-05 | 7.00e-04 |
| fem | 100 | cubic | 4.37e-06 | 1.74e-07 | 2.39e-04 |
| dem | 1000 | linear | 1.25e-05 | 6.28e-07 | 1.17e-05 |
| dem | 1000 | cubic | 2.58e-08 | 7.47e-10 | 3.52e-06 |

Observed order in the resolved regime: linear **1.59**,
cubic **3.23** (asymptotic 2 and 4; the gap is the
finite sweep range plus the natural-spline end conditions).  In the *unresolved*
regime the error is O(1) for both schemes -- at 100 Hz the L2 error is
1.132 (linear) / 1.128 (cubic) -- i.e. no
interpolation scheme can recover a 2.6 kHz component from a 100 Hz producer; the
exchange rate, not the scheme, is the binding constraint.

Work error of the DEM -> LBM exchange (1 kHz -> 10 kHz): 3.52e-06
(cubic) / 1.17e-05 (linear), both far inside the 1 % energy gate.
FEM -> DEM at 100 Hz is the weak link: work error 7.00e-04
(linear) / 2.39e-04 (cubic) for *this* smooth signal.

### 3.4 End-to-end chain

Stage errors combined in quadrature: **0.506 %** (gate 1 %),
dominated by the DEM -> LBM momentum closure (0.506 %),
not by the energy ledger (0.00387 %).

## 4. Figures

(PNG files, also written to `artifacts/cross_solver_conservation/`:
`fig1_fem_stress_and_failure_face.png`, `fig2_dem_energy_ledger.png`,
`fig3_lbm_momentum_closure.png`, `fig4_multirate_exchange_error.png`.)

![FEM stress field, failure-face traction, LEFM G(x)](fig1_fem_stress_and_failure_face.png)

![DEM energy ledger and crack length vs resolution](fig2_dem_energy_ledger.png)

![LBM Poiseuille profile and DEM->LBM momentum closure](fig3_lbm_momentum_closure.png)

![Multirate exchange error and work error vs rate](fig4_multirate_exchange_error.png)

## 5. Notes, traps, limitations

1. **Secant vs naive damage.** The bilinear cohesive law's damage must be the
   *secant* form; the naive `(theta_max-theta_0)/(theta_c-theta_0)` makes
   `work != stored + dissipated` by up to 1.2x.  The self-audit (C3) is what
   caught it.
2. **Fracture-energy convention.** `theta_c = theta_0 + 2 Phi_f/M_f` (softening
   area only) double-counts the peak elastic triangle and gives an *effective*
   fracture energy `G_f + sigma_f^2 h/(6E)`.  `theta_c = 2 Phi_f/M_f` is the
   Griffith-consistent choice and is what is used here.
3. **Negative tangent stiffness.** A rate-independent damage law is numerically
   unstable once a hinge passes the peak (the snap-back mode grows faster than
   the lattice frequency); the per-step damage rate limit stabilises it and adds
   a small reported excess dissipation (2.34e-04 J
   at the finest mesh, i.e. process-zone hinges that are damaged but not broken).
4. **Kelvin-Voigt hinge damping was rejected**: `eta omega_max >> 1` made the
   damping moment ~300x the elastic moment at the lattice frequency and blew the
   explicit scheme up; nodal dashpots with a modal damping ratio are stable.
5. **Start-up impulse.** Charging the moving-wall start-up (0 -> v_ice) to the
   audit window inflates the momentum closure to 2-2.5 %; pre-solving the flow
   with the wall already at `v_ice(0)` removes it.  This is a coupling-protocol
   requirement, not a physics result.
6. **Model reduction.** The DEM chain is a 1D curvature-hinge reduction of the
   beam at the failure face; it carries the bending energy exactly but not the
   3D stress state near the crack front.  The fixture is a *verification* rig for
   the audit machinery, not a validated ice-fracture model.
7. **Not covered:** immersed-boundary (as opposed to moving-wall) treatment,
   multi-fragment (many-body) DEM at the wall, and 3D plate FEM.

## 6. Error verdict against the gates

| key quantity | measured | gate | pass |
|---|---|---|---|
| FEM->DEM mapping | 0.3 % | 1 % | yes |
| FEM->DEM energy closure | 0.00387 % | 1 % | yes |
| DEM->LBM momentum closure | 0.506 % | 2 % | yes |
| LBM wall flux vs analytic | 0.016 % | 2 % | yes |
| multirate work error (DEM->LBM) | 3.52e-06 | 1 % | yes |
| chain energy closure | 0.506 % | 1 % | yes |

**Is every audited quantity below its gate?** Yes -- all twelve frozen criteria
return PASS, and the worst quantity is the DEM->LBM momentum closure
(1.26 % for the lightest fragment, still
inside the 2 % momentum gate but **outside** a 1 % gate; recorded honestly).
No quantity is answered "withheld"; the crack-length comparison against the
static LEFM arrest length is *not* a pass/fail criterion because dynamic
overshoot is expected physics, and it is reported as a ratio
(2.29x) instead.
