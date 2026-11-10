# Conservative cross-solver field exchange (LBM/DEM → FEM) — P475W1-N05

Generated 2026-11-10T07:26:09Z from `runs/P475W1-N05/generate.py` (every number below is
injected from the computed payload `artifacts/P475W1-N05/acceptance.json`).

## 1. Physics background

Coupling a kinetic solver (LBM) or a granular solver (DEM) to a structural FE code
means transferring a *surface traction* or a *discrete contact force* onto FEM nodes.
An exchange operator is admissible only if it is

1. **conservative** — `∫_Γ t dA = Σ_i f_i` and `∫_Γ r×t dA = Σ_i r_i×f_i`;
2. **consistent** — the transferred load reproduces the tractions the FEM code would
   have assembled itself (partition of unity that reproduces linear coordinates);
3. **auditable** — the residuals are measured, not assumed, and a deliberately
   non-conservative operator must be *detected*.

Two operators are implemented (`src/tensorphysics/field_exchange/mapping.py`):

* `map_panels`: cell-centred traction panels (LBM wall data) → nodal forces by exact
  overlap-area weighting. Each panel/target-cell intersection is clipped
  (Sutherland–Hodgman) and integrated with a degree-2 triangle rule, which integrates
  bilinear shape functions exactly, so `f_i = ∫_Γ t N_i dA`. Because the bilinear
  basis reproduces linear coordinates (`Σ_i N_i x_i = x`), force *and* moment
  conservation are algebraic identities.
* `map_contacts`: DEM contact forces → nodal forces with the bilinear partition of
  unity (exact force conservation). For a **planar interface and pure forces the
  moment is conserved exactly as well**, because the weights reproduce the
  coordinates: `Σ_i N_i (r_i × F) = (Σ_i N_i r_i) × F = r × F`. A
  **minimum-norm moment completion** is therefore only needed when the source
  carries a contact couple (rolling resistance / bonded contact), when the
  interface is curved, or when the weights do not reproduce coordinates (e.g.
  nearest-node lumping). Both branches are audited below.

### Verification chains

| case | source | exchange | target | reference |
|---|---|---|---|---|
| A | LBM D2Q9 Couette wall shear | `map_panels` (8×1 panels) | plane-stress Q4 membrane | `u = τL²/(2Et)` (exact) |
| B | LBM D2Q9 Poiseuille wall pressure | `map_panels` / `map_panels_to_basis` | BFS Kirchhoff plate (SSSS) | Navier series of the same field; 0.00406 q a⁴/D |
| C | Hertzian DEM contact forces | `map_contacts` (+ moment completion) | cantilever plate strip | Euler–Bernoulli/plate-strip closed form |

## 2. Method and settings

* Exchange: rectangular target grid, exact polygon clipping, degree-2 triangle rule;
  moment completion = minimum-norm least squares on the containing cell's 4 nodes.
* LBM: in-repo reference D2Q9 BGK channel with half-way bounce-back, Guo forcing for
  the pressure-driven case, steady state driven to a wall-shear change < 1e-8;
  lattice → SI by viscosity/velocity similarity (`LatticeScale`).
* FEM: `PlaneStressQ4` (2×2 Gauss), `PlateBFS` (16-DOF Bogner–Fox–Schmit, hard simple
  support, `w=0` on the boundary), `PlateStripBending` (cubic Hermite, rigidity
  `D = Et³/[12(1−ν²)]`). All targets accept an arbitrary nodal load vector.
* Audits: source resultants are integrated independently with 4×4 Gauss on the panel
  grid; target resultants are summed at nodes; residuals are relative to the nodal
  L1 load/moment measure.

Parameter tables (case A / B / C):

* A: water ρ=998 kg/m³, ν=1.004e-6 m²/s, gap 1 mm, U=0.05 m/s,
  ny=32, τ_relax=0.5963, dx=3.125e-05 m,
  dt=3.122e-05 s; membrane L=0.01 m, W=0.002 m,
  t=0.0005 m, E=2.100e+06 Pa, ν=0.0.
* B: same lattice scale; plate a=2.500e-04 m, b=1.000e-03 m,
  t=2e-05 m, E=2.100e+09 Pa, ν=0.3,
  D=1.538e-06 N·m, mean pressure 3.999e-02 Pa.
* C: strip L=0.1 m, width=0.02 m, t=0.002 m,
  E=2.100e+09 Pa, D=1.5385 N·m; spheres R=0.005 m,
  E=7.000e+10 Pa; Hertz forces ['0.5967', '1.0961', '0.5967'] N.

## 3. Results

### 3.1 LBM source fields

| quantity | value |
|---|---|
| Couette profile rel. L2 error vs `U y/h` | 9.080e-09 |
| Couette wall shear τ_w (measured / exact) | 5.009960e-02 / 5.009960e-02 Pa |
| Couette τ_w rel. error | 2.332e-08 |
| Poiseuille profile rel. L2 error vs parabola | 1.271e-03 |
| Poiseuille wall shear rel. error vs ρgh/2 | 1.980e-02 |
| Poiseuille pressure drop over channel | 7.9979e-02 Pa |

### 3.2 Exchange conservation

| case | force residual | moment residual |
|---|---|---|
| A panels→membrane | 1.902e-15 | 1.192e-16 |
| C contacts→plate (completed) | 1.940e-16 | 3.966e-34 |
| C contacts→plate (raw PU lumping) | 1.940e-16 | 6.904e-17 |
| C negative control (nearest node) | 0.000e+00 | 1.155e-01 |
| C contact couple (raw, force-only) | — | 1.000e+00 of the applied couple lost |
| C contact couple (completed) | — | 6.948e-15 of the applied couple lost |

The contact-couple row is the case that genuinely needs the completion: a rocking
couple of 1.0e-03 N·m cannot be
represented by nodal forces alone, the raw operator loses essentially all of it, and
the minimum-norm completion restores it to round-off.

### 3.3 Deformation vs reference

| case | model | reference | rel. error |
|---|---|---|---|
| A | u_tip = 2.385695e-09 m | 2.385695e-09 m | 2.332e-08 |
| B | w_c = -1.301553e-12 m | -1.301558e-12 m | 3.284e-06 |
| B (uniform, literature) | -4.124771e-13 m | -4.122373e-13 m | 5.818e-04 |
| C | see fig. 5 | closed form | 8.060e-06 (max over 3 contacts) |

Energy audit: plate strain vs external work 5.975e-11, strip
6.390e-08, DEM interface work defect 1.037e-05,
panel-exchange work defect 9.902e-16.

### 3.4 Criterion table

| id | criterion | value | threshold | verdict |
|---|---|---|---|---|
| C1_FORCE_PANEL | LBM wall-traction -> membrane: relative force residual | 1.902e-15 | <= 1e-06 | PASS |
| C2_MOMENT_PANEL | LBM wall-traction -> membrane: relative moment residual | 1.192e-16 | <= 1e-06 | PASS |
| C3_FORCE_CONTACT | DEM contact -> plate: relative force residual | 1.940e-16 | <= 1e-06 | PASS |
| C4_MOMENT_CONTACT | DEM contact -> plate: relative moment residual | 3.966e-34 | <= 1e-06 | PASS |
| C5_NEGCTRL_MOMENT | negative control (nearest-node lumping) moment residual must exceed the 1% band | 1.155e-01 | >= 1e-02 | PASS |
| C6_LBM_COUETTE_PROFILE | LBM Couette velocity profile vs U*y/h (relative L2) | 9.080e-09 | <= 1e-02 | PASS |
| C7_LBM_COUETTE_TAU | LBM Couette wall shear vs mu*U/h | 2.332e-08 | <= 3e-02 | PASS |
| C8_MEMBRANE_TIP | membrane tip displacement vs tau*L^2/(2*E*t) | 2.332e-08 | <= 1e-02 | PASS |
| C9_LBM_POISEUILLE_PROFILE | LBM Poiseuille profile vs parabola (relative L2) | 1.271e-03 | <= 1e-02 | PASS |
| C10_LBM_POISEUILLE_TAU | LBM Poiseuille wall shear vs rho*g*h/2 | 1.980e-02 | <= 3e-02 | PASS |
| C11_PLATE_NAVIER | mapped pressure -> plate centre deflection vs Navier series | 3.284e-06 | <= 3e-02 | PASS |
| C12_PLATE_LITERATURE | uniform-pressure plate vs tabulated 0.00406 q a^4/D | 5.818e-04 | <= 1e-02 | PASS |
| C13_CONTACT_ANALYTIC | mapped contact forces -> strip deflection vs closed form | 8.060e-06 | <= 1e-02 | PASS |
| C14_ENERGY_PLATE | plate strain energy vs external work (energy audit) | 5.975e-11 | <= 1e-06 | PASS |
| C15_ENERGY_STRIP | strip strain energy vs external work (energy audit) | 6.390e-08 | <= 1e-06 | PASS |
| C16_INTERFACE_WORK | DEM interface work defect (sum f_i u_i vs sum P_j w_j) | 1.037e-05 | <= 1e-02 | PASS |
| C17_PLATE_ENERGY_EXCHANGE | plate exchange work defect (sum f_i u_i vs int t u dA) | 9.902e-16 | <= 1e-02 | PASS |
| C18_COUPLE_COMPLETED | contact couple carried after moment completion (relative to the applied couple) | 6.948e-15 | <= 1e-06 | PASS |
| C19_COUPLE_RAW_DETECTED | contact couple lost by raw force-only lumping must be detected | 1.000e+00 | >= 1e-02 | PASS |

**Passed 19/19 criteria.**

## 4. Figures

- `artifacts/P475W1-N05/fig01_lbm_profiles.png`
- `artifacts/P475W1-N05/fig02_pressure_field.png`
- `artifacts/P475W1-N05/fig03_conservation.png`
- `artifacts/P475W1-N05/fig04_plate_deflection.png`
- `artifacts/P475W1-N05/fig05_contact_deflection.png`

## 5. Pitfalls, limitations, scope

* The LBM in this repository is a **reference implementation for generating source
  data**, not the TensorLBM production solver; the coupling verification is therefore
  about the *exchange operator*, and LBM accuracy is reported separately (C6–C10).
* The Poiseuille wall pressure is not measured from an inlet/outlet pressure BC:
  the channel is force-driven and the pressure field follows from the steady momentum
  balance `dp/dx = −ρg`, which is why `dp` is reported as a consequence of the
  balance rather than a separate measurement.
* Target grids must be **rectangular** (`QuadGrid`); warped/isoparametric faces need
  the patch inverse and are left to the mortar path in `interface_operators.mortar_nd`.
* The mapping is **static**: no time interpolation or sub-cycling is performed, and the
  coupled system is solved in one pass (no fixed-point iteration between LBM/DEM and
  FEM). A strongly coupled (added-mass dominated) problem would need sub-iteration;
  this is out of scope here and does not affect the conservation identities.
* Small-deflection (Kirchhoff/plane-stress) theory only; membrane/shear locking at
  very thin plates is not addressed.
* Contact detection and overlap evolution belong to DEM; here the overlaps are inputs
  and the Hertz force is evaluated from them.
* The moment completion re-distributes forces within the containing surface cell only;
  it is a minimum-norm correction, so it adds no net energy beyond what the
  conservation requirement demands, but it is a local (cell-wise) operation — a
  globally optimal load transfer would need a mortar/L2 projection.
* Transfer is force-based: DEM particle rotations are passed as couples only in the
  dedicated couple demonstration, not in the beam-deflection chain.

## 6. Error verdict (vs the < 3% standard)

Every gate is a *relative* error against an independent reference:
* C1_FORCE_PANEL: 1.902e-15 < 1e-06 → PASS
* C2_MOMENT_PANEL: 1.192e-16 < 1e-06 → PASS
* C3_FORCE_CONTACT: 1.940e-16 < 1e-06 → PASS
* C4_MOMENT_CONTACT: 3.966e-34 < 1e-06 → PASS
* C5_NEGCTRL_MOMENT: 1.155e-01 < 1e-02 → PASS
* C6_LBM_COUETTE_PROFILE: 9.080e-09 < 1e-02 → PASS
* C7_LBM_COUETTE_TAU: 2.332e-08 < 3e-02 → PASS
* C8_MEMBRANE_TIP: 2.332e-08 < 1e-02 → PASS
* C9_LBM_POISEUILLE_PROFILE: 1.271e-03 < 1e-02 → PASS
* C10_LBM_POISEUILLE_TAU: 1.980e-02 < 3e-02 → PASS
* C11_PLATE_NAVIER: 3.284e-06 < 3e-02 → PASS
* C12_PLATE_LITERATURE: 5.818e-04 < 1e-02 → PASS
* C13_CONTACT_ANALYTIC: 8.060e-06 < 1e-02 → PASS
* C14_ENERGY_PLATE: 5.975e-11 < 1e-06 → PASS
* C15_ENERGY_STRIP: 6.390e-08 < 1e-06 → PASS
* C16_INTERFACE_WORK: 1.037e-05 < 1e-02 → PASS
* C17_PLATE_ENERGY_EXCHANGE: 9.902e-16 < 1e-02 → PASS
* C18_COUPLE_COMPLETED: 6.948e-15 < 1e-06 → PASS
* C19_COUPLE_RAW_DETECTED: 1.000e+00 < 1e-02 → PASS

All deformation/consistency criteria are far below 3%; conservation residuals are at
round-off level (1.9e-15 … 4.0e-34), and the
negative control is correctly rejected at 11.5% moment error.
