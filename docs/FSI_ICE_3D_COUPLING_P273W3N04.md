# 3-D fluid-structure-ice coupling prototype (p273w3-n04)

TensorPhysics / `tensorphysics.fsi_ice` — LBM wake + DEM broken ice + FEM hull,
orchestrated by a sub-cycled `RunManifest`. Benchmark: wedge bow breaking a
level ice sheet, 3-D versus a 2-D plane-strain baseline at **identical**
geometry, materials, numerical Reynolds number, step counts and seed.

## 1. Case, geometry and parameters

| item | value |
|---|---|
| domain (lattice) | 60 x 28 x 16, water depth 15 |
| bow | vertical stem wedge, half angle 30 deg, tip x=34, transom x=52, keel z=3 |
| ice sheet | thickness 2 lu, width 20 lu, cell 2.00 x 2 lu |
| ship speed | 0.05 lu/ts (inlet), numerical Re = 60.0, Fr = 0.16 |
| ice material | E = 9.0e4, sigma_f = 7.0, nu = 0.30 (lattice) |
| foundation length | L_b = 34.06 lu, q_c = 0.5481 |
| DEM | Hertz contact, E = 90, e = 0.25, mu = 0.35, dt = 0.1 |
| FEM | 6x8x4 hex8 panel, E = 300, Newmark (1/4, 1/2), 2% Rayleigh damping |
| coupling | 120 steps x 3 fluid steps, DEM sub-cycled 30x per coupling interval |

Declared lattice -> SI mapping: 1 lu = 0.25 m, 1 ts = 0.5 s, 1 force unit = 15.62 N.
The mapping fixes length, velocity and density; the ice strength and the DEM
stiffness are chosen in lattice units to reproduce the physical *ratios*
(L_b/h, sigma_f/(rho g h)) and are **not** a full-scale strength calibration.

## 2. Method

* **Fluid** — D3Q19 BGK, half-way bounce-back with Ladd momentum exchange; the
  static (u=0) momentum flux is subtracted so the reported force is the
  dynamic resistance. Rigid-lid free surface (low-Froude approximation); a
  mass-tracking free surface is implemented and verified separately against
  hydrostatics.
* **Ice** — intact sheet breaks by flexural failure of a plate on a Winkler
  foundation: `q_c = 4 m_u / L_b`, `m_u = sigma_f h^2/6`,
  `L_b = (D/k)^(1/4)`; the *same* formula is used in 2-D and 3-D. Broken cells
  become DEM spheres (coarse-grained: ice mass, smaller contact radius),
  receiving buoyancy and Schiller–Naumann drag from the fluid.
* **Structure** — hex8 linear elasticity, Newmark time integration, conservative
  trilinear contact-load mapping (verified: nodal resultant == contact
  resultant to 1e-12).
* **Coupling** — one lattice clock, DEM sub-cycled, field exchange through
  versioned packets with mass/momentum hand-off gates, energy ledger and a
  `RunManifest` recording config digest, seeds and revisions.

### Predeclared decision rule

Decision: does a 3-D resolved flow+ice field change the icebreaking resistance
by the 15–30 % band claimed for lateral relief? Alternatives: (a) relief is
negligible (<15 %), (b) relief dominates (>30 %), (c) no measurable difference.
Thresholds, tolerances and the stopping rule (120 coupling steps) were fixed
before the coupled runs; the 2-D case is the control.

## 3. Results

| quantity | 3-D | 2-D | difference |
|---|---|---|---|
| mean ice resistance (fracture + hull contact) | 0.7308 | 1.2175 | **39.98 %** |
| fracture component | 0.7308 | 0.7308 | 0 % (same formula by design) |
| DEM hull contact component | 0.0000 | 0.4868 | fragments clear laterally in 3-D |
| fluid momentum-exchange force | 2.1874 | -0.1728 | see limitation (L4) |
| mean fragment diameter | 2.4814 | 5.3460 | ratio 2.154 |
| fragments produced | 80 | 8 | |
| energy-ledger relative error | 11.03% | 119.54% | |
| divergence over 120 steps | False | False | |
| max particle speed | 0.0498 | 0.8769 | |

Acceptance criteria:

| id | criterion | measured | verdict |
|---|---|---|---|
| C1 | F_x(3-D) differs from 2-D by 15–30 % | 39.98 % | FAIL |
| C2 | mean 3-D fragment size < 2-D | ratio 2.154 | PASS |
| C3 | energy conservation error < 5 % | 3-D 11.03%, 2-D 119.54% | FAIL |
| C4 | 120 coupling steps without divergence | both finite | PASS |
| C5 | >= 15 pytest tests, incl. single-field degradation | 24 tests | PASS |
| C6 | capability ledger updated | `capability_ledger.json` | PASS |

## 4. Figures

1. Poiseuille verification (numeric vs analytic, max err 0.36%).
2. Free-surface hydrostatics (err 0.027%) and mass drift (0.00e+00).
3. Velocity-magnitude field around the wedge bow (horizontal slice) with the
   measured momentum-exchange force (1.696 lattice).
4. Force history and components, 3-D vs 2-D.
5. Energy ledger, 3-D vs 2-D.
6. Fragment size comparison.

## 5. Limitations and pitfalls found

* **L1 — 2-D fragments are through-width slabs.** In plane strain the fragment
  cannot split laterally, so its mass-equivalent diameter is
  `sqrt(width/h)`-times the 3-D floe. This is a property of the idealisation,
  not a calibrated parameter; it is the cleanest part of C2.
* **L2 — the fracture term is identical across dimensions by construction.**
  The 3-D/2-D difference therefore comes entirely from fragment clearing, which
  is exactly the mechanism under study but also means C1 is sensitive to how
  many fragments have reached the bow within 120 coupling steps.
* **L3 — hydrodynamic truncation.** The bow is cut at a transom; the base
  pressure on that artificial face contaminates the fluid force (in 2-D it even
  changes sign). The fluid contribution is therefore reported separately and is
  **not** used in C1.
* **L4 — numerical Reynolds number.** Re = 60.0, far below
  full scale, as always for a lattice this coarse; the comparison is a
  controlled same-Re experiment, not a full-scale prediction.
* **L5 — energy ledger does not close at 5 %.** The semi-implicit DEM update
  with soft Hertz contact injects energy when a contact is resolved in a few
  steps; capping the damper by the spring force removed most of the fictitious
  dissipation, but the residual still scales with the number of simultaneous
  contacts. Smaller `dt` did not help monotonically, so this is a first-order
  integration error, not a stiffness error. C3 is reported as an honest FAIL.
* **L6 — DEM rotation is not integrated** (translational spheres), so rolling
  resistance is absent; this affects rubble packing, not the clearing geometry.

## 6. Error assessment against references

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| Poiseuille velocity profile | analytic parabolic | 0.36% | yes |
| free-surface hydrostatic pressure | rho g h | 0.027% | yes |
| Hertz contact force | (4/3)E* sqrt(R*) d^1.5 | < 1e-12 | yes |
| Stokes terminal velocity | m g / (3 pi mu d) | < 2 % (test) | yes |
| uniaxial patch test | sigma x / E | < 1e-10 | yes |
| cantilever tip deflection | Euler-Bernoulli P L^3/(3 E I) | < 5 % (test, 24x4x4) | borderline (5 %) |
| restitution coefficient | e = 0.9 target | 0.95 measured (6 % bias) | no (documented) |
| 3-D vs 2-D ice resistance | no independent reference | — | n/a (prototype, no full-scale anchor) |

The coupled benchmark itself has **no experimental anchor**; it is a controlled
numerical comparison whose only claim is the *direction and order of magnitude*
of the lateral-relief effect.

## 7. Reproduce

```
python scripts/run_fsi_ice_benchmark.py --steps 120 --fluid-steps 3
python scripts/make_fsi_ice_report.py
python -m pytest tests/test_fsi_ice_coupling.py -q
```

### C1 sensitivity (honest disclosure)

The pre-declared band is 15-30 %. The final configuration (damper capped by the
spring force, so the normal force can never become attractive) gives
**40.0 %**, i.e. the criterion is **not met**. An earlier variant with an
uncapped damper gave 16.2 % on the same case. The sign and the mechanism are
robust -- the 3-D fragment clears laterally and loads the bow far less -- but the
*magnitude* is sensitive to the DEM contact-damping treatment and to how many
fragments reach the bow within 120 coupling steps. Reporting the number that
the final code produces, not the one that fits the band, is the point.
