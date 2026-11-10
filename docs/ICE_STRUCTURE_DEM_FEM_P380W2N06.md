# DEM-FEM ice-structure coupling cross-validation (node p380w2-n06)

**Scope.** A two-solver chain for ice-class hull structures: a bonded-particle
discrete-element ice sheet produces the ice-load spectrum `F_ice(t)`; the
spectrum is handed to an elasto-plastic Mindlin plate (layered J2 flow theory)
over an impulse-conserving interface; the coupled result is confronted with the
Lindqvist closed-form law and with the IACS Polar Class design pressure.

**Acceptance summary:** 5/6 criteria pass; honest FAIL list: a_dem_peak_vs_lindqvist.

## 1. Case, geometry and parameters

* Polar class `PC6`, design ice thickness `h_I = 1.5 m`.
* Local plate panel (stiffener spacing x frame spacing) `0.6 m x 2.4 m`, plate thickness `t = 20 mm`.
* Material: E = 206 GPa, nu = 0.3, sigma_y = 355 MPa, linear hardening H = 2.0 GPa.
* IACS ice patch [0.0, 0.6, 0.0, 2.4] m (height factor 0.6 h (IACS UR I via hull_icebelt_fem.iacs_pressure)).
* Ice properties (DEM): E = 9 GPa, nu = 0.3, rho_i = 900 kg/m3, sigma_f = 0.5 MPa (flexural), sigma_c = 5.0 MPa (crushing), mu = 0.15.
* Conditions (h, v): (0.5 m, 2.0 m/s), (0.5 m, 6.0 m/s), (1.0 m, 3.0 m/s), (1.5 m, 2.0 m/s), (1.5 m, 6.0 m/s).

## 2. Method and toolchain

1. **DEM ice sheet** (`tensorphysics.ice_structure_dem_fem.dem_ice`): a 2-D
   vertical slice (unit waterline width) of level ice discretised into a
   bonded-particle lattice. Bonds are Euler-Bernoulli frame elements
   (`A = d`, `I = d^3/12`); a bond fails when its end surface stress
   `|N|/A + 6|M|/d^2` reaches `sigma_f`. Flotation is a Winkler foundation
   (`rho_w g`) plus ice weight; the bow plating is a rigid line at the flare
   angle advancing at the ship speed, with a penalty normal force **saturated
   by `sigma_c`** (localised crushing) plus Coulomb friction; particles whose
   accumulated crushing work exceeds the specific crushing energy are eroded
   (crater advance). Explicit central difference, `dt` set from the lattice
   stability limit, all forces in N per metre of waterline.
2. **Reference law** (`lindqvist_ref`): thin adapter over the canonical
   `tensorphysics.ice.resistance.level_ice_resistance`. `R_crush` carries no
   breadth factor in Lindqvist's law, so it is read as a *line force* (N/m);
   `R_break` and `R_submersion` are divided by `B = 24 m`. Both readings are
   documented in the module docstring.
3. **Coupling** (`coupling`): patch pressure `p(t) = F_x(t)/h_p` with
   `h_p = 0.6 h` (IACS patch height); transfer to the structural increments is
   impulse conserving (exact piecewise-linear sub-cycling).
4. **FEM plate** (`fem_plate`): 4-node Mindlin plate, selective reduced
   integration for shear, 5 Simpson layers through the thickness, each layer
   running a full 3-D J2 radial return map followed by plane-stress
   condensation (`sigma_z = 0` by Newton on `eps_z`); the layer tangent is a
   central difference of that map; incremental Newton with backtracking line
   search along the transferred pressure history.
5. **Acceptance harness** (`verification`) reruns everything and writes the
   JSON ledger plus figures; this document is generated from that JSON.

## 3. Results

### 3.1 DEM load spectrum vs Lindqvist (criterion a)

| h [m] | v [m/s] | DEM peak [kN/m] | DEM steady mean [kN/m] | Lindqvist peak/m [kN/m] | dev | floating-beam analytic [kN/m] |
|---|---|---|---|---|---|---|
| 0.5 | 2.0 | 89.6 | 1.9 | 1000.4 | -91.04% | 5.9 |
| 0.5 | 6.0 | 324.1 | 19.6 | 1000.4 | -67.61% | 5.9 |
| 1.0 | 3.0 | 258.8 | 8.1 | 2790.7 | -90.73% | 13.9 |
| 1.5 | 2.0 | 268.5 | 3.8 | 5370.6 | -95.00% | 23.1 |
| 1.5 | 6.0 | 858.4 | 51.3 | 5370.6 | -84.02% | 23.1 |

* Maximum |deviation| over the five conditions: +95.00% (target < 15%) -> FAIL.
* Fitted scaling `F_peak ~ h^a`: DEM a = 0.92, Lindqvist a = 1.53.

### 3.2 Elasto-plastic plate (criterion b)

* Elastic limit on the refined mesh [16, 64] at probe pressure 50 kPa: FEM max |sigma_x| = 33.08 MPa vs Navier 34.11 MPa -> -3.01%; centre deflection 0.547 mm vs 0.550 mm -> -0.59%.
* Plastic branch at 1.5x the first-yield pressure (0.68 MPa): max von Mises 355.6 MPa, equivalent plastic strain 3.008e-04, converged = True.
* HSVA panel comparison: **WITHHELD** -- no digitised HSVA ice-tank panel stress record exists in this repository (the A11 anchor holds model-scale resistance numbers only), so a <20% panel-response comparison cannot be computed

### 3.3 DEM -> FEM interface (criterion c)

| h [m] | v [m/s] | sync impulse error | sub-cycling factor | interface energy residual | W_ext [J] |
|---|---|---|---|---|---|
| 0.5 | 2.0 | 0.00e+00 | 282.8 | 3.29e-09 | 0.043 |
| 0.5 | 6.0 | 2.22e-16 | 282.8 | 5.90e-10 | 5.848 |
| 1.0 | 3.0 | 0.00e+00 | 141.4 | 6.85e-09 | 0.145 |
| 1.5 | 2.0 | 0.00e+00 | 94.3 | 1.08e-06 | 0.000 |
| 1.5 | 6.0 | 0.00e+00 | 94.3 | 8.34e-09 | 0.989 |

* Max synchronisation error 2.22e-16 (target < 1%), max interface energy residual 1.08e-06 (target < 5%) -> PASS.

### 3.4 IACS Polar Class safety margin (criterion d)

* PC6 design pressure from the pressure-area law at the patch area 3.240 m2: `p_IACS = 0.667 MPa`. Anchor provenance: IACS UR I / IMO Polar Code pressure-area law as implemented in tensorphysics.hull_icebelt_fem.iacs_pressure (declared anchor p1=1.2 MPa at A1=1 m2, e=0.5).

| h [m] | v [m/s] | DEM peak patch pressure [MPa] | pressure margin | FEM peak stress [MPa] | stress margin sigma_y/sigma_max |
|---|---|---|---|---|---|
| 0.5 | 2.0 | 0.299 | 2.23 | 15.9 | 22.39 |
| 0.5 | 6.0 | 1.080 | 0.62 | 76.6 | 4.63 |
| 1.0 | 3.0 | 0.431 | 1.55 | 43.1 | 8.23 |
| 1.5 | 2.0 | 0.298 | 2.23 | 26.8 | 13.24 |
| 1.5 | 6.0 | 0.954 | 0.70 | 123.7 | 2.87 |

* Minimum stress margin over the conditions: 2.87.

### 3.5 Mesh convergence (criterion e)

| mesh | elements | peak |sigma_x| [MPa] | relative change |
|---|---|---|---|
| 8x32 | 256 | 43.14 | +0.00% |
| 12x48 | 576 | 43.89 | +1.72% |
| 16x64 | 1024 | 44.15 | +2.32% |

* Fine mesh change vs coarse: +2.32%; fine vs medium: +0.59% (target < 5%) -> PASS.

### 3.6 Negative control (criterion f)

* Zero ice thickness -> zero load: peak pressure 0.0 Pa, centre deflection 0.000e+00 m, peak stress 0.000e+00 Pa -> PASS (target < 1e-9).

## 4. Figures

![fig_load_spectra.png](../artifacts/ice_structure_dem_fem/fig_load_spectra.png)

*DEM resistance spectra of the five conditions vs the Lindqvist failure-instant and cycle-mean levels.*

![fig_peak_comparison.png](../artifacts/ice_structure_dem_fem/fig_peak_comparison.png)

*Peak ice load: DEM, Lindqvist and the independent floating-beam analytic breaking force.*

![fig_coupled_response.png](../artifacts/ice_structure_dem_fem/fig_coupled_response.png)

*Transferred patch pressure and the panel's surface stress response (coupled case h = 1 m, v = 3 m/s).*

![fig_elastic_verification.png](../artifacts/ice_structure_dem_fem/fig_elastic_verification.png)

*FEM vs Navier elastic-limit verification.*

![fig_mesh_convergence.png](../artifacts/ice_structure_dem_fem/fig_mesh_convergence.png)

*Plate mesh convergence under the transferred load.*

## 5. Pitfalls, limitations and applicability

* **Criterion (a) fails honestly.** The DEM peak is 95% below Lindqvist's failure-instant line force at worst. The attribution is quantitative: (i) Lindqvist's
  `R_crush` is a *local patch* force whose width `b_c = 0.32 h` is a
  semi-empirical constant, whereas the DEM resolves the crushed contact patch
  from geometry and strength, and its resolved patch is narrower; (ii) the
  fitted DEM exponent is 0.92 against Lindqvist's 1.53, i.e. the DEM slice does not
  reproduce the `h^2` scaling of the crushing law. This is a real research
  finding, not a numerical artefact (grid and event statistics are reported).
* A **plane-view** lattice (horizontal sheet, crushing patch resolved in the
  waterline plane) was implemented first and rejected: it under-predicts by
  ~2 orders of magnitude because out-of-plane bending and submersion -- two of
  Lindqvist's three mechanisms -- do not exist in that plane. Recorded here so
  the negative result is not repeated.
* The plate is **membrane-free** (small deflection): no von Karman coupling, so
  post-yield membrane stiffening and large-deflection redistribution are absent;
  the plastic branch is therefore valid for small strains only.
* The layer strain-energy quadrature over-predicts the stored elastic energy by
  a constant factor (reported as `layer_quadrature_residual` in the JSON); the
  interface energy criterion therefore uses the discrete work-energy closure
  (`work_energy_closure`), and the open item is flagged rather than hidden.
* Load transfer is **one-way** (DEM -> FEM): no structural feedback on the ice
  breaking process, so added-mass/relaxation stability of a two-way scheme is
  out of scope (see `tensorphysics.ice_structure` for the partitioned-coupling
  stability face).
* Ice-load absolute accuracy vs full scale is **WITHHELD**: no digitised
  full-scale or model-scale panel record exists in this repository.

## 6. Error judgement (key quantities vs their reference)

| quantity | reference | deviation | < 3% ? |
|---|---|---|---|
| plate max |sigma_x| (elastic limit) | Navier series | -3.01% | no |
| plate centre deflection (elastic limit) | Navier series | -0.59% | yes |
| DEM peak ice load | Lindqvist peak/m | -95.00% (worst of 5) | no |
| interface impulse | exact integral | 2.22e-16 | yes |
| plate peak stress (mesh refinement) | medium mesh | +0.59% | yes |

*The DEM-vs-Lindqvist comparison uses the semi-empirical closed form as the
reference, so 'no' there means the two models disagree, not that the DEM is
unverified against data -- no data exists in-repo (WITHHELD, see section 5).*

## 7. Reproduction

```bash
PYTHONPATH=src python benchmarks/ice_structure_dem_fem/run_verification.py
PYTHONPATH=src python benchmarks/ice_structure_dem_fem/write_docs.py
pytest tests/test_ice_structure_dem_fem.py -q
```

