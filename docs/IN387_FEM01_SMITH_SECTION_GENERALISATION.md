# IN387-FEM01 — Smith progressive collapse: grid convergence, section-type generalisation and a real KCS midship section ultimate-moment handoff

**Node**: IN387-FEM01 (TensorPhysics, physics domain) · **Module**:
`tensorphysics.smith_section_benchmark` · **Tests**:
`tests/test_smith_section_benchmark.py` (16 tests) ·
**Acceptance record**: `artifacts/in387_fem01/in387_fem01_acceptance.json`

## 1. What this case is about

Classical Smith-method progressive collapse (Smith, Trans. SNAME 85, 1977) is the
industry-standard method for the ultimate strength of a plated structure: the
section is cut into axial fibers, every fiber carries a *load-end shortening
curve*, plane sections are imposed and the load is increased until the
integrated moment peaks.  The repository already had two specialised
implementations — a single-stiffener panel with the mid-span column equilibrium
(`elastoplastic_collapse.progressive_collapse`) and a damaged hull-girder section
with a fixed 3-fiber web (`hull_redundancy.smith`).  Neither can be used for the
questions that decide a hull-girder safety case:

* does the answer depend on the mesh, and by how much (grid convergence);
* does the method carry over between section *types* — double bottom, stiffened
  panel, box girder, a full midship section;
* what is the ultimate moment of a **real** ship's midship section (not a box
  idealisation), and how does it compare with the wave-induced demand.

This node answers those three questions with one component model
(`section.py`), one solver (`solver.py`), one convergence layer
(`convergence.py`) and a benchmark + handoff layer (`benchmarks.py`,
`handoff.py`, `acceptance.py`).

### Physical model

* **Geometry**: plating strips (horizontal, vertical, inclined: bilge, hopper,
  topside tank, hatch coaming) and stiffener groups (lumped, area-exact), with
  rational fiberisation; `n_fibers`/`n_web_fibers` are the mesh parameters.
* **Element curves** (compression positive, shortening strain):
  plating — elastic to `sigma_cr`, von Karman post-buckling branch
  `sqrt(sigma_cr E s)` up to the plate strength, then linear softening to a
  residual plateau; stiffeners — J2 elasto-plastic capped by the minimum of
  squash / weak-axis column (Johnson-Ostenfeld) / tripping (flat bar,
  `GJ/I_p`); tension — elastic then perfectly plastic.
  Two plating strength laws are implemented and reported separately:
  `von_karman` (perfect plate, `sqrt(sigma_cr sigma_y)`) and `faulkner`
  (imperfection-inclusive, `(2/beta - 1/beta^2) sigma_y`).
* **Section equilibrium**: pure bending, `N = 0` solved for the neutral-axis
  strain at every curvature, `M = sum A sigma z`; the peak of `M(kappa)` is
  refined by a golden-section search, so `M_u` does not depend on the curvature
  step.  Ultimate moments are reported **signed** (positive sagging).
* **Failure-mode attribution**: the *first* element to reach its own ultimate
  (smallest collapse strain), restricted to elements that actually carry the
  section (>= 5% of the section resultant) — the classical Smith attribution,
  made robust against small appendages near the neutral axis.

### Benchmark set

| family | geometry source | reference used | evidence |
|---|---|---|---|
| ISSC-2009-style double-bottom unit | reconstructed arrangement (bottom 18 mm, inner bottom 13 mm, 0.80 m longitudinals, 2.40 m frames, centre girder) | self-convergence + element-wise formula | RECONSTRUCTED |
| Paik-style panel, beta in [1.5, 3.0] | plate thickness derived from beta (a = 3.2 m, b = 0.8 m, HT32) | CSR-type element-wise formula | RECONSTRUCTED |
| box girder (stocky and slender) | four-plate square box | closed-form limit analysis | RECONSTRUCTED |
| **KCS midship section** | published particulars (Lpp 230 m, B 32.2 m, D 19.0 m, T 10.8 m, Cb 0.651, Cm 0.977), representative 3600 TEU scantlings | wave-induced demand (IACS + strip model) | REAL GEOMETRY / representative scantlings |

**Honesty statement on the literature anchors.**  The experimental collapse
moments of the Dow (1991) and Gordo & Guedes Soares (1996) box-girder tests, and
the ISSC-2009 committee reference values, are **not** reproduced here: the data
are not available in the repository and are not fabricated offline.  They are
recorded as `NOT_SUPPORTED` in the acceptance record.  What *is* verified is
(a) grid convergence against the Richardson limit, (b) the panel strength against
an independent element-wise design formula, and (c) the box/panel collapse
moments against closed-form limit analysis (statics only, no element curves).

## 2. How it was done

1. `SectionModel` + `PlateStrip`/`StiffenerGroup` represent all four section
   families; fiberisation conserves area and first moment exactly (tested).
2. The vectorised element-curve kernel evaluates the whole section in one pass,
   so a three-level mesh study costs milliseconds.
3. `SmithSectionSolver` sweeps curvature, solves `N = 0` by bisection,
   golden-section-refines the peak, and attributes the failure mode.
4. `convergence.mesh_convergence_study` runs the three-level Richardson protocol
   with a predeclared 1% tolerance; a constant sequence is reported as
   *mesh independent* and a non-monotone one as a *successive-difference bound*,
   never as a fabricated order.
5. `acceptance.py` evaluates C1-C6 with literal thresholds and returns the
   numbers plus their evidence class; `scripts/run_in387_fem01.py` (this file's
   generator) writes the record, the figures and this report.

## 3. Results

### C1 — grid convergence (Richardson, h -> 0, tolerance 1%)

| study | levels | finest value | observed order p | criterion value | verdict |
|---|---|---|---|---|---|
| double-bottom unit, axial sigma_u | [1, 2, 4, 8] | 199.6045 MPa | mesh independent | 0.00e+00 | PASS |
| KCS midship section, M_u sagging | [4, 8, 16] | 4.4777 GNm | 2.29 | 4.74e-05 | PASS |

Coarse levels that fall outside the asymptotic range (KCS, `n_fibers = 1, 2`)
give 4.4874 and
4.4930 GNm; they
are shown in the figure but are **not** used for the extrapolation.  The
double-bottom axial strength is mesh independent by construction: under uniform
shortening every stiffener and plate element sits at its own strength and the
only fiberised element (the centre girder) has a uniform stress, so the integrated
strength is an area-weighted sum — the study confirms the expected degeneracy
rather than manufacturing a convergence order.

![C1](../artifacts/in387_fem01/figures/fig1_mesh_convergence.png)

### C2 — stiffened-panel beta series (band [0.4, 0.8], CSR-type difference <= 10%)

| beta | t [mm] | sigma_u [MPa] | sigma_u/sigma_y | CSR-type | diff [%] | mode |
|---|---|---|---|---|---|---|
| 1.5 | 20.86 | 308.7 | 0.980 | 0.886 | 10.67 | stiffener_column_buckling |
| 2.0 | 15.64 | 241.8 | 0.768 | 0.772 | 0.61 | plate_buckling |
| 2.5 | 12.51 | 216.5 | 0.687 | 0.692 | 0.62 | plate_buckling |
| 3.0 | 10.43 | 198.9 | 0.631 | 0.637 | 0.84 | plate_buckling |

At `beta = 2.0`: `sigma_u/sigma_y = 0.768`
(in band: True), relative difference from the CSR-type
element-wise formula `0.61%`
(<= 10%: True).

**Element-law sensitivity at beta = 2** (declared before the comparison):
Faulkner law gives `0.768`
(plate_buckling), von Karman perfect-plate law gives
`0.934`
(stiffener_column_buckling).  The two laws bracket the *modelled*
uncertainty of the plating strength; the C2 band [0.4, 0.8] is consistent with
the imperfection-inclusive law only.  This is a real physical statement: the
perfect-plate effective width over-estimates imperfect ship plating, and the
choice of plating law is the dominant modelling decision at this slenderness —
larger than the discretisation error by two orders of magnitude.

![C2](../artifacts/in387_fem01/figures/fig2_panel_beta_series.png)

### C3 — box girder (reference band [0.85, 1.15])

| case | beta flange | M_u [MNm] | M_ref [MNm] | ratio | mode | verdict |
|---|---|---|---|---|---|---|
| stocky | 1.16 | 0.6777 | 0.6777 | 1.000 | yielding | PASS |
| slender | 3.82 | 1.4169 | 1.6334 | 0.867 | plate_buckling | PASS |

The stocky box collapses at **exactly** its full-plastic moment (ratio
`1.0000`), which is the strong check of the collapse
machinery: the peak search, the strength-balanced neutral axis and the plastic
mechanism all have to be right for the two numbers to coincide.  The slender box
lands at `0.867` of the limit-analysis reference
(statics with element caps) and satisfies the two-sided bound
`0.905 <= M_u = 1.417 <= 2.258` MNm.  The residual
~13.3% shortfall of the incremental solution
against the rigid-plastic reference is the elastic-plastic transition (the peak is
reached before the whole section is plastic), which the limit analysis cannot see.

![C3](../artifacts/in387_fem01/figures/fig3_box_girder_collapse.png)

### C4 — real KCS midship section and the load-capacity handoff

**Geometry gate** (`Cm` from the modelled underwater outline, no fitting to the answer):

| item | value |
|---|---|
| Cm from the closed-form outline | 0.977000 |
| Cm from the waterline integration of the section outline | 0.976998 |
| Cm published (KCS) | 0.977 |
| Cm error | -1.11e-16 |
| midship area, closed form [m^2] | 339.762 |
| midship area, outline integration [m^2] | 339.761 |
| derived bilge radius [m] | 4.317 |
| all gate checks | True |

The two area evaluations are independent: the first is the closed form used to
*derive* the bilge radius, the second integrates the section outline keel->deck
edge (the value a CAD section-extraction supplies).  They agree to
0.0002%,
which is the check that the idealisation and the outline describe the same
section.

Section properties of the idealised midship section: area
`2.353 m^2`, elastic neutral axis `z = 7.211 m`,
`I = 136.79 m^4`, `S_deck = 10.37 m^3`, `S_keel = 18.97 m^3`.

| quantity | GN m |
|---|---|
| IACS UR S11 sagging | 2.573 |
| IACS UR S11 hogging | 2.142 |
| design-wave strip model | 2.474 |
| long-term spectral upper reference | 4.366 |
| Smith M_u sagging | 4.473 |
| Smith M_u hogging | 5.177 |

Margins: sagging `1.74`, hogging `2.01`,
governing `sagging` -> **margin 1.74
(required 1.5, passes: True)**.

**Honest caveat.**  Under the *stricter* spectral definition (long-term 1e-8 from
the strip RAO + coarse climate model) the margin is
`1.02`, i.e. **below 1.5**
(`passes_long_term_upper = False`).  That number is
not used to overwrite the criterion because the strip RAO has neither diffraction
nor radiation damping and is known to over-predict the extreme moment — but it is
reported, and it means the C4 verdict is *conditional on the demand definition*:
with the IACS design-wave demand the section has 74% reserve; with the
uncalibrated spectral upper reference it has 2%.

![C4](../artifacts/in387_fem01/figures/fig4_kcs_midship_section.png)

![handoff](../artifacts/in387_fem01/figures/fig5_load_capacity_handoff.png)

### C5 — failure-mode identification (3/3 correct, requirement 2/3)

| case | expected | observed | governing group |
|---|---|---|---|
| slender plating (beta=3.0), flanged stiffener | plate_buckling | plate_buckling | panel_plate |
| stocky plating (beta=0.9), slender flat bar (h/t=44) | stiffener_tripping | stiffener_tripping | panel_stiffener |
| stocky box girder in bending (beta=1.16) | yielding | yielding | top_flange |

![C5](../artifacts/in387_fem01/figures/fig6_failure_modes.png)

### C6 — test suite

`test_smith_section_benchmark.py`: 16 tests collected,
pytest exit code 0, runtime
17.9 s, verdict
**PASS**.

## 4. Points to watch (traps and limitations)

1. **beta convention matters.**  `beta = (b/t) sqrt(sigma_y/E)` with the plate
   buckling coefficient `k = 4` implies `sigma_cr / sigma_y = 3.615 / beta^2`;
   `beta = 2` is therefore only *marginally* slender (`sigma_cr = 0.9 sigma_y`).
   Mixing this beta with the "`sigma_cr = sigma_y/beta^2`" convention silently
   moves the whole series by a factor 1.9 in slenderness.
2. **Web lumping destroys the plastic moment.**  Treating a vertical strip as one
   fiber at its centroid throws away its entire `A sigma |z - z_p|`
   contribution; the reference statics therefore integrate vertical plating over
   its extent (a test pins the hand-computed plastic moment).
3. **Peak search must be sign-aware.**  A hogging analysis returns negative
   moments; maximising the signed value finds the *first* sample and reports a
   meaningless capacity.  The solver maximises `|M|` and extends the curvature
   range automatically if the peak lands on the sweep boundary.
4. **Attribution by utilisation alone mislabels sections.**  An element at its cap
   next to the neutral axis, or a small appendage, can be "saturated" without
   limiting the moment; the classifier uses the *first element to collapse*
   weighted by the element's share of the section resultant.
5. **The KCS scantlings are representative, not a drawing.**  The hull-form test
   case publishes geometry, not structure.  The section reproduces the published
   principal dimensions and `Cm`, and its structural arrangement is the ship's
   (single skin, double bottom, hopper, deck strips, hatch coaming, topside
   tank); the plate thicknesses are class-typical values.  `M_u` and the margin
   carry that caveat.
6. **Element law >> discretisation.**  Discretisation error is 0.005% while the
   plating-law choice is ~20%: effort belongs in the element curves (tests,
   imperfection sensitivity), not in finer fibers.
7. **No KCS geometry in TensorCAD.**  The TensorCAD package available in this
   workspace (0.22) exposes parametric hull models (Wigley, SubOff) and a hull
   mesh builder, but no KRISO container-ship hull-form definition, so the
   midship outline could **not** be imported from CAD as the node suggested.
   The integration point is implemented instead: `midship_area_from_outline`
   consumes any half-breadth outline and `KCSMidshipSection.outline()` emits the
   one used here, both gated on the published `Cm`.  Wiring a real
   section-extraction is then a data-supply problem, not a modelling change.
8. **Topside tank omission changes the answer.**  Without the topside tank the
   neutral axis drops to 6.6 m and the deck modulus to 8.9 m^3; the sagging
   capacity falls by ~10%.  The section is therefore sensitive to upper-wing
   structure, which is the first thing to check against a real drawing.

## 5. Error assessment (is it < 3%?)

| quantity | reference | error | < 3%? |
|---|---|---|---|
| stocky box `M_u` | closed-form full-plastic moment | 0.000% | **yes** |
| KCS `M_u` mesh convergence | Richardson `h -> 0` (levels 4/8/16) | 0.005% | **yes** |
| panel `sigma_u` at beta = 2 | CSR-type element-wise formula | 0.61% | **yes** (0.6%, limit 10%) |
| slender box `M_u` | rigid-plastic limit analysis with caps | 13.3% | no (elastic-plastic transition, inside the 15% band) |
| KCS `M_u` vs experiment | none available (no published ultimate strength for the KCS) | n/a | **not applicable — declared** |
| box girder vs Dow / Gordo & Soares tests | not available in-repo | n/a | **NOT_SUPPORTED — declared, not fabricated** |

## 6. Acceptance summary

| criterion | passes |
|---|---|
| C1 | True |
| C2 | True |
| C3 | True |
| C4 | True |
| C5 | True |
| C6 | True |

All six criteria pass on the numbers quoted above.  The two declared
substitutions/deviations are: (i) the literature *experimental* box-girder and
ISSC reference values are replaced by closed-form limit analysis and an
independent element-wise design formula because the datasets are not available
(this is recorded as `NOT_SUPPORTED` in the JSON record); (ii) the KCS
*scantlings* are representative values of the class, because the hull-form test
case does not publish structure.  Both are stated in the figures, the JSON
record and this report.

## 7. Reproduce

```bash
python -m pytest tests/test_smith_section_benchmark.py -q      # 16 tests
python scripts/run_in387_fem01.py                              # record + figures + this report
```
