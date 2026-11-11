# FEM→DEM ``failure_faces.json`` cross-library hand-off: from the frozen interface to a closed physical conservation chain

**Node** p495w4-n07 (TensorPhysics) · mission 13949 · branch ``agent/m13949``
**Scope** six three-point-bend cases: span L = 2 / 4 / 6 m × two load rates (quasi-static, dynamic rise time 2 ms)

---

## 1. Case introduction

### 1.1 Physical background

A structural member that fractures in bending hands its stored elastic energy to
debris.  When the FEM and DEM solvers are different codes, the hand-off is where
physics is usually lost: the fracture surface geometry, the energy that the new
surfaces absorb, and the kinematic state of the material that becomes debris all
cross a file boundary.  This node verifies that the frozen ``failure_faces`` v3
packet plus a sealed release sidecar carry enough information for a DEM consumer
to reproduce *mass, momentum and energy* to stated bands, and it quantifies every
residual instead of asserting closure.

### 1.2 Geometry and material

The bar is a metre-scale, simply supported ice beam (columnar freshwater ice,
laboratory range) driven in three-point bending at midspan.  Self weight is
included in all cases.

| parameter | symbol | value | unit |
|---|---|---|---|
| span | L | 2 / 4 / 6 | m |
| depth | h | 0.80 | m |
| width | b | 0.60 | m |
| Young's modulus | E | 9.0 | GPa |
| Poisson's ratio | ν | 0.33 | — |
| density | ρ | 917 | kg/m³ |
| tensile strength | σ_t | 1.50 | MPa |
| fracture energy | G_f | 5.0 | J/m² |
| FEM element length | Δx | 50 | mm |
| DEM fragment size | s | 50 | mm (cube) |
| fragments per case | N | 7680, 7680, 15360, 15360, 23040, 23040 | — |
| cross-section area | A | 0.480 | m² |
| second moment | I | 0.0256 | m⁴ |

Load rates: quasi-static (static equilibrium path, cross-head 1×10⁻⁴ m/s) and
dynamic (force ramp reaching the static failure load in 2 ms,
Newmark average acceleration, no algorithmic damping).

## 2. Method

1. **FEM (producer).** Euler–Bernoulli beam elements, Hermite cubics, consistent
   mass, 40 elements (Δx = 50 mm).
   Supports at both ends; quasi-static path solves the displacement-controlled
   static problem by bisection on the cross-head stroke (`solve_static`), the
   dynamic path integrates a force ramp with Newmark (β = 1/4, γ = 1/2).
   Fracture is declared at the first load level whose extreme-fibre tensile
   bending stress reaches σ_t.
2. **Export.** A frozen ``failure_faces`` **v3.0.0** packet (validated by
   ``tensorphysics.failure_handoff.schema.validate_packet``: JSON-Schema subset +
   ``jsonschema`` + semantic invariants + SHA-256 seal) carries
   192 crack-plane faces with position, unit normal,
   area, traction and surface energy; a separate sealed ``release_state.json``
   carries the kinematic release state (outside the frozen schema, referenced by
   the packet digest).
3. **Release dynamics.** After fracture the two halves are free bodies; their
   free vibration is integrated with Newmark until the kinetic-energy peak.  The
   run is sink-free and energy conserving, and its closure is reported as
   ``fe_release_closure``.
4. **DEM (consumer).** Fragments are element-sized cubes over the whole bar
   (N = elements × cross-section cells); velocities are the consistent-mass
   element averages of the release field, and each fragment carries its element's
   residual strain energy plus the packet surface energy on the crack layer.
   The surface energy ``G_f A`` is drawn from the *dispersive* part of the release
   motion so that the centre-of-mass momentum is untouched.  The cloud is then
   integrated by ``RigidDEM`` (free flight + ground impact, restitution
   0.30, friction 0.40, dt = 2.0 ms,
   60 ms).
5. **Gates.** Mass (FEM bar mass vs fragment masses), momentum (consistent FE
   momentum vs lumped fragment momentum), energy (released mechanical energy vs
   fragment KE + residual strain + sub-fragment KE + surface + ΔPE), schema and a
   two-sample KS test on fragment size vs element size.

## 3. Results

### 3.1 Acceptance table (machine-checkable criteria)

| case | fragments | fracture mode | P_f (kN) | DAF | U₁ (J) | T₁ (J) | mass res. | momentum res. | energy res. | KS p |
|---|---|---|---|---|---|---|---|---|---|---|
| L2m_qs | 7680 | quasi_static | 192.5 | 1.00 | 14.16 | 0.00 | 1.31e-13 | 1.74e-13 | 2.89e-05 | 1.000 |
| L2m_dyn | 7680 | dynamic | 177.4 | 0.95 | 14.49 | 3.74 | 1.31e-13 | 1.76e-14 | 2.25e-05 | 1.000 |
| L4m_qs | 15360 | quasi_static | 88.5 | 1.00 | 28.52 | 0.00 | 1.47e-13 | 1.81e-13 | 3.50e-05 | 1.000 |
| L4m_dyn | 15360 | dynamic | 167.3 | 1.92 | 20.67 | 42.68 | 1.47e-13 | 8.71e-13 | 1.08e-05 | 1.000 |
| L6m_qs | 23040 | quasi_static | 51.5 | 1.00 | 44.78 | 0.00 | 1.52e-13 | 2.61e-11 | 5.19e-05 | 1.000 |
| L6m_dyn | 23040 | dynamic | 152.8 | 2.99 | 22.06 | 88.56 | 1.52e-13 | 2.52e-12 | 7.28e-06 | 1.000 |

Sweep verdict: mass PASS (max 1.52e-13 < 1%),
momentum PASS (max 2.61e-11 < 2%),
energy PASS (max 5.19e-05 < 5%),
schema PASS, KS PASS (min p = 1.000 > 0.05).

### 3.2 Energy ledger

| case | E_released (J) | KE_frag (J) | U_res (J) | KE_subgrid (J) | E_surface (J) | ΔPE (J) | residual | residual w/o sub-grid |
|---|---|---|---|---|---|---|---|---|
| L2m_qs | 13.392 | 9.333 | 1.802 | 0.663 | 2.400 | -0.806 | 2.89e-05 | 4.952e-02 |
| L2m_dyn | 17.447 | 14.366 | 2.380 | 0.892 | 2.400 | -2.591 | 2.25e-05 | 5.117e-02 |
| L4m_qs | 22.304 | 27.482 | 4.605 | 0.809 | 2.400 | -12.993 | 3.50e-05 | 3.632e-02 |
| L4m_dyn | 58.187 | 93.142 | 4.204 | 0.810 | 2.400 | -42.370 | 1.08e-05 | 1.393e-02 |
| L6m_qs | 23.256 | 86.142 | 6.918 | 0.774 | 2.400 | -72.979 | 5.19e-05 | 3.333e-02 |
| L6m_dyn | 96.902 | 277.364 | 8.173 | 0.796 | 2.400 | -191.832 | 7.28e-06 | 8.225e-03 |

### 3.3 FEM verification against closed-form theory

| case | FEM δ (m) | closed-form δ (m) | δ err | FEM σ (MPa) | σ err | FEM ω₁ (rad/s) | ω err |
|---|---|---|---|---|---|---|---|
| L2m_qs | 1.3889e-04 | 1.3889e-04 | 7.8e-12 | 1.455 | 2.45% | 16187.0 | 1.5e-06 |
| L2m_dyn | 1.3889e-04 | 1.3889e-04 | 7.8e-12 | 1.455 | 2.45% | 16187.0 | 1.5e-06 |
| L4m_qs | 5.5556e-04 | 5.5556e-04 | 1.4e-11 | 1.450 | 1.14% | 4046.7 | 5.1e-07 |
| L4m_dyn | 5.5556e-04 | 5.5556e-04 | 1.4e-11 | 1.450 | 1.14% | 4046.7 | 5.1e-07 |
| L6m_qs | 1.2500e-03 | 1.2500e-03 | 1.7e-10 | 1.415 | 0.66% | 1798.6 | 6.2e-07 |
| L6m_dyn | 1.2500e-03 | 1.2500e-03 | 1.7e-10 | 1.415 | 0.66% | 1798.6 | 6.2e-07 |

Mesh convergence of the fracture-deciding quantity (first case, L2m_qs):

| Δx (mm) | elements | stress error | deflection error | free–free ω error |
|---|---|---|---|---|
| 200 | 10 | 9.80% | 2.5e-15 | 4.8e-04 |
| 100 | 20 | 4.89% | 7.6e-13 | 3.3e-05 |
| 50 | 40 | 2.45% | 7.8e-12 | 1.5e-06 |
| 25 | 80 | 1.22% | 2.3e-11 | 5.1e-07 |

### 3.4 DEM integrator invariants

| case | contacts | dissipated (J) | energy closure (KE+PE+internal+diss) | horizontal momentum residual |
|---|---|---|---|---|
| L2m_qs | 192 | 4.26 | 1.36e-15 | 0.00e+00 |
| L2m_dyn | 384 | 8.65 | 1.72e-15 | 0.00e+00 |
| L4m_qs | 672 | 15.17 | 3.78e-14 | 0.00e+00 |
| L4m_dyn | 864 | 24.31 | 5.42e-14 | 0.00e+00 |
| L6m_qs | 1128 | 28.64 | 1.65e-12 | 0.00e+00 |
| L6m_dyn | 1320 | 48.26 | 2.11e-12 | 0.00e+00 |

### 3.5 Negative controls (every control must turn its gate red)

| case | control | gate | measured | verdict |
|---|---|---|---|---|
| L2m_qs | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L2m_qs | inflate_velocity_20pct | momentum | 2.406e-02 | red (as expected) |
| L2m_qs | inflate_velocity_20pct | energy | 2.347e-01 | red (as expected) |
| L2m_qs | coarsen_fragments_2x | fragment_size_consistency | 5.649e-108 | red (as expected) |
| L2m_qs | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |
| L2m_dyn | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L2m_dyn | inflate_velocity_20pct | momentum | 2.000e-01 | red (as expected) |
| L2m_dyn | inflate_velocity_20pct | energy | 2.659e-01 | red (as expected) |
| L2m_dyn | coarsen_fragments_2x | fragment_size_consistency | 5.649e-108 | red (as expected) |
| L2m_dyn | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |
| L4m_qs | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L4m_qs | inflate_velocity_20pct | momentum | 1.825e-01 | red (as expected) |
| L4m_qs | inflate_velocity_20pct | energy | 3.515e-01 | red (as expected) |
| L4m_qs | coarsen_fragments_2x | fragment_size_consistency | 0.000e+00 | red (as expected) |
| L4m_qs | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |
| L4m_dyn | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L4m_dyn | inflate_velocity_20pct | momentum | 2.000e-01 | red (as expected) |
| L4m_dyn | inflate_velocity_20pct | energy | 4.133e-01 | red (as expected) |
| L4m_dyn | coarsen_fragments_2x | fragment_size_consistency | 0.000e+00 | red (as expected) |
| L4m_dyn | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |
| L6m_qs | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L6m_qs | inflate_velocity_20pct | momentum | 2.000e-01 | red (as expected) |
| L6m_qs | inflate_velocity_20pct | energy | 6.197e-01 | red (as expected) |
| L6m_qs | coarsen_fragments_2x | fragment_size_consistency | 0.000e+00 | red (as expected) |
| L6m_qs | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |
| L6m_dyn | drop_5pct_fragments | mass | 5.000e-02 | red (as expected) |
| L6m_dyn | inflate_velocity_20pct | momentum | 2.000e-01 | red (as expected) |
| L6m_dyn | inflate_velocity_20pct | energy | 5.574e-01 | red (as expected) |
| L6m_dyn | coarsen_fragments_2x | fragment_size_consistency | 0.000e+00 | red (as expected) |
| L6m_dyn | drop_one_face | failure_faces_schema | 0.000e+00 | red (as expected) |

## 4. Figures

![Energy ledger](fig1_energy_ledger.png)

*Figure 1 — energy ledger per case (left) and the energy residual against the 5 % band (right).*

![Conservation residuals](fig2_conservation_residuals.png)

*Figure 2 — the three conservation residuals over the six cases on a log scale, with the acceptance bands.*

![FEM verification](fig3_fem_verification.png)

*Figure 3 — (a) FEM vs closed-form deflection/stress/frequency, (b) dynamic amplification of the failure load.*

![Fragment cloud](fig4_fragment_cloud.png)

*Figure 4 — (a) DEM fragment cloud at the hand-off instant (colour = speed), (b) DEM flight/impact ledger.*

![Release history](fig5_release_history.png)

*Figure 5 — post-fracture release: strain/kinetic/potential energy and the conserved total.*

![KS test](fig6_size_ks.png)

*Figure 6 — fragment-size consistency (ECDF) with the coarsened negative control.*

## 5. Caveats, limitations and applicability

1. **Element-sized fragment idealisation.** The DEM cloud is a *rigid,
   translational* idealisation of the fractured bar: fragments do not bond and do
   not collide with each other (ground contact only).  The audit therefore
   verifies the *hand-off and the conservation bookkeeping*, not a predictive
   debris field.
2. **Sub-fragment kinetic energy.** At the fragment size equal to the element
   size, the intra-element velocity dispersion cannot be carried by rigid
   fragments.  It is measured in the FEM and handed over as fragment internal
   energy (max 5.3% of the release-relevant kinetic energy in this
   sweep).  Without that term the ledger residual would be
   5.1% in the worst case
   (L2m_dyn) — reported in Table 3.2, not hidden.
3. **Surface energy is a prescribed sink.** ``G_f A`` comes from the material
   data and the packet geometry; the chain verifies that everything else closes
   around it, and the dispersive draw ratio ``G_f A / KE_disp`` stays ≤
   0.21, so the
   sink never exceeds the dispersive kinetic energy (no negative scale factor).
4. **Bending-only kinematics.** The Euler–Bernoulli model has one transverse dof;
   crack opening appears as a moment release, not as a local opening
   displacement.  Shear deformation is neglected (L/h = 2.5–7.5; the static
   deflection error against Euler–Bernoulli is a mesh error, see Table 3.3).
5. **Dynamic loading** is modelled as a force ramp (rise time 2 ms).  A
   displacement-controlled rigid cross-head at the same rate produces
   non-physical local stresses (curvature concentration under the load) and is
   deliberately not used.
6. **Scope of the claims.** No experimental benchmark exists for the synthetic
   six-case sweep; the verification anchors are closed-form beam theory (<3 %)
   and the internal conservation identities.  The capability is therefore
   *Implemented and audited*, not *experimentally validated*.

## 6. Error assessment against references (explicit)

| quantity | reference | error | < 3 %? |
|---|---|---|---|
| midspan deflection (all cases) | Euler–Bernoulli + self weight | max 1.70e-10 | yes |
| extreme-fibre stress (all cases) | P L/4 + q L²/8 | max 2.45% | yes |
| free–free ω₁ of the half bar | 22.3733 √(EI/ρA a⁴) | max 1.50e-06 | yes |
| mass conservation | FEM bar mass | max 1.52e-13 | yes |
| momentum conservation | consistent FE momentum | max 2.61e-11 | yes |
| energy conservation | released mechanical energy | max 5.19e-05 | yes |
| fragment size distribution | FEM element size | KS p ≥ 1.00 (α = 0.05) | n/a (test) |

All acceptance criteria of the campaign node are met for all six cases; the
residual that is **not** below 3 % is the stress verification bound (2.4 % worst
case is below 3 %) and the *rigid-fragment* energy residual without the
sub-grid transfer (up to 5.1%),
which is a documented resolution limit of element-sized rigid fragments.

## 7. Audit chain

* packets: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/cases/<case>/failure_faces.json`` (schema v3.0.0, SHA-256 sealed)
* release sidecars: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/cases/<case>/release_state.json``
* ledgers: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/cases/<case>/conservation.json``
* controls: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/cases/<case>/negative_controls.json``
* sweep summary: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/summary.json``
* figures: ``/root/tp-agent/work/m13949/artifacts/p495w4_n07/figures``

Packet digests per case:

* L2m_qs: packet `3dcff2d0c7fdf854…`, release `2878ae66ca1688fb…`
* L2m_dyn: packet `1d39ab423ab9cad4…`, release `8ea186da3b8cfe91…`
* L4m_qs: packet `ecf5c413f56dea7f…`, release `175e5739e808cf77…`
* L4m_dyn: packet `4d009e2effd51fa6…`, release `39fc5bec0c3aae6c…`
* L6m_qs: packet `0712d8bf149188a9…`, release `2854700a6a0527ff…`
* L6m_dyn: packet `868b915df3f019bd…`, release `a3eb9d4992de4577…`

## 8. Reproduction

```bash
PYTHONPATH=src python scripts/run_p495w4_n07.py          # artifacts + figures + report
PYTHONPATH=src pytest tests/test_p495w4_n07_beam_handoff.py -q
```

## 9. Test evidence

| item | value |
|---|---|
| node test module | `tests/test_p495w4_n07_beam_handoff.py` |
| tests | 58 passed |
| coverage — `beam_fem.py` | 96 % (branch) |
| coverage — `beam_cases.py` | 99 % (branch) |
| regression of the existing hand-off suites | `test_failure_handoff_schema/conservation`, `test_ag29w1_n07_fem_dem_handoff`: 67 passed |
| published report | platform report id 13444 (mission 13949) |
