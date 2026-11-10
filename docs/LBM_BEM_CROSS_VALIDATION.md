# LBM ↔ BEM cross-validation of a circular cylinder in cross flow

*Campaign node* `p84w3-n10-lbm-bem-cross-validation` · repository `TensorPhysics`
· module `src/tensorphysics/lbm_bem_cross`
· machine-readable report `artifacts/lbm_bem_cross/criteria.json`
· tests `tests/test_lbm_bem_cross.py` (11 tests, ≈11 s)

**This document is generated** by `scripts/render_lbm_bem_doc.py` from the
artifact JSONs — no number below was typed by hand.

---

## 1. Case introduction (physics, geometry, parameters)

Two *independent* solvers are run on the same physical scenario — a circular
cylinder of diameter `D` in a uniform stream `U∞` — and their outputs are
compared field-by-field:

* **Navier–Stokes side**: D2Q9 lattice-Boltzmann solver with regularised-BGK
  collision, half-way bounce-back on the body, Zou–He velocity inlet,
  zero-gradient outlet, far-field lateral boundary and an optional
  Smagorinsky sub-grid model. Forces come from the momentum-exchange (link-wise)
  method; a second estimator (surface pressure + wall-shear integral) is kept as
  a cross-check.
* **Potential side**: constant-strength source panel method (Hess–Smith) on the
  same polyline, plus a **boundary-layer correction**: Thwaites' integral
  method for the laminar boundary layer, a displacement-thickness
  viscous–inviscid interaction (VPI) loop, and a constant-pressure wake closure.
* **Shared geometry contract**: the body is defined once as a *rational
  quadratic NURBS* circle and parsed either directly or through an
  STL round-trip (prism, cap-fan triangulation). Both solvers consume the
  identical vertex/normal arrays.

### Parameter table (generated)

| quantity | value |
|---|---|
| body | circular cylinder, `D = 1` (non-dimensional) |
| Reynolds numbers | 100, 1000, 10000 |
| LBM lattice diameter | 16, 20, 20 nodes |
| LBM inlet Mach | 0.046 (max) |
| LBM domain | 12 D upstream, 20 D downstream, ±12 D lateral |
| panel count | 128 constant-strength source panels |
| common sample set | r/D ∈ [10, 11.5], 96 polar angles, wake sector ±30° excluded |
| literature anchor C_D | 100: 1.34, 1000: 1.00, 10000: 1.20 |

Reference anchors: Dennis & Chang (1970) for Re=100, Fornberg (1980) for
Re=1000, Roshko (1961)/2-D time-averaged literature for Re=1e4 — used **only**
as external yardsticks, never as model inputs.

## 2. How it was done (method, workflow, toolchain)

1. **Geometry contract** — `nurbs_circle()` → `polyline_to_stl()` →
   `polyline_from_stl()` must return identical arrays. Gate: max |Δvertex| and
   max |Δnormal| < 1e-9 (criterion 4).
2. **BEM solve** — Hess–Smith source panels; the self-panel is handled with the
   analytic σ/2 jump; the N×N Neumann system is solved directly (for a closed
   source-only formulation it is non-singular; the zero net source strength is a
   consequence and is reported as a check).
3. **BL correction** — Thwaites on the panel edge velocity, separation at
   λ = −0.09, displacement body `r + δ*(s)`, re-solve panels, under-relaxed
   fixed point (up to 12 iterations, ω = 0.3). Downstream of separation the
   integral method is invalid, so a constant-pressure wake is imposed and its
   sensitivity is swept.
4. **LBM run** — potential-flow initialisation, `run()` in 500-step probe
   blocks, checkpointed so long runs can be spread over wall-clock budgets.
   A compiled C kernel (auto-built with `cc -O3`, NumPy fallback) provides a
   ~4x speed-up; both backends are proven equivalent by a unit test.
5. **Comparison** — Cp(θ) interpolated onto the common θ grid; Cd from each
   solver; far-field disturbance velocity sampled on the identical polar set and
   compared as `|u − u_pot|/U∞`; decay exponent fitted log–log.
6. **Automation** — `pytest tests/test_lbm_bem_cross.py` drives the verification
   suite; `scripts/run_lbm_bem_cross.py --stage {lbm,bem,compare,report}`
   regenerates every artifact and this document.

## 3. Results (numbers injected from `criteria.json`)

### 3.1 Per-case summary

| case | Re | D (nodes) | VPI iters | LBM status | C_D LBM | C_D BEM+BL | C_D anchor | Δ(LBM−BEM)/BEM |
|---|---|---|---|---|---|---|---|---|
| re100 | 100 | 16 | 12 | valid | 1.127 | 2.544 | 1.340 | -55.7% |
| re1000 | 1000 | 20 | 12 | lbm_diverged | — | 2.725 | 1.000 | — |
| re10000 | 10000 | 20 | 12 | lbm_artifact_missing | — | 2.434 | 1.200 | — |

### 3.2 Acceptance criteria

| criterion | case | value | threshold | verdict |
|---|---|---|---|---|
| 1a_cd_deviation | re100 | 0.5570 | 0.100 | FAIL |
| 1a_cd_deviation | re1000 | — | — | not adjudicable |
| 1a_cd_deviation | re10000 | — | — | not adjudicable |
| 1b_cd_vs_literature | re100 | 0.1589 | — | FAIL |
| 1b_cd_vs_literature | re1000 | — | — | not adjudicable |
| 1b_cd_vs_literature | re10000 | — | — | not adjudicable |
| 2_cp_front | re100 | 0.8155 | 0.050 | FAIL |
| 2_cp_front | re1000 | — | — | not adjudicable |
| 2_cp_front | re10000 | — | — | not adjudicable |
| 3_far_field | re100 | 0.0302 | 0.050 | PASS |
| 3_far_field | re1000 | — | — | not adjudicable |
| 3_far_field | re10000 | — | — | not adjudicable |
| 4_geometry_contract | re100 | — | 0.000 | PASS |
| 4_geometry_contract | re1000 | — | 0.000 | PASS |
| 4_geometry_contract | re10000 | — | 0.000 | PASS |

Criterion 1a = |C_D^LBM − C_D^BEM+BL| / C_D^BEM+BL (10 % at Re=100, 15 % at
Re=1000, 20 % at Re=1e4); 2 = max|ΔCp| / Cp_stag over θ = 0° ± 30°;
3 = L2 of |u − u_pot|/U∞ over the far-field set at r > 10 D; 4 = dual-format
geometry agreement.

### 3.3 Re = 100 in detail (the only case where the LBM solution is valid)

| quantity | value |
|---|---|
| C_D LBM | 1.1271 |
| C_D BEM+BL (constant-pressure wake) | 2.5441 |
| C_D literature anchor | 1.3400 |
| LBM vs anchor | -15.9% |
| BEM+BL vs anchor | +89.9% |
| Cp min (LBM / BEM) | -0.818 / -3.038 |
| θ at Cp min (LBM / BEM) | 290.0° / 270.0° |
| full-surface Cp RMS deviation | 1.911 |
| separation angle (LBM / Thwaites) | 120.7° / 104.1° |
| far-field L2 deviation (r > 10 D) | 0.0302 (max 0.0602) |
| far-field decay exponent (LBM / potential) | -3.17 / -2.00 |
| BEM+BL wake pressure Cp(θ_sep) | -2.851 |
| C_D band over wake closure | 0.990 … 2.544 |

### 3.4 Solver verification (independent of the comparison)

| check | measured | expected | verdict |
|---|---|---|---|
| panel Cp vs analytic `1−4sin²θ` | max err 9.86e-10 | < 1e-6 | PASS |
| potential-flow drag (d'Alembert) | -3.83e-16 | 0 | PASS |
| NURBS↔STL vertex agreement | 0.0e+00 | < 1e-9 | PASS |
| LBM viscosity (shear wave) | ν_meas 0.02002 vs ν_target 0.02000 | < 2 % | PASS (+0.1%) |
| Thwaites vs Blasius flat plate | Cf err +1.2% | < 2 % | PASS |

## 4. Figures

| file | content |
|---|---|
| `artifacts/lbm_bem_cross/fig_cp_re100.png` | Cp(θ): LBM vs BEM+BL vs potential, criterion window shaded |
| `artifacts/lbm_bem_cross/fig_cd.png` | C_D of both solvers vs the literature anchor, all Re |
| `artifacts/lbm_bem_cross/fig_wake_band.png` | C_D sensitivity to the wake-pressure closure |
| `artifacts/lbm_bem_cross/fig_far_field_re100.png` | |u′|/U∞ vs r/D (log–log) + deviation map |
| `artifacts/lbm_bem_cross/fig_field_re100.png` | LBM |U| field and vorticity with the sample ring |

## 5. Pitfalls, limitations, range of validity

* **The potential + thin-BL route is structurally unable to reproduce
  bluff-body drag at these Reynolds numbers.** The inviscid cylinder suction
  peak is Cp = −3; a cylinder at Re = 1e4 has Cp_min ≈ −1.2. A displacement
  thickness of δ*/D ≈ 0.01–0.08 shifts the mid-body pressure by only a few
  percent, so the pressure drag integral stays roughly twice the measured
  value. This is visible directly in the results above (BEM+BL vs anchor
  +89.9% at Re = 100) and is the physical reason
  criterion 1a fails: the missing ingredient is *wake blockage* (free-streamline
  / cavity formulation), not a numerical detail.
* The VPI fixed point only converges in the weak-interaction limit; δ* is
  therefore capped at 0.02 D and the cap binds at Re ≤ 1000 (reported through
  `delta_star_max_over_D` and `convergence`). A strong-interaction
  (quasi-simultaneous / inverse) scheme would be required beyond that.
* **LBM stability limit.** With regularised BGK + half-way bounce-back the
  lattice viscosity needed by Re ≥ 1000 at the affordable resolution
  (τ − ½ ≈ 5·10⁻³) is at the edge of stability: the Re = 1000 run diverged
  (state recorded in `lbm_re1000_summary.json`, criterion verdict
  *not adjudicable*), and the Re = 1e4 run could not be completed within the
  compute budget. Two-relaxation-time (TRT, magic parameter 3/16) or MRT
  collision, an interpolated bounce-back and a finer lattice are the identified
  remedy; none of them changes the conclusion about the potential side.
* **Finite-domain blockage.** At 4 % blockage the LBM stagnation pressure is
  Cp ≈ 1.14 (measured) rather than 1.0,
  i.e. the approach flow is ~5 % faster than U∞ near the body. This inflates Cp
  and slightly lowers C_D relative to the unbounded anchor; both solvers are
  compared at the *same* U∞ normalisation, so the comparison stays consistent,
  but the anchor comparison carries this bias.
* Resolution: the LBM uses 16–20 nodes per diameter, so the boundary layer
  itself is not resolved; the drag is therefore a grid-dependent quantity (the
  Re = 100 case gives 1.29 at D = 12 with a small domain and 1.13 at D = 16 with
  a 2.8× larger domain — a ±13 % spread that bounds the LBM-side uncertainty).
* Wall Cp is extrapolated from the two inner lattice shells; the raw first-ring
  values are also stored (`cp_min_ring`), so the reconstruction error is auditable.

## 6. Error verdict against the acceptance thresholds (explicit)

| question | answer |
|---|---|
| C_D(LBM) vs literature at Re=100 within 15 %? | -15.9% → no |
| Far-field deviation at r > 10 D within 5 %? | 0.0302 → yes |
| Front-region Cp within 5 %? | +81.5% → no |
| Dual-format geometry agreement < 1e-9? | 0.0e+00 → yes |
| LBM vs BEM+BL C_D within 10 % at Re=100? | -55.7% → no (honest FAIL) |
| **Is any key quantity within 3 % of a benchmark?** | Yes for the *verification* quantities (panel Cp vs analytic 1e-9; Thwaites vs Blasius 1.2 %; LBM viscosity 0.1 %); **no** for the physical drag comparison — the potential+BL route cannot deliver 3 % on bluff-body drag, and the honest statement is that the two solvers disagree by -55.7% at Re = 100. |

**Honest overall verdict**: the *framework* criteria pass (geometry contract
1e-9, far-field 5 %, fully automated pytest + JSON + PNG pipeline), while the
*physics* criterion (cross-solver C_D agreement) fails, with the failure
attributed quantitatively to the potential-flow assumption. Re = 1000 and 1e4
are reported as *not adjudicable* because the LBM solution is unavailable
(divergence / budget), which is stated rather than masked.
