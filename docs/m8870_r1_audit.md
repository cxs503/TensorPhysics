# m8870 R1 — `geometric_nonlinear` accuracy & verification integrity audit

**Mission**: m8870 (TensorPhysics PR #367 reopened, "管到底" round 1)
**Repo**: TensorPhysics · branch `agent/m14934`
**Artifacts**: `artifacts/m8870_r1/` (`summary.json` + per-topic JSON + `fig_*.png`)
**Regenerate**: `python docs/m8870_r1_audit.py`
**Tests**: `python -m pytest tests/test_geometric_nonlinear.py tests/test_geometric_nonlinear_verification.py tests/test_geometric_nonlinear_accuracy.py` → 51 passed

---

## 0. Why this round exists

PR #367 delivered the four modules of the geometric-nonlinear verification suite
(`elastica.py`, `von_karman_plate.py`, `shallow_arch.py`, `convergence.py`) plus
`tests/test_geometric_nonlinear.py`. Those files are on `main` and their tests pass,
so the *files* are not the problem. The problem is that three of the four
"verification" routines could not fail, and two of the numerical kernels carried
systematic errors that the self-referential checks could not see:

| # | Defect | Type | Evidence |
|---|--------|------|----------|
| D1 | `ElasticaExact._compute_integrals` used `theta = theta_B t^2`, which does **not** remove the `1/sqrt(1-t)` endpoint singularity → load `P` **~1.2–2.1 % low**, tip deflection ~0.3 % low | wrong numerics | `elastica_quadrature.json` |
| D2 | `verify_energy_conservation_elastica` compared the strain energy with `P·delta`. For a dead load `U = ∫P dδ ≠ Pδ`, so it reported **50–72 % "errors"** that had nothing to do with the solution | wrong physics in the criterion | `elastica_energy.json` |
| D3 | `verify_energy_conservation_plate` computed `U_membrane = W_ext − U_bending`, then re-added them → residual **identically 0**, a check that can never fail | tautology (MOCK-grade) | `plate_energy.json` |
| D4 | `full_verification_plate` compared the closed-form plate law against `camp_berger_reference`, which is generated *from the same law* → error identically 0 | tautology | `plate_convergence.json` |
| D5 | `convergence_order_plate` returned the constant `4.0` ("theoretical order") without solving anything | STUB | `plate_convergence.json` |
| D6 | `VonMisesTruss.find_limit_points`: load reported with the **upward** sign on the window `[-0.5H, 2.5H]` → `q_limit_upper` was the value at the window edge, i.e. **4.7× the true limit load** at H/L=0.1 | sign + window artefact | `arch_limit_load.json` |
| D7 | dead, incorrect `VonKarmanPlateExact._ode_system` (Bessel-type operator instead of the von Kármán compatibility operator, no `Eh` factor) — never called, but misleading | dead code | grep: no call sites |
| D8 | `ElasticaFEMSolver` converges at observed order **1.00**, not the `O(h²)` the test asserted | formulation limitation (documented, not changed) | `summary.json` |

---

## 1. D1 — elastica quadrature (fixed)

The elastica first integral gives `ds = dθ / sqrt((2P/EI)(sinθ_B − sinθ))`.
At `θ → θ_B` the integrand behaves like `(θ_B − θ)^(−1/2)`: integrable, but
Gauss–Legendre on `θ = θ_B t²` still sees a `1/sqrt(1−t)` singularity at `t = 1`.
The fix uses the singularity-removing substitution

```
theta = theta_upper (1 - t^2),      |dtheta| = 2 theta_upper t dt,   t in (0,1]
sin(a) - sin(b) = 2 cos((a+b)/2) sin((a-b)/2)         (cancellation-free)
```

which makes the integrand analytic at `t = 0`, so the 160-node rule converges spectrally
(nodes cached with `lru_cache`; the un-cached version cost 250 s per load-deflection curve).

| θ_B [rad] | P module | P reference (adaptive quad) | err after fix | err before fix |
|---|---|---|---|---|
| 0.02 | 0.040006 | 0.040006 | 3.0e-10 % | 1.222 % |
| 0.10 | 0.200736 | 0.200736 | 1.3e-10 % | 1.222 % |
| 0.50 | 1.099675 | 1.099675 | 1.2e-10 % | 1.243 % |
| 1.10 | 3.815552 | 3.815552 | 6.2e-10 % | 1.377 % |
| 1.50 | 14.796895 | 14.796895 | 4.2e-10 % | **2.064 %** |

Independent cross-check: an entirely different method (direct BVP solve of
`EI θ'' + P cosθ = 0` with `P` as an unknown parameter, `scipy.solve_bvp`)
reproduces the fixed values to < 1e-4 relative (`test_against_independent_bvp`).

Two guards were added: `θ_B > π/2` has no monotone first integral
(`sinθ_B − sinθ` changes sign) and now raises `ValueError` instead of silently
returning a truncated integral; `compute_shape_at_load` brackets its root find
strictly below `π/2`.

---

## 2. D2/D3/D4/D5 — verification routines made falsifiable

### 2.1 Elastica energy criterion (now genuine)

Correct statement for a monotonically applied dead tip load: `U = ∫₀^δ P dδ'`
(work of deformation). The routine now integrates the load path numerically and
additionally checks the generalised-force identity `dU/dδ = P`.

| θ_B | `U_strain` | `∫P dδ` | residual | `U/(Pδ)` | `dU/dδ` vs `P` |
|---|---|---|---|---|---|
| 0.3 | 0.060265 | 0.060265 | 1.3e-05 % | 0.490 | 0 |
| 0.7 | 0.335687 | 0.335687 | 8.5e-05 % | 0.435 | 0 |
| 1.1 | 0.882971 | 0.882973 | 3.0e-04 % | 0.328 | 0 |
| 1.3 | 1.331598 | 1.331606 | 6.2e-04 % | 0.282 | 3e-08 % |

`U/(Pδ)` falls from 0.49 to 0.28: this is exactly why the old `U = Pδ` criterion
produced the bogus 50–72 % numbers. The new residual (≤ 6e-4 %) is limited only by
the trapezoidal load-path quadrature.

### 2.2 Plate energy criterion (now genuine)

`VonKarmanPlateFEM.energy(u)` integrates `½∫(N·ε + M·κ) r dr` (×2π) with the same
Green–Lagrange kinematics as the residual, split into membrane/bending.
`load_path_work(history)` integrates `∫P dw₀` over the Newton load path.
Neither quantity is derived from the other.

| w₀/h | W_ext [J] | U_FE [J] | residual |
|---|---|---|---|
| 0.5 | 12.42 | 12.42 | 0.03 % |
| 1.0 | 55.76 | 55.70 | 0.10 % |
| 2.0 | 340.96 | 339.81 | 0.34 % |
| 3.0 | 1240.09 | 1230.85 | 0.75 % |

Cross-check of the two independent loops: `dU/du` reproduces `2π·f_int` dof by dof
(`test_energy_gradient_equals_internal_force`, tolerance 1e-4).

### 2.3 Plate convergence order (now computed)

`convergence_order_plate` runs the FE on 8/16/32 elements at a fixed load and
estimates the order from the Richardson ratio of successive differences:
**p = 2.32** (previously: a hard-coded `4.0`).

### 2.4 Closed-form plate law vs FE (no longer self-comparison)

| w₀/h target | FE w₀/h (n_elem=16) | error |
|---|---|---|
| 0.5 | 0.4937 | 1.27 % |
| 1.0 | 0.9770 | 2.30 % |
| 2.0 | 1.9799 | 1.01 % |
| 3.0 | 3.0464 | 1.55 % |

Mesh sensitivity 16→32 elements: **0.30 %**, i.e. the FE is converged and the
2.3 % is a property of the constant-coefficient cubic law, not of the FE.
Fitting `P̄ = w̄ + c w̄³` to the FE curve gives `c = 0.443 (w₀/h=0.5)` down to
`c = 0.354 (w₀/h=3)` around the literature value `c(0.3) = 0.376`. The
coefficient is now defined once (`plate_cubic_coefficient(nu)`) and reused by
`camp_berger_reference` (previously duplicated as a literal).

---

## 3. D6 — von Mises truss limit load (fixed)

The apex equilibrium gives the applied **downward** load `q = −2 N sinβ`
(`N < 0` in compression); the module returned `+2 N sinβ`, and searched
`v ∈ [−0.5H, 2.5H]`, where the largest value sits on the far tension branch at the
window edge. Therefore `q_limit_upper` was a truncation artefact (and
`budiansky_roth_reference` published it as a "literature limit load"):
at H/L = 0.1 it returned 13.81 MN instead of 2.94 MN.

Fix: downward-positive load, window `v ∈ [0, 2H]` (the snap-through branch — both
limit points are interior), and `upper_is_interior` / `lower_is_interior` flags so
a boundary maximum can never masquerade as a limit point again.
`ShallowArchModel.analytical_limit_load()` now evaluates the closed form
`q(h) = −2 N(h) h/L1(h)`, `h ∈ (0,H]`, instead of delegating to the grid search.

| H/L | grid `q_limit_upper` | closed form | error |
|---|---|---|---|
| 0.10 | 2.941312 MN | 2.941312 MN | 2e-06 % |
| 0.20 | 20.700024 MN | 20.700029 MN | 2e-05 % |
| 0.30 | 57.872278 MN | 57.872281 MN | 4e-06 % |

(pre-fix errors: **264 % – 367 %**)

---

## 4. D8 — elastica FEM is first order (documented, not changed)

`ElasticaFEMSolver` is a lumped rigid-link model with piecewise-constant rotation.
Against the corrected exact solution the observed order at `P = 2.0` is
**1.00** (errors 5.3e-2, 2.7e-2, 1.3e-2, 6.7e-3, 3.4e-3 for n = 20…320).
The old test asserted `1 < p < 3` and passed only marginally (p ≈ 1.00); it now
pins `0.85 < p < 1.5` and additionally requires monotone error reduction.
A second-order element is a separate work item — it is *not* claimed here.
`full_verification_elastica` accepts the first-order window and reports
`convergence_order_expected = "first order (lumped rigid-link model)"`.

---

## 5. Error judgement (vs the requirement "> 15 % unacceptable", target < 3 %)

| Quantity | Reference | Error | < 3 % ? |
|---|---|---|---|
| Elastica `P(θ_B)` | adaptive quadrature (limit 400) | 6e-10 % | yes |
| Elastica `P(θ_B)` | independent BVP solve of the strong form | < 1e-4 relative | yes |
| Elastica `δ(θ_B)` | adaptive quadrature | 6e-13 relative | yes |
| Elastica energy balance | work of deformation `∫P dδ` | ≤ 6e-4 % | yes |
| Elastica generalised force | `dU/dδ` vs `P` | ≤ 3e-8 % | yes |
| Plate closed-form law `P(w₀)` | FE (mesh-converged, 0.30 % mesh sensitivity) | ≤ 2.30 % | yes |
| Plate energy balance | FE strain energy vs dead-load work | ≤ 0.75 % | yes |
| Plate convergence order | Richardson over 8/16/32 elements | p = 2.32 (reported, not asserted against theory) | n/a |
| von Mises limit load | closed-form maximum of `q(h)` | ≤ 2e-5 % | yes |

Nothing in this round is withheld: every number above is regenerated by
`docs/m8870_r1_audit.py` into `artifacts/m8870_r1/*.json`.

---

## 6. Pitfalls / limits of this round

* **The good-looking test suite was the trap.** 31 tests were green while three
  verification routines were tautological and two kernels biased. Any check whose
  reference is generated by the code under test (D3/D4) is worthless; every fixed
  check now has an independent side (adaptive quadrature, BVP, FE, closed form).
* **`1/sqrt` endpoint singularities are not removed by `θ = θ_B t²`.** Only a
  substitution that makes the square-root argument behave like `t²` at the
  singular end removes them; the residual bias was 1–2 %, i.e. the same size as
  the acceptance tolerance it was supposed to police.
* **`np.polynomial.legendre.leggauss` is expensive** (O(n²) root finding). Calling
  it inside a 100-point shape loop made one load-deflection curve take 250 s;
  caching makes it 0.2 s.
* **Sign conventions in limit-point searches are load-bearing.** A sign flip plus
  an arbitrary displacement window produced a "limit load" 4.7× too large that
  then propagated into `budiansky_roth_reference`. Boundary extrema must be
  flagged, not reported.
* **Point load on a von Kármán plate** has a logarithmic singularity at the centre;
  the FE model applies `P/(2π)` per radian and is validated against the linear
  limit (`w₀ = Pa²/(16πD)`, error 0.03 %) — the comment in `solve()` that "we drop
  2π" refers to the per-radian formulation, not to a missing factor.
* Not addressed in this round: second-order elastica element (D8), the
  multi-element `ShallowArchModel.solve_arc_length` path (only the von Mises
  truss was re-validated), and Poisson dependence of `c(nu)` beyond the linear
  interpolation in `[0.15, 0.45]`.