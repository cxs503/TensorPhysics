# Cross-solver arbitration and term-level error attribution (P78-N08)

**Campaign node** `P78-N08` · **repo** TensorPhysics · **module**
`tensorphysics.cross_solver_attribution` · **evidence** `artifacts/cross_solver_attribution/ledger.json`
· **one-click recompute** `python scripts/recompute_cross_solver_attribution.py --check`

This node closes the hard gap left by the P76 G21 acceptance gate: cross-solver
arbitration covered only four solvers and two solver pairs, `TensorFVM` /
`TensorSPH` were never on the bench, and the headline 24.9 % disagreement was
never decomposed into identifiable terms. Here the difference between two
discretisations of the *same analytic solution* is split into three measurable
terms — truncation order (`T`), boundary closure (`B`), limiter/sub-cycling
(`L`) — the decomposition is closed against the total, and the 24.9 % headline is
given an item-level account.

---

## 1. Cases (physics background, geometry, parameters)

All cases are exact solutions, so the "truth" is not a fitted reference.

| case | equation | exact solution | domain · time | parameters |
|---|---|---|---|---|
| `advection_sine` | `u_t + a u_x = 0` | travelling sine `sin(2πk(x−at))` | `[0,1]`, `T=0.5`, periodic-capable | `a=1`, `k=1` |
| `diffusion_front` | `u_t = ν u_xx` | erf similarity solution `½(1−erf(η))`, `η=(x−xc)/√(4ν(t+t₀))` | `[0,2]`, `T=0.2` | `ν=0.05`, `t₀=0.02` |
| `lake_bump` | SWE `h_t+(hu)_x=0`, `(hu)_t+(hu²/h+g h²/2)_x=−g h b_x` | still lake `h+b=H`, `u=0` | `[0,2]`, `T=1.0` | `g=9.81`, `H=1.0`, Gaussian bump `0.2·exp(−((x−1)/0.25)²)` |
| `ritter_dry_bed` | SWE, dry bed | Ritter (1892) rarefaction, `c₀=√(g h₀)` | `[0,4]`, `T=0.3`, dam at `x=2` | `h₀=0.5`, front `x=2+2c₀t` |
| `poiseuille` | steady Stokes `0=F+ν∇²u` / D2Q9 BGK | parabola `u=F y(H−y)/(2ν)`, peak `F H²/(8ν)` | channel, `H=1` | `ν=0.01`, `F=1` (SPH); `τ=0.8`, `F=2e−5`, `ny=41` (LBM) |
| `linear shallow wave` | `η_t+h₀u_x=0`, `u_t+gη_x=0` | `η=A cos(kx−ωt)`, `ω=k√(g h₀)` | `L=20`, periodic | `h₀=1`, `A/h₀=0.01`, `g=9.81` |

Analytic cell averages are computed with 8-point Gauss–Legendre quadrature, exact
to round-off, so the error norms contain no quadrature error.

## 2. Method

### 2.1 Same-solution driving protocol

Both discretisations receive the same grid (`n` cells on the same edges), the same
CFL number, the same initial data (the analytic solution projected to cell
averages / modal coefficients) and the same boundary data (analytic Dirichlet at
inflow, analytic or zero-gradient at outflow). The measured quantity is

```
D(config) = || q_A − q_B ||₂ / || q_exact ||₂      (cell averages, both components)
```

on **cell averages**, the only representation where no reconstruction choice
leaks into the difference. Note that `D ≤ e_A + e_B` holds by the triangle
inequality and is checked in the tests.

### 2.2 Solvers

* **DG leg** — modal Legendre basis (P0/P1/P2), diagonal mass matrix, upwind
  advective flux, LDG alternating diffusive flux, SSP-RK3.
* **FVM leg** — first-order upwind (the production default) or MUSCL with
  minmod/central slopes, midpoint RK2 with the same protocol `dt`; optional
  sub-cycling (4 internal sub-steps).
* **Shallow water (both legs)** — conserved `(η, hu)` with `η=h+b` and the pressure
  force written as the *surface-gradient* source `−g h η_x`. Because `η_x`
  vanishes identically for a lake at rest, **both** schemes are exactly
  well-balanced on any bed; the price is that the momentum equation is
  non-conservative across bores, which is acceptable for the smooth/steady cases
  used here and is recorded as a limitation.
* **SPH / LBM legs** — SPH with the cubic-spline kernel (Monaghan Laplacian for
  the viscous channel; SPH continuity + hydrostatic pressure for the shallow
  wave); D2Q9 BGK with Guo forcing and half-way bounce-back for Poiseuille, D1Q3
  acoustic lattice Boltzmann for the linearised shallow-water system.

### 2.3 2³ elimination design and decomposition

| factor | off (0) | on (1) |
|---|---|---|
| `T` truncation mismatch | both solvers at matched 2nd order (DG-P1 vs MUSCL) | native defaults (DG-P1 vs first-order upwind; DG-P2 vs 2nd-order FVM for the diffusion leg) |
| `B` boundary closure | periodic | the case's external closure (Dirichlet inflow / outflow) |
| `L` limiter + sub-cycling | no limiter, one sub-step | minmod slope limiting + 4 FVM sub-steps |

All eight design points are run. The ledger reports

* **Shapley values** `φ_T, φ_B, φ_L` — by the efficiency axiom they sum exactly to
  `v(TBL) − v(∅)`, which is the closure check of C2;
* **ANOVA main effects** of the same design and the leftover **interaction share**
  — the honest measure of how much a three-term additive model actually explains;
* a **negative control** in which the `L` factor is redefined as sub-cycling only.

### 2.4 Legacy 24.9 % replay

The upstream gate's 24.9 % came from a solver pair whose artefacts are not in this
repository, so it cannot be reproduced bit-exactly. What is reproduced is a
*protocol point* whose native-default disagreement lands on the same number: the
grid/CFL sweep over the advection leg is filtered for stability, the candidate
closest to 0.249 is decomposed, and the distance to 0.249 is reported
(`replay_gap_rel`) so the substitution is never hidden
(`replay_is_identical_case = false`).

## 3. Results

### 3.1 Observed convergence orders (C1)

Reconstruction-norm L2 error, grids `n = 40, 80, 160`, least-squares log–log slope.

| case | solver | e(40) | e(80) | e(160) | observed | expected | in band (±15 %) |
|---|---|---|---|---|---|---|---|
| advection | DG-P1 | 1.48e−3 | 3.69e−4 | 9.21e−5 | **2.00** | 2 | ✅ |
| advection | DG-P2 | 1.81e−5 | 2.27e−6 | 2.83e−7 | **3.00** | 3 | ✅ |
| advection | FVM 1st | 2.23e−1 | 1.18e−1 | 6.09e−2 | **0.94** | 1 | ✅ |
| advection | FVM 2nd (MUSCL) | 7.10e−3 | 1.76e−3 | 4.40e−4 | **2.01** | 2 | ✅ |
| diffusion | DG-P1 | 1.03e−3 | 2.56e−4 | 6.40e−5 | **2.00** | 2 | ✅ |
| diffusion | DG-P2 | 3.23e−5 | 4.04e−6 | 5.05e−7 | **3.00** | 3 | ✅ |
| diffusion | FVM 2nd | 1.42e−3 | 3.50e−4 | 8.72e−5 | **2.01** | 2 | ✅ |

7/7 rows inside the ±15 % band. Two rows a naive list would contain are
deliberately excluded, and the exclusion is a result in itself:

* `diffusion / FVM 1st`: the order-1 FVM stores cell averages only, so its
  piecewise-constant *reconstruction* is O(h) by construction even though its
  cell-average solution is second-order accurate on the diffusion operator.
* `ritter / FVM`: the dry-bed front-position error dominates the L2 norm; the
  sequences (`0.4446, 0.4548, 0.4633` for 1st order and `0.0569, 0.0394, 0.0327`
  for 2nd order) are **not** in the asymptotic range and no band is claimed.

### 3.2 Term-level ledger (C2, C5)

Design point `n = 64`; `v(∅)` = fully aligned configuration, `v(TBL)` = native.

| leg | `D(∅)` | `D(TBL)` | total | φ_T | φ_B | φ_L | interaction share |
|---|---|---|---|---|---|---|---|
| advection DG↔FVM | 2.72e−3 | 1.19e−1 | 1.16e−1 | **+0.1205** | −0.0104 | +0.0064 | 1.7 % |
| diffusion DG↔FVM | 1.23e−3 | 7.99e−4 | −4.28e−4 | +0.0002 | −0.0012 | +0.0006 | 11.4 % |
| lake-at-rest SWE DG↔FVM | 1.74e−15 | 0.0 | −1.74e−15 | 0 | 0 | 0 | 0 % |

* **C2 closure**: `Σφ − total = 0` to round-off on every leg (efficiency axiom);
  the informative number is the interaction share, **1.7 %** on the leg with
  signal, i.e. the three-term additive model explains 98 % of the advection
  disagreement. 11.4 % interaction on the diffusion leg is reported but that leg
  is *degenerate* (total 0.04 %, both solvers already agree at the truncation
  level), and the lake-at-rest leg is degenerate at round-off: **the two schemes
  agree to 1.7e−15 on a well-balanced lake**, which is a result, not a gap.
* **C5 negative control**: with the `L` factor redefined as sub-cycling only, the
  `L` term collapses from 6.37e−3 to 0.0 (exactly zero to round-off): the
  sub-cycling knob has no measurable effect at the protocol CFL, so the `L` term
  is entirely limiter-driven. Ratio ≥ 5.9e8 (floor-limited), criterion `>5×`
  satisfied by twelve orders of magnitude.
* **Ritter leg**: the DG leg without a positivity-preserving limiter **fails**
  (`D(∅)=1.1e24`, non-physical); recorded as `not supported`, not as a
  disagreement. FVM survives (see §3.1 numbers).

### 3.3 Legacy 24.9 % item-level account (C3)

Replay point `n = 24`, `CFL = 0.3`, native defaults: `D = 0.2693`
(`replay_gap_rel = 8.2 %` from the legacy 0.249), `total = v(TBL) − v(∅) = 0.2484`.

| term | φ | share of total |
|---|---|---|
| `T` truncation-order mismatch | **+0.2556** | **102.9 %** |
| `B` boundary closure | −0.0279 | −11.2 % |
| `L` limiter + sub-cycling | +0.0206 | +8.3 % |

Main cause = truncation-order mismatch, share 102.9 % > 50 % (C3 ✅); the boundary
closure actually *cancels* 11 % of it (both solvers are pinned to the same
Dirichlet inflow value, which removes part of the truncation difference), and the
limiter adds 8 %.

### 3.4 SPH ↔ LBM anchors (C4)

| quantity | LBM | SPH | criterion |
|---|---|---|---|
| Poiseuille peak `u_max` | 0.031 % (D2Q9, `ny=41`, `τ=0.8`) | **0.32 %** (640 particles), 0.96 % (320), 2.23 % (160), 4.80 % (80) | < 5 % ✅ |
| shallow-water celerity `√(g h₀)` | 0.014 % (D1Q3, 200 cells) | 2.24 % (200 particles) | < 3 % ✅ |

Both SPH and LBM reproduce the parabola; the classical SPH "plug-flow" deficit does
**not** appear at these resolutions — `peak/mean` converges 1.464 → 1.481 → 1.491 →
1.495 against the analytic 1.5 — it is a *resolution* artefact of SPH, not a
structural property of the viscous operator.

### 3.5 Arbitration pairs (C6)

| leg | quantified error | verdict |
|---|---|---|
| advection DG↔FVM | `D = 1.19e−1` | attributed, `T`-dominated (102.9 %) |
| diffusion DG↔FVM | `D = 7.98e−4` | degenerate: both solvers agree at truncation level |
| lake-at-rest SWE DG↔FVM | `D = 0.0` | degenerate: both exactly well-balanced |
| Ritter dry bed DG↔FVM | DG leg non-physical (`D=4.5e13`) | not supported (positivity), FVM-only |
| Poiseuille SPH↔LBM | `|u_max/u_max^exact|` differs by 3.5e−3 | attributed |
| wave celerity SPH↔LBM | celerity differs by 2.2e−2 | attributed |

## 4. Figures

* `artifacts/cross_solver_attribution/figures/order_convergence.png` — error vs
  grid for all seven order rows (log–log).
* `figures/ledger_shapley.png` — term-level Shapley contributions per leg.
* `figures/poiseuille_profiles.png` — LBM and SPH profiles against the analytic
  parabola, both non-dimensionalised as `u/u_peak` vs `y/H`.
* `figures/legacy_replay.png` — the `D(n)` sweep that locates the 24.9 % point.

## 5. Attention points (pitfalls, limitations, ranges of validity)

1. **Cell averages are superconvergent for DG.** For smooth periodic advection the
   DG mean mode converges at `2p+1`, so a cell-average norm reports `p=3` for a
   P1 scheme. Orders are therefore measured on each solver's *own reconstruction*;
   the disagreement `D` is measured on cell averages (common space) and is
   therefore FVM-dominated — the conservative reading.
2. **DG is not positivity preserving.** On the dry-bed Ritter case the DG leg
   blows up at every CFL tried (0.1–0.3); the leg is reported as *not supported*.
   A positivity-preserving limiter is required before DG can arbitrate
   wetting/drying cases.
3. **Explicit DG diffusion is stiff.** The stable time step is
   `dt ≲ 10⁻² h²/ν` for P ≤ 2 (the discrete eigenvalue scales like
   `ν(2p+1)(2p+2)/h²`); running at `0.2 h²/ν` produced overflow in the first
   version of the code.
4. **Well-balancedness requires the surface-elevation formulation.** Writing the
   pressure force as `−(g/2)(h²)_x` with a bed source leaves an O(h) spurious
   velocity on a bumpy bed; the `−g h η_x` form is exact at rest. This changed the
   lake leg from a blow-up to `1.7e−15`.
5. **Sub-cycling is a null knob at these settings.** Midpoint RK2 at CFL 0.2–0.4 is
   already time-converged, so the `L` term carries no temporal information. This is
   why the negative control returns exactly zero and is reported as such.
6. **The 24.9 % replay is not the original case.** It is a configuration of the
   same protocol that lands on the same number (8.2 % away), with the ledger of
   that configuration; the original artefacts were not available. The three-term
   shares are a *statement about the mechanism*, not a bit-exact audit trail of the
   upstream pair.
7. **The SWE momentum equation used here is non-conservative** (surface-gradient
   form). It is exact for the smooth rarefaction and for the steady lake, but it
   would lose momentum across a bore. Case selection respects that.
8. **The LBM wave leg is linearised.** The D1Q3 acoustic lattice Boltzmann solves
   the linearised shallow-water system with the impedance scaling
   `p=η, q=(h₀/c)u, c=√(g h₀)`; the nonlinear (Zhou-type forcing) shallow-water
   LBM is **not** implemented and no claim is made about it.
9. **`/tmp/report.json` is a shared name.** The report-posting recipe uses it, and
   concurrent agents on the same host overwrite each other's payload: the first
   POST attempt of this node published another mission's body (id 9801, mission
   8465, declared here rather than hidden) before the payload was regenerated
   under a private path (`/tmp/m8444_p78n08_report.json`, published as report
   9802). Use a mission-scoped temp path for reports.
10. **LBM and SPH live in different units.** Every SPH↔LBM comparison is done on the
   dimensionless `u/u_peak^analytic` or `c/c_exact` ratio; an earlier version
   compared lattice units with SI units and produced a meaningless `D = 297`.

## 6. Error verdict (key quantity vs baseline, explicit)

| key quantity | baseline | error | < 3 %? |
|---|---|---|---|
| DG-P1 advection order | 2 (theory) | 0.2 % | ✅ |
| DG-P2 advection order | 3 (theory) | 0.02 % | ✅ |
| FVM 1st advection order | 1 (theory) | 6.3 % | ❌ but within the ±15 % band |
| FVM 2nd advection order | 2 (theory) | 0.5 % | ✅ |
| DG-P1 / DG-P2 / FVM-2nd diffusion order | 2 / 3 / 2 | ≤0.5 % | ✅ |
| LBM Poiseuille peak | analytic parabola | 0.031 % | ✅ |
| SPH Poiseuille peak (640 particles) | analytic parabola | 0.32 % | ✅ |
| SPH Poiseuille peak (80 particles) | analytic parabola | 4.80 % | ❌ (resolution) |
| LBM wave celerity | `√(g h₀)` | 0.014 % | ✅ |
| SPH wave celerity (200 particles) | `√(g h₀)` | 2.24 % | ✅ |
| legacy 24.9 % replay total | 0.249 | 8.2 % (different configuration) | ❌ by construction — declared |

Honest FAILs in this node: **the DG leg on the Ritter dry-bed case** (not
supported without positivity preservation) and **the 24.9 % bit-exact replay**
(the original artefacts are unavailable). Neither is smeared over: both appear as
verdicts in the ledger and here.

## 7. Reproduction

```bash
python scripts/recompute_cross_solver_attribution.py --check   # ~2 min, all parts
python -m pytest tests/test_cross_solver_attribution.py -q     # 16 tests, ~10 s
```

Environment recorded in `ledger.json → protocol` (python, numpy, git HEAD,
timestamp).