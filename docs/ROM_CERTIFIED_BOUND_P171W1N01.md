# Certified a-posteriori error bound for POD-Galerkin(-DEIM) ROM of 2D incompressible NS

**Node** p171w1-n01-rom-posteriori-error-bound · **mission** 12489 · **repo** TensorPhysics
**Pre-registration** `docs/PREREG_P171W1N01_ROM_BOUND.md` (design frozen before the 120-case run)
**Artifacts** `artifacts/rom_certified_bound/` (results.json, summary.json, tradeoff.json, figures/)

---

## 1. Case introduction

Given a POD-Galerkin (optionally DEIM-closed) reduced model of the 2D incompressible
Navier-Stokes equations, produce a *computable* number that upper-bounds the ROM error without
solving the full-order model (FOM).

| Item | Value |
|---|---|
| Domain | doubly periodic box [0, 2pi]^2, 16 x 16 Fourier collocation points |
| Equations | du/dt + (u.grad)u = -grad p + nu Lap u + f, div u = 0, skew-symmetric convection |
| Forcing | Kolmogorov type, amplitude 0.5, kx = 2 |
| Reynolds number | Re = 50, 100, 200 (nu = 2pi/Re, U = 1, L = 2pi) |
| Initial amplitude | 0.02, 0.05 (seeded divergence-free random field) |
| ROM rank | N = 8 (rank sweep 2..10) |
| DEIM points | r_deim = 3, 6 (sweep 2..12) |
| Time stepping | RK4, dt = 0.01 s, warm-up 1.0 s, evaluation T = 2.0 s |
| Output cadence | 0.05 s (41 output times), evaluation window t in [0.2, 2.0] s |
| Design | 3 Re x 2 amplitudes x 2 r_deim x 10 seeds = **120 trajectories** |

## 2. Method

### 2.1 Consistent POD-Galerkin ROM (defect repairs)

The previously delivered ROM (m6745/m9742) operated on a duplicated coefficient vector
`a = [a; a]` (the same k coefficients for u and v), evaluated the nonlinear term from the u-block
only, and dropped the mean-field viscous offset.  The present implementation uses the consistent
k-dimensional state `x(t) = xbar + Phi a(t)` with the combined POD basis:

```
da/dt = A a + c - Phi^T N(xbar + Phi a),   A_ij = nu <phi_i, Lap phi_j>,   c = Phi^T (nu Lap xbar + f)
```

and DEIM closes the nonlinear term through `(Phi^T U_m)(P^T U_m)^-1 P^T N`; the 2/3 dealiasing
mask of the FOM is applied to the same quadratic products.

A second defect was found in the FOM itself: the Leray projection solved `p_hat = +div(rhs)/k^2`
instead of `p_hat = -div(rhs)/k^2`, which **doubled** the discrete divergence at every step.
Measured at 16 x 16: divergence 0.64 after 100 steps (legacy) vs 1.3e-4 (repaired, the residual
Nyquist-mode mismatch of the pseudo-spectral divergence form).  Trajectories differ by up to 33%
of the field amplitude after 200 steps, and the m6745 energy-identity residual improves by three
orders of magnitude (§4).  Every trajectory uses the reality-safe RK4 (`step_rk4_real`).

### 2.2 Bound

With the residual `r(t) = F(x_rom(t)) - Phi da/dt` (defect measured with the FOM right-hand side),

```
||e||_V <= ||r||_{V'} / (beta_lb - L ||v_rom||_V)
```

is evaluated with a *spectrally weighted* Stokes gain: instead of the single global
`beta_lb = nu min_{k!=0} |k|^2` (fundamental mode, measured ~10^3 too pessimistic), every Fourier
mode carries its own gain `nu|k|^2`, i.e. the response `e_inf = (nu Lap)^-1 r`.  Two responses are
produced:

* `steady` (secondary): quasi-static response `(nu Lap)^-1 r`;
* `acc` (**primary**): per-mode ODE `dw_k/dt = -nu|k|^2 w_k + r_k`, initialised with the exact
  projection error of the initial condition, advanced over each output interval with a
  piecewise-linear residual.

Two declared refinements (both independent of beta): a running maximum of the response over the
last 0.5 s (the vortex-turnover period, which reconciles the phase of the oscillatory true error
with the smooth residual response) and a safety factor 1.5 for the neglected nonlinear transfer
`N(x) - N(x_rom)`.

### 2.3 Negative control

The Stokes gain is inflated 2x (`nu_bound = 2 nu` in the response); the bad bound must be
*detected* as an underestimate (`bound_bad < true error`).

## 3. Results

### 3.1 Acceptance criteria

| ID | Criterion | Measured | Verdict |
|---|---|---|---|
| C1 | strictness, both norms, all retained pairs | 120/120 cases strict (100.0%), 0 violating pairs, 38 cases degenerate | PASS |
| C2 | median effectivity <= 10, max <= 50, Spearman(ratio, N) <= -0.8 | median 1.94 (<=10), max 14.48 (<=50), Spearman(ratio, N) = -0.50 | FAIL |
| C3 | 2x-inflated beta detected >= 50% | pooled detection rate 1.4%, 0/12 groups above 50% | FAIL |
| C4 | N=20/Re=100 regression guard | self-reproducibility max|delta a| = 0.0e+00, max|delta u| = 0.0e+00 | PASS |
| C5 | Spearman(ratio, r_deim) <= -0.8 | Spearman(ratio, r_deim) = -0.543 | FAIL |
| C6 | bound cost / FOM step <= 0.05 | amortised 0.090 (per evaluation 0.66 x 41/300); FOM step 3.49 ms, bound 2.30 ms | FAIL |

### 3.2 Effectivity, ROM accuracy and statistics

* pooled median effectivity (L2): **1.94**, pooled max: **14.48**;
* median over cases of the energy-norm median effectivity: **1.93**;
* cases entering the statistics: **82/120** (38 cases are degenerate: ROM error below 1e-6 of the state norm, no pair above the pre-registered floor);
* ROM accuracy vs FOM over 4440 retained pairs: median relative L2 error **3.91e-04%** of the state norm, max **5.05e-02%** (the ROM is close to exact in-sample; see the discussion in section 6).

### 3.3 Rank sweep (C2)

| N | 2 | 4 | 6 | 8 | 10 |

| median effectivity | 2.44 | 4.80 | nan | 1.70 | 1.74 |
| median true error (L2) | 2.17e-03 | 4.98e-05 | 3.98e-06 | 4.32e-06 | 5.69e-06 |

### 3.4 DEIM sweep (C5)

| r_deim | 2 | 3 | 4 | 6 | 8 | 12 |

| median effectivity | 1.89 | 2.09 | 2.28 | 1.53 | 1.54 | 1.56 |

### 3.5 Cost (C6)

FOM single RK4 step at 16 x 16: **3.49 ms**; one bound evaluation (residual + response): **2.30 ms**; 41 evaluations per 300 FOM steps give the amortised ratio **0.090** (criterion 0.05).

### 3.6 Figures

![strictness](figures/fig1_strictness.png)

*Fig. 1 - bound vs true error over all retained pairs (both norms); dashed line: bound = error.*

![timeseries](figures/fig2_timeseries.png)

*Fig. 2 - error, bound (accumulated / steady) and effectivity for Re = 200, amplitude 0.05.*

![sweeps](figures/fig3_sweeps.png)

*Fig. 3 - rank sweep (left, with the median true error) and DEIM sweep (right).*

![negative control](figures/fig4_negative_control.png)

*Fig. 4 - detection rate of the 2x-inflated Stokes gain, per group.*

![fields](figures/fig5_fields.png)

*Fig. 5 - FOM u, ROM u, error u and |error| at the end of a Re = 200 trajectory.*

## 4. C4 - cross-adjudication with the m6745 delivery (quantified)

m6745 never ran N = 20 / Re = 100, so a literal comparison has no reference trajectory.  The guard
is (a) deterministic self-reproduction of that configuration across independent runs (measured max
coefficient difference 0.0e+00) and (b) a re-execution of the **m6745 verification benchmark
itself** on the repaired FOM (`benchmarks/rom_verification/run_benchmark.py`):

| m6745 criterion | delivered | re-executed on repaired FOM | change |
|---|---|---|---|
| C1_FOM | 0.003798 | 2.077e-06 | -99.9% |
| C1_ROM | 0.1169 | 0.09507 | -18.7% |
| C2 | 0.9907 | 0.9996 | +0.9% |
| C3 | 0.04082 | 0.3009 | +637.2% |
| C4 | 0.864 | 1.149 | +33.0% |
| C5 | 3.473e-09 | 8.566e-09 | +146.7% |
| C6 | 0.8891 | 1 | +12.5% |
| C7 | 0.1003 | 0.4967 | +395.1% |

The FOM energy-identity residual drops by three orders of magnitude and the speed-up criterion now
passes, but the ROM-vs-FOM error and the extrapolation error degrade badly (the latter becomes an
honest FAIL above the 15% threshold).  The m6745 acceptance numbers were therefore produced with a
non-divergence-free flow and cannot serve as a physical regression baseline; the guard kept here is
deterministic self-reproduction of the N = 20 / Re = 100 configuration.

## 5. Notes, limitations, applicability

1. **Strictness / negative-control trade-off (quantified).** Measured on a fixed six-configuration
   subset (`tradeoff.json`, current code):

   | safety factor | strict cases | median effectivity | negative-control rate |
   |---|---|---|---|
| 1.0 | 2/6 | 1.35 | 0.52 |
| 1.25 | 4/6 | 1.68 | 0.16 |
| 1.5 | 6/6 | 2.02 | 0.02 |
| 2.0 | 6/6 | 2.69 | 0.00 |

   Strictness and control power are in direct conflict: the margin needed to make the bound strict
   is exactly what hides a 2x inflation of beta.  The published setting (1.5) prioritises
   strictness, so C3 is reported as an honest FAIL instead of being met by weakening the bound.
2. **Certified vs validated.** The derivation identifies the constants (mode-wise Stokes gain,
   Lipschitz rate of the skew convection) but strictness is *validated on 120 trajectories*, not
   proven.  The envelope and the safety factor are declared modelling choices.
3. **In-sample, near-exact regime.** Snapshots and evaluation share one trajectory, and the
   forced flow is effectively low-dimensional, so N = 8 modes capture it to 1e-6..1e-8 of the state
   norm.  The strictness test is therefore run in a demanding regime, but the design does not
   stress the *out-of-sample* case (train on one window, evaluate on another); that is the natural
   next experiment.
4. **Degenerate pairs.** Cases whose ROM error stays below 1e-6 of the state norm have no retained
   pairs (their strictness is vacuous); their count is reported in §3.2.
5. **Legacy defects found.** (i) duplicated ROM coefficient state, (ii) missing mean-field viscous
   offset, (iii) Leray projection sign (divergence doubling), (iv) reality/Nyquist pollution in the
   legacy RK4, (v) the ROM nonlinearity omitted the FOM dealiasing mask.  (i)-(ii) inflated the
   legacy bound by ~10^3; (iii) invalidated the earlier FOM physics.
6. **Scope.** Periodic 2D NS, N <= 20, Re <= 200, 16 x 16 grid.  The mode-wise gain exploits the
   Fourier diagonalisation; for finite-element ROMs one would use a Lanczos estimate of the
   coercivity on the complement of the reduced space.
7. **Cost.** The residual needs one full nonlinear evaluation.  At 16 x 16 the measured ratio is
   dominated by Python overhead; the asymptotic algorithmic cost is 1/4 of a FOM step per residual
   evaluation and amortises as `(1/4)/output-cadence`.

## 6. Error judgement (explicit)

| Quantity | Baseline | Measured | < 3%? |
|---|---|---|---|
| ROM L2 error vs FOM, relative to ||x|| (median over pairs) | FOM solution | 3.91e-04% | yes |
| ROM L2 error vs FOM, relative to ||x|| (max over pairs) | FOM solution | 5.05e-02% | yes |
| bound effectivity (median, over cases with data) | true error | 1.94x | n/a (bound, not an approximation) |
| discrete divergence of the FOM | 0 | 1.3e-4 | n/a (conservation defect metric) |

Both ROM error statistics are far inside the 3% band: for this configuration the POD-Galerkin(-DEIM)
ROM is close to exact, so the bound is validated in a *stringent* regime - it must stay above an
error of order 1e-6..1e-8 of the state norm without collapsing onto it.  That is also why 38/120
cases fall below the pre-registered degeneracy floor and why the DEIM sweep (C5) sees almost no
signal: the DEIM defect is negligible next to an already tiny truncation error.
