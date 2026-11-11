# Pre-registration — p171w1-n01 ROM a-posteriori error bound study

**Node**: p171w1-n01-rom-posteriori-error-bound (mission 12489, TensorPhysics)
**Design frozen**: before the 120-case production run (`artifacts/rom_certified_bound/results.json`)
**Amendments**: recorded in §7 (all made before the production run, informed by pilot probes).

## 1. Question

Does the residual-type bound `||r||_{V'} / (beta_lb - L ||v_rom||_V)`, evaluated with a
*spectrally weighted* Stokes gain and the accumulated per-mode response, upper-bound the true
POD-Galerkin(-DEIM) ROM error on 120 independently seeded trajectories, with controlled
tightness, and can it be computed for a small fraction of a FOM step?

## 2. Design (12 groups x 10 trajectories = 120 cases)

| Factor | Levels |
|---|---|
| Reynolds number | 50, 100, 200 (`nu = 2*pi/Re`, U=1, L=2*pi) |
| Initial amplitude | 0.02, 0.05 (scaling of the seeded divergence-free field) |
| DEIM points `r_deim` | 3, 6 |
| Trajectory seed | 100..109 |

Fixed: grid 16x16, periodic [0,2pi]^2, Kolmogorov forcing amp 0.5 / kx 2,
warm-up 1.0 s, collection T = 2.0 s, dt = 0.01, snapshot/output interval 0.05 s
(41 output times), ROM rank N = 8.  The FOM is integrated with the reality-safe RK4
(`step_rk4_real`) and the corrected Leray projection.

## 3. Estimators (primary / secondary declared now)

* Residual `r(t) = F(x_rom(t)) - Phi * da/dt(t)` evaluated with the FOM right-hand side.
* `steady` (secondary): `e_inf = (nu Lap)^-1 r` (quasi-static Stokes response), L2 and V norms.
* `acc` (**primary**): per-mode ODE `dw_k/dt = -nu|k|^2 w_k + r_k`, initialised with the exact
  projection error `x(0) - (xbar + Phi a(0))`, advanced over each output interval with a
  piecewise-linear residual; L2 and V norms.
* Primary estimator refinements (declared, both beta-independent):
  * running maximum of the response over the last 0.5 s (10 output intervals) — reconciles the
    phase of the oscillatory true error with the smooth residual response;
  * safety factor `s = 1.5` for the neglected nonlinear transfer `N(x) - N(x_rom)`.
* Negative control: the same estimator with the Stokes gain inflated 2x (`nu_scale = 2`,
  i.e. `beta_lb -> 2 beta_lb`); the bad bound must be *detected* as an underestimate.

## 4. Metrics and analysis protocol (fixed before the run)

* Evaluation window: `t in [0.2, 2.0]` (the first 0.2 s is the initial projection transient in
  which FOM and ROM still coincide to rounding).
* Degenerate pairs: a pair `(bound, err)` is excluded when `err < 1e-6 * ||x_fom||`;
  excluded count is reported per case.
* `ratio = bound / true_error` computed separately for the L2 and the V norms.
* Case statistic = median and max of the ratio over its window; pooled statistic = median of case
  medians and max over all pairs.
* Strictness: a case is strict if **every** retained pair satisfies `bound >= err` in **both**
  norms. C1 rate = strict cases / 120.

## 5. Machine acceptance criteria (from the campaign node)

| ID | Criterion | Pass condition |
|---|---|---|
| C1 | strictness | 120/120 cases strict in both norms (1 violation => FAIL) |
| C2 | tightness | pooled median ratio <= 10, pooled max ratio <= 50, and Spearman(N, median ratio) <= -0.8 over N in {2,4,6,8,10} |
| C3 | negative control power | pooled underestimate-detection rate of the 2x-inflated bound >= 0.5 |
| C4 | m6745 cross-check / regression | (a) N=20, Re=100 pipeline reproduces itself across processes to L2 rel. diff <= 1e-8; (b) cross-adjudication against the m6745 delivered benchmark numbers reported with root cause |
| C5 | DEIM monotonicity | Spearman(r_deim, median ratio) <= -0.8 at fixed N, over r_deim in {2,3,4,6,8,12} |
| C6 | cost | amortised bound-evaluation time / FOM single-step time <= 0.05 |

A criterion that cannot be met is reported as honest FAIL with the measured numbers; it is not
redefined after the fact.

## 6. Known limitation declared in advance

The bound is *validated empirically*; the derivation identifies the constants but the strictness
guarantee is a discrete-time, quasi-static argument plus a Grönwall factor, not a computer-checked
theorem.  The envelope window and the safety factor are declared modelling choices; §5 of the
report quantifies the strictness/power trade-off they induce.

## 7. Amendments before the production run (with reasons)

| # | Amendment | Reason |
|---|---|---|
| A1 | output interval 0.04 s -> 0.05 s (41 instead of 51 outputs) | cost budget (C6): each output costs one FOM residual evaluation |
| A2 | added the 0.5 s running-maximum envelope | pilot probes: pointwise effectivity dips to 0.57-0.68 because the true error oscillates through minima while the residual response is smooth |
| A3 | added the declared safety factor s = 1.5 | same cause; a strictness/power trade-off curve is reported (`artifacts/rom_certified_bound/tradeoff.json`) rather than a single tuned number |
| A4 | moved from piecewise-constant to piecewise-linear residual in the accumulator | removes a first-order lag when the residual grows inside an interval (free: no extra residual evaluation) |
| A5 | replaced the legacy FOM by the reality-safe RK4 and the corrected Leray projection | defect found in the legacy FOM: the projection sign doubled the divergence; the legacy RK4 accumulated spurious imaginary parts |

Pilot probes (not part of the 120-case evidence) used Re = 50/100/200, seeds 100-102 and are
reproducible from `runs/probe/`.