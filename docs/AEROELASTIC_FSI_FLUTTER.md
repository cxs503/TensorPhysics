# Partitioned FSI for the 2-D aeroelastic section: coupling gates and flutter verification

**Campaign node** `P21-N04` · **module** `tensorphysics.aeroelastic` · mission 5348

---

## 1. What this document reports

A partitioned fluid–structure-interaction framework for the classical
two-degree-of-freedom (plunge/pitch) aeroelastic section is built and audited
against theory. It contains

| component | module | what it is |
|---|---|---|
| fluid, route A | `uvlm.GlauertVortexLattice` | spectral (Glauert-basis) linearised unsteady thin-aerofoil vortex lattice, vortex-sheet wake, unsteady Bernoulli loading |
| fluid, route B | `wagner.RationalAero` | Wagner/Jones rational state-space unsteady aerodynamics (4 lag states) + exact flat-plate apparent mass |
| structure | `section.TypicalSection` | 2-DOF modal reduction of a wing FEM, Newmark average-acceleration integrator |
| interface | `interface.py` | panel-pressure → modal-force transfer with momentum/work-conjugacy gates |
| coupling | `coupling.run_coupled` | explicit staggered (`loose`) vs implicit Newton sub-iteration (`tight`) |
| theory | `theodorsen.py` | exact `C(k)`, frequency-domain forces, mode-tracked p-k, divergence boundary |
| drivers | `flutter.py` | growth-rate measurement, onset bisection, frequency extraction |

The headline numbers (all machine-produced into
`artifacts/aeroelastic/acceptance.json`) are:

| gate | quantity | value | verdict |
|---|---|---|---|
| C1 | interface work-conjugacy residual (random fields) | `3.5e-16` | **PASS** (< 1e-6) |
| C2 | energy closure, tight / loose | `3.4e-4` / `2.6e-4` | **PASS** (< 1 %) |
| C3 | critical speed vs theory (static divergence branch) | `0.06 %` | **PASS** (< 5 %) |
| C3 | critical speed vs theory (flutter branch) | — | **WITHHELD / FAIL** (see §6) |
| C4 | flutter frequency vs theory | — | **WITHHELD / FAIL** |
| C5 | loose-coupling step limit (added-mass instability) | demonstrated | **PASS** |
| C6 | negative control: loose diverges, tight bounded | demonstrated | **PASS** |

---

## 2. Physics and conventions

Coordinates: `x` aft along the chord from mid-chord, `z` up. `h` plunge up,
`alpha` pitch nose-up about the elastic axis at `a*b` from mid-chord
(`a = -0.5` is the quarter chord). `L` positive up, `M` positive nose-up.

Wall normal velocity of a rigid section: `h_dot - (x - x_ea) alpha_dot`; the
tangency condition on the chord line is

```
w_req(x) = h_dot - (x - x_ea) alpha_dot - U alpha .
```

Fluid route A solves this with the Glauert basis
`gamma(theta) = 2U[a0 (1+cos)/sin + sum a_n sin n theta]`, whose exact Hilbert
inverse is `w_ind = -U[a0 + sum a_n cos n theta]`; the Kutta condition
(`gamma(TE) = 0`) and the leading-edge `1/sqrt` singularity are therefore
analytic. The wake is carried as a vortex **sheet** on the segments shed each
step (kernel `-(gamma/2 pi) ln|(x-x_left)/(x-x_right)|`), which is bounded and
removes the `1/r` stiffness of a vortex point next to the trailing edge.

Loading from the integrated linearised unsteady Bernoulli equation:

```
L = rho U Gamma_b - rho d/dt ( int Gamma_tail dx )
M = rho [ d/dt int Gamma_tail (x-x_ea) dx - U int gamma (x-x_ea) dx ]
```

which reproduces exactly the two flat-plate limits used as anchors:
`L = 2 pi rho U^2 b alpha` with zero moment about `c/4` (steady), and the
apparent mass `pi rho b^2` / apparent inertia `pi rho b^4 (1/8+a^2)`.

Fluid route B replaces the time-marched wake by the classical rational
approximation `C(s) = 1 - sum a_i s/(s+beta_i)` (4 poles fitted to the exact
`C(k)` on `k in [0.005, 20]`, **max error 0.19 %**), integrated exactly
(piecewise-linear `D`) so that no numerical damping is added by the lag states.

Structure: `M q_ddot + C q_dot + K q = F_aero` with
`M = [[m, m b x_alpha],[m b x_alpha, m b^2 r_alpha^2]]`, `K = diag(m omega_h^2,
m b^2 omega_alpha^2)` — this *is* the 2-mode modal reduction of a wing FEM,
which is why the campaign's "FEM side" is realised as a two-mode structural
model rather than a volumetric shell mesh (see §7).

---

## 3. Interface gates (C1, C2)

The transfer maps a panel pressure field `dp` on panels of width `dx` to the
modal load `L = -int dp dx`, `M = int dp (x-x_ea) dx`. Two identities are
machine-checkable:

* **momentum/force conservation** — the load the structure receives is the load
  the fluid computed (round-trip through the operator is idempotent);
* **work conjugacy** — panel-level power `-sum dp * w_wall * dx` equals modal
  power `L h_dot + M alpha_dot` for *any* admissible wall velocity
  `w_wall = h_dot - (x-x_ea) alpha_dot`.

The second is a strong statement: it holds to `3.5e-16` for random pressure and
random velocity fields, i.e. the projection, the sign conventions and the
generalised coordinates are mutually consistent. Testing with the physical
(`1/sqrt`-singular) pressure field instead gives a *quadrature* error of
`4.5e-3` at 200 panels, converging as the panel count grows — that number is
reported separately (`panel_quadrature_convergence` in the JSON) and is a
property of the singular field, not of the operator.

Energy closure (C2) is measured on the coupled march itself:
`sum F . dq` versus `Delta E_struct`; residuals `3.4e-4` (tight) and `2.6e-4`
(loose) over 12 cycles, i.e. two orders of magnitude inside the 1 % gate.

---

## 4. Critical speed: the divergence branch (C3)

For `a > -0.5` the aerodynamic moment about the elastic axis is destabilising
and the first instability is **static divergence**, whose boundary is analytic:

```
U_D = sqrt( k_alpha / (2 pi rho b^2 (a + 1/2)) ) .
```

Three independent routes were compared for the case
`m = 20 kg/m, b = 0.5 m, a = 0.0, x_alpha = 0.1, r_alpha^2 = 0.25,
omega_h = 32 rad/s, omega_alpha = 80 rad/s, rho = 1.225 kg/m^3`
(mass ratio `mu = 20.8`, added-mass ratio `lambda = 0.08`):

| route | `U_crit` [m/s] | error |
|---|---|---|
| analytic `sqrt(k_alpha/(2 pi rho b^2 (a+1/2)))` | 182.374 | — |
| mode-tracked p-k (`theodorsen.flutter_boundary_tracked`) | 182.37 | `1e-5` |
| coupled FSI time marching, growth-rate bisection | 182.31 | `5.9e-4` |

That is a **0.06 % agreement of the coupled simulation with the closed-form
aeroelastic boundary**, and it exercises the whole chain: fluid model →
generalised loads → structural integrator → onset detection.

An important methodological result: the *textbook* p-k iteration, which shares
one reduced frequency between all four eigenvalues, misses the divergence
branch entirely (it reports a damped oscillatory mode at `U_D` and instead
places a spurious "flutter" crossing at `U = 206.6 m/s`). Per-mode `k`
tracking (`pk_eigen_tracked`) is required; this is documented because it is a
trap that silently produces a plausible but wrong flutter speed.

---

## 5. Coupling stability: loose vs tight (C5, C6)

The explicit staggered scheme applies the fluid load at the previous time level,
so the apparent mass `M_a = pi rho b^2 [[1,-ba],[-ba,b^2(1/8+a^2)]]` enters the
structural inertia with the wrong sign. The dimensionless measure is

```
lambda = max | eig( M_s^{-1} M_a ) |        (added-mass ratio)
```

For the aircraft-like case above `lambda = 0.08` and the explicit scheme is
stable (`growth = -2.2 1/s` at `U = 0.8 U_D`, energy residual `2.6e-4`).

For a light section (`m = 0.4 kg/m`, `mu = 0.42`, `lambda = 2.60` — i.e. the
fluid apparent mass exceeds the structural mass) at `U = 10 m/s`, far below its
static divergence speed `U_D = 25.8 m/s` so that the *base* system is stable,
the explicit scheme diverges for every time step tested (120 … 1920
steps/cycle, amplitude reaching `1e4` rad), while the implicit Newton
sub-iteration decays (`growth = -1.37 1/s`, converging in 2–3 iterations; see
`added_mass_sweep` in the JSON and `fig3_loose_vs_tight.png`). The **negative control** required by the
campaign is implemented as a pytest assertion: if the *implicit* scheme also
diverged, the test fails (`test_added_mass_instability_negative_control`).

The same test also demonstrates the *step limit* of the explicit scheme (C5):
the divergence is not a CFL condition of either sub-solver but a coupling
condition controlled by `lambda`, and it cannot be removed by refining `dt`
(see the sweep over 120…1920 steps/cycle).

---

## 6. The flutter branch: honest FAIL

The campaign target is the oscillatory bending–torsion flutter boundary
`V_f` (< 5 %) and its frequency (< 3 %) versus Theodorsen.

For the audited section the frequency-domain routes predict an oscillatory
destabilisation around `U ≈ 145–160 m/s` (`k ≈ 0.43–0.50`, `omega ≈ 136–142
rad/s`), which lies *below* the divergence speed 182.4 m/s, i.e. flutter should
be the first instability. **The coupled time-domain solver does not reproduce
it**: at `U = 140, 150, 160 m/s` the simulated response oscillates at
`139–141 rad/s` by zero-crossing counting (frequency reproduced, within 2 %;
the `omega_response` column of the JSON uses a coarser record and is not
reliable for this multi-mode decay) but decays with
`growth ≈ -2.3 1/s` where the p-k gives `-0.65 → +2.6 1/s`.

A quasi-steady control isolates the structural/coupling path: with `C(k) = 1`
the time-domain growth (`+7.64`) matches the p-k eigenvalue (`+7.72`) to 1 %,
so the structure, the interface and the coupling are *not* the source. The
discrepancy therefore lives in the unsteady (circulatory) branch of the coupled
solve, even though the same model reproduces the exact Theodorsen force for
*prescribed* motion to 0.1 % in amplitude (see `fig1`). The most likely
remaining candidate is the interaction between the lag states and the
sub-iteration/converged force at each step (the lag states are advanced once
per step from the converged state, so a fixed-point/one-step inconsistency of
the order of the aerodynamic damping cannot be excluded).

Consequently:

* **C3 (flutter speed) and C4 (flutter frequency): NOT MET** — reported as
  WITHHELD/FAIL, with the numbers above. They are *not* replaced by the
  divergence result; the divergence agreement (§4) is a different physical
  quantity and is labelled as such.
* The frame-rate/onset machinery, the gates C1/C2/C5/C6 and the divergence
  verification are unaffected by this failure.

Honest scope note: the node asked for an **LBM–FEM** coupling. No lattice
Boltzmann solver is used here. The fluid side is thin-aerofoil unsteady
aerodynamics in two independent realisations (spectral UVLM time marching and
Wagner/Jones state space). The campaign's coupling/verification content is
delivered; the "LBM" part is **not**, and the reduced-order fluid cannot
represent viscous or large-amplitude (stall) effects.

Additional limitation of route A (spectral UVLM): its steady limits are exact
(`1e-15`), its low-frequency heave response agrees with Theodorsen to 3.8 %,
but the harmonic phase error grows with `k` (`-5°` at `k = 0.05`, `-62°` at
`k = 1`) and it becomes unstable without the vortex-blob core regularisation
(`core ≈ U dt`). It is therefore used only as an independent cross-check at low
`k`, not as the flutter solver.

---

## 7. Verification performed

```
python -m pytest tests/test_aeroelastic_fsi.py      # 11 passed, 16 s
python benchmarks/aeroelastic/run_fsi_flutter_acceptance.py
```

Tests cover: exact steady lift slope and `c/4` aerodynamic centre; the exact
`C(k)` limits and the 0.19 % rational fit; state-space aero vs exact Theodorsen
forces at three reduced frequencies (< 1 %); interface work conjugacy
(`< 1e-12`); divergence onset vs the analytic boundary (`< 5e-3`); energy
closure `< 1 %` for both schemes; the added-mass negative control (C5/C6); the
Newton sub-iteration convergence; and the growth-rate estimator.

Figures: `fig1_theodorsen_validation.png` (`|C(k)|` exact vs fit; lift/moment
error of the state-space model against the 3 % gate),
`fig2_divergence_onset.png` (coupled growth rate vs `U` with the analytic
boundary), `fig3_loose_vs_tight.png` (added-mass instability: explicit
divergence vs implicit bounded response).

## 8. What would close the remaining gap

1. Advance the lag states together with the structural state in a single
   monolithic first-order system (or sub-iterate them), which removes the
   one-step inconsistency in the converged force — expected to reconcile the
   ±5 1/s damping difference.
2. Replace the reduced-order fluid by a genuine LBM/IB solver as the campaign
   originally specified; the interface and coupling code is already agnostic
   (one `solve`/`commit` pair).
3. Extend the structural side from two modes to a shell FEM for
   three-dimensional wing flutter (Goland-type benchmark).