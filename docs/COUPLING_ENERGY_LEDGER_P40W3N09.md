# Coupling-energy ledger, node P40W3-N09: frozen acceptance criteria and the machine verdict

**Campaign node** `P40W3-N09` · **mission** 8645 (re-opened from 7607 / PR #178)
· **repo** TensorPhysics · **module** `tensorphysics.coupling_energy`
· **record** `artifacts/p40w3b_n09/verification.json` · **criterion digest** `809e40fd6556ede3844ff5d190a8886849d17c49b1713382e6b44a58ca85a2b6`
· **verdict** **REAL** (15 PASS, 0
FAIL, 0 WITHHELD)

---

## 1. Case introduction

A partitioned multi-physics simulation is only as trustworthy as its interface
budget. Three couplings share one contract: **every energy or impulse that
crosses an interface must be measured by the solver that produced it and booked
as the identical number by the solver that received it.** The conserved quantity
is evaluated from the solvers themselves at every exchange instant,

```
eps(t) = | sum_sides E_side(t) - Q_external_in(t) - (same at t = 0) | / scale
```

so the residual is a *physical* non-conservation and not a bookkeeping identity.

| coupling | physics | what must close |
| --- | --- | --- |
| (a) thermal | 2D square cavity `L = 0.1 m`; lower half Q4 conduction FEM, upper half D2Q9 passive-scalar LBM; interface at `y = L/2`; Dirichlet `dT = 50 K` at `y = 0` | energy: `E_fem + E_lbm = Q_in` |
| (b) mechanical | unsupported plane-stress plate (0.2 m) struck edge-on by two DEM discs | momentum: `P_plate + P_particles = const` (Newton's third law) |
| (c) three-field | clamped hull panel with a hot lower face, water column above, ice discs hitting the free edge | energy chain across LBM / FEM / DEM |

Configuration of the production record, read from the artifact (not
transcribed): thermal `dt_coarse = 1.563 s`,
`n_steps = 1280` at `r = 1`, `t_end =
2000.000 s`, identical diffusivity
`alpha = 1.000e-06 m^2/s` on both
sides (so the exact solution is a single 1-D slab); mechanical `dt =
4.059e-06 s`, `t_end = 0.002 s`,
contact stiffness `5.000e+06 N/m`, maximum
overlap `1.157e-04 m`; three-field thermal step
`25.000 s` vs mechanical step
`3.247e-06 s` (time-scale separation
`7.699e+06`).

## 2. What this work adds: the ledger gets a judge

Before this change the acceptance gates lived inline in the campaign runner
(`scripts/run_coupling_energy_contract.py`) as ~100 lines of ad-hoc comparisons.
Four weaknesses followed from that placement, and each is a way for a false
claim to survive:

1. **A stored record could not be re-gated** — the booleans existed only while
   the runner was alive.
2. **Partial evidence looked like a pass** — a two-ratio suite still returned
   `PASS` for every gate it happened to evaluate.
3. **No criterion provenance** — nothing recorded which thresholds the record
   was judged against, so editing a threshold silently re-scored old artifacts.
4. **Vacuous passes** — `eps = 0` is trivially satisfied by a run in which the
   two physics never touched; the momentum residual of an uneventful impact is
   exactly zero, which reads as perfect conservation.

`tensorphysics/coupling_energy/acceptance.py` answers these with four mechanisms:

* **Frozen criteria.** Every gate is a `Criterion` carrying id, coupling,
  quantity, requirement string, gate class, rationale, baseline and the *numeric
  threshold*. The set is hashed into
  `criteria_digest = 809e40fd6556ede3...` and written into the
  record, so a record can be re-judged against the criterion set it claims to
  satisfy. (Finding: the first version kept the numbers inside the checking
  functions, so the digest attested a requirement text whose threshold was not
  the enforced one; `test_frozen_thresholds_are_the_enforced_ones` pins the fix.)
* **Three-valued outcomes** (`PASS` / `FAIL` / `WITHHELD`). A gate whose input is
  absent is `WITHHELD`, never `PASS`; `FAIL` is reserved for measured
  violations.
* **Vacuity guards.** `mechanical_contact_engaged` (max overlap > 0 in every
  shared run) and `three_field_physics_engaged` (`Q_in > 0` and
  `E_contact > 0`) are mandatory, so a record with uneventful physics can never
  reach `REAL`. Not hypothetical: the chain run at `n_macro = 30` has zero
  contact energy and its `4e-16` residual was indistinguishable from a genuine
  closure before the guard existed.
* **Re-gating and drift audit.** `evaluate_record()` reproduces the verdict from
  `verification.json` alone; `audit_drift()` reports criterion-set drift, status
  flips and numeric drift between two records.

Reproduction (the record below was produced by exactly this command):

```bash
python3 scripts/run_coupling_energy_contract.py --out artifacts/p40w3b_n09
python3 -m tensorphysics.coupling_energy.acceptance artifacts/p40w3b_n09/verification.json
python3 scripts/report_coupling_acceptance.py     # regenerates this document
```

## 3. Results

### 3.1 Verdict and criterion table

Verdict **REAL**; criterion digest `809e40fd6556ede3844ff5d190a8886849d17c49b1713382e6b44a58ca85a2b6`;
15 PASS / 0 FAIL / 0
WITHHELD over 15 frozen criteria.

| criterion | gate | frozen requirement | measured | status |
| --- | --- | --- | --- | --- |
| `thermal_energy_conservation` | mandatory | eps_max < 1e-12 over every shared LBM<->FEM run | 2.231e-13 | **PASS** |
| `thermal_profile_accuracy` | mandatory | rel-L2 temperature error < 3 % vs exact 1-D slab (r = 1) | 0.019 | **PASS** |
| `thermal_falsification_ratio` | mandatory | independent-flux control residual > 1e3 x shared residual | 1.763e+12 | **PASS** |
| `mechanical_momentum_conservation` | mandatory | max |P(t)-P(0)| / |P(0)| < 1e-10 (unsupported plate) | 1.423e-13 | **PASS** |
| `mechanical_contact_engaged` | mandatory | max contact overlap > 0 in every shared mechanical run | 1.120e-04 | **PASS** |
| `mechanical_energy_convergence` | evidence | energy drift monotone decreasing in r and < 5 % at r = 8 | {'1': 0.15471683205285708, '2': 0.07332520127006628, '4': 0.03524064611970841, '8': 0.016813541999185787} | **PASS** |
| `mechanical_falsification_ratio` | mandatory | lagged-reaction control momentum residual > 1e2 x shared | 1.407e+13 | **PASS** |
| `three_field_energy_chain` | mandatory | eps_max < 1e-3 over every shared LBM<->FEM<->DEM run | 4.175e-06 | **PASS** |
| `three_field_thermal_subchain` | mandatory | thermal sub-chain residual < 1e-12 inside the chain | 1.400e-14 | **PASS** |
| `three_field_physics_engaged` | mandatory | every shared chain run has external heat input > 0 and contact energy > 0 | {'q_external_total_J': 11277.66420784264, 'max_contact_energy_J': 0.2503398879917597} | **PASS** |
| `three_field_falsification_ratio` | mandatory | independent-flux control residual > 1e2 x shared | 1.217e+05 | **PASS** |
| `interpolation_order_scan` | mandatory | measured exchange order 1.0 / 2.0 / 3.0 within (-0.3, +0.3) / (-0.3, +0.3) / (-0.5, +0.5) | {'zoh': 0.9982695568376241, 'linear': 1.9683578251844187, 'quadratic': 2.8147134503430338} | **PASS** |
| `plate_first_frequency` | mandatory | FEM clamped-free axial frequency within 3 % of c/(4L) | 0.003 | **PASS** |
| `matrix_completeness` | mandatory | 48 runs = 4 ratios x 3 schemes x 3 couplings + 4 x 3 controls | 48 | **PASS** |
| `falsification_arm_coverage` | mandatory | each of the three couplings contributes >= 1 independent run | {'thermal': 4, 'mechanical': 4, 'three_field': 4} | **PASS** |

### 3.2 (a) thermal: conservation residual and profile accuracy

| r | scheme | coarse steps | eps_max [-] | profile rel-L2 [%] |
| --- | --- | --- | --- | --- |
| 1 | zoh | 1280 | 2.181e-13 | 1.86 |
| 1 | linear | 1280 | 2.231e-13 | 1.86 |
| 1 | quadratic | 1280 | 2.219e-13 | 1.86 |
| 2 | zoh | 640 | 6.940e-14 | 3.58 |
| 2 | linear | 640 | 6.973e-14 | 3.58 |
| 2 | quadratic | 640 | 6.785e-14 | 3.58 |
| 4 | zoh | 320 | 5.916e-14 | 6.62 |
| 4 | linear | 320 | 5.809e-14 | 6.62 |
| 4 | quadratic | 320 | 5.889e-14 | 6.62 |
| 8 | zoh | 160 | 2.795e-14 | 11.31 |
| 8 | linear | 160 | 2.753e-14 | 11.32 |
| 8 | quadratic | 160 | 2.666e-14 | 11.32 |

The ledger stays at round-off for all twelve shared combinations (max
2.231e-13) while the profile error is monotone in the sub-cycling ratio: the
*coupling* is exact, the *sub-cycling* is first order. The reconstruction order
moves the solution error only in the third decimal, which is reported as it is.

### 3.3 (b) mechanical: Newton's third law and the energy budget

| r | scheme | max momentum residual [-] | max energy drift [-] | max overlap [m] |
| --- | --- | --- | --- | --- |
| 1 | zoh | 1.385e-13 | 0.157 | 1.157e-04 |
| 1 | linear | 1.423e-13 | 0.155 | 1.153e-04 |
| 1 | quadratic | 1.395e-13 | 0.156 | 1.154e-04 |
| 2 | zoh | 1.373e-13 | 0.078 | 1.142e-04 |
| 2 | linear | 1.386e-13 | 0.073 | 1.133e-04 |
| 2 | quadratic | 1.385e-13 | 0.076 | 1.136e-04 |
| 4 | zoh | 1.386e-13 | 0.041 | 1.136e-04 |
| 4 | linear | 1.408e-13 | 0.035 | 1.124e-04 |
| 4 | quadratic | 1.401e-13 | 0.039 | 1.129e-04 |
| 8 | zoh | 1.409e-13 | 0.026 | 1.132e-04 |
| 8 | linear | 1.418e-13 | 0.017 | 1.120e-04 |
| 8 | quadratic | 1.411e-13 | 0.021 | 1.125e-04 |

Momentum is conserved to 1.423e-13 at every ratio: one scalar contact force
per contact, applied with opposite sign to the two sides. The energy drift is a
property of the explicit DEM sub-stepping, not of the coupling, and falls by
≈2 per doubling of `r` (0.155 at r=1 →
0.017 at r=8); it is registered as an *evidence*
criterion (a regression stays visible, but cannot overturn the verdict).

### 3.4 (c) three-field energy chain

| r | scheme | eps_total | eps_thermal | eps_mechanical | Q_in [J] | E_contact [J] |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | zoh | 4.175e-06 | 8.845e-16 | 0.123 | 1.128e+04 | 0.264 |
| 1 | linear | 4.122e-06 | 8.845e-16 | 0.121 | 1.128e+04 | 0.262 |
| 1 | quadratic | 4.148e-06 | 8.845e-16 | 0.122 | 1.128e+04 | 0.262 |
| 2 | zoh | 1.624e-06 | 2.991e-15 | 0.062 | 1.402e+04 | 0.258 |
| 2 | linear | 1.535e-06 | 2.991e-15 | 0.058 | 1.402e+04 | 0.255 |
| 2 | quadratic | 1.578e-06 | 2.991e-15 | 0.060 | 1.402e+04 | 0.256 |
| 4 | zoh | 7.295e-07 | 8.156e-15 | 0.033 | 1.561e+04 | 0.256 |
| 4 | linear | 6.318e-07 | 8.156e-15 | 0.028 | 1.561e+04 | 0.252 |
| 4 | quadratic | 6.828e-07 | 8.156e-15 | 0.031 | 1.561e+04 | 0.253 |
| 8 | zoh | 4.362e-07 | 1.400e-14 | 0.021 | 1.612e+04 | 0.255 |
| 8 | linear | 2.829e-07 | 1.400e-14 | 0.014 | 1.612e+04 | 0.250 |
| 8 | quadratic | 3.406e-07 | 1.400e-14 | 0.016 | 1.612e+04 | 0.252 |

The chain residual (max 4.175e-06) is dominated by the mechanical sub-system,
while the thermal sub-chain alone closes to 1.400e-14. That is why the
thermal sub-chain has its own mandatory criterion: the total gate alone would
hide a broken thermal exchange behind the mechanical drift.

### 3.5 Falsification arm

| coupling | shared eps_max | independent control eps_max | ratio |
| --- | --- | --- | --- |
| thermal | 2.231e-13 | 0.362 | 1.623e+12 |
| mechanical | 1.423e-13 | 1.996 | 1.403e+13 |
| three_field | 4.175e-06 | 0.488 | 1.170e+05 |

Removing the shared interface number destroys the contract by 5-14 orders of
magnitude. A gate that cannot be broken cannot certify anything.

### 3.6 Instrument checks

| scheme | declared order | measured order | deviation [%] |
| --- | --- | --- | --- |
| zoh | 1.0 | 0.998 | 0.17 |
| linear | 2.0 | 1.968 | 1.58 |
| quadratic | 3.0 | 2.815 | 6.18 |

FEM physics check: clamped-free axial frequency 1868.069 Hz
against the analytic `c/(4L) = 1863.390 Hz`, i.e.
0.25 % relative error at `nx = 20`.

### 3.7 Smoke mode is not certified

`--quick` runs (coarse grid, short horizon) measure a thermal profile error of
≈3.4 % and therefore **fail** the 3 % accuracy gate. That is the intended
behaviour: the gate distinguishes "the contract holds at production resolution"
from "a smoke run finished". Only the production record above is certified.

## 4. Figures

The runner writes these next to the record (`artifacts/p40w3b_n09/`, listed in
`figures.json`):

| file | content |
| --- | --- |
| `fig_thermal_energy.png` | stored energy, external input, interface transfer, and the residual per sub-cycling ratio |
| `fig_thermal_profile.png` | coupled temperature profile against the exact 1-D slab solution |
| `fig_interpolation_order.png` | measured exchange-quadrature order vs. declared |
| `fig_mechanical_momentum.png` | impact energy partition and the momentum residual with its control |
| `fig_three_field_chain.png` | three-field energy chain and residual vs `r` |
| `fig_residual_vs_ratio.png` | residual of all three couplings vs sub-cycling ratio |

## 5. Notes, traps and limitations

* **The vacuous-pass trap is real.** A chain run short enough that the ice never
  reaches the plate reports a chain residual of 4e-16: a perfect closure of a
  coupling that never happened. Both vacuity guards were written because of it.
* **Threshold provenance matters more than the thresholds.** The first version
  of the module kept the numbers in the checking code; the digest then certified
  text it did not enforce. Thresholds now live in the criterion.
* **`WITHHELD` is not a soft `FAIL`.** A partial matrix is withheld, not failed:
  the record did not test the full contract, and the honest statement is
  "insufficient evidence", not "the contract is broken".
* **The three-field clocks are not synchronised.** Thermal (s) and mechanical
  (µs) physics differ by ~1e5; each macro step advances both by one of their own
  stable steps. The ledger is a snapshot statement, so no common physical time
  is required — but thermomechanical coupling (thermal stress / expansion) is
  **not** modelled and no claim is made about it.
* **Energy drift is an integrator property**, registered as evidence with a
  convergence requirement rather than as a conservation claim.
* **The interface wall is first order.** The one-node Dirichlet LBM wall sits
  half a cell outside the first node, which is why the profile error — not the
  instantaneous wall value — is the accuracy criterion.

## 6. Error adjudication against benchmarks

### 6.1 Key quantities versus reference values

| quantity | reference | measured | error | < 3 % ? |
| --- | --- | --- | --- | --- |
| thermal profile, r = 1 (production grid) | exact 1-D slab series | 1.86 % | 1.86 % | **yes** |
| thermal profile, r = 8 | exact 1-D slab series | 11.32 % | 11.32 % | no (sub-cycling error, see 6.2) |
| FEM plate first axial frequency | c/(4L) = 1863.39 Hz | 1868.07 Hz | 0.25 % | **yes** |
| exchange order, zero | declared 1 | 0.998 | 0.17 % | **yes** |
| exchange order, first | declared 2 | 1.968 | 1.58 % | **yes** |
| exchange order, second | declared 3 | 2.815 | 6.18 % | no (see 6.2) |

### 6.2 Explicit answers

* **Thermal profile vs exact solution: 1.86 % at r = 1 —
  yes, < 3 %.** At `r = 8` the error is 11.32 %,
  which is **not** < 3 %: that is documented sub-cycling error (≈ first order in
  `r`), and is why the accuracy criterion is frozen at the finest exercised
  ratio while the `r`-dependence is reported in §3.2 instead of hidden.
* **FEM plate frequency: 0.25 % — yes, < 3 %.**
* **Exchange order, zero and first: 0.17 % and
  1.58 % — yes, < 3 %.** The quadratic
  scheme measures 2.815 against a declared 3
  (6.18 %), i.e. **not < 3 %**: the
  fit uses a finite refinement window `h = 0.05…0.0125`, so the observed order
  is an asymptotic estimate, not the analytic exponent. The criterion therefore
  admits ±0.5 for the quadratic scheme and states so openly.
* **Conservation residuals have no benchmark to be within 3 % of.** The
  reference is machine round-off; the measured residuals are 2.231e-13
  (energy), 1.423e-13 (momentum) and 4.175e-06 (chain) — 8-13 orders of
  magnitude above round-off but 5-14 orders below the falsification arm (§3.5).
  They are asserted as absolute thresholds; the quantity in this study that *can*
  be compared with a benchmark is the falsification ratio.

No quantity here is quoted as "within 3 %" without naming its reference above.
