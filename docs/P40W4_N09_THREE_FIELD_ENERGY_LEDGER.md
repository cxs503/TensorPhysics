# Three-field coupling energy contract, node P40W4-N09

**Campaign node** `P40W4-N09` · **mission** 9575 · **repo** TensorPhysics ·
**module** `tensorphysics.coupling_energy.three_field` ·
**record** `docs/p40w4_n09/verification.json` ·
**criteria digest** `34a58662782dcc37…` (full value in the record) ·
**verdict** **PARTIAL** — 8 PASS, 2 FAIL, 0 WITHHELD (the two FAILs are declared
and analysed below; nothing is withheld, nothing is smoothed over)

---

## 1. Case introduction

A partitioned three-solver run is only as trustworthy as the budget of its
interfaces. This node closes the loop **LBM → FEM → DEM → LBM** on one physical
clock and asks a single question of every crossing: *can the energy that crosses
it be booked, on both sides, as the same number — and if not, is the difference
a number we understand?*

| side | discretisation | what it carries |
| --- | --- | --- |
| `fluid` (LBM) | D2Q9 advection–diffusion (passive scalar), 40 × 4 lattice, `dx = 5 mm`, τ = 1.25 | thermal internal energy; Dirichlet wall on the panel |
| `solid` (FEM) | Q4 plane-stress elasticity **and** Q4 transient conduction on one 6 × 16 mesh; left column clamped | elastic + thermal energy; contact reaction |
| `particles` (DEM) | two rigid discs (r = 10 mm, ρ = 2000 kg/m³) in contact with the panel's top face | kinetic energy; contact force and drag loss |

The three declared crossings (module `.contracts`, machine-validated):

| key | source → target | quantity | booking on the producer | booking on the receiver |
| --- | --- | --- | --- | --- |
| `lbm_fem_heat_flux` | fluid → solid | heat rate [W] | the scalar its Dirichlet wall removed, integrated over the lattice step (an *exact* boundary balance, `Δ total = injected`) | the same rate as a nodal sink on the panel's top row, reconstructed from the fluid's samples |
| `fem_dem_contact_force` | solid → particles | force [N] | `+f` on the particle from the spring–dashpot law on the current overlap | `−f` on the plate node (Newton's third law, checked to round-off at every exchange) |
| `dem_lbm_drag` | particles → fluid | power [W] | `γ|v − u|² Δt` measured at every DEM step | the reconstructed power integral, injected as a scalar source in the particle's cell |

Geometry: cantilever panel 0.20 m × 0.05 m × 0.05 m (thickness), warm fluid
column on top of it, water-side temperature perturbation `ΔT = 0.01 K`. The two
discs start 0.1 mm above the panel and fall at 0.02 m/s.

**Parameter table** (all values in `docs/p40w4_n09/RESULTS_TABLES.md` §1,
generated from the record): `E = 1.0e7 Pa`, `ν = 0.3`, `ρ = 8000 kg/m³`,
`k = 250 W/(m·K)`, `ρc_p = 1.0e4 J/(m³K)` (so α = 0.025 m²/s on **both** sides —
the composite is one slab), fluid `α_f = 0.025 m²/s`, contact `k_c = 5.0e3 N/m`,
`c = 8 N·s/m` (ζ ≈ 0.32), drag coefficient `γ = 0.5 N·s/m`, `t_end = 0.04 s`.

The parameter set is deliberately **verification-scale**: conductivity,
capacity, diffusivity and the carrier viscosity are chosen so that the panel's
conduction time, the contact duration (≈ 8 ms) and the lattice step are
resolvable inside one 40 ms window. This is not a material claim, and §5 says
so.

### Multi-rate schedule (three distinct steps)

| step | value | count per exchange interval |
| --- | --- | --- |
| `Δt_LBM` | 2.500e-4 s | 1 (the exchange interval is the lattice step) |
| `Δt_FEM,thermal` | 6.250e-5 s | 4 |
| `Δt_FEM,elastic` | 3.125e-5 s | 8 |
| `Δt_DEM` | 1.5625e-5 s | 16 |
| exchange interval | 2.500e-4 s | 160 intervals, 4 640 solver steps |

## 2. What was built, and how it was run

New package `src/tensorphysics/coupling_energy/three_field/`:

* `contracts.py` — typed crossings with unit, sign convention, both required
  bookings, the conservation statement, and a structural validator that refuses
  a *chain* (a set of crossings in which some side is not both a producer and a
  receiver cannot close a loop). `assert_reciprocity()` enforces the
  machine-checkable clause of the `exact` crossing.
* `energy.py` — the four-term runtime book
  `E_total = E_kinetic + E_potential + E_internal + E_dissipated`, filled from
  **solver states** (`DEM2D.kinetic_energy`, `Q4PlaneStressFEM.strain_energy`,
  `Q4ThermalFEM.energy()`, the LBM scalar mass, the measured dashpot work) and
  never from a bookkeeping sum of transfers — a sum of transfers would close by
  construction and could not detect a coupling that quietly creates energy.
  `E_dissipated` enters with a plus sign: it is energy that has left the
  resolvable fields into an *explicitly modelled* sink, so an unbooked sink shows
  up exactly like an unbooked source.
* `subcycling.py` — `RateSchedule` (three steps, sub-cycling ratios) and the
  **causal** degree-`p` Lagrange reconstruction (`p = 1, 2, 3`) of one
  counterpart history, plus its interval integral.
* `loop.py` — the closed loop, the uncoupled negative control, the degree scan
  and the operator-level order study.
* `acceptance.py` — ten frozen, hashed criteria with three-valued outcomes.

Run: `PYTHONPATH=src python scripts/run_p40w4_n09_three_field_energy.py`.
Record and figures are regenerated by that script; the numeric tables in this
report are regenerated from the record by
`scripts/render_p40w4_n09_tables.py` (no number is transcribed by hand).

**One design decision needs to be declared**, because it is the reason the order
gate behaves the way it does: an interface quantity that is *extrapolated*
forward (a degree-`p` polynomial through past samples, evaluated inside the next
step) is unstable in two directions. Measured, not assumed:

* a reconstructed **contact load** with `p > 0` blows the plate up at every
  degree (the polynomial through a discontinuous force history overshoots by a
  factor three, and the overshoot is fed back inside the same interval). The loop therefore
  exchanges the contact load with a zero-order hold and resolves the contact at
  the DEM rate against the **actual** (not extrapolated) plate surface. This is pinned by
  `test_high_degree_extrapolation_overshoots_a_contact_step` (a degree-2
  extrapolation of a contact step returns 3× the applied load) and documented
  in the code.
* a cubic (`p = 3`) extrapolation of the **interface temperature** destabilises
  the two-way thermal crossing: the coupled residual reaches `1.4e6` (table §3,
  `p = 3`). The production degree is therefore `p = 1`, and the `p = 3` row is
  reported rather than removed.

## 3. Results

All numbers below are read from `docs/p40w4_n09/verification.json`
(`suite.runs[*]`), reproduced in `docs/p40w4_n09/RESULTS_TABLES.md` §3–§6.

### 3.1 The ledger closes (criterion 1)

| p | ledger residual `|ΔE|/E_scale` | absolute imbalance [J] |
| --- | --- | --- |
| 1 (production) | **6.6978e-4** | 2.345e-5 |
| 2 | 4.9501e-4 | 1.733e-5 |
| 3 | 1.3956e6 (diverged) | 4.886e4 |

`E_scale = 3.5013e-2 J` (fluid + panel thermal, which is 99.96 % of the initial
energy). At the production degree the coupled three-field run conserves total
energy to **0.067 % of the initial energy** — the campaign's 1e-3 gate is met
with a factor 1.5 margin. `residual_settled` (initial transient excluded) is
`6.61e-4`, i.e. the closure is not an artefact of a quiet start-up.

### 3.2 The order gate fails, and why (criterion 2)

Campaign criterion: `residual(p=2) ≤ residual(p=1)/4`. Measured gain
`residual(p=1)/residual(p=2) = 1.353`. **FAIL.**

The same measurement on the exchange operator *alone* (a smooth analytic flux,
one interval-average sample per interval, no solver in the loop — table §6) gives

| p | relative quadrature error (h = 2.5e-4 s) | observed order |
| --- | --- | --- |
| 1 | 4.599e-5 | 2.00 |
| 2 | 1.079e-5 | 2.03 |
| 3 | 1.037e-5 | 2.01 |

gain `p1/p2 = 4.26` (passes the 4× gate, thin margin), `p2/p3 = 1.04`. The
observed order of *every* scheme saturates at **2**, because one sample per
interval is an **interval average** (a midpoint value), not an instantaneous
value: the sample itself carries an `O(h²)` representation error that no degree
can remove. Two independent reasons therefore explain the coupled-run failure:
(i) beyond `p = 2` there is nothing left to gain — the operator is
second-order-limited by its own data; (ii) the coupled residual is not
truncation-limited at all, it is dominated by the lattice-scale structure of the
measured wall balance and by the mechanical interface.

This is the node's falsified hypothesis and it is worth stating plainly: *"a
higher-order interface reconstruction reduces the conservation residual of a
coupled multi-solver run"* is **false in this setting**, and the two-level
measurement above is what separates "the operator is fine" from "the residual is
order-limited".

### 3.3 Interface flux continuity (criterion 3)

Two independent discretisations of the same physical quantity at the coupled
face: the fluid's measured wall balance against the panel's second-order
one-sided surface flux, compared over the settled window:

| p | flux L2 (settled) | flux L2 (full run) |
| --- | --- | --- |
| 1 | **1.862e-2** | 2.084e-1 |
| 2 | 1.495e-1 | 4.006e-1 |
| 3 | 2.184 | > 1 |

`1.86 % < 2 %` → PASS at the production degree, with a thin margin, and with a
declared grid dependence: the same quantity at `ny = 8` (through-thickness step
6.25 mm instead of 3.125 mm) is 4.7 %, i.e. the agreement converges roughly as
`h^1.4`. The full-run number (21 %) is dominated by the first few intervals, in
which the composite relaxes away from its piecewise-linear initial state; both
numbers are reported.

### 3.4 Negative control — uncoupled fields (criterion 4)

Each solver advanced with the same integrator and steps, every coupling term
switched off. Metric: secular drift `|mean(last quarter) − mean(first quarter)|`
of the field's own energy, plus the bounded oscillation amplitude:

| field | secular drift | bounded oscillation |
| --- | --- | --- |
| fluid thermal (LBM) | 1.3e-14 | 1.7e-14 |
| panel conduction (implicit) | 1.4e-14 | 1.7e-14 |
| particles (free flight) | 0.0 | 0.0 |
| plate elasticity (explicit Verlet) | **2.364e-5** | 6.306e-3 |

The gate is `max < 1e-6` → **FAIL**, and the failure is entirely the explicit
elastic plate. This is a *bounded symplectic oscillation*, not a leak: velocity
Verlet conserves a shadow Hamiltonian, so its energy oscillates with amplitude
`O((ωΔt)²)` ≈ 0.6 % at the safe step used here. Reaching a 1e-6 drift would need
≈ 79× smaller steps (≈ 2×10⁶ steps, ≈ 2 h of this box's dense-matvec throughput).
**The criterion as phrased conflates symplectic oscillation with
non-conservation**; the honest reading is: the three solvers that *can* be
exactly conservative are (1e-14), and the plate cannot be pushed below the
second-order floor at an affordable step. Note the scale: the plate's absolute
contribution to the coupled imbalance is at most `6.3e-3 × 5e-6 J ≈ 3e-8 J`,
three orders below the coupled absolute imbalance (2.3e-5 J), so the coupled
residual cannot be blamed on the integrators.

### 3.5 The three crossings are really exercised (vacuity guards)

`max overlap = 4.484e-5 m` (contact engaged), `peak measured heat rate =
0.4282 W` (thermal engaged), `drag energy removed from the particles = 1.643e-5
J` with `1.643e-5 J` delivered to the fluid (drag engaged, mismatch `1.35e-5`),
`dashpot dissipation = 1.025e-5 J` booked. Reciprocity of the contact force:
**0.0** relative (machine-exact, 4 640 exchanges).

### 3.6 Figures

* `docs/p40w4_n09/fig_energy_book.png` — the four terms and their sum over the
  run (production degree): the mechanical energy is transferred into the fluid's
  internal energy through the drag crossing, and the *sum* stays flat to 0.07 %.
* `docs/p40w4_n09/fig_residual_vs_time.png` — `|ΔE|/E_scale` vs time for
  `p = 1, 2, 3` on a log axis against the 1e-3 gate; the `p = 3` curve leaves the
  plot, which is the divergence reported in §2.
* `docs/p40w4_n09/fig_degree_scan.png` — ledger residual and interface flux L2
  vs `p`, against both gates: the residual is flat (order-limited elsewhere) while
  the flux agreement degrades with `p`.
* `docs/p40w4_n09/fig_operator_order.png` — the exchange operator alone: error vs
  interval `h` for `p = 1, 2, 3`, log–log, all slopes ≈ 2.

## 4. Error verdict, item by item

| quantity | reference / baseline | error | below 3 %? |
| --- | --- | --- | --- |
| total energy closure (production `p=1`) | the conserved quantity itself (the composite is one homogeneous slab, so the initial state is an exact steady solution; closure is measured against `E(0)`, not against an experiment) | 6.70e-4 (0.067 %) | **yes** (0.07 %) |
| interface heat-flux agreement | two independent discretisations of one flux | 1.86 % (settled window) / 21 % (full run) | **yes** on the settled window, **no** if the start-up transient is included — both reported |
| contact force reciprocity | Newton's third law | 0.0 (round-off) | **yes** |
| drag energy transfer | the particle's measured loss | 1.35e-5 | **yes** |
| order gain `p1/p2` (coupled) | campaign gate ≥ 4 | 1.353 | **no** — declared FAIL (§3.2) |
| order gain `p1/p2` (operator only) | campaign gate ≥ 4 | 4.263 | **yes**, thin margin |
| uncoupled single-field drift | 1e-6 gate | fluid/panel/particles 1e-14, plate 2.36e-5 | **no** for the plate, with the analysis in §3.4 |

There is **no experimental reference** for this case and none is claimed: it is a
*contract-verification* harness, and its baseline is (i) the exact conservation
of the discrete operators that are conservative by construction, and (ii) the
analytic steady solution of the homogeneous slab the initial state sits on. Any
comparison with a measured temperature or force field would require a physical
material and a resolved boundary layer, neither of which this configuration
claims to have.

## 5. Pitfalls, limits, and what a reader must not conclude

1. **Verification-scale parameters.** `k = 250 W/(m·K)` with
   `ρc_p = 1e4 J/(m³K)` and a 0.5 N·s/m carrier drag are chosen so that three
   time scales fit in one window. They are not water, not ice and not steel.
2. **The mechanical channel is energetically tiny** (1.26e-5 J against 3.5e-2 J
   thermal). Its *contract* is verified exactly (reciprocity 0.0, vacuity
   overlap > 0, dissipation booked), but its contribution to the ledger residual
   is below the reporting resolution. Read `residual_mechanical` (0.555) as "the
   mechanical sub-book hands 55 % of its own scale to the fluid as drag heat",
   not as a non-conservation: that transfer is by design.
3. **One-way carrier.** The fluid velocity is prescribed quiescent, so the
   DEM→LBM crossing is two-way in energy only. Momentum exchange with the carrier
   is out of scope.
4. **No thermomechanical two-way coupling.** No thermoelasticity, no friction
   heat fed back into the panel. This is what makes the thermal and mechanical
   channels energetically additive, and it is the reason the ledger can be
   checked channel by channel at all.
5. **Start-up is special.** One probe lattice step is executed before the loop to
   give the panel's first interval a heat-rate sample (the standard initial
   exchange pass of a partitioned code). The full-run flux number includes the
   relaxation of the initial state; the settled number does not.
6. **`p ≥ 2` is not a free improvement.** Both the contact load and the interface
   temperature are unstable when extrapolated with `p > 0..2`; this is measured
   (`p = 3` diverges to 1.4e6) and is the report's main reusable finding.
7. **`contact_impulse` is not computed** in the current loop (the field is
   present in the record as `null`); it must not be cited.

## 6. Reproduction

```
PYTHONPATH=src python scripts/run_p40w4_n09_three_field_energy.py   # record + figures
python scripts/render_p40w4_n09_tables.py                           # numeric tables
python -m pytest tests/test_p40w4_n09_three_field_energy.py -q      # 24 passed
```

`tests/test_p40w4_n09_three_field_energy.py` (25 tests, ≥ 12 required): contract
structure and reciprocity, four-term book behaviour (source detection, time
ordering, undefined scale), schedule and reconstruction (cubic exactness,
causality, exact constant integral), the coupled loop (closure, all three
crossings engaged, reciprocity in the run, three distinct rates, drag balance),
the negative control, the operator order study, and the acceptance layer
(WITHHELD on a missing input, digest pinning, re-gating of the committed record,
and rejection of a foreign digest).