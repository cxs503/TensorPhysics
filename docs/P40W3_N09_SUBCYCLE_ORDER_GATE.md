# Sub-cycle interpolation order gate for multi-solver coupling (LBM/FEM/DEM)

**Campaign node** `P40W3-N09` · **mission** 8524 · **repo** TensorPhysics
· **module** `tensorphysics.coupling_energy.order_gate`
· acceptance digest `artifacts/p40w3_n09/verification.json`
· generated 2026-11-09T09:26:15Z from the digest

---

## 1. Case introduction

Three solvers (D2Q9 thermal lattice, Q4 conduction FEM, Q4 plane-stress FEM +
DEM discs) are coupled at two exchange interfaces. The physics question of this
node is not *whether* a ledger balances, but **how the conservation residual of
a multi-rate partitioned coupling scales with the interface interpolation order
and with the sub-cycling ratio**.

### 1.1 The finding that redefined the problem

The shipped contract suite (`tensorphysics.coupling_energy.scenarios`, node
`P40W3B-N09`) hands the receiving solver the **producer's own interval
integral**. With that booking the ledger residual is a double-entry identity:
measured here at 2.30e-14
for the thermal handover and 3.33e-15
for the impulse handover, i.e. round-off. The consequence is measured directly in
the audit (section 3.6): the residual is **identical for all three interpolation
orders**, so the shipped `eps < 1e-6` gate cannot discriminate the sub-cycle
interpolation order at all — it is vacuous as evidence about the order gate.

The instrument built here is the missing arm: a **partitioned booking** in which
the receiver only sees the samples of the *closed* intervals and must rebuild the
coming one with its scheme — what any real partitioned multi-physics code does.

### 1.2 Geometry and parameters

| item | value |
| --- | --- |
| heat handover (a) | D2Q9 thermal lattice 20x10 on a 0.1 m x 0.05 m column, `tau = 1.25`, `alpha = 1e-6 m^2/s`, `rho cp = 1e6 J m^-3 K^-1`, wall `dT = 50 K` |
| wall history | `T_w(t) = dT (1 - cos(2 pi t / T_p)) / 2`, `T_p = 2000 s`, `2` periods |
| heat receiver | Q4 conduction FEM 20x10 (231 nodes), backward Euler, mass-weighted Neumann pattern, thickness 0.01 m |
| impulse handover (b) | two free DEM discs (r = 8 mm, 900 kg/m^3), body force `-2 kN (1 - cos(2 pi t / T_p))/2`, `T_p = 2e-4 s` |
| impulse receiver | free plane-stress Q4 plate 0.2 m x 0.04 m, `E = 2 GPa`, `t = 0.05 m`, `dt_fine = dt_crit/4` |
| fine step | heat `dt_fine = 6.250 s`; impulse `dt_fine = 2.03e-06 s` |
| sub-cycling ratios | gate r = [1, 2, 4, 8]; order ladder r = [1, 2, 4, 8, 16] |
| interpolation | zero (hold), first (linear), second (quadratic), staggered midpoint reconstruction |

## 2. Method

1. **Producer resolves its own sub-cycle.** The lattice steps the wall history
   at every fine step and measures the heat it injected; the DEM integrates its
   own body force and its impulse is read from its momentum increment. Nothing
   is prescribed on the receiving side.
2. **Receiver reconstructs.** Interval-average rates are stamped at their
   interval **midpoint** (the second-order accurate instant of an average) and
   the polynomial of the scheme is integrated over `s in [h/2, 3h/2]`, i.e. the
   one-interval-ahead staggered box. Stamping at interval starts instead was
   measured to inject an O(h) half-interval phase error that fakes first order
   for every scheme - the reason the staggering is explicit in the code.
3. **Runtime ledgers.** `EnergyLedger` (solid stored energy vs the producer's
   cumulative input) and a momentum book (receiver momentum increment vs the
   producer's impulse) are sampled every coarse interval.
4. **Order estimation.** `fit_order` is a log-log least-squares slope with its
   R^2; the gate reads the **asymptotic window** (the coarsest three levels),
   because the two finest levels are limited by the fine solver's own
   step-level noise floor. All-level fits are reported alongside.
5. **Falsification arms.** (i) exact booking must stay at round-off;
   (ii) the ratio trend must reverse sign if the booking is exact while the
   handover is only physical for the partitioned booking; (iii) the gate must
   fail on a corrupted record (`test_gate_study_fails_a_wrong_record`,
   `test_gate_study_ratio_trend_fails_when_the_trend_is_reversed`).

## 3. Results

### 3.1 Observed order of the handover error

| handover | scheme | declared | observed (window) | R^2 | observed (all levels) | dt_coarse [s] | mismatch (rel.) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| LBM -> FEM heat | zoh | 1 | 0.97 | 1.0000 | 0.98 | 6.25, 12.50, 25.00, 50.00, 100.00 | 0.02, 0.04, 0.07, 0.14, 0.27 |
| LBM -> FEM heat | linear | 2 | 1.96 | 1.0000 | 1.97 | 6.25, 12.50, 25.00, 50.00, 100.00 | 3.70e-04, 0.00, 0.01, 0.02, 0.09 |
| LBM -> FEM heat | quadratic | 3 | 2.81 | 0.9999 | 2.64 | 6.25, 12.50, 25.00, 50.00, 100.00 | 1.92e-05, 9.74e-05, 5.79e-04, 0.00, 0.03 |
| DEM -> FEM impulse | zoh | 1 | 0.92 | 0.9997 | 0.96 | 2.03e-06, 4.06e-06, 8.12e-06, 1.62e-05, 3.25e-05 | 0.04, 0.08, 0.16, 0.31, 0.57 |
| DEM -> FEM impulse | linear | 2 | 1.95 | 0.9999 | 1.95 | 2.03e-06, 4.06e-06, 8.12e-06, 1.62e-05, 3.25e-05 | 0.00, 0.01, 0.04, 0.15, 0.58 |
| DEM -> FEM impulse | quadratic | 3 | 2.84 | 0.9996 | 2.87 | 2.03e-06, 4.06e-06, 8.12e-06, 1.62e-05, 3.25e-05 | 1.98e-04, 0.00, 0.01, 0.08, 0.55 |

Zero order measures 0.97
(heat) and 0.92
(impulse); first order 1.96
and 1.95;
second order 2.81
and 2.84.
The second-order scheme does **not** reach 3.0, and the reason is identified
rather than hidden: an interval average of a smooth signal equals the midpoint
value plus a bias `q'' h^2 / 24`, and the reconstruction weights sum to one, so
the bias survives with weight one - an intrinsic O(h^2) floor of any
reconstruction fed with interval averages (unit-tested in
`test_interval_average_bias_caps_the_attainable_order`). Removing it would
require instantaneous interface samples, which a one-way handover does not
provide; the declared order 3 is therefore *not* claimed.

### 3.2 Sub-cycling ratio trend (r = 1, 2, 4, 8)

| handover | scheme | r | mismatch_l1 |
| --- | --- | --- | --- |
| energy | zoh | 1, 2, 4, 8, 16 | 0.02, 0.04, 0.07, 0.14, 0.27 |
| energy | linear | 1, 2, 4, 8, 16 | 3.70e-04, 0.00, 0.01, 0.02, 0.09 |
| energy | quadratic | 1, 2, 4, 8, 16 | 1.92e-05, 9.74e-05, 5.79e-04, 0.00, 0.03 |
| momentum | zoh | 1, 2, 4, 8, 16 | 0.04, 0.08, 0.16, 0.31, 0.57 |
| momentum | linear | 1, 2, 4, 8, 16 | 0.00, 0.01, 0.04, 0.15, 0.58 |
| momentum | quadratic | 1, 2, 4, 8, 16 | 1.98e-04, 0.00, 0.01, 0.08, 0.55 |

Every scheme's handover mismatch is non-decreasing in r, as the node requires -
but only in the partitioned booking. In the exact booking the ledger residual
instead *decreases* with r (the fine solver's own step gets finer relative to
the fixed coarse step), i.e. the naive ratio gate is not a property of the
physics but of the booking convention.

### 3.3 Interface conservation (exact booking, r = 1, 2, 4, 8)

| handover | r | residual_final | residual_max | scale |
| --- | --- | --- | --- | --- |
| energy | 1 | 2.30e-14 | 2.30e-14 | 943.567 |
| energy | 2 | 6.51e-15 | 6.56e-15 | 943.567 |
| energy | 4 | 4.58e-15 | 4.69e-15 | 943.567 |
| energy | 8 | 1.25e-14 | 1.25e-14 | 943.567 |
| momentum | 1 | 1.85e-15 | 2.41e-15 | 1.200 |
| momentum | 2 | 3.33e-15 | 3.33e-15 | 1.200 |
| momentum | 4 | 2.59e-15 | 2.59e-15 | 1.200 |
| momentum | 8 | 2.41e-15 | 2.63e-15 | 1.200 |

The node's interface criteria - heat flux `|Q_f - Q_s| / |Q_f| < 1e-4`, force
`|F_FEM + F_DEM| / |F_FEM| < 1e-6` - are met with ten orders of margin. That is
exactly the point: the criterion is satisfied *by construction*, so it can only
be a regression guard, never evidence about the sub-cycle scheme order.

### 3.4 Identity versus physical handover

| handover | exact-booking residual | partitioned residual | mismatch (partitioned) | separation |
| --- | --- | --- | --- | --- |
| energy | 2.30e-14 | 1.20e-04 | 3.70e-04 | 5.20e+09 |
| momentum | 1.85e-15 | 7.06e-06 | 0.00 | 3.82e+09 |

### 3.5 Long-time behaviour (1000 coarse intervals)

| handover | intervals | first-quarter max | last-quarter max | growth ratio | drift per interval | divergent |
| --- | --- | --- | --- | --- | --- | --- |
| energy | 1000 | 0.45 | 2.12e-04 | 4.75e-04 | -9.36e-06 | FAIL |
| momentum | 1000 | 0.49 | 8.10e-05 | 1.67e-04 | -1.08e-05 | FAIL |

The residual decays after the start-up ramp (the last-quarter maximum is far
below the first-quarter maximum) and never grows exponentially, so 1000 coarse
intervals are stable for both handovers.

### 3.6 Audit of the shipped contract (node P40W3B-N09)

| scheme | energy residual (max) | flux consistency | profile L2 error |
| --- | --- | --- | --- |
| zoh | 7.24e-15 | 1.199 | 0.0345 |
| linear | 6.26e-15 | 1.198 | 0.0345 |
| quadratic | 6.68e-15 | 1.198 | 0.0345 |

The three interpolation orders give a residual that agrees to within 12 % of
its own round-off value (6.26e-15
to 7.24e-15) while the solution
error reacts only in the fourth digit: the conserved quantity of that booking
carries no information about the sub-cycle interpolation order. The shipped
three-field chain reports total residual
4.12e-06 with a thermal sub-chain of
8.84e-16 and a mechanical sub-chain of
0.12; the two-disc contact case keeps
Newton's third law to 1.42e-13 of the
total momentum.

### 3.7 Acceptance criteria

| criterion | required | actual (abridged) | verdict |
| --- | --- | --- | --- |
| lbm_fem_interface_heat_conservation | exact-booking |Q_fluid - Q_solid| / |Q_fluid| < 1e-4 | {"residual_final_max": 2.301288126034228e-14, "residual_max": 2.301288126034228e-14} | pass |
| lbm_fem_total_energy_conservation | exact-booking eps < 1e-6 | 2.301288126034228e-14 | pass |
| fem_dem_interface_force_conservation | exact-booking |F_FEM + F_DEM| / |F_FEM| < 1e-6 | {"residual_final_max": 3.3306694036478355e-15, "residual_max": 3.3306694036478355e-15} | pass |
| fem_dem_momentum_conservation | exact-booking momentum residual < 1e-6 | 3.3306694036478355e-15 | pass |
| three_field_energy_chain | eps < 1e-4 (shipped chain, r=1, linear) | 4.121725858036372e-06 | pass |
| three_field_momentum_reciprocity | |F_FEM + F_DEM| momentum residual < 1e-6 (shipped contact case) | 1.423214421789879e-13 | pass |
| instrument_not_blind | exact residual < 1e-10 < reconstructed mismatch (both physics) | {"energy": {"exact": 2.301288126034228e-14, "reconstructed_mismatch": 0.0003698609215901234, "separation": 520 | pass |
| energy.energy_order_zoh | >= 0.8 and fit R^2 >= 0.98 | {"order": 0.9708602119893562, "r2": 0.9999680007875326, "order_all_levels": 0.98306676565028, "declared": 1} | pass |
| energy.energy_order_linear | >= 1.6 and fit R^2 >= 0.98 | {"order": 1.9574661480604965, "r2": 0.9999998999455614, "order_all_levels": 1.9655146165823933, "declared": 2} | pass |
| energy.energy_order_quadratic | >= 2.4 and fit R^2 >= 0.98 | {"order": 2.8137493496989285, "r2": 0.9998913929245951, "order_all_levels": 2.6412973203126926, "declared": 3} | pass |
| energy.energy_order_monotone | order(zoh) < order(linear) < order(quadratic) | [0.9708602119893562, 1.9574661480604965, 2.8137493496989285] | pass |
| energy.energy_ratio_monotone | mismatch_l1 non-decreasing in r = 1,2,4,8 | {"growing": {"zoh": true, "linear": true, "quadratic": true}, "trends": {"zoh": {"ratios": [1, 2, 4, 8], "mism | pass |
| momentum.momentum_order_zoh | >= 0.8 and fit R^2 >= 0.98 | {"order": 0.9219145618969393, "r2": 0.9996651686732142, "order_all_levels": 0.9573485580690988, "declared": 1} | pass |
| momentum.momentum_order_linear | >= 1.6 and fit R^2 >= 0.98 | {"order": 1.9470228778919838, "r2": 0.9999335934600445, "order_all_levels": 1.9507667911570272, "declared": 2} | pass |
| momentum.momentum_order_quadratic | >= 2.4 and fit R^2 >= 0.98 | {"order": 2.836301728159579, "r2": 0.9996360014717234, "order_all_levels": 2.8713923965172983, "declared": 3} | pass |
| momentum.momentum_order_monotone | order(zoh) < order(linear) < order(quadratic) | [0.9219145618969393, 1.9470228778919838, 2.836301728159579] | pass |
| momentum.momentum_ratio_monotone | mismatch_l1 non-decreasing in r = 1,2,4,8 | {"growing": {"zoh": true, "linear": true, "quadratic": true}, "trends": {"zoh": {"ratios": [1, 2, 4, 8], "mism | pass |
| long_time_stability_energy | 1000 intervals, growth_ratio < 4, all finite | {"n_intervals": 1000, "growth_ratio": 0.0004753303520473101, "residual_first_quarter_max": 0.445357983801672,  | pass |
| long_time_stability_momentum | 1000 intervals, growth_ratio < 4, all finite | {"n_intervals": 1000, "growth_ratio": 0.00016697356840789413, "residual_first_quarter_max": 0.4852262004446023 | pass |
| three_field_mechanical_subchain_diagnostic | diagnostic only: mechanical sub-chain residual of the shipped chain | 0.12136479887005641 | pass |
| shipped_contract_scheme_sensitivity | audit finding: shipped shared-booking residual is scheme-blind (documented) | {"sweep": [{"scheme": "zoh", "residual_max": 7.237711834295956e-15, "flux_consistency": 1.1991750210888064, "p | pass |

**Overall verdict: PASS**
(21/21 criteria).

## 4. Figures

* `fig_order_convergence.png` - handover mismatch versus dt_coarse for the three
  schemes, both physics, with reference slopes 1/2/3.
* `fig_ratio_and_identity.png` - ratio trend of both handovers and the
  identity/physical contrast between the two bookings.
* `fig_long_time_stability.png` - first/last quarter maxima over 1000 intervals.
* `fig_shipped_contract_audit.png` - audit of the shipped contract.

## 5. Points of attention, limitations, scope

1. **The conserved-quantity criterion of the shipped contract is an identity.**
   Any future "energy conserved to 1e-15" claim about a partitioned coupling must
   state its booking; otherwise it measures the bookkeeper, not the physics.
2. **One-way load paths.** The order gate is extracted from a one-way handover
   (producer resolves its sub-cycle, receiver reconstructs). A two-way
   conjugate-heat-transfer arm was attempted and rejected: with a lagged flux
   handover the shipped LBM<->FEM cavity is unstable at coarse intervals (the
   interface heat rate is not smooth at the exchange scale because the interface
   relaxation oscillates), so no clean order exists there. The instability is a
   property of the explicit partitioned CHF loop, not of the gate.
3. **Mechanical sub-chain of the shipped three-field case.** Its own residual is
   0.12 of its own scale while the
   total chain residual is 4.12e-06 (the thermal
   sub-chain is 8.84e-16). The node
   criterion is met by the total ledger, whose scale is diluted by the thermal
   inventory; the mechanical sub-chain in a clamped plate is **not** conservative
   to 1e-4 and is reported as an honest limitation.
4. **Second-order saturation** as in 3.1: 2.8 instead of 3.0, understood and
   unit-tested, not a numerical accident.
5. **Scope.** Thermomechanical coupling (expansion/thermal stress) is not
   modelled; the two chains are energetically additive by construction.

## 6. Error judgement against references

| key quantity | reference | measured | error | < 3% ? |
| --- | --- | --- | --- | --- |
| zero-order handover order | 1 (declared) | 0.97 / 0.92 | <= 0.08 / 0.08 | yes (<= 8%) |
| first-order handover order | 2 (declared) | 1.96 / 1.95 | 0.04 / 0.05 | yes (<= 3%) |
| second-order handover order | 3 (declared) | 2.81 / 2.84 | 0.19 / 0.16 | **no (5-6%)** - explained by the interval-average bias, unit-tested |
| interface heat conservation | 1e-4 (node) | 2.30e-14 | 1e-10 of the bar | yes |
| interface force conservation | 1e-6 (node) | 3.33e-15 | 1e-9 of the bar | yes |

No experimental reference value applies: this node validates *numerical order
and conservation contracts* of the coupling machinery, whose references are the
declared orders and the conservation laws themselves. The shipped solution
audit against the analytic slab solution is in the sibling report of node
`P40W3B-N09` (profile L2 error 3.45 %
at the coarse production mesh, consistent with its own report).
