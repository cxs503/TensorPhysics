# Coupling time-exchange operator algebra, second order, and the interval-complete conservation ledger

**Campaign node** `p223w3-n06-physics-coupling-exchange` · **mission** 9722 ·
**repo** TensorPhysics · **module** `tensorphysics.coupling_energy.{
exchange_operators, exchange_mms, time_exchange, exchange_gate}` ·
acceptance record `artifacts/p223w3_n06/verification.json` (sha256
`85976e1963dbeb7dc12123867df1e4b53e44a3fecf92cf3a15bb941c2dad63c9` in `artifacts/p223w3_n06/SHA256SUMS`) · generated from the record
2026-11-10T14:22:20+00:00 · commit `41d4a5161edd` on `agent/m9722`
(loongarch64, Python 3.11.6, numpy 2.4.6).

---

## 1. Case introduction

### 1.1 Physical background

Partitioned multi-physics coupling (fluid ↔ structure, weld heat source ↔ plate,
fluid ↔ particle) exchanges one interface quantity per coarse step. Two defects
are reported by the coupling contract that the wake / weld / settling chains
share:

1. **end-stamped injection** — the interface flux of the closed coarse interval
   is held over the coming interval. The receiver therefore applies a quantity
   that belongs to the past, and the resulting bias is ``O(dt)``: the defect
   stops decreasing once it reaches the level set by the producer's own fine step.
   The contract sees this as a *staircase saturation* near **1.2e-3**.
2. **missing exchange-interval attribution** — only the sub-cycle sample of the
   interval that closes at the exchange instant is booked, so the heat carried by
   the other ``r-1`` fine sub-intervals of the window is never booked. At
   sub-cycling ratios ``r = 8, 16`` that is **87.5 % / 93.75 %** of the exchanged
   quantity, the reported **87–94 %** loss.

The node's physical goal is to replace that practice with a **time-centred
(Crank–Nicolson type midpoint) exchange operator** and an **interval-complete
ledger** in which every interface flux belongs to exactly one exchange interval,
``Σ intervals = total exchange``, and the unattributed share is identically zero.

### 1.2 Geometry and parameters

There is no ship geometry in this node: the object under test is the exchange
operator, not a hull. The measurable instrument is a **coupling manufactured
solution** (analytic producer field, analytic receiver field, analytic interface
flux with a closed-form interval integral) plus **three chains** in which the
exchange drives a real receiver. The receiver is linear and forced
piecewise-constant, so its interval update is *exact* (matrix exponential /
closed form) and every measured error belongs to the exchange, not to the
receiver's own time scheme. The reference is `solve_ivp` (DOP853,
`rtol = 1e-12`).

| chain | physics | exchanged quantity | period [s] | fine step [s] | run [s] |
| --- | --- | --- | --- | --- | --- |
| viv | VIV lock-in force exchange onto a 1-DOF structure | F_L(t) [N] → v_s(t) [m/s] | 0.625 | 5.0e-04 | 2.5 |
| weld | weld-toe heat exchange onto a lumped conduction node | q_weld(t) [W] → T_s(t) [K] | 1.0 | 2.0e-03 | 1.5 |
| settling | Stokes settling particle driven by the exchanged fluid velocity | u_f(t) [m/s] → v_p(t) [m/s] | 0.25 | 5.0e-04 | 1.0 |

Exchange-cadence ladders use ``r = [1, 2, 4, 8, 16]`` fine steps per coarse
interval, i.e. the exchange is refreshed every one of the 5
cadences while the fine solver step is *held fixed* — the only knob a real
partitioned code has.

### 1.3 The operator algebra

| scheme | declared order | causal | quadrature rule |
| --- | --- | --- | --- |
| `lag` | 1 | True | end-stamped hold of the newest closed sample (the shipped practice) |
| `centered` | 2 | False | Crank-Nicolson style midpoint of both interval endpoints |
| `predictor_corrector` | 2 | True | trapezoid on a linearly predicted endpoint, deferred correction on close |

`centered` needs ``q(t_hi)`` and is therefore the *idealised* member;
`predictor_corrector` predicts that endpoint by linear extrapolation
(``q̂ = 2q_lo − q_prev``, order `1.956`)
and applies a deferred correction when the realised sample arrives, after which
it reproduces the CN interval integral **exactly**
(``max relative difference = 0.0``).
The stamp convention is part of the algebra, not of the driver: `StampRule`
declares whether a published sample is an endpoint value or a window average
displaced by ``offset``, and optionally de-stamps it.

---

## 2. Method

1. **Manufactured solution first.** A harmonic two-tone flux with a closed-form
   antiderivative is exchanged on grids ``h = 1/16 … 1/256`` of the period. No
   ODE, no solver, no ledger: the measured exponent is the operator's temporal
   order and nothing else.
2. **The three chains.** `viv` (SDOF oscillator forced by the exchanged lift
   force), `weld` (lumped weld-toe node heated by the exchanged flux) and
   `settling` (Stokes particle relaxing to the exchanged fluid velocity) are each
   driven through the exact linear receiver with the `solve_ivp` trajectory as
   oracle.
3. **Exchange-cadence ladders.** For each chain × exchange arm the coarse step is
   refined by refreshing the exchange more often (``r`` down to 1) while the fine
   step stays fixed. Per level the run reports the L1 handover defect
   ``Σ|Q_applied − Q_exact| / Σ|Q_exact|``, the receiver state error against the
   oracle, and the interval-attribution audit.
4. **Interval-complete ledger.** Each produced sub-window share is booked into the
   interval that owns it, the receiver books the integral it applied, and a
   deferred correction is booked on the same interval. Double bookings, double
   closes and missing intervals are refused.
5. **Falsification arms.** (i) with a true endpoint convention the CN operator
   must be second order on every chain; (ii) the end-stamped window convention
   must plateau (order ≈ 0) and the repair must break the plateau; (iii)
   tail-only booking must lose ``(r-1)/r``; (iv) the gate must reject a ladder
   whose defect no longer converges (`test_gate_recomputes_the_order_from_the_ladder`),
   and it re-derives every exponent from the stored ``(h, error)`` pairs instead
   of trusting a stored order field.
6. **Tool chain.** `pytest` (34 new tests), a deterministic runner
   (`scripts/run_p223w3_n06_exchange.py`) that writes the verification record,
   the compact ladders and four figures, and a renderer
   (`scripts/render_p223w3_n06_doc.py`) that writes this document from the record.

---

## 3. Results

### 3.1 Operator order on the manufactured solution

| arm | defect (coarsest h) | defect (finest h) | observed order | R² | coarsest h |
| --- | --- | --- | --- | --- | --- |
| lag (end-stamped hold) | 1.07e-01 | 6.70e-03 | `0.997` | 1.00000 | 0.06250 |
| centered (CN midpoint) | 1.90e-02 | 7.33e-05 | `2.004` | 0.99999 | 0.06250 |
| predictor-corrector, provisional | 9.57e-02 | 4.31e-04 | `1.956` | 0.99972 | 0.06250 |
| predictor-corrector, corrected | 1.90e-02 | 7.33e-05 | `2.004` | 0.99999 | 0.06250 |
| window average, end-stamped (raw) | 4.99e-02 | 3.35e-03 | `0.978` | 0.99965 | 0.06250 |
| window average, de-stamped | 1.89e-02 | 7.11e-05 | `2.011` | 0.99998 | 0.06250 |

The lag operator converges at first order
(`0.997`), the CN midpoint at second
(`2.004`), and the causal predictor-corrector
is second order *before* the correction
(`1.956`) and identical
to CN after it. Turning the stamp convention on shows the root cause of the
reported defect as a separate, measurable effect: consuming a window average as
an endpoint value collapses the order to
`0.978`; the de-stamping rule restores
`2.011`.

### 3.2 Interval attribution

| sub-cycling ratio r | unattributed share (measured) | loss [%] | released fraction | analytic 1−1/r |
| --- | --- | --- | --- | --- |
| 2 | 4.99e-01 | 49.9 % | 50.07 % | 50.000 % |
| 4 | 7.49e-01 | 74.9 % | 25.10 % | 75.000 % |
| 8 | 8.74e-01 | 87.4 % | 12.62 % | 87.500 % |
| 16 | 9.36e-01 | 93.6 % | 6.37 % | 93.750 % |

Tail-only booking loses 87.4 % at ``r = 8`` and
93.6 % at ``r = 16``: the reported 87–94 % band is
reproduced from first principles, without tuning, by the single rule "only the
closing sub-window is booked". Publishing every sub-window average instead gives
an unattributed share of **exactly zero** and a handover defect of
1.39e-13 —
each window average *is* the exact integral of its window, so the exchange
carries no quadrature error at all.

Ledger controls (the anomalies it must catch):

| control | outcome |
| --- | --- |
| `double_booking_refused` | True |
| `double_close_refused` | True |
| `open_interval_detected` | True |
| `missing_interval_detected` | True |
| `centered_requires_endpoint` | True |
| `endpoint_offset_refused` | True |

### 3.3 The staircase, and what the repair removes

| chain | arm | defect (coarsest cadence) | defect (finest cadence) | observed order | R² | state error (finest) | plateau |
| --- | --- | --- | --- | --- | --- | --- | --- |
| viv | lag + end stamp | 4.26e-02 | 5.03e-03 | `0.775` | 0.9929 | 5.27e-03 | 8.38e-03 |
| viv | CN + end stamp | 2.56e-03 | 2.51e-03 | `0.006` | 0.5540 | 2.63e-03 | 2.51e-03 |
| viv | CN + de-stamp | 4.42e-04 | 6.71e-06 | `1.567` | 0.8209 | 7.92e-07 | 7.24e-06 |
| weld | lag + end stamp | 6.76e-02 | 8.00e-03 | `0.773` | 0.9927 | 2.67e-03 | 1.33e-02 |
| weld | CN + end stamp | 4.51e-03 | 4.00e-03 | `0.035` | 0.5096 | 1.33e-03 | 3.99e-03 |
| weld | CN + de-stamp | 1.74e-03 | 2.51e-05 | `1.595` | 0.8017 | 5.14e-08 | 2.53e-05 |
| settling | lag + end stamp | 6.79e-02 | 8.00e-03 | `0.776` | 0.9928 | 3.97e-04 | 1.33e-02 |
| settling | CN + end stamp | 4.53e-03 | 4.00e-03 | `0.037` | 0.5509 | 1.99e-04 | 4.00e-03 |
| settling | CN + de-stamp | 1.74e-03 | 2.51e-05 | `1.593` | 0.7999 | 2.19e-06 | 2.52e-05 |

With the end-stamped window convention the CN operator's defect is **flat**
(plateau ratio
`1.13`
at worst), the fitted order collapses to
≈ `0.037`:
refining the exchange cadence buys nothing. The weld-chain plateau of the state
error is 1.33e-03
against the 1.20e-03 reported by the contract (ratio
`1.109`).
That agreement is not fitted: the plateau level is set by the half-fine-step
phase offset of the published sample, and the weld chain's fine step is
``2.0e-3 s``.

The repair (CN midpoint + de-stamping) cuts the finest-cadence defect by
`159`×
and the state error by
`521`×
on the worst chain, and the ladder spread grows again by
`69`×
across the cadences, i.e. the plateau is gone. The remaining error of the
repaired operator is still one order short of two on the *defect* metric
(`1.595` on the weld
chain) because the de-stamping slope is one-sided; the *state* error already
converges at
`1.966`.

### 3.4 Error judgement against the reference

Reference: the analytic interval integral (defect) and the `solve_ivp` oracle at
`rtol = 1e-12` (state error). "≤ 3 %" is the platform criterion; it is answered
per row, and a row that is **not** below 3 % is marked as such rather than
smoothed over.

| chain | arm | worst interval defect | final state error vs oracle | ≤ 3 % ? |
| --- | --- | --- | --- | --- |
| viv | lag + end stamp | 6.70 % | 0.53 % | **no** |
| viv | CN + end stamp | 0.40 % | 0.26 % | yes |
| viv | CN + de-stamp | 0.20 % | 0.00 % | yes |
| viv | CN, endpoint convention | 0.08 % | 0.00 % | yes |
| viv | every sub-window | 0.00 % | 0.00 % | yes |
| weld | lag + end stamp | 10.64 % | 0.27 % | **no** |
| weld | CN + end stamp | 0.71 % | 0.13 % | yes |
| weld | CN + de-stamp | 0.30 % | 0.00 % | yes |
| weld | CN, endpoint convention | 0.34 % | 0.00 % | yes |
| weld | every sub-window | 0.00 % | 0.00 % | yes |
| settling | lag + end stamp | 10.67 % | 0.04 % | **no** |
| settling | CN + end stamp | 0.71 % | 0.02 % | yes |
| settling | CN + de-stamp | 0.30 % | 0.00 % | yes |
| settling | CN, endpoint convention | 0.34 % | 0.00 % | yes |
| settling | every sub-window | 0.00 % | 0.00 % | yes |

Reading — and this is the reason the gate is *not* an accuracy threshold:

* the lag/end-stamp practice **violates** the 3 % criterion
  (worst measured 10.7 %
  over the three chains) as soon as the exchange cadence is coarse;
* the CN + end-stamp practice **passes** the 3 % criterion
  (0.71 %
  worst interval, 0.26 %
  final state error) while being *first order in nothing*: its order is
  `0.035` and its
  defect does not move when the cadence is refined 16×. The 3 % threshold cannot
  see the defect; only the order/plateau structure can.

The repaired operator and the interval-complete exchange sit two to nine orders of
magnitude inside the 3 % threshold at every cadence.

### 3.5 Figures

* `artifacts/p223w3_n06/fig1_operator_order.png` — log–log defect versus exchange
  step for the six MMS arms; the slopes are the declared orders (1, 2, 2, 2) and
  the two stamp-convention arms (1 → 2 after de-stamping).
* `artifacts/p223w3_n06/fig2_staircase_repair.png` — the three chains, defect
  versus coarse step for lag + end stamp, CN + end stamp, CN + de-stamp and the
  sub-window arm: the plateau of the reported practice and its removal.
* `artifacts/p223w3_n06/fig3_attribution_loss.png` — measured loss versus
  sub-cycling ratio against ``1 − 1/r`` and the reported 87–94 % band.
* `artifacts/p223w3_n06/fig4_chain_trajectories.png` — receiver state against the
  oracle for the three chains at ``r = 8``, broken versus repaired exchange.

### 3.6 Acceptance criteria

| criterion | required | measured (abridged) | verdict |
| --- | --- | --- | --- |
| `algebra_three_operators` | 3 schemes, orders [1, 2, 2] | {"lag": 1, "centered": 2, "predictor_corrector": 2} | **pass** |
| `mms_order_lag` | 0.85 <= order <= 1.15 and R^2 >= 0.999 | {"order": 0.99726930633309, "r2": 0.9999989701604994, "stored_order": 0.99726930633309} | **pass** |
| `mms_order_centered` | 1.85 <= order <= 2.15 and R^2 >= 0.999 | {"order": 2.004252294981196, "r2": 0.9999914175394103, "stored_order": 2.004252294981196} | **pass** |
| `mms_order_predictor_corrector` | provisional >= 1.75 and corrected 1.85..2.15, R^2 >= 0.999 | {"provisional_order": 1.9556291791975953, "corrected_order": 2.004252294981196, "provisional_r2": 0.9997180272690804, "corrected_r2": 0.99999141753... | **pass** |
| `predictor_corrector_equals_centered` | max relative difference <= 1e-12 | {"max_rel_diff": 0.0} | **pass** |
| `mms_window_stamp_first_order` | 0.85 <= order <= 1.15 | {"order": 0.9781228328168609, "r2": 0.999654078630938} | **pass** |
| `mms_window_destamp_second_order` | >= 1.85 and R^2 >= 0.999 | {"order": 2.010752661242295, "r2": 0.9999827491189456} | **pass** |
| `stamp_rule_linear_exact` | both residuals <= 1e-15 | {"left_residual": 0.0, "right_residual": 0.0, "offset": 0.25} | **pass** |
| `attribution_complete_exact` | unattributed <= 1e-15, deficit <= 1e-12, handover defect <= 1e-10 | {"viv": {"unattributed": 0.0, "deficit": 1.5111600664889234e-14, "max_defect": 1.3928237038859205e-13}, "weld": {"unattributed": 0.0, "deficit": 2.... | **pass** |
| `attribution_tail_loss_band` | 0.85 <= loss(r=8) <= 0.90 and 0.92 <= loss(r=16) <= 0.95 | {"loss_r8": 0.87383978184105, "loss_r16": 0.9362570401631113, "predicted": {"2": 0.5, "4": 0.75, "8": 0.875, "16": 0.9375}} | **pass** |
| `attribution_ledger_refuses_anomalies` | all four controls true | {"double_booking_refused": true, "double_close_refused": true, "open_interval_detected": true, "missing_interval_detected": true, "centered_require... | **pass** |
| `endstamp_staircase_saturates` | fitted order <= 0.30 and max/min defect <= 1.30 on all three chains | {"viv": {"order": 0.005884278386746202, "plateau_defect": 0.0025133455598440233, "plateau_state_error": 0.002517504952143025, "plateau_ratio": 1.01... | **pass** |
| `repair_removes_plateau` | defect gain >= 50x and state gain >= 50x on all three chains | {"viv": {"finest_defect": 6.711311879521238e-06, "finest_state_error": 7.920181847485008e-07, "defect_gain": 374.49392979533485, "state_gain": 3178... | **pass** |
| `repair_restores_cadence_dependence` | max/min defect >= 3x on all three chains (5-level ladder) | {"viv": 65.91670502038824, "weld": 69.44073368954325, "settling": 69.22796277917291} | **pass** |
| `weld_plateau_matches_reported` | 0.6e-3 <= plateau <= 2.4e-3 (reported 1.2e-3) | {"plateau_state_error": 0.0013310397759347505, "reported": 0.0012, "ratio": 1.109199813278959} | **pass** |
| `lag_endstamp_first_order_bias` | order <= 1.05 on all three chains | {"viv": [0.7753439782619219, 0.9928900458144081], "weld": [0.7732993553502401, 0.9926671577125601], "settling": [0.7756166657376151, 0.992833478611... | **pass** |
| `endpoint_convention_second_order` | 1.85 <= order <= 2.15 on all three chains | {"viv": 2.0000346320453444, "weld": 1.999246984579385, "settling": 1.9996923981963546} | **pass** |

**Overall verdict: PASS** (17/17
criteria).

---

## 4. Note / caveats

1. **The m7031 contract document is not in this worktree.** Both reported limits
   were therefore treated as *hypotheses* and reconstructed mechanistically: the
   87–94 % loss follows from "only the closing sub-window is booked"
   (measured 87.4 % / 93.6 %
   at ``r = 8/16``), and the 1.2e-3 plateau follows from the half-fine-step phase
   offset of an end-stamped window average (measured
   1.33e-03).
   Reproducing both numbers from a single mechanism is evidence for the
   reconstruction, not proof of it; if the contract's mechanism differs, the
   discriminating test is the fine-step scaling (the plateau must move with
   `dt_fine`, not with the cadence).
2. **VIV / weld / settling are TensorPhysics-side reconstructions.** The three
   names are the chains named by the node, but their models here (SDOF oscillator,
   lumped weld node, Stokes particle) are linear reference models, not the
   production chain models of those disciplines. The invariant content of this
   node is the *exchange operator and its ledger*, which are chain-agnostic by
   construction; swapping in a production chain changes the flux history, not the
   algebra.
3. **One-sided de-stamping slope.** The repaired defect converges at
   `1.595`
   on the weld chain, not 2.0, because the de-stamping slope uses the closed
   history only. A second-order one-sided slope (needing one more history sample)
   would close the gap; it was not added because it changes the operator
   interface and is not needed for the plateau removal.
4. **Receiver exactness is a choice, not a claim about production codes.** The
   receivers integrate their linear ODE exactly so that the measured error is
   attributable to the exchange. A production receiver adds its own time-scheme
   error, which would mask the exchange order further (that masking is exactly the
   staircase phenomenon).
5. **Scope.** No structural/nonlinear receiver, no moving-mesh interface, no
   multi-interface network; the ledger is per interface, and a network audit
   (``Σ interfaces``) is out of scope for this node.

## 5. Audit chain

* Code: `src/tensorphysics/coupling_energy/exchange_operators.py`
  (algebra + ledger), `exchange_mms.py` (MMS ladder), `time_exchange.py`
  (chains + cadence ladders), `exchange_gate.py` (study + criteria).
* Tests: `tests/test_coupling_exchange_operators.py` (16) and
  `tests/test_coupling_exchange_order_gate.py` (18).
* Record: `artifacts/p223w3_n06/verification.json` carries the full study, the gate
  verdicts and the sha256 of every other artifact; its own digest
  `85976e1963dbeb7d…` is in `artifacts/p223w3_n06/SHA256SUMS`
  (a record cannot carry its own hash — writing it would change the file):
  ladders `3dea242a7ab21ae2…`,
  fig1 `980f9cc41049a049…`,
  fig2 `2a047dd3adba88a4…`,
  fig3 `d6cc88459350c5e7…`,
  fig4 `7eb1191c14620a15…`.
* Reproduce: `python3 scripts/run_p223w3_n06_exchange.py` then
  `python3 scripts/render_p223w3_n06_doc.py`.
