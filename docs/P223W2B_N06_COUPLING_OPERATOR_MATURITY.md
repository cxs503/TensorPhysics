# Coupling time-exchange operator maturity: second-order algebra and an interval-complete conservation ledger

Campaign node `p223w2b-n06` (TensorPhysics). Baseline = `main` at
`bfd8911db3d7093f7c5d34baea30aadef7affda2`. Every number below is written by
`scripts/run_p223w2b_n06_maturity.py` from `artifacts/p223w2b_n06/maturity.json`
(sha256 `f4558b113b63c688ecba9a4f160bfaf5558917c6c83fab911e00ff1bc5460db0`); nothing is transcribed by hand.

## 1. The case

A partitioned solver hands an interface flux from a producer to a receiver once
every `r` fine steps. Two defects were reported by the coupling contract:

1. **end-stamped hand-over** — the producer publishes the *average of the last
   fine window* and the receiver treats it as the value at the exchange instant,
   so every sample carries a half-fine-step phase offset and the receiver takes
   an `O(dt)` bias;
2. **interval attribution** — only the sub-cycle share of the interval that
   closes at the exchange instant is booked, so `(r-1)/r` of the exchanged heat
   (87.5 % at `r = 8`, 93.75 % at `r = 16`) is never attributed to any interval.

The contract saw the consequence as a *staircase*: refining the exchange cadence
lowered the error until it saturated near `1.2e-3` on the weld chain and stopped
moving. Both defects are properties of the exchange operator, not of the
receivers, so they are measurable with no receiver in the loop.

| item | value |
| --- | --- |
| chains | `viv` (SDOF oscillator, `dt_fine = 5.0e-4 s`), `weld` (lumped weld-toe conduction node, `dt_fine = 2.0e-3 s`), `settling` (Stokes particle, `dt_fine = 5.0e-4 s`) |
| interface quantity | analytic flux with a closed-form interval integral (MMS) |
| exchange operators | `lag` (end-stamp hold, order 1), `centered` (Crank–Nicolson midpoint, order 2), `predictor_corrector` (causal, deferred correction) |
| de-stamping | order 1 (two-sample one-sided slope) and **order 2 (three-sample Newton forward difference, new in this node)** |
| cadence ladder | `r = 1, 2, 4, 8, 16` at the shipped fine step |
| `dt` ladder | fine step halved 5× at fixed `r = 2` |
| reference | exact interval integral (defect); `solve_ivp` DOP853 at `rtol = 1e-12` (state) |

### What was actually changed

* `StampRule` gained a de-stamping **order** (`order = 1 | 2`). Order 1 leaves the
  residual rate bias `-(d^2/2) q2` — a cadence-independent floor, which is
  exactly why the reported staircase saturated. Order 2 is Newton's forward
  difference over three coarse samples (`q_lo, q_prev, q_prev2`), whose residual
  bias `O(d h^2 q3)` sits *below* the trapezoid's own `O(h^2)` term.
* The second history sample is threaded through the operator interface
  (`q_prev2`), including the deferred correction of the causal
  `predictor_corrector` arm, and the transfer record now carries the raw
  previous sample (`q_prev_raw`) so the correction de-stamps with the same rule
  as the provisional step instead of silently falling back to order 1.
* `StampRule.effective_order()` reports the order actually available from the
  supplied history, so an operator that lacks the second sample degrades openly.
* The interval-complete ledger is exercised end-to-end (`IntervalExchangeLedger`)
  with **absolute** share accounting, which stays well defined when the exchanged
  flux has zero mean (the VIV force does — the aggregate *signed* share it used
  before is degenerate there).

## 2. Method

Manufactured solution: the operator alone integrates an analytic flux with a
closed-form interval integral, so a measured exponent can be attributed to the
operator algebra and to nothing else. Chains: the receiver is linear and its
update for a piecewise-constant rate is *exact* (matrix exponential / closed
form), so every residual measured through a chain belongs to the exchange.

The gate recomputes each exponent from the stored `(step, defect)` pairs — it
never trusts a stored `order` field — and returns three-valued verdicts.

## 3. Results

### 3.1 (a) manufactured-solution order

| arm | defect (coarsest `h`) | defect (finest `h`) | observed order | R² |
| --- | --- | --- | --- | --- |
| `centered_destamp1` | 5.497e-02 | 2.519e-04 | `1.9477` | 0.999735 |
| `centered_destamp2` | 3.678e-02 | 1.422e-04 | `2.0040` | 1.000000 |
| `centered_endpoint` | 1.905e-02 | 7.325e-05 | `2.0043` | 0.999991 |
| `centered_endstamp` | 9.204e-02 | 6.695e-03 | `0.9527` | 0.998593 |
| `lag_endpoint` | 1.065e-01 | 6.699e-03 | `0.9973` | 0.999999 |
| `lag_endstamp` | 1.841e-01 | 1.339e-02 | `0.9527` | 0.998593 |

Baseline `lag` still measures first order; the CN midpoint and both de-stamped
window arms exceed the 1.9 line. The order-2 de-stamping is exact on a quadratic
sample stream (unit test), and the causal `predictor_corrector` arm reproduces
the CN integrals to round-off after its deferred correction.

### 3.2 (b) interval-complete conservation ledger

| chain | identity residual | unattributed | lag handover deficit | CN+de-stamp(2) deficit | sub-window deficit | tail control (r = 8) |
| --- | --- | --- | --- | --- | --- | --- |
| settling | 0.0e+00 | 0.0e+00 | 0.064 | 2.131e-03 | 5.7e-14 | 0.875 |
| viv | 0.0e+00 | 0.0e+00 | 0.040 | 5.459e-04 | 1.6e-13 | 0.875 |
| weld | 0.0e+00 | 0.0e+00 | 0.064 | 2.116e-03 | 1.7e-16 | 0.874 |

Every produced sub-interval share is booked into exactly one exchange interval
and applied there; the produced-vs-released identity closes at machine level, the
unattributed share is identically zero, every interval is closed exactly once,
and the tail-booking control still reproduces the reported `(r-1)/r` loss. The
ledger's negative controls (double booking, double close, undeclared interval,
missing interval, closed-but-unapplied interval) are all refused or reported:
{"double_booking_refused": true, "double_close_refused": true, "missing_interval_detected": true, "unapplied_interval_detected": true, "undeclared_interval_refused": true}.

### 3.3 (c) weld staircase

Cadence refinement at the shipped fine step (the reported plateau):

| arm | defect (coarsest `h`) | defect (finest `h`) | defect order | state order | state error (finest `h`) | plateau defect |
| --- | --- | --- | --- | --- | --- | --- |
| `centered_destamp1` | 1.742e-03 | 2.508e-05 | `1.595` | `1.966` | 5.135e-08 | 2.532e-05 |
| `centered_destamp2` | 2.116e-03 | 1.257e-05 | `1.862` | `2.459` | 2.477e-08 | 6.223e-05 |
| `centered_endpoint` | 2.140e-03 | 8.378e-06 | `1.999` | `3.431` | 1.328e-09 | 5.859e-05 |
| `centered_endstamp` | 4.511e-03 | 4.000e-03 | `0.035` | `-0.001` | 1.333e-03 | 3.993e-03 |
| `lag_endpoint` | 6.360e-02 | 4.000e-03 | `0.997` | `0.998` | 1.333e-03 | 9.305e-03 |
| `lag_endstamp` | 6.756e-02 | 8.000e-03 | `0.773` | `0.774` | 2.666e-03 | 1.330e-02 |

`dt` halving at fixed cadence ratio `r = 2` (the
refinement the criterion names):

| arm | defect (`dt` coarsest) | defect (`dt` finest) | defect order | R² | state order | state error (finest) |
| --- | --- | --- | --- | --- | --- | --- |
| `centered_destamp1` | 2.508e-05 | 9.816e-08 | `1.999` | 1.000000 | `2.663` | 8.877e-11 |
| `centered_destamp2` | 3.753e-05 | 1.472e-07 | `1.999` | 1.000000 | `2.501` | 7.845e-11 |
| `centered_endpoint` | 3.351e-05 | 1.309e-07 | `2.000` | 1.000000 | `1.662` | 8.619e-11 |
| `centered_endstamp` | 4.000e-03 | 2.500e-04 | `1.000` | 1.000000 | `1.000` | 8.333e-05 |
| `lag_endpoint` | 8.000e-03 | 5.000e-04 | `1.000` | 1.000000 | `1.000` | 1.667e-04 |
| `lag_endstamp` | 1.200e-02 | 7.500e-04 | `1.000` | 1.000000 | `1.000` | 2.500e-04 |

Two different questions, two different answers, and this is the substance of the
maturity node:

* **cadence refinement** is the only knob a partitioned code has, and it is
  *structurally capped*: the published sample is the average of a fine window of
  width `2d`, which differs from the value at its own midpoint by `(d^2/6) q2`.
  No de-stamping can recover sub-window information, so at a fixed fine step the
  flux-defect staircase has an `O(d^2)` floor. Order-2 de-stamping halves that
  floor (`2.51e-05 -> 1.26e-05`
  on the weld chain; at the coarser cadences the wider stencil is slightly
  *worse*, `2.12e-03` vs
  `1.74e-03`) but it cannot
  remove it. The
  repair for a *fixed* fine step is not a better de-stamping but the
  interval-complete sub-window exchange, whose defect is at round-off.
* **`dt` halving** halves the window as well, so every `d`-induced floor halves
  twice; the restored operator then falls at second order on the defect metric
  as well as on the receiver state, and lands four orders of magnitude below the
  `1.2e-3` plateau.

### 3.4 (d) no-delay energy residual

| chain | arm | dt | interface-energy defect | stored-energy error | < 1e-5 ? |
| --- | --- | --- | --- | --- | --- |
| settling | `centered_endpoint` | 2.500e-04 | 2.620e-06 | 7.206e-09 | yes |
| settling | `predictor_corrector_endpoint` | 2.500e-04 | 2.620e-06 | 7.206e-09 | yes |
| viv | `centered_endpoint` | 2.500e-04 | 1.048e-06 | 1.708e-06 | yes |
| viv | `predictor_corrector_endpoint` | 2.500e-04 | 1.048e-06 | 1.708e-06 | yes |
| weld | `centered_endpoint` | 1.000e-03 | 2.094e-06 | 3.345e-10 | yes |
| weld | `predictor_corrector_endpoint` | 1.000e-03 | 2.094e-06 | 3.345e-10 | yes |

### 3.5 (e) three-chain regression

| gate | verdict | criteria passed |
| --- | --- | --- |
| flux / attribution gate (`exchange_gate`) | **pass** | 17/17 |
| energy gate (`exchange_energy`) | **pass** | 14/14 |

### 3.6 Acceptance table (machine-checkable criteria)

| criterion | required | measured (abridged) | verdict |
| --- | --- | --- | --- |
| `mms_time_order` | 0.9 <= lag <= 1.1; centered, destamp1, destamp2 all >= 1.9 | lag=0.99726930633309, centered=2.004252294981196, window_destamp1=1.9477035138608225, window_destamp2=2.0039892802101225 | **pass** |
| `interval_complete_identity` | identity residual <= 1e-12, unattributed == 0, all negative controls caught | per_chain={'settling': {'unattributed': 0.0, 'identity_residual': 0.0, 'booked_once': True, 'tail_control': {'booked_once': False, 'deficit_abs_share': 0.8749999999999334, 'expected_loss': 0.875, 'ratio': 8, 'unattrib... | **pass** |
| `weld_staircase_second_order` | dt-halving defect order >= 1.9 and finest defect < 3e-4 | dt_arm='centered_destamp2', dt_defect_order=1.998553390581774, dt_defect_r2=0.9999999158459013, dt_state_order=2.501012774644131 | **pass** |
| `no_delay_energy_residual` | energy_defect <= 1e-5 and reference_energy_error <= 1e-5 | caliper={'settling': {'centered_endpoint': {'energy_defect': 2.6204361829958627e-06, 'reference_energy_error': 7.20609960458714e-09, 'dt': 0.00025}, 'predictor_corrector_endpoint': {'energy_defect': 2.6204361829958627... | **pass** |
| `three_chain_no_regression` | flux gate pass (and energy gate pass when run) | flux={'failed': [], 'passed': 17, 'total': 17, 'verdict': 'pass'}, energy={'failed': [], 'passed': 14, 'total': 14, 'verdict': 'pass'} | **pass** |

overall: **pass** (5/5)

## 4. Figures

* `fig1_mms_order.png` — (a) quadrature defect vs exchange step for every arm; slopes are the measured orders.
* `fig2_weld_staircase.png` — (c) left: cadence refinement reproduces the 1.2e-3 plateau; right: dt halving restores second order.
* `fig3_interval_attribution.png` — (b) left: handover deficit vs cadence; right: identity residual at round-off vs the tail-booking control.
* `fig4_no_delay_energy.png` — (d) no-delay exchange energy defect vs dt per chain, below the 1e-5 line.
* `fig5_interval_defect_map.png` — per-interval defect field (interval index x cadence) for the lag hold and the order-2 de-stamped CN exchange.

## 5. Caveats and limits

* The chains are 0-D lumped receivers, so no spatial field contour exists; the
  field-like figure is the per-interval defect map (interval index × cadence).
* The de-stamping repair uses the *producer's own* coarse history. A real code
  that publishes only window averages at the exchange instant has exactly this
  information, so the result transfers; a code that publishes fine-resolution
  history could do better (`q_prev2` at fine spacing), which is not exercised
  here.
* `predictor_corrector` is causal but needs one interval of deferred
  correction: the reported integrals are post-correction. A receiver that cannot
  absorb a correction impulse must accept the provisional (first-order-in-time)
  rate.
* The energy criterion is stated on relative defects normalised per chain; the
  absolute membranes differ by chain (J, J/kg), so cross-chain comparison of the
  raw numbers is not meaningful.
* Criterion (c) is met on the `dt` ladder. On the cadence ladder the flux defect
  order is capped near `1.86`
  by the window-smoothing floor, while the state error on the same ladder
  converges at `1.97`.
  We report both rather than quoting the favourable one.
* Round-off floor: identity residuals are exact zeros here, but the ledger
  compares float sums, so on other hardware the same audit can be expected to
  read `~1e-16` rather than `0`.

## 6. Error judgement against the reference

Reference = exact interval integral (flux defect) and the `rtol = 1e-12`
`solve_ivp` oracle (state error, energy). The platform criterion is "≤ 3 %"; it is
answered per row and never smoothed over.

| quantity | reference | worst measured | ≤ 3 % ? |
| --- | --- | --- | --- |
| lag + end stamp, interval defect (coarsest cadence) | exact integral | 1.064e-01 | **no** |
| CN + end stamp, interval defect (coarsest cadence) | exact integral | 7.093e-03 | yes |
| CN + de-stamp(2), interval defect (coarsest cadence) | exact integral | 3.350e-03 | yes |
| CN + de-stamp(2), state error (finest `dt`) | `solve_ivp` oracle | 7.845e-11 | yes |
| no-delay energy defect (worst chain) | stored-energy budget | 2.620e-06 | yes |

## 7. Provenance

```json
{
  "arms": {
    "centered_destamp1": {
      "recenter": true,
      "recenter_order": 1,
      "scheme": "centered",
      "stamp_offset_ratio": 0.5
    },
    "centered_destamp2": {
      "recenter": true,
      "recenter_order": 2,
      "scheme": "centered",
      "stamp_offset_ratio": 0.5
    },
    "centered_endpoint": {
      "scheme": "centered",
      "stamp_offset_ratio": 0.0
    },
    "centered_endstamp": {
      "scheme": "centered",
      "stamp_offset_ratio": 0.5
    },
    "lag_endpoint": {
      "scheme": "lag",
      "stamp_offset_ratio": 0.0
    },
    "lag_endstamp": {
      "scheme": "lag",
      "stamp_offset_ratio": 0.5
    },
    "subcycle_complete": {
      "scheme": "centered",
      "subcycle": true
    }
  },
  "cadence_ratios": [
    1,
    2,
    4,
    8,
    16
  ],
  "generated_utc": "2026-11-10T22:49:38+00:00",
  "git_branch": "agent/m11636",
  "git_commit": "bfd8911db3d7093f7c5d34baea30aadef7affda2",
  "node": "p223w2b-n06",
  "numpy": "2.4.6",
  "payload_sha256": "f4558b113b63c688ecba9a4f160bfaf5558917c6c83fab911e00ff1bc5460db0",
  "platform": "Linux-6.6.52-1.7.lns23.loongarch64-loongarch64-with-glibc2.38",
  "python": "3.11.6",
  "script": "scripts/run_p223w2b_n06_maturity.py"
}
```
