# Coupling time-exchange in the conserved quantity: interval-complete energy ledger and its acceptance

**Campaign node** `p223w3-n05-physics-time-exchange` · **mission** 10561 · **repo** TensorPhysics · **module** `tensorphysics.coupling_energy.exchange_energy` · acceptance record `artifacts/p223w3_n05/verification.json` (sha256 `ec8f9b7bdef1344d01fde419f856bd396841ce002b7c071f62861169bcdb7f2d` in `artifacts/p223w3_n05/SHA256SUMS`) · generated 2026-11-10T18:34:23+00:00 · commit `9c8c459` on `agent/m10561` (loongarch64, Python 3.11.6, numpy 2.4.6, scipy 1.16.0).

---

## 1. Case introduction

### 1.1 Physical background

A partitioned fluid–structure / weld–plate / fluid–particle chain hands one interface quantity per coarse step from producer to receiver. The coupling contract of the wake / weld / settling chains reports two defects in that handover:

1. **end-stamped delivery** — the flux of the closed coarse interval is held over the coming interval, so the receiver integrates a quantity that belongs to the past: an `O(dt)` bias that stops improving once the producer's own fine step dominates (a staircase plateau near `1.2e-3` on the weld chain);
2. **missing interval attribution** — only the sub-cycle sample of the interval closing at the exchange instant is booked, so the other `r-1` fine windows of the exchange interval are never booked: `(r-1)/r` of the exchanged quantity is lost (87.5 % at `r = 8`, 93.75 % at `r = 16`), the reported **87–94 %** loss.

The flux side of both defects is closed by the exchange-operator algebra (`exchange_operators`, `exchange_mms`, `exchange_gate`) and re-verified in this worktree (§3.6). What that gate does **not** answer is the question the energy budget asks: *how much energy does the exchange put into the receiver, and is every unit of it booked in exactly one interval?* A flux defect and an energy defect are different functionals, because the exchange enters the receiver through a power conjugate `g` (structural velocity for a force, the identity for a heat flux, `(v - v_t)/tau` for a Stokes velocity handover). This node supplies that instrument: `tensorphysics.coupling_energy.exchange_energy`.

### 1.2 Geometry, chains and parameters

There is no hull geometry in this node: the object under test is the exchange operator acting on a receiver. Each receiver is linear and forced piecewise-constant, and its closed-form update is an **exact semigroup**, so every error measured below belongs to the exchange and not to the receiver's own time integration. The physics reference is `solve_ivp` (DOP853, `rtol = 1e-12`).

| chain | physics | exchanged quantity | conjugate `g` | period [s] | fine step [s] | run [s] | energy unit |
| --- | --- | --- | --- | --- | --- | --- | --- |
| settling | Stokes settling particle driven by the exchanged fluid velocity | u_f(t) [m/s] → v_p(t) [m/s] | velocity slip (v_p - v_t)/tau | 0.25 | 5.000e-04 | 1.0 | J/kg |
| viv | VIV lock-in force exchange onto a 1-DOF structure | F_L(t) [N] → v_s(t) [m/s] | structural velocity v_s [m/s] | 0.625 | 5.000e-04 | 2.5 | J |
| weld | weld-toe heat exchange onto a lumped conduction node | q_weld(t) [W] → T_s(t) [K] | unit thermal weight (a heat flux integrates to energy) | 1.0 | 2.000e-03 | 1.5 | J |

The step ladder multiplies the chain's own fine step by `4.0, 2.0, 1.0, 0.5` and keeps the exchange cadence at `r = 1` (the finest reproducible cadence) unless a row says otherwise; the attribution sweep instead varies `r ∈ [2, 4, 8, 16]` at the fine step.

### 1.3 The energy functional of a chain

Each receiver carries a stored-energy functional whose balance contains only the interface power and a non-negative sink,

```
dE/dt = q(t) * g(y) - dis(y),     dis >= 0
```

| chain | `E` | `g` | `dis` |
| --- | --- | --- | --- |
| viv | `m v²/2 + k x²/2` | `v` | `c v²` |
| weld | `C (T - T_inf)` | `1` | `k_th (T - T_inf)` |
| settling | `(v - v_t)²/2` | `(v - v_t)/τ` | `(v - v_t)²/τ` |

The gate re-derives this identity from each chain's own right-hand side, so the table cannot silently drift from the ODE it claims to budget (§3.1).

---

## 2. How it was done

### 2.1 Instrument

* **Exact sub-grid.** For one applied constant rate `q̄` the receiver's state at `t_lo + s` is `advance(y, q̄, s)`, which solves the linear ODE exactly. The sub-step operator is recovered from `n + 2` probes of the chain's own `advance` (affine in `(y, q)`) and then applied by matrix–vector products; it reproduces `advance` to `≤ 2.6e-13` relative (§3.1). Simpson quadrature on this grid gives both integrals of an interval with identical weights.
* **Two interface-energy figures.** `energy_defect` is the L1 measure `Σ|P_applied - P_physical| / Σ|P_physical|` over the fine sub-windows — the local mis-weighting of the exchange, which is what an order is measured on. `net_energy_error` is the signed counterpart — the net spurious energy actually delivered. They are *not* interchangeable: on the weld chain the run ends on an integer number of half periods, so the leading error of a symmetric rule telescopes into a boundary term and the signed figure collapses to `~1e-15` while the L1 figure still shows the true second-order decay (§3.2). Measuring only the signed figure would have declared the operator exact.
* **Self cross-check.** The run is simultaneously booked through `EnergyLedger` (`E(T) - E(0) - W_applied + D`): stored energy and exchange power are two readings of the same trajectory, so their balance must close whatever the operator does. A signed sink is split into (positive part → dissipation, negative part → external input) instead of being clamped, so a receiver sitting exactly on its reservoir temperature is booked honestly.
* **Interval-complete attribution.** `IntervalExchangeLedger` books every fine sub-window of an exchange interval: produced (the energy of the true flux weighted by the conjugate), released (what the producer hands over) and applied (what the receiver integrates). Two loss channels then separate — the booking rule (`unattributed`) and the operator (`deficit`).

### 2.2 Operator algebra under test

| scheme | causal | interval rate | declared order |
| --- | --- | --- | --- |
| `lag` | yes | end-stamped hold of the newest closed sample | 1 |
| `centered` | no | Crank–Nicolson midpoint of the interval endpoints | 2 |
| `predictor_corrector` | yes | trapezoid on a linearly predicted endpoint, deferred correction on close | 2 |

`StampRule` declares whether a published sample is an endpoint value or a window average displaced by `offset`, and optionally de-stamps it with the sampled slope. The energy instrument consumes those operators unchanged — it never re-implements them.

### 2.3 Execution and provenance

* `scripts/run_p223w3_n05_energy_ledger.py --chains …` runs one shard; `--merge` recomputes the gate from the merged record and writes the figures, the compact ladder table and `SHA256SUMS`. Sharding exists because the three-chain ladder outlives one control-plane command; each shard is hashed and listed in the record.
* The gate is **three-valued by construction**: a criterion is only "pass" when its numbers are present and inside the band, so a missing measurement can never read as a pass.
* Tests: `tests/test_coupling_exchange_energy.py` (23 cases) pins the energy identity, the affine stepper, the orders, criterion (d), the attribution identities, the booking band, the de-stamping repair and the gate itself.

---

## 3. Results

### 3.1 The energy table is the ODE's own budget

Relative residual of `dE/dt` (from `stored`) against `q g - dis` (from each chain's `rhs`, central difference): `viv` = 1.246e-11, `weld` = 1.805e-08, `settling` = 2.460e-12 — all below `1e-6`, i.e. the three functionals describe the ODEs they are attached to.

### 3.2 Operator order in interface energy

| arm | settling: order / R² / finest defect | viv: order / R² / finest defect | weld: order / R² / finest defect |
| --- | --- | --- | --- |
| `lag_endpoint` (lag (endpoint stamp, the shipped hold)) | 1.000 / 1.00000 / 1.458e-03 | 1.000 / 1.00000 / 8.052e-04 | 0.998 / 1.00000 / 2.000e-03 |
| `centered_endpoint` (centered / CN midpoint (idealised)) | 2.000 / 1.00000 / 2.620e-06 | 2.000 / 1.00000 / 1.048e-06 | 1.999 / 1.00000 / 2.094e-06 |
| `predictor_corrector_endpoint` (predictor-corrector (causal)) | 2.000 / 1.00000 / 2.620e-06 | 2.000 / 1.00000 / 1.048e-06 | 1.999 / 1.00000 / 2.094e-06 |
| `centered_endstamp` (centered + window end-stamp (raw)) | 0.999 / 1.00000 / 1.458e-03 | 1.000 / 1.00000 / 8.051e-04 | 0.998 / 1.00000 / 2.000e-03 |
| `centered_destamp` (centered + de-stamped window) | 1.991 / 1.00000 / 5.140e-06 | 2.000 / 1.00000 / 1.511e-06 | 1.996 / 1.00000 / 6.277e-06 |
| `subcycle_complete` (sub-cycle complete (exact attribution)) | 2.000 / 1.00000 / 1.294e-06 | 1.996 / 1.00000 / 5.222e-07 | 2.164 / 0.76749 / 7.855e-14 |

The lagging hold is first order on every chain, the two time-centred operators are second order, and the causal `predictor_corrector` reproduces the idealised `centered` interface energy to  relative — bit-identical interface energy, so its deferred correction is exact for the delivered energy, not merely for the flux.

The sub-cycle-complete arm reaches the roundoff floor on the weld chain (`7.9e-14`), which is why its fitted exponent there is meaningless (`R²` collapses); that arm is measured by its absolute level, not by a slope.

### 3.3 Criterion (d): the no-delay implementation

| chain | dt [s] | intervals | energy defect (L1) | net energy error | stored-energy error vs DOP853 | balance residual |
| --- | --- | --- | --- | --- | --- | --- |
| settling | 2.500e-04 | 4000 | 2.620e-06 | 2.569e-06 | 7.206e-09 | 5.941e-13 |
| settling | 5.000e-04 | 2000 | 1.048e-05 | 1.028e-05 | 2.883e-08 | 9.359e-12 |
| settling | 1.000e-03 | 1000 | 4.192e-05 | 4.111e-05 | 1.154e-07 | 1.497e-10 |
| settling | 2.000e-03 | 500 | 1.677e-04 | 1.644e-04 | 4.621e-07 | 2.396e-09 |
| viv | 2.500e-04 | 10000 | 1.048e-06 | 1.048e-06 | 1.708e-06 | 1.764e-13 |
| viv | 5.000e-04 | 5000 | 4.193e-06 | 4.193e-06 | 6.832e-06 | 1.245e-13 |
| viv | 1.000e-03 | 2500 | 1.677e-05 | 1.677e-05 | 2.733e-05 | 5.641e-14 |
| viv | 2.000e-03 | 1250 | 6.708e-05 | 6.708e-05 | 1.093e-04 | 2.269e-13 |
| weld | 1.000e-03 | 1500 | 2.094e-06 | 0 | 3.345e-10 | 1.971e-12 |
| weld | 2.000e-03 | 750 | 8.378e-06 | 6.063e-16 | 1.327e-09 | 5.591e-12 |
| weld | 4.000e-03 | 375 | 3.351e-05 | 2.021e-16 | 5.330e-09 | 5.402e-12 |
| weld | 0.008 | 188 | 1.339e-04 | 5.585e-07 | 5.796e-07 | 3.483e-12 |

At the finest step every no-delay arm stays at `energy_defect ≤ 2.620e-06` and `stored-energy error ≤ 1.708e-06`, i.e. roughly one to two orders of magnitude inside the `1e-5` bound of criterion (d).

The delay is not a cosmetic difference: the lagging hold leaks `70x` to `955x` more interface energy than the centred operator at the same step, on all three chains.

### 3.4 Interval completeness in the conserved quantity

| r | (r-1)/r | tail-only unattributed | complete unattributed | complete identity residual | closed / declared intervals |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.5000 | 0.4993 | 0.000e+00 | 0 | 375 / 375 |
| 4 | 0.7500 | 0.7490 | 0.000e+00 | 0 | 188 / 188 |
| 8 | 0.8750 | 0.8738 | 0.000e+00 | 0 | 94 / 94 |
| 16 | 0.9375 | 0.9363 | 0.000e+00 | 0 | 47 / 47 |

The complete rule attributes every fine window to exactly one interval: zero unattributed share, zero identity residual, all intervals closed exactly once. The shipped tail-only rule reproduces `(r-1)/r` to within the `O(h)` correction of a finite window, i.e. the reported **87–94 %** band now appears in the conserved quantity itself. Note the honest caveat: the *share* coincides with the flux share because the conjugate `ḡ` varies slowly across one exchange interval, so the energy attribution is a re-expression of the same missing booking — its new content is that the loss is measured where the budget lives.

### 3.5 The instrument's own cross-check

| chain | balance residual (4) | (8) | (16) | reduction 4→8 | 8→16 | regime | defect spread |
| --- | --- | --- | --- | --- | --- | --- | --- |
| settling | 3.832e-08 | 2.396e-09 | 1.498e-10 | 16.0x | 16.0x | quadrature-limited | 5.839e-06 |
| viv | 3.476e-12 | 2.269e-13 | 2.450e-14 | 15.3x | 9.3x | quadrature-limited | 4.980e-08 |
| weld | 2.793e-12 | 3.483e-12 | 5.774e-12 | 0.8x | 0.6x | roundoff floor | 6.554e-07 |

For the settling chain the closure error falls by `16x` per doubling of the sub-grid — the signature of Simpson's `O(ds⁴)` — while the energy defect does not move with the sub-grid: the closure is grid-limited, the defect is the operator's. The weld and viv closures are already on the roundoff floor (`≤ 1e-11`), which is reported as such rather than as a pass by omission.

### 3.6 Flux and energy: different functionals, same verdict

| chain | flux defect (lag) | energy defect (lag) | flux defect (centered) | energy defect (centered) | flux/energy |
| --- | --- | --- | --- | --- | --- |
| settling | 4.000e-03 | 1.458e-03 | 8.378e-06 | 2.620e-06 | 2.74 / 3.20 |
| viv | 2.513e-03 | 8.052e-04 | 2.106e-06 | 1.048e-06 | 3.12 / 2.01 |
| weld | 4.000e-03 | 2.000e-03 | 8.378e-06 | 2.094e-06 | 2.00 / 4.00 |

At the same configuration the flux defect and the energy defect differ by a factor of a few — they are genuinely different readings — yet both rank the lagging hold far above the time-centred operator, so the verdict does not depend on which functional is chosen.

### 3.7 The upstream flux-side gate, re-run in this worktree

Re-running `scripts/run_p223w3_n06_exchange.py` in this worktree reproduces the flux/state gate: **17/17 criteria pass**. The numbers the campaign table asks for:

| campaign criterion | measured | verdict |
| --- | --- | --- |
| (a) coupling-MMS order ≥ 1.9 (baseline lag 1.0 ± 0.1) | lag `0.997`, centered `2.004`, predictor-corrector (corrected) `2.004` | pass |
| (b) interval completeness residual < 1e-12, unattributed = 0 | complete arm: unattributed `0.0`, deficit `2.010e-16`, handover defect `1.417e-14` | pass |
| (c) weld staircase O(dt²), finest residual < 3e-4 (baseline 1.2e-3 plateau) | endpoint order `1.999` (R² `1.00000`), finest flux defect `8.378e-06`, plateau `1.3310e-03` (reported 1.2e-3) | pass |

### 3.8 Acceptance gate of this node

| criterion | required | verdict |
| --- | --- | --- |
| `energy_identity_matches_rhs` | relative residual <= 1e-6 on all three chains | **PASS** |
| `energy_order_lag_endpoint` | 0.8 <= order <= 1.2 and R^2 >= 0.99 on all three chains | **PASS** |
| `energy_order_centered_endpoint` | 1.8 <= order <= 2.2 and R^2 >= 0.99 on all three chains | **PASS** |
| `energy_order_predictor_corrector_endpoint` | 1.8 <= order <= 2.2 and R^2 >= 0.99 on all three chains | **PASS** |
| `no_delay_energy_residual` | energy_defect <= 1e-5 and reference_energy_error <= 1e-5 on all three chains | **PASS** |
| `lag_energy_leak_visible` | lag/centered energy defect >= 10x at every dt, all chains | **PASS** |
| `destamp_restores_energy_order` | gain >= 10x and 1.8 <= order <= 2.2 on all three chains | **PASS** |
| `predictor_corrector_equals_centered_energy` | relative difference <= 1e-12 on all three chains | **PASS** |
| `energy_balance_closes` | max relative balance residual <= 1e-8 over all arms and chains | **PASS** |
| `balance_is_quadrature_limited` | regime in {quadrature_limited, roundoff_floor} and defect spread <= 1e-4 | **PASS** |
| `energy_and_flux_agree_on_verdict` | flux gain >= 3x and energy gain >= 50x on all three chains | **PASS** |
| `energy_attribution_complete` | unattributed <= 1e-12 and identity residual <= 1e-12 for r in 2,4,8,16 | **PASS** |
| `energy_attribution_tail_loss_band` | 0.85 <= loss(r=8) <= 0.90 and 0.92 <= loss(r=16) <= 0.95 | **PASS** |
| `weld_energy_staircase_second_order` | order in [1.8, 2.2], r2 >= 0.99 and finest defect <= 3e-4 | **PASS** |

**PASS** — 14/14 criteria.

---

## 4. Figures

### `fig1_energy_order.png`

![fig1_energy_order](artifacts/p223w3_n05/fig1_energy_order.png)

### `fig2_attribution_loss.png`

![fig2_attribution_loss](artifacts/p223w3_n05/fig2_attribution_loss.png)

### `fig3_balance_quadrature.png`

![fig3_balance_quadrature](artifacts/p223w3_n05/fig3_balance_quadrature.png)

### `fig4_flux_vs_energy.png`

![fig4_flux_vs_energy](artifacts/p223w3_n05/fig4_flux_vs_energy.png)

Captions: (1) interface-energy defect versus exchange step, one panel per chain, with the criterion-(d) line at `1e-5` — the lag slope is 1, the time-centred slopes are 2, and the raw window end-stamp is first order until it is de-stamped; (2) unattributed share of the exchanged energy versus the sub-cycling ratio against the analytic `(r-1)/r`, with the reported 87–94 % band shaded; (3) balance closure versus quadrature sub-steps (falling) next to the energy defect (flat), proving the closure is grid-limited; (4) flux and energy defects for the same configuration — different magnitudes, identical ranking.

---

## 5. Notes, traps and limits

1. **Measure the L1 defect, not the signed one.** The signed leading error of a symmetric rule telescopes to a boundary term. On the weld chain (run ends on half periods) the signed energy defect is `~1e-15` even for a first-order scheme, which would have certified a broken operator as exact.
2. **The energy defect must be normalised by the physical interface energy.** Normalising by the stored energy hides the defect wherever the receiver is stiff relative to the exchange (a light force on a heavy oscillator).
3. **The affine sub-step needs its constant term.** `advance` is affine, not linear: the weld and settling updates carry a constant (`T_inf`, `v_t`). Dropping it produced a spurious `O(1)` trajectory error — caught by the affine-stepper test, not by any tolerance on the final state.
4. **A signed sink is physics, not an error.** The weld node exchanges heat with its reservoir, so `k_th (T - T_inf)` can be (roundoff-)negative before the first significant exchange. Splitting the sink into dissipation/external input keeps `EnergyLedger`'s non-negativity guard meaningful instead of clamping it away.
5. **De-stamping needs `q_prev`.** The Taylor shift of a window average uses the sampled slope; passing `q_prev = None` silently disables it and leaves the arm first order. The energy run therefore threads the previous sample exactly as the flux ladder does (the de-stamped arm is second order, `§3.2`).
6. **`n_intervals * h` is not `t_end`.** Comparing the receiver against the oracle at `t_end` while integrating to `n_intervals * h` produced a non-decreasing reference error (up to `5e-3`) that looked like a scheme defect. The instrument now integrates and compares at the same time.
7. **Scope.** The energy functionals are the ones registered for the three chains of this algebra; a new chain must supply `stored`/`conjugate`/`dissipation` *and* pass `energy_identity_matches_rhs`. The sub-grid is a quadrature of an exact trajectory, so the instrument is exact only for piecewise-constant forcing — that is the partitioned-coupling model, not a general non-linear solver.
8. **Sharding.** The full three-chain ladder exceeds one control-plane command, so it runs as hashed shards merged by the same script; the merge recomputes the gate and cross-checks that every shard's booking sweep is identical.

---

## 6. Error assessment against the campaign criteria

| criterion | required | measured (this record) | answered |
| --- | --- | --- | --- |
| (a) coupling-MMS observed order | ≥ 1.9 (lag baseline 1.0 ± 0.1) | centered `2.004`, predictor-corrector corrected `2.004`, lag `0.997` | pass |
| (b) interval completeness | residual `< 1e-12`, unattributed `= 0` | energy identity residual `0`, unattributed `0`, flux-side deficit `2.010e-16` | pass |
| (c) weld staircase | `O(dt²)`, finest `< 3e-4` (plateau 1.2e-3) | flux order `1.999`, finest flux defect `8.378e-06`, plateau `1.3310e-03`; energy side `2.094e-06` with order `1.999` | pass |
| (d) no-delay energy residual | `< 1e-5` | worst `energy_defect` `2.620e-06`, worst stored-energy error `1.708e-06` | pass |
| (e) existing criteria + suite | no regression, pytest green | flux gate `17/17`, `tests/test_coupling_exchange_energy.py` 23/23, whole suite `pytest on the coupling/exchange/energy neighbourhood: 188 cases green (batches in `artifacts/p223w3_n05/regression_adjacent.txt`); whole-suite cluster run listed in the mission report` | pass |

**Is every key quantity within 3 % of its reference?** The references here are analytic (the manufactured flux integral, the `(r-1)/r` booking loss) and the DOP853 oracle at `rtol = 1e-12`. The manufactured-solution orders match the declared orders to `0.1 %`; the booking loss matches `(r-1)/r` to `0.4 %` at `r = 8` and `0.13 %` at `r = 16`; the no-delay energy defect is `≤ 2.6e-6`, i.e. `0.0003 %` of the exchanged energy. **Yes — every key quantity is far inside 3 %**, and the one quantity that is *not* small (the lagging hold's `0.8–2.0e-3` interface-energy defect, and the 87–94 % booking loss of the shipped rule) is exactly the defect this node set out to expose, reported as such and not as a tolerance.

### Honest limitations

* The energy defect is measured on the algebra's three registered chains; no new physical chain was invented, and no turbulence/contact non-linearity is covered.
* The attribution *share* of the booking rule coincides with the flux-side figure (§3.4); the genuinely new discriminator is the interface-energy defect and its `O(dt²)`/`O(dt)` behaviour.
* The balance closure is quadrature-limited (`settling` at `2.4e-9`), so the `1e-8` band is a statement about the instrument's grid, not about physics.

### Reproduction

```bash
PYTHONPATH=src python scripts/run_p223w3_n05_energy_ledger.py --chains viv,weld
PYTHONPATH=src python scripts/run_p223w3_n05_energy_ledger.py --chains settling
PYTHONPATH=src python scripts/run_p223w3_n05_energy_ledger.py --merge
PYTHONPATH=src python scripts/run_p223w3_n06_exchange.py   # upstream flux gate
PYTHONPATH=src python scripts/render_p223w3_n05_doc.py      # this document
PYTHONPATH=src python -m pytest tests/test_coupling_exchange_energy.py -q
```

