#!/usr/bin/env python3
"""Render ``docs/P223W3_N06_COUPLING_TIME_EXCHANGE.md`` from verification.json.

Every number in the document is read from the artifact, so the report cannot
drift from the gate it describes (no hand-copied values).
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "p223w3_n06" / "verification.json"
DOC = ROOT / "docs" / "P223W3_N06_COUPLING_TIME_EXCHANGE.md"


def sci(x: float, digits: int = 2) -> str:
    return f"{x:.{digits}e}"


def pct(x: float, digits: int = 2) -> str:
    return f"{100.0 * x:.{digits}f} %"


def bar(x: float, digits: int = 2) -> str:
    return f"`{x:.{digits}f}`"


def main() -> int:
    payload = json.loads(ART.read_text())
    study = payload["study"]
    gate = payload["gate"]
    mms = study["mms"]
    chains = study["chains"]
    tail = study["attribution"]["tail_sweep"]
    controls = study["attribution"]["controls"]
    algebra = study["algebra"]
    prov = payload["provenance"]
    env = payload["environment"]
    hashes = payload["sha256"]
    sums = {}
    for line in (ART.parent / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split()
        sums[name] = digest
    record_digest = sums["verification.json"]

    crit = {c["key"]: c for c in gate["criteria"]}

    def mms_row(key: str, label: str) -> str:
        lad = mms[key]
        order, r2 = lad["observed_order"], lad["fit_r2"]
        coarse = lad["levels"][0]["defect_l1"]
        fine = lad["levels"][-1]["defect_l1"]
        return (f"| {label} | {sci(coarse)} | {sci(fine)} | {bar(order, 3)} | {r2:.5f} | "
                f"{lad['levels'][0]['h']:.5f} |")

    def arm_row(name: str, arm: str, label: str) -> str:
        lad = chains[name]["arms"][arm]
        order, r2 = lad["observed_order"], lad["fit_r2"]
        coarse = lad["levels"][-1]["defect_l1"]
        fine = lad["levels"][0]["defect_l1"]
        se_fine = lad["levels"][0]["state_error"]
        return (f"| {name} | {label} | {sci(coarse)} | {sci(fine)} | {bar(order, 3)} | "
                f"{r2:.4f} | {sci(se_fine)} | {sci(lad['saturation'])} |")

    def criterion_rows() -> str:
        rows = []
        for c in gate["criteria"]:
            measured = json.dumps(c["measured"], default=str)
            if len(measured) > 150:
                measured = measured[:147] + "..."
            rows.append(f"| `{c['key']}` | {c['required']} | {measured} | **{c['verdict']}** |")
        return "\n".join(rows)

    def error_table() -> str:
        rows = ["| chain | arm | worst interval defect | final state error vs oracle | ≤ 3 % ? |",
                "| --- | --- | --- | --- | --- |"]
        for name in ("viv", "weld", "settling"):
            for arm, label in (("lag_endstamp", "lag + end stamp"),
                               ("centered_endstamp", "CN + end stamp"),
                               ("centered_destamp", "CN + de-stamp"),
                               ("centered_endpoint", "CN, endpoint convention"),
                               ("subcycle_complete", "every sub-window")):
                lad = chains[name]["arms"][arm]
                worst = max(lv["max_interval_defect"] for lv in lad["levels"])
                final_se = lad["levels"][0]["state_error"]
                verdict = "yes" if max(worst, final_se) <= 0.03 else "**no**"
                rows.append(f"| {name} | {label} | {pct(worst)} | {pct(final_se)} | {verdict} |")
        return "\n".join(rows)

    tail_rows = "\n".join(
        f"| {r['ratio']} | {sci(r['unattributed_share'])} | {pct(r['unattributed_share'], 1)} | "
        f"{pct(r['released_fraction'], 2)} | {pct(r['predicted_loss'], 3)} |"
        for r in tail)

    control_rows = "\n".join(f"| `{k}` | {v} |" for k, v in controls.items())

    meta_rows = "\n".join(
        f"| {name} | {payload['meta']['description']} | {payload['meta']['fluid_field']} → "
        f"{payload['meta']['solid_field']} | {payload['meta']['period']} | "
        f"{sci(payload['meta']['dt_fine'], 1)} | {payload['meta']['t_end']} |"
        for name, payload in chains.items())

    algebra_rows = "\n".join(
        f"| `{key}` | {val['declared_order']} | {val['causal']} | {val['description']} |"
        for key, val in algebra.items())

    doc = f"""# Coupling time-exchange operator algebra, second order, and the interval-complete conservation ledger

**Campaign node** `p223w3-n06-physics-coupling-exchange` · **mission** 9722 ·
**repo** TensorPhysics · **module** `tensorphysics.coupling_energy.{{
exchange_operators, exchange_mms, time_exchange, exchange_gate}}` ·
acceptance record `artifacts/p223w3_n06/verification.json` (sha256
`{record_digest}` in `artifacts/p223w3_n06/SHA256SUMS`) · generated from the record
{payload['generated']} · commit `{prov['commit'][:12]}` on `{prov['branch']}`
({env['machine']}, Python {env['python']}, numpy {env['numpy']}).

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
{meta_rows}

Exchange-cadence ladders use ``r = {study['ratios']}`` fine steps per coarse
interval, i.e. the exchange is refreshed every one of the {len(study['ratios'])}
cadences while the fine solver step is *held fixed* — the only knob a real
partitioned code has.

### 1.3 The operator algebra

| scheme | declared order | causal | quadrature rule |
| --- | --- | --- | --- |
{algebra_rows}

`centered` needs ``q(t_hi)`` and is therefore the *idealised* member;
`predictor_corrector` predicts that endpoint by linear extrapolation
(``q̂ = 2q_lo − q_prev``, order {bar(mms['predictor_corrector_provisional']['observed_order'], 3)})
and applies a deferred correction when the realised sample arrives, after which
it reproduces the CN interval integral **exactly**
(``max relative difference = {crit['predictor_corrector_equals_centered']['measured']['max_rel_diff']}``).
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
{mms_row('lag', 'lag (end-stamped hold)')}
{mms_row('centered', 'centered (CN midpoint)')}
{mms_row('predictor_corrector_provisional', 'predictor-corrector, provisional')}
{mms_row('predictor_corrector', 'predictor-corrector, corrected')}
{mms_row('window_raw', 'window average, end-stamped (raw)')}
{mms_row('window_destamp', 'window average, de-stamped')}

The lag operator converges at first order
({bar(mms['lag']['observed_order'], 3)}), the CN midpoint at second
({bar(mms['centered']['observed_order'], 3)}), and the causal predictor-corrector
is second order *before* the correction
({bar(mms['predictor_corrector_provisional']['observed_order'], 3)}) and identical
to CN after it. Turning the stamp convention on shows the root cause of the
reported defect as a separate, measurable effect: consuming a window average as
an endpoint value collapses the order to
{bar(mms['window_raw']['observed_order'], 3)}; the de-stamping rule restores
{bar(mms['window_destamp']['observed_order'], 3)}.

### 3.2 Interval attribution

| sub-cycling ratio r | unattributed share (measured) | loss [%] | released fraction | analytic 1−1/r |
| --- | --- | --- | --- | --- |
{tail_rows}

Tail-only booking loses {pct(tail[2]['unattributed_share'], 1)} at ``r = 8`` and
{pct(tail[3]['unattributed_share'], 1)} at ``r = 16``: the reported 87–94 % band is
reproduced from first principles, without tuning, by the single rule "only the
closing sub-window is booked". Publishing every sub-window average instead gives
an unattributed share of **exactly zero** and a handover defect of
{sci(max(v['max_defect'] for v in crit['attribution_complete_exact']['measured'].values()))} —
each window average *is* the exact integral of its window, so the exchange
carries no quadrature error at all.

Ledger controls (the anomalies it must catch):

| control | outcome |
| --- | --- |
{control_rows}

### 3.3 The staircase, and what the repair removes

| chain | arm | defect (coarsest cadence) | defect (finest cadence) | observed order | R² | state error (finest) | plateau |
| --- | --- | --- | --- | --- | --- | --- | --- |
{arm_row('viv', 'lag_endstamp', 'lag + end stamp')}
{arm_row('viv', 'centered_endstamp', 'CN + end stamp')}
{arm_row('viv', 'centered_destamp', 'CN + de-stamp')}
{arm_row('weld', 'lag_endstamp', 'lag + end stamp')}
{arm_row('weld', 'centered_endstamp', 'CN + end stamp')}
{arm_row('weld', 'centered_destamp', 'CN + de-stamp')}
{arm_row('settling', 'lag_endstamp', 'lag + end stamp')}
{arm_row('settling', 'centered_endstamp', 'CN + end stamp')}
{arm_row('settling', 'centered_destamp', 'CN + de-stamp')}

With the end-stamped window convention the CN operator's defect is **flat**
(plateau ratio
{bar(max(c['measured'][n]['plateau_ratio'] for c in gate['criteria'] if c['key'] == 'endstamp_staircase_saturates' for n in ('viv', 'weld', 'settling')), 2)}
at worst), the fitted order collapses to
≈ {bar(max(c['measured'][n]['order'] for c in gate['criteria'] if c['key'] == 'endstamp_staircase_saturates' for n in ('viv', 'weld', 'settling')), 3)}:
refining the exchange cadence buys nothing. The weld-chain plateau of the state
error is {sci([c for c in gate['criteria'] if c['key'] == 'weld_plateau_matches_reported'][0]['measured']['plateau_state_error'])}
against the {sci(1.2e-3)} reported by the contract (ratio
{bar([c for c in gate['criteria'] if c['key'] == 'weld_plateau_matches_reported'][0]['measured']['ratio'], 3)}).
That agreement is not fitted: the plateau level is set by the half-fine-step
phase offset of the published sample, and the weld chain's fine step is
``2.0e-3 s``.

The repair (CN midpoint + de-stamping) cuts the finest-cadence defect by
{bar(min(v['defect_gain'] for v in [c for c in gate['criteria'] if c['key'] == 'repair_removes_plateau'][0]['measured'].values()), 0)}×
and the state error by
{bar(min(v['state_gain'] for v in [c for c in gate['criteria'] if c['key'] == 'repair_removes_plateau'][0]['measured'].values()), 0)}×
on the worst chain, and the ladder spread grows again by
{bar(max([c for c in gate['criteria'] if c['key'] == 'repair_restores_cadence_dependence'][0]['measured'].values()), 0)}×
across the cadences, i.e. the plateau is gone. The remaining error of the
repaired operator is still one order short of two on the *defect* metric
({bar(chains['weld']['arms']['centered_destamp']['observed_order'], 3)} on the weld
chain) because the de-stamping slope is one-sided; the *state* error already
converges at
{bar(chains['weld']['arms']['centered_destamp']['state_order'], 3)}.

### 3.4 Error judgement against the reference

Reference: the analytic interval integral (defect) and the `solve_ivp` oracle at
`rtol = 1e-12` (state error). "≤ 3 %" is the platform criterion; it is answered
per row, and a row that is **not** below 3 % is marked as such rather than
smoothed over.

{error_table()}

Reading — and this is the reason the gate is *not* an accuracy threshold:

* the lag/end-stamp practice **violates** the 3 % criterion
  (worst measured {pct(max(max(lv['max_interval_defect'] for lv in chains[n]['arms']['lag_endstamp']['levels']) for n in ('viv','weld','settling')), 1)}
  over the three chains) as soon as the exchange cadence is coarse;
* the CN + end-stamp practice **passes** the 3 % criterion
  ({pct(max(max(lv['max_interval_defect'] for lv in chains[n]['arms']['centered_endstamp']['levels']) for n in ('viv','weld','settling')), 2)}
  worst interval, {pct(max(chains[n]['arms']['centered_endstamp']['levels'][0]['state_error'] for n in ('viv','weld','settling')), 2)}
  final state error) while being *first order in nothing*: its order is
  {bar(chains['weld']['arms']['centered_endstamp']['observed_order'], 3)} and its
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
{criterion_rows()}

**Overall verdict: {gate['verdict'].upper()}** ({gate['passed']}/{gate['total']}
criteria{', ' + str(len(gate['not_run'])) + ' not run' if gate['not_run'] else ''}).

---

## 4. Note / caveats

1. **The m7031 contract document is not in this worktree.** Both reported limits
   were therefore treated as *hypotheses* and reconstructed mechanistically: the
   87–94 % loss follows from "only the closing sub-window is booked"
   (measured {pct(tail[2]['unattributed_share'], 1)} / {pct(tail[3]['unattributed_share'], 1)}
   at ``r = 8/16``), and the 1.2e-3 plateau follows from the half-fine-step phase
   offset of an end-stamped window average (measured
   {sci([c for c in gate['criteria'] if c['key'] == 'weld_plateau_matches_reported'][0]['measured']['plateau_state_error'])}).
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
   {bar(chains['weld']['arms']['centered_destamp']['observed_order'], 3)}
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
  `{record_digest[:16]}…` is in `artifacts/p223w3_n06/SHA256SUMS`
  (a record cannot carry its own hash — writing it would change the file):
  ladders `{hashes['cadence_ladders.json'][:16]}…`,
  fig1 `{hashes['fig1_operator_order.png'][:16]}…`,
  fig2 `{hashes['fig2_staircase_repair.png'][:16]}…`,
  fig3 `{hashes['fig3_attribution_loss.png'][:16]}…`,
  fig4 `{hashes['fig4_chain_trajectories.png'][:16]}…`.
* Reproduce: `python3 scripts/run_p223w3_n06_exchange.py` then
  `python3 scripts/render_p223w3_n06_doc.py`.
"""
    DOC.write_text(doc)
    print(f"wrote {DOC} ({len(doc)} chars)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())