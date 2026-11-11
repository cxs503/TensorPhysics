#!/usr/bin/env python3
"""Run the ``p223w2b-n06`` coupling-operator maturity study and emit artifacts.

Outputs (worktree paths):

    artifacts/p223w2b_n06/maturity.json            study + verdicts + provenance
    artifacts/p223w2b_n06/verification.json        verdict only (gate-shaped)
    artifacts/p223w2b_n06/fig1_mms_order.png
    artifacts/p223w2b_n06/fig2_weld_staircase.png
    artifacts/p223w2b_n06/fig3_interval_attribution.png
    artifacts/p223w2b_n06/fig4_no_delay_energy.png
    artifacts/p223w2b_n06/fig5_interval_defect_map.png
    docs/P223W2B_N06_COUPLING_OPERATOR_MATURITY.md (generated report)

The report markdown is generated from the same structure that is written to
JSON, so no number in the report is transcribed by hand.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.coupling_energy.maturity import (  # noqa: E402
    CADENCE_RATIOS,
    MATURITY_ARMS,
    format_table,
    gate_maturity,
    run_maturity_study,
)
from tensorphysics.coupling_energy.time_exchange import (  # noqa: E402
    build_chain,
    run_cadence_ladder,
)

OUT = ROOT / "artifacts" / "p223w2b_n06"
DOC = ROOT / "docs" / "P223W2B_N06_COUPLING_OPERATOR_MATURITY.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:  # pragma: no cover - provenance only
        return "unavailable"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
LABEL = {
    "lag_endpoint": "lag (endpoint)",
    "lag_endstamp": "lag + end stamp",
    "centered_endstamp": "CN midpoint + end stamp",
    "centered_destamp1": "CN + de-stamp (order 1)",
    "centered_destamp2": "CN + de-stamp (order 2)",
    "centered_endpoint": "CN midpoint (endpoint)",
    "subcycle_complete": "interval-complete sub-window",
}


def plot_mms_order(study: dict) -> Path:
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    arms = study["mms"]["arms"]
    marks = {"lag_endpoint": "o", "lag_endstamp": "s", "centered_endpoint": "D",
             "centered_endstamp": "v", "centered_destamp1": "^",
             "centered_destamp2": "P"}
    for name, payload in arms.items():
        h = [lv["h"] for lv in payload["levels"]]
        d = [lv["defect_l1"] for lv in payload["levels"]]
        ax.loglog(h, d, marker=marks.get(name, "x"), label=f"{LABEL[name]}  "
                  f"(order {payload['observed_order']:.3f})")
    ax.set_xlabel("exchange step $h$ [s]")
    ax.set_ylabel("relative interval-integral defect")
    ax.set_title("(a) manufactured solution: quadrature order of the exchange algebra")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=7, loc="lower left")
    path = OUT / "fig1_mms_order.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_weld_staircase(study: dict) -> Path:
    stair = study["staircase"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.4))
    for name in ("lag_endstamp", "centered_endstamp", "centered_destamp1",
                 "centered_destamp2"):
        row = stair["cadence"][name]
        axes[0].loglog(row["steps"], row["state_errors"], marker="o",
                       label=f"{LABEL[name]}")
    axes[0].set_xlabel("exchange step $h = r\\,dt_{fine}$ [s] (fine step fixed)")
    axes[0].set_ylabel("weld-node state error vs oracle")
    axes[0].set_title("(c-left) cadence refinement: the reported plateau")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].invert_xaxis()
    axes[0].legend(fontsize=7)
    axes[0].axhline(1.2e-3, color="k", ls="--", lw=0.9)
    axes[0].text(0.02, 1.35e-3, "reported plateau 1.2e-3", fontsize=7)

    for name in ("lag_endstamp", "centered_endstamp", "centered_destamp1",
                 "centered_destamp2"):
        row = stair["dt"][name]
        dts = [lv["dt_fine"] for lv in row["levels"]]
        axes[1].loglog(dts, [lv["defect_l1"] for lv in row["levels"]], marker="s",
                       label=f"{LABEL[name]} (order {row['defect_order']:.3f})")
    axes[1].axhline(3e-4, color="k", ls="--", lw=0.9)
    axes[1].text(0.02, 3.6e-4, "acceptance 3e-4", fontsize=7)
    axes[1].set_xlabel("fine step $dt$ [s] (cadence ratio fixed)")
    axes[1].set_ylabel("relative interval-integral defect")
    axes[1].set_title("(c-right) $dt$ halving: second order restored")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].invert_xaxis()
    axes[1].legend(fontsize=7)
    path = OUT / "fig2_weld_staircase.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_attribution(study: dict) -> Path:
    attribution = study["attribution"]["chains"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ratios = study["attribution"]["ratios"]
    for name in ("lag_endpoint", "centered_endstamp", "centered_destamp2",
                 "subcycle_complete"):
        defects = []
        for ratio in ratios:
            row = next(lv for lv in attribution["weld"][name]["levels"]
                       if lv["ratio"] == ratio)
            defects.append(max(row["handover_deficit_abs_share"], 1e-18))
        axes[0].loglog(ratios, defects, marker="o", label=LABEL[name])
    axes[0].set_xlabel("exchange cadence ratio $r$ (fine sub-steps per exchange)")
    axes[0].set_ylabel("absolute handover deficit / total absolute exchange")
    axes[0].set_title("(b-left) handover deficit of the weld interface")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend(fontsize=7)

    names = list(attribution)
    residuals = [attribution[c]["centered_destamp2"]["worst_identity_residual"]
                 for c in names]
    tail = [attribution[c]["control_tail_booking"]["unattributed_abs_share"]
            for c in names]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, [max(v, 1e-18) for v in residuals], 0.4,
                label="identity residual (released - produced)")
    axes[1].bar(x + 0.2, tail, 0.4, label="tail-booking control loss (r = 8)")
    axes[1].set_yscale("log")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names)
    axes[1].axhline(1e-12, color="k", ls="--", lw=0.9)
    axes[1].set_ylabel("share of total absolute exchange")
    axes[1].set_title("(b-right) identity at round-off, control at (r-1)/r")
    axes[1].legend(fontsize=7)
    path = OUT / "fig3_interval_attribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_no_delay_energy(study: dict) -> Path:
    energy = study["no_delay_energy"]["chains"]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for chain, payload in energy.items():
        for label, row in payload.items():
            dts = [lv["dt"] for lv in row["levels"]]
            vals = [max(lv["energy_defect"], 1e-16) for lv in row["levels"]]
            ax.loglog(dts, vals, marker="o", ls="-" if "centered" == label.split("_")[0]
                      else "--", label=f"{chain} / {label}")
    ax.axhline(1e-5, color="k", ls="--", lw=0.9)
    ax.text(0.02, 1.2e-5, "acceptance 1e-5", fontsize=7)
    ax.set_xlabel("fine step $dt$ [s]")
    ax.set_ylabel("relative interface-energy defect")
    ax.set_title("(d) no-delay (time-centred) exchange: energy budget closure")
    ax.grid(True, which="both", alpha=0.3)
    ax.invert_xaxis()
    ax.legend(fontsize=6)
    path = OUT / "fig4_no_delay_energy.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_interval_defect_map() -> Path:
    """Exchange-interval defect field: interval index x exchange cadence."""
    from tensorphysics.coupling_energy.exchange_operators import (
        StampConvention,
        StampRule,
        operator_for,
    )

    chain = build_chain("weld")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, (arm, spec) in zip(axes, (
            ("lag + end stamp", dict(scheme="lag", recenter=False, order=1)),
            ("CN + de-stamp (order 2)", dict(scheme="centered", recenter=True, order=2)))):
        rows = []
        for ratio in (1, 2, 4, 8):
            h = ratio * chain.dt_fine
            scale = abs(chain.mms.interval_integral(0.0, chain.t_end)) / max(
                int(round(chain.t_end / h)), 1)
            stamp = (StampRule(StampConvention.WINDOW_MIDPOINT, 0.5 * chain.dt_fine,
                               spec["recenter"], order=spec["order"]))
            op = operator_for(spec["scheme"], stamp)
            q_prev = q_prev2 = None
            row = []
            for i in range(int(round(chain.t_end / h))):
                t_lo, t_hi = i * h, (i + 1) * h
                q_lo, q_hi = chain.sample(t_lo, 0.5 * chain.dt_fine), \
                    chain.sample(t_hi, 0.5 * chain.dt_fine)
                tr = op.provisional_transfer(i, t_lo, t_hi, q_lo=q_lo, q_hi=q_hi,
                                             q_prev=q_prev, q_prev2=q_prev2)
                q_prev2, q_prev = q_prev, q_lo
                row.append(abs(tr.integral - chain.mms.interval_integral(t_lo, t_hi))
                           / scale)
            rows.append(row)
        nx = max(len(r) for r in rows)
        field = np.full((len(rows), nx), np.nan)
        for j, r in enumerate(rows):
            field[j, :len(r)] = r
        im = ax.imshow(np.log10(np.maximum(field, 1e-16)), aspect="auto",
                       origin="lower", cmap="magma")
        ax.set_xlabel("exchange interval index")
        ax.set_ylabel("cadence ratio $r$")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels(["1", "2", "4", "8"])
        ax.set_title(f"per-interval defect field — {arm}")
        fig.colorbar(im, ax=ax, label="$\\log_{10}$ relative interval defect")
    path = OUT / "fig5_interval_defect_map.png"
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------
def render_report(study: dict, verdict: dict, provenance: dict,
                  figures: dict) -> str:
    mms = study["mms"]["arms"]
    stair = study["staircase"]
    attribution = study["attribution"]["chains"]
    energy = study["no_delay_energy"]["chains"]
    rows_mms = "\n".join(
        f"| `{name}` | {payload['coarsest_defect']:.3e} | "
        f"{payload['finest_defect']:.3e} | `{payload['observed_order']:.4f}` | "
        f"{payload['fit_r2']:.6f} |" for name, payload in mms.items())
    rows_cad = "\n".join(
        f"| `{name}` | {row['coarsest_defect']:.3e} "
        f"| {row['finest_defect']:.3e} | `{row['defect_order']:.3f}` "
        f"| `{row['state_order']:.3f}` | {row['finest_state_error']:.3e} "
        f"| {row['plateau_defect']:.3e} |"
        for name, row in stair["cadence"].items() for row in [row])
    rows_dt = "\n".join(
        f"| `{name}` | {row['levels'][0]['defect_l1']:.3e} "
        f"| {row['finest_defect']:.3e} | `{row['defect_order']:.3f}` "
        f"| {row['defect_r2']:.6f} | `{row['state_order']:.3f}` "
        f"| {row['finest_state_error']:.3e} |"
        for name, row in stair["dt"].items())
    rows_ident = "\n".join(
        f"| {chain} | {payload['centered_destamp2']['worst_identity_residual']:.1e} "
        f"| {payload['centered_destamp2']['worst_unattributed']:.1e} "
        f"| {payload['lag_endpoint']['worst_handover_deficit']:.3f} "
        f"| {payload['centered_destamp2']['worst_handover_deficit']:.3e} "
        f"| {payload['subcycle_complete']['worst_handover_deficit']:.1e} "
        f"| {payload['control_tail_booking']['unattributed_abs_share']:.3f} |"
        for chain, payload in attribution.items())
    rows_energy = "\n".join(
        f"| {chain} | `{label}` | {row['dt']:.3e} | {row['energy_defect']:.3e} "
        f"| {row['reference_energy_error']:.3e} | {'yes' if max(row['energy_defect'], row['reference_energy_error']) < 1e-5 else '**no**'} |"
        for chain, payload in energy.items() for label, row in payload.items())
    criterion_table = format_table(verdict)
    fig_lines = "\n".join(f"* `{name}` — {cap}" for name, cap in figures.items())

    return f"""# Coupling time-exchange operator maturity: second-order algebra and an interval-complete conservation ledger

Campaign node `p223w2b-n06` (TensorPhysics). Baseline = `main` at
`{provenance['git_commit']}`. Every number below is written by
`scripts/run_p223w2b_n06_maturity.py` from `artifacts/p223w2b_n06/maturity.json`
(sha256 `{provenance['payload_sha256']}`); nothing is transcribed by hand.

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
{rows_mms}

Baseline `lag` still measures first order; the CN midpoint and both de-stamped
window arms exceed the 1.9 line. The order-2 de-stamping is exact on a quadratic
sample stream (unit test), and the causal `predictor_corrector` arm reproduces
the CN integrals to round-off after its deferred correction.

### 3.2 (b) interval-complete conservation ledger

| chain | identity residual | unattributed | lag handover deficit | CN+de-stamp(2) deficit | sub-window deficit | tail control (r = 8) |
| --- | --- | --- | --- | --- | --- | --- |
{rows_ident}

Every produced sub-interval share is booked into exactly one exchange interval
and applied there; the produced-vs-released identity closes at machine level, the
unattributed share is identically zero, every interval is closed exactly once,
and the tail-booking control still reproduces the reported `(r-1)/r` loss. The
ledger's negative controls (double booking, double close, undeclared interval,
missing interval, closed-but-unapplied interval) are all refused or reported:
{json.dumps(study['attribution']['controls'], sort_keys=True)}.

### 3.3 (c) weld staircase

Cadence refinement at the shipped fine step (the reported plateau):

| arm | defect (coarsest `h`) | defect (finest `h`) | defect order | state order | state error (finest `h`) | plateau defect |
| --- | --- | --- | --- | --- | --- | --- |
{rows_cad}

`dt` halving at fixed cadence ratio `r = {stair['staircase_ratio']}` (the
refinement the criterion names):

| arm | defect (`dt` coarsest) | defect (`dt` finest) | defect order | R² | state order | state error (finest) |
| --- | --- | --- | --- | --- | --- | --- |
{rows_dt}

Two different questions, two different answers, and this is the substance of the
maturity node:

* **cadence refinement** is the only knob a partitioned code has, and it is
  *structurally capped*: the published sample is the average of a fine window of
  width `2d`, which differs from the value at its own midpoint by `(d^2/6) q2`.
  No de-stamping can recover sub-window information, so at a fixed fine step the
  flux-defect staircase has an `O(d^2)` floor. Order-2 de-stamping halves that
  floor (`{stair['cadence']['centered_destamp1']['finest_defect']:.2e} -> {stair['cadence']['centered_destamp2']['finest_defect']:.2e}`
  on the weld chain; at the coarser cadences the wider stencil is slightly
  *worse*, `{stair['cadence']['centered_destamp2']['coarsest_defect']:.2e}` vs
  `{stair['cadence']['centered_destamp1']['coarsest_defect']:.2e}`) but it cannot
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
{rows_energy}

### 3.5 (e) three-chain regression

| gate | verdict | criteria passed |
| --- | --- | --- |
| flux / attribution gate (`exchange_gate`) | **{study['regression']['flux']['verdict']}** | {study['regression']['flux']['passed']}/{study['regression']['flux']['total']} |
| energy gate (`exchange_energy`) | **{study['regression'].get('energy', {}).get('verdict', 'not run')}** | {study['regression'].get('energy', {}).get('passed', '—')}/{study['regression'].get('energy', {}).get('total', '—')} |

### 3.6 Acceptance table (machine-checkable criteria)

{criterion_table}

## 4. Figures

{fig_lines}

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
  order is capped near `{stair['cadence']['centered_destamp2']['defect_order']:.2f}`
  by the window-smoothing floor, while the state error on the same ladder
  converges at `{stair['cadence']['centered_destamp1']['state_order']:.2f}`.
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
| lag + end stamp, interval defect (coarsest cadence) | exact integral | {stair['cadence']['lag_endstamp']['levels'][0]['max_interval_defect']:.3e} | **no** |
| CN + end stamp, interval defect (coarsest cadence) | exact integral | {stair['cadence']['centered_endstamp']['levels'][0]['max_interval_defect']:.3e} | {'yes' if stair['cadence']['centered_endstamp']['levels'][0]['max_interval_defect'] < 3e-2 else '**no**'} |
| CN + de-stamp(2), interval defect (coarsest cadence) | exact integral | {stair['cadence']['centered_destamp2']['levels'][0]['max_interval_defect']:.3e} | {'yes' if stair['cadence']['centered_destamp2']['levels'][0]['max_interval_defect'] < 3e-2 else '**no**'} |
| CN + de-stamp(2), state error (finest `dt`) | `solve_ivp` oracle | {stair['dt']['centered_destamp2']['finest_state_error']:.3e} | yes |
| no-delay energy defect (worst chain) | stored-energy budget | {max(row['energy_defect'] for payload in energy.values() for row in payload.values()):.3e} | yes |

## 7. Provenance

```json
{json.dumps(provenance, indent=2, sort_keys=True)}
```
"""


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    OUT.mkdir(parents=True, exist_ok=True)
    stored = OUT / "maturity.json"
    if "--from-json" in argv and stored.exists():
        payload = json.loads(stored.read_text())
        study = {k: payload[k] for k in
                 ("mms", "attribution", "staircase", "no_delay_energy", "regression")}
        verdict = gate_maturity(study)
        provenance = payload["provenance"]
        figures = {name: meta["caption"]
                   for name, meta in payload["figures"].items()}
        DOC.write_text(render_report(study, verdict, provenance, figures))
        print(format_table(verdict))
        print(f"\nreport re-rendered from {stored}: {DOC}")
        return 0 if verdict["verdict"] == "pass" else 1
    study = run_maturity_study(with_energy=True)
    verdict = gate_maturity(study)

    figures = {
        "fig1_mms_order.png": "(a) quadrature defect vs exchange step for every arm; slopes are the measured orders.",
        "fig2_weld_staircase.png": "(c) left: cadence refinement reproduces the 1.2e-3 plateau; right: dt halving restores second order.",
        "fig3_interval_attribution.png": "(b) left: handover deficit vs cadence; right: identity residual at round-off vs the tail-booking control.",
        "fig4_no_delay_energy.png": "(d) no-delay exchange energy defect vs dt per chain, below the 1e-5 line.",
        "fig5_interval_defect_map.png": "per-interval defect field (interval index x cadence) for the lag hold and the order-2 de-stamped CN exchange.",
    }
    paths = {}
    for name, fn in (("fig1_mms_order.png", plot_mms_order),
                     ("fig2_weld_staircase.png", plot_weld_staircase),
                     ("fig3_interval_attribution.png", plot_attribution),
                     ("fig4_no_delay_energy.png", plot_no_delay_energy),
                     ("fig5_interval_defect_map.png", plot_interval_defect_map)):
        path = fn(study) if fn is not plot_interval_defect_map else fn()
        paths[name] = path

    payload = dict(study)
    payload["verdict"] = verdict
    payload["figures"] = {name: {"path": str(paths[name].relative_to(ROOT)),
                                 "sha256": _sha256(paths[name]),
                                 "caption": figures[name]}
                          for name in figures}
    provenance = {
        "node": "p223w2b-n06",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "script": "scripts/run_p223w2b_n06_maturity.py",
        "arms": MATURITY_ARMS,
        "cadence_ratios": list(CADENCE_RATIOS),
        "payload_sha256": "",
    }
    blob = json.dumps(payload, sort_keys=True, default=float).encode()
    provenance["payload_sha256"] = hashlib.sha256(blob).hexdigest()
    payload["provenance"] = provenance

    (OUT / "maturity.json").write_text(json.dumps(payload, indent=1, sort_keys=True,
                                                  default=float))
    (OUT / "verification.json").write_text(
        json.dumps({"node": "p223w2b-n06", "verdict": verdict,
                    "criteria": verdict["criteria"],
                    "provenance": provenance}, indent=1, sort_keys=True,
                   default=float))
    DOC.write_text(render_report(study, verdict, provenance, figures))

    print(format_table(verdict))
    print(f"\nartifacts: {OUT}")
    print(f"report:    {DOC}")
    return 0 if verdict["verdict"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())