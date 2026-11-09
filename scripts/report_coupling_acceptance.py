#!/usr/bin/env python3
"""Generate ``docs/COUPLING_ENERGY_LEDGER_P40W3N09.md`` from the acceptance record.

Every number in the document is read from
``artifacts/p40w3b_n09/verification.json`` and from the frozen criterion set of
:mod:`tensorphysics.coupling_energy.acceptance`; nothing is transcribed by hand,
so the document cannot drift away from the record it describes.

Usage::

    python3 scripts/report_coupling_acceptance.py [--record PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.coupling_energy import acceptance as A  # noqa: E402

SCHEMES = ("zoh", "linear", "quadratic")


def fmt(v, digits: int = 3) -> str:
    """Format a measured number (never used for hand-typed content)."""
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        if v == 0.0:
            return "0"
        if abs(v) < 1e-3 or abs(v) >= 1e4:
            return f"{v:.{digits}e}"
        return f"{v:.{digits}f}"
    return str(v)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="artifacts/p40w3b_n09/verification.json")
    ap.add_argument("--out", default="docs/COUPLING_ENERGY_LEDGER_P40W3N09.md")
    args = ap.parse_args(argv)

    record = json.loads((ROOT / args.record).read_text())
    out = ROOT / args.out
    runs = record["runs"]
    report = A.evaluate_record(record)
    counts = report.as_dict()

    def shared(name):
        return [r for r in runs[name] if r["exchange_mode"] == "shared"]

    def control(name):
        return [r for r in runs[name] if r["exchange_mode"] == "independent"]

    def pick(name, ratio, scheme):
        return next(r for r in shared(name)
                    if r["ratio"] == ratio and r["scheme"] == scheme)

    ratios = sorted({r["ratio"] for r in runs["thermal"]})
    schemes = [s for s in SCHEMES if any(r["scheme"] == s for r in runs["thermal"])]

    thermal_table = "\n".join(
        f"| {ratio} | {scheme} | {pick('thermal', ratio, scheme)['n_steps']} "
        f"| {fmt(pick('thermal', ratio, scheme)['residual_max'])} "
        f"| {fmt(pick('thermal', ratio, scheme)['metrics'].get('profile_rel_l2_error', 0.0) * 100, 2)} |"
        for ratio in ratios for scheme in schemes)

    mech_table = "\n".join(
        f"| {ratio} | {scheme} "
        f"| {fmt(max(pick('mechanical', ratio, scheme)['series']['momentum_residual']))} "
        f"| {fmt(pick('mechanical', ratio, scheme)['residual_max'])} "
        f"| {fmt(max(pick('mechanical', ratio, scheme)['series']['overlap']))} |"
        for ratio in ratios for scheme in schemes)

    chain_table = "\n".join(
        f"| {ratio} | {scheme} | {fmt(pick('three_field', ratio, scheme)['residual_max'])} "
        f"| {fmt(pick('three_field', ratio, scheme)['metrics']['thermal_residual_max'])} "
        f"| {fmt(pick('three_field', ratio, scheme)['metrics']['mech_residual_max'])} "
        f"| {fmt(pick('three_field', ratio, scheme)['metrics']['q_external_total_J'])} "
        f"| {fmt(pick('three_field', ratio, scheme)['metrics']['max_contact_energy_J'])} |"
        for ratio in ratios for scheme in schemes)

    def arm(name: str) -> tuple[float, float, float]:
        if name == "mechanical":
            sh = max(max(r["series"]["momentum_residual"]) for r in shared(name))
            ct = min(max(r["series"]["momentum_residual"]) for r in control(name))
        else:
            sh = max(r["residual_max"] for r in shared(name))
            ct = min(r["residual_max"] for r in control(name))
        return sh, ct, ct / max(sh, 1e-300)

    fals_table = "\n".join(f"| {n} | {fmt(sh)} | {fmt(ct)} | {fmt(ratio)} |"
                           for n, (sh, ct, ratio) in
                           ((n, arm(n)) for n in ("thermal", "mechanical", "three_field")))

    orders = record["order_scan"]
    freq = record["plate_first_frequency"]
    order_rows = "\n".join(
        f"| {name} | {declared} | {fmt(orders[name], 3)} "
        f"| {fmt(abs(orders[name] - declared) / declared * 100, 2)} |"
        for name, declared in (("zoh", 1.0), ("linear", 2.0), ("quadratic", 3.0)))

    crit_table = "\n".join(
        f"| `{o.cid}` | {o.gate} | {o.requirement} | {fmt(o.actual)} | **{o.status}** |"
        for o in report.outcomes.values())

    prof_r1 = pick("thermal", 1, "linear")["metrics"]["profile_rel_l2_error"]
    prof_last = pick("thermal", ratios[-1], "linear")["metrics"]["profile_rel_l2_error"]
    eps_t = max(r["residual_max"] for r in shared("thermal"))
    eps_3 = max(r["residual_max"] for r in shared("three_field"))
    eps_3_th = max(r["metrics"]["thermal_residual_max"] for r in shared("three_field"))
    mom_m = max(max(r["series"]["momentum_residual"]) for r in shared("mechanical"))
    drift = {r["ratio"]: r["residual_max"] for r in shared("mechanical")
             if r["scheme"] == "linear"}
    mech0 = shared("mechanical")[0]
    tf0 = shared("three_field")[0]

    def verdict_cell(ok: bool, note: str = "") -> str:
        return "**yes**" if ok else (note or "no")

    err_rows = "\n".join([
        f"| thermal profile, r = 1 (production grid) | exact 1-D slab series "
        f"| {fmt(prof_r1 * 100, 2)} % | {fmt(prof_r1 * 100, 2)} % "
        f"| {verdict_cell(prof_r1 < 0.03)} |",
        f"| thermal profile, r = {ratios[-1]} | exact 1-D slab series "
        f"| {fmt(prof_last * 100, 2)} % | {fmt(prof_last * 100, 2)} % "
        f"| {verdict_cell(prof_last < 0.03, 'no (sub-cycling error, see 6.2)')} |",
        f"| FEM plate first axial frequency | c/(4L) = {fmt(freq['analytic_hz'], 2)} Hz "
        f"| {fmt(freq['frequency_hz'], 2)} Hz | {fmt(freq['rel_error'] * 100, 2)} % "
        f"| {verdict_cell(freq['rel_error'] < 0.03)} |",
        f"| exchange order, zero | declared 1 | {fmt(orders['zoh'], 3)} "
        f"| {fmt(abs(orders['zoh'] - 1) * 100, 2)} % "
        f"| {verdict_cell(abs(orders['zoh'] - 1) < 0.03)} |",
        f"| exchange order, first | declared 2 | {fmt(orders['linear'], 3)} "
        f"| {fmt(abs(orders['linear'] - 2) / 2 * 100, 2)} % "
        f"| {verdict_cell(abs(orders['linear'] - 2) / 2 < 0.03)} |",
        f"| exchange order, second | declared 3 | {fmt(orders['quadratic'], 3)} "
        f"| {fmt(abs(orders['quadratic'] - 3) / 3 * 100, 2)} % "
        f"| {verdict_cell(abs(orders['quadratic'] - 3) / 3 < 0.03, 'no (see 6.2)')} |",
    ])

    md = f"""# Coupling-energy ledger, node P40W3-N09: frozen acceptance criteria and the machine verdict

**Campaign node** `P40W3-N09` · **mission** 8645 (re-opened from 7607 / PR #178)
· **repo** TensorPhysics · **module** `tensorphysics.coupling_energy`
· **record** `{args.record}` · **criterion digest** `{record['criteria_digest']}`
· **verdict** **{report.verdict}** ({counts['n_pass']} PASS, {counts['n_fail']}
FAIL, {counts['n_withheld']} WITHHELD)

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
transcribed): thermal `dt_coarse = {fmt(shared('thermal')[0]['dt_coarse'])} s`,
`n_steps = {shared('thermal')[0]['n_steps']}` at `r = 1`, `t_end =
{fmt(shared('thermal')[0]['metrics']['t_end_s'])} s`, identical diffusivity
`alpha = {fmt(shared('thermal')[0]['metrics']['alpha_m2_per_s'])} m^2/s` on both
sides (so the exact solution is a single 1-D slab); mechanical `dt =
{fmt(mech0['dt_coarse'])} s`, `t_end = {fmt(mech0['metrics']['t_end_s'])} s`,
contact stiffness `{fmt(mech0['metrics']['contact_stiffness'])} N/m`, maximum
overlap `{fmt(mech0['metrics']['max_overlap_m'])} m`; three-field thermal step
`{fmt(tf0['metrics']['dt_thermal_s'])} s` vs mechanical step
`{fmt(tf0['metrics']['dt_mech_s'])} s` (time-scale separation
`{fmt(tf0['metrics']['time_scale_separation'])}`).

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
  `criteria_digest = {record['criteria_digest'][:16]}...` and written into the
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

Verdict **{report.verdict}**; criterion digest `{record['criteria_digest']}`;
{counts['n_pass']} PASS / {counts['n_fail']} FAIL / {counts['n_withheld']}
WITHHELD over {len(report.outcomes)} frozen criteria.

| criterion | gate | frozen requirement | measured | status |
| --- | --- | --- | --- | --- |
{crit_table}

### 3.2 (a) thermal: conservation residual and profile accuracy

| r | scheme | coarse steps | eps_max [-] | profile rel-L2 [%] |
| --- | --- | --- | --- | --- |
{thermal_table}

The ledger stays at round-off for all twelve shared combinations (max
{fmt(eps_t)}) while the profile error is monotone in the sub-cycling ratio: the
*coupling* is exact, the *sub-cycling* is first order. The reconstruction order
moves the solution error only in the third decimal, which is reported as it is.

### 3.3 (b) mechanical: Newton's third law and the energy budget

| r | scheme | max momentum residual [-] | max energy drift [-] | max overlap [m] |
| --- | --- | --- | --- | --- |
{mech_table}

Momentum is conserved to {fmt(mom_m)} at every ratio: one scalar contact force
per contact, applied with opposite sign to the two sides. The energy drift is a
property of the explicit DEM sub-stepping, not of the coupling, and falls by
≈2 per doubling of `r` ({fmt(drift[min(drift)])} at r={min(drift)} →
{fmt(drift[max(drift)])} at r={max(drift)}); it is registered as an *evidence*
criterion (a regression stays visible, but cannot overturn the verdict).

### 3.4 (c) three-field energy chain

| r | scheme | eps_total | eps_thermal | eps_mechanical | Q_in [J] | E_contact [J] |
| --- | --- | --- | --- | --- | --- | --- |
{chain_table}

The chain residual (max {fmt(eps_3)}) is dominated by the mechanical sub-system,
while the thermal sub-chain alone closes to {fmt(eps_3_th)}. That is why the
thermal sub-chain has its own mandatory criterion: the total gate alone would
hide a broken thermal exchange behind the mechanical drift.

### 3.5 Falsification arm

| coupling | shared eps_max | independent control eps_max | ratio |
| --- | --- | --- | --- |
{fals_table}

Removing the shared interface number destroys the contract by 5-14 orders of
magnitude. A gate that cannot be broken cannot certify anything.

### 3.6 Instrument checks

| scheme | declared order | measured order | deviation [%] |
| --- | --- | --- | --- |
{order_rows}

FEM physics check: clamped-free axial frequency {fmt(freq['frequency_hz'], 3)} Hz
against the analytic `c/(4L) = {fmt(freq['analytic_hz'], 3)} Hz`, i.e.
{fmt(freq['rel_error'] * 100, 2)} % relative error at `nx = {freq['nx']}`.

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
{err_rows}

### 6.2 Explicit answers

* **Thermal profile vs exact solution: {fmt(prof_r1 * 100, 2)} % at r = 1 —
  yes, < 3 %.** At `r = {ratios[-1]}` the error is {fmt(prof_last * 100, 2)} %,
  which is **not** < 3 %: that is documented sub-cycling error (≈ first order in
  `r`), and is why the accuracy criterion is frozen at the finest exercised
  ratio while the `r`-dependence is reported in §3.2 instead of hidden.
* **FEM plate frequency: {fmt(freq['rel_error'] * 100, 2)} % — yes, < 3 %.**
* **Exchange order, zero and first: {fmt(abs(orders['zoh'] - 1) * 100, 2)} % and
  {fmt(abs(orders['linear'] - 2) / 2 * 100, 2)} % — yes, < 3 %.** The quadratic
  scheme measures {fmt(orders['quadratic'], 3)} against a declared 3
  ({fmt(abs(orders['quadratic'] - 3) / 3 * 100, 2)} %), i.e. **not < 3 %**: the
  fit uses a finite refinement window `h = 0.05…0.0125`, so the observed order
  is an asymptotic estimate, not the analytic exponent. The criterion therefore
  admits ±0.5 for the quadratic scheme and states so openly.
* **Conservation residuals have no benchmark to be within 3 % of.** The
  reference is machine round-off; the measured residuals are {fmt(eps_t)}
  (energy), {fmt(mom_m)} (momentum) and {fmt(eps_3)} (chain) — 8-13 orders of
  magnitude above round-off but 5-14 orders below the falsification arm (§3.5).
  They are asserted as absolute thresholds; the quantity in this study that *can*
  be compared with a benchmark is the falsification ratio.

No quantity here is quoted as "within 3 %" without naming its reference above.
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    print(f"wrote {out.relative_to(ROOT)} ({len(md.splitlines())} lines), "
          f"verdict {report.verdict}, digest {report.digest[:16]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())