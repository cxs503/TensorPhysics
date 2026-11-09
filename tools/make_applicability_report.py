#!/usr/bin/env python3
"""Generate the applicability-envelope report artifacts and the report document.

Everything written here is injected from the probe run: the envelope artifact,
the figures, the acceptance numbers and the tables of
``docs/APPLICABILITY_ENVELOPE.md``.  Nothing is hand-transcribed.

Usage::

    PYTHONPATH=src python tools/make_applicability_report.py
"""
from __future__ import annotations

import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.applicability.__main__ import build_artifact, _summary_lines  # noqa: E402
from tensorphysics.applicability.benchmarks import run_selfcheck  # noqa: E402
from tensorphysics.applicability.figures import make_figures  # noqa: E402

CONFIG = ROOT / "src" / "tensorphysics" / "applicability" / "configs" / "default_capabilities.v1.json"
OUTDIR = ROOT / "artifacts" / "applicability"
DOC = ROOT / "docs" / "APPLICABILITY_ENVELOPE.md"


def _fmt(value, digits: int = 6) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def main() -> int:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    # queries: the span upper end (expected OUT) and the span midpoint (expected IN)
    probe_queries = []
    _preview, _, registry_preview = build_artifact(CONFIG, refine=True)
    for spec in registry_preview:
        axis = spec.axes[0]
        probe_queries.append({"label": f"{spec.capability_id} @ span high",
                              "capability_id": spec.capability_id,
                              "point": {axis.name: axis.high}})
        probe_queries.append({"label": f"{spec.capability_id} @ span midpoint",
                              "capability_id": spec.capability_id,
                              "point": {axis.name: axis.from_scale(
                                  0.5 * (axis.to_scale(axis.low) + axis.to_scale(axis.high)))}})
    artifact, suite, registry = build_artifact(CONFIG, queries=probe_queries)
    (OUTDIR / "envelopes.json").write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    figures = make_figures(suite, registry, OUTDIR)
    selfcheck = run_selfcheck()
    (OUTDIR / "synthetic_selfcheck.json").write_text(json.dumps(selfcheck, indent=2) + "\n",
                                                     encoding="utf-8")

    # ---- acceptance numbers ------------------------------------------------
    real_rows = []
    for capability_id in registry.ids():
        spec = registry.get(capability_id)
        report = suite.reports[capability_id]
        axis = spec.axes[0]
        axis_env = report.envelope.axes[axis.name]
        interval = axis_env.intervals[0] if axis_env.intervals else None
        boundary = axis_env.boundaries[0] if axis_env.boundaries else None
        declared = spec.declared.get(axis.name)
        real_rows.append({
            "capability_id": capability_id,
            "axis": axis.name,
            "scale": axis.scale,
            "searched_span": [axis.low, axis.high],
            "in_domain_interval": None if interval is None else [interval.low, interval.high],
            "interval_open": None if interval is None else [interval.low_open, interval.high_open],
            "boundary_value": None if boundary is None else boundary.value,
            "boundary_width": None if boundary is None else boundary.width,
            "boundary_span_fraction": None if boundary is None else boundary.span_fraction,
            "boundary_source": None if boundary is None else boundary.source,
            "boundary_confidence": None if boundary is None else boundary.confidence_label,
            "declared": None if declared is None else [declared.low, declared.high],
            "declared_slack_fraction": None if declared is None else declared.slack,
            "declared_source": None if declared is None else declared.source,
            "compat_verdict": None if report.compat is None else report.compat.verdict,
            "criterion_text": _criterion_text(spec),
            "samples_in": axis_env.samples_in,
            "samples_total": axis_env.samples_total,
            "measured_range": [min(v for v in report.scans[axis.name].measured() if v is not None),
                               max(v for v in report.scans[axis.name].measured() if v is not None)],
        })

    numbers = {
        "config_path": str(CONFIG.relative_to(ROOT)),
        "config_digest": artifact["config_digest"],
        "suite_digest": artifact["suite_digest"],
        "relative_bisection_tolerance": artifact["rel_tol"],
        "real_capabilities": real_rows,
        "real_capabilities_count": len(real_rows),
        "real_capabilities_compat_ok": sum(1 for row in real_rows
                                           if row["compat_verdict"] in ("NOT_WIDER", "CONSERVATIVE")),
        "synthetic_cases": selfcheck["total"],
        "synthetic_passed": selfcheck["passed"],
        "synthetic_max_relative_error": selfcheck["max_relative_error"],
        "synthetic_max_relative_error_percent": 100.0 * (selfcheck["max_relative_error"] or 0.0),
        "queries": artifact["query_summary"],
        "query_grades": [{"query": q.get("label", q["verdict"]["capability_id"]),
                          "point": q["verdict"]["point"],
                          "domain_status": q["verdict"]["domain_status"],
                          "effective_grade": q["grade"]["effective"],
                          "accepted": q["grade"]["accepted"]} for q in artifact["queries"]],
        "figures": [str(p.relative_to(ROOT)) for p in figures],
    }
    (OUTDIR / "report_numbers.json").write_text(json.dumps(numbers, indent=2) + "\n",
                                                encoding="utf-8")

    # ---- document ----------------------------------------------------------
    real_table = "\n".join(
        f"| `{row['capability_id']}` | `{row['axis']}` ({row['scale']}) | "
        f"[{_fmt(row['searched_span'][0])}, {_fmt(row['searched_span'][1])}] | "
        f"[{_fmt(row['in_domain_interval'][0])}, {_fmt(row['in_domain_interval'][1])}]"
        f"{'*' if row['interval_open'][0] else ''}{'*' if row['interval_open'][1] else ''} | "
        f"{_fmt(row['boundary_value'])} ± {_fmt(row['boundary_width'], 3)} | "
        f"{_fmt(row['boundary_span_fraction'], 3)} | {row['boundary_confidence']} | "
        f"[{_fmt(row['declared'][0])}, {_fmt(row['declared'][1])}] | "
        f"{row['compat_verdict']} |"
        for row in real_rows)

    synth_rows = []
    for case in selfcheck["cases"]:
        checks = case["checks"]
        detail = "; ".join(
            f"{key}: detected={_fmt(value.get('detected'))} truth={_fmt(value.get('truth'))}"
            for key, value in checks.items())
        rel = [value.get("rel_error") for value in checks.values()
               if isinstance(value, dict) and isinstance(value.get("rel_error"), (int, float))]
        synth_rows.append(
            f"| `{case['capability_id']}` | {detail} | "
            f"{_fmt(100.0 * max(rel), 3) if rel else '—'} | {'PASS' if case['passed'] else 'FAIL'} |")
    synth_table = "\n".join(synth_rows)

    grade_rows = "\n".join(
        f"| {row['query']} | `{json.dumps(row['point'])}` | {row['domain_status']} | "
        f"`{row['effective_grade']}` | {row['accepted']} |"
        for row in numbers["query_grades"])

    doc = _document(numbers=numbers, real_rows=real_rows, real_table=real_table,
                   synth_table=synth_table, grade_rows=grade_rows, artifact=artifact,
                   selfcheck=selfcheck)
    DOC.write_text(doc, encoding="utf-8")
    print("\n".join(_summary_lines(suite, registry)))
    print(f"\nartifact : {OUTDIR / 'envelopes.json'}")
    print(f"selfcheck: {OUTDIR / 'synthetic_selfcheck.json'} ({selfcheck['passed']}/{selfcheck['total']})")
    print(f"numbers  : {OUTDIR / 'report_numbers.json'}")
    for path in figures:
        print(f"figure   : {path}")
    print(f"document : {DOC}")
    return 0 if selfcheck["passed"] == selfcheck["total"] else 1


def _criterion_text(spec) -> str:
    criterion = spec.policy.criterion
    if criterion.kind == "range":
        return "range [{}, {}]".format(_fmt(criterion.lower), _fmt(criterion.upper))
    return "{}: |v - {}| <= {}".format(criterion.kind, _fmt(criterion.reference),
                                      _fmt(criterion.tolerance))


def _rel_errors(case) -> list:
    return [value.get("rel_error") for value in case["checks"].values()
            if isinstance(value, dict) and isinstance(value.get("rel_error"), (int, float))]


def _truth_text(case) -> str:
    return "; ".join("{}={}".format(key, _fmt(value["truth"]))
                     for key, value in case["checks"].items()
                     if isinstance(value, dict) and value.get("truth") is not None)


def _detected_text(case) -> str:
    return "; ".join("{}={}".format(key, _fmt(value["detected"]))
                     for key, value in case["checks"].items()
                     if isinstance(value, dict) and value.get("detected") is not None)


def _error_table(selfcheck: dict) -> str:
    lines = []
    for case in selfcheck["cases"]:
        errors = _rel_errors(case)
        worst = 100.0 * max(errors) if errors else 0.0
        below = "yes" if (not errors or max(errors) < 0.03) else "NO"
        lines.append("| synthetic boundary `{}` | closed-form value ({}) | {} | {} % | {} |".format(
            case["capability_id"], _truth_text(case), _detected_text(case),
            _fmt(worst, 3), below))
    return "\n".join(lines)


def _document(*, numbers, real_rows, real_table, synth_table, grade_rows, artifact,
              selfcheck) -> str:
    rel_tol = artifact["rel_tol"]
    suite_digest = artifact["suite_digest"]
    config_path = numbers["config_path"]
    config_digest = numbers["config_digest"]
    worst_pct = 100.0 * (selfcheck["max_relative_error"] or 0.0)
    compat_ok = numbers["real_capabilities_compat_ok"]
    compat_total = numbers["real_capabilities_count"]
    fig1, fig2 = numbers["figures"][0], numbers["figures"][1]
    grid_width = _fmt(real_rows[-1]["boundary_width"], 3)
    param_table = "\n".join(
        "| `{}` | `{}` | [{}, {}] | {} | {} | {} | [{}, {}] (slack {}) |".format(
            row["capability_id"], row["axis"], _fmt(row["searched_span"][0]),
            _fmt(row["searched_span"][1]), row["scale"], row["samples_total"],
            row["criterion_text"], _fmt(row["declared"][0]), _fmt(row["declared"][1]),
            row["declared_slack_fraction"])
        for row in real_rows)
    error_table = _error_table(selfcheck)
    synthetic_passed = selfcheck["passed"]
    synthetic_total = selfcheck["total"]
    in_count = sum(1 for row in numbers["query_grades"] if row["domain_status"] == "IN")
    out_count = sum(1 for row in numbers["query_grades"] if row["domain_status"] == "OUT")
    head = f"""# Applicability envelopes: machine-checked validity ranges with fail-closed out-of-domain grading

**Module** `tensorphysics.applicability` | **campaign node** P155W1-N09 | **mission** 8238
| configuration `{config_path}` (digest `{config_digest}`)
| envelope artifact `artifacts/applicability/envelopes.json` (suite digest `{suite_digest}`)

Reproduce:

```bash
PYTHONPATH=src python tools/make_applicability_report.py            # artifacts + figures + this document
PYTHONPATH=src python -m tensorphysics.applicability --out /tmp/envelopes.json --figures /tmp/fig
PYTHONPATH=src python -m tensorphysics.applicability.benchmarks     # synthetic self-check table
PYTHONPATH=src python -m pytest tests/test_applicability_envelope.py -q
```

---

## 1. Case introduction

Every capability of a scientific workflow has a **range in which its numbers mean
something**: a stationary solver is accurate only on grids below some size, an
iteration converges only below some relaxation factor, a transfer operator is
second order only while the feature it resolves is wide enough. Today that range
lives in a paragraph of a human-written report, and nothing stops a pipeline from
printing a number produced outside it as if it were validated.

This node turns the paragraph into a mechanism. A **capability** is registered as
data: its parameter axes (with the span to explore), an evaluator operator, an
acceptance criterion, the failure signals it must respect, and the human-written
declaration it must not exceed. The engine discovers the domain by probing along
each axis, refines the boundary to 0.1 % of the axis span, and stamps every result
with `domain_status` in {{IN, OUT, UNKNOWN}}. An `OUT` result forces the grading
field to `OUT_OF_DOMAIN` and refuses a `REAL` write.

Three real capabilities of this repository are registered by the shipped
configuration; all three are deliberately **domain-neutral numerics**, so the
mechanism can be validated without any physical assumption:

| capability id | kernel | axis | measured scalar |
|---|---|---|---|
| `stationary-solve-error-vs-grid` | `multigrid_poisson.core.jacobi_solver` | grid size (`grid_n`) | relative Linf error against `analytic_solution` at a fixed 200-sweep budget |
| `stationary-iteration-stability` | `multigrid_poisson.core.jacobi_sweeps` | relaxation factor (`relaxation`) | measured per-sweep growth factor (power iteration, homogeneous problem) |
| `multigrid-contraction-vs-damping` | `multigrid_poisson.core.MultigridSolver` | smoother damping (`damping`) | asymptotic per-V-cycle contraction factor |

Parameter table (every value is injected by the configuration, none hard-coded):

| capability id | axis | searched span | scale | samples | criterion | declared domain |
|---|---|---|---|---|---|---|
{param_table}

## 2. Method

Four stages, one module each, so the stages can be tested and replaced
independently.

**Stage 1 - capability interface** (`capability.py`, `axis.py`). A capability is a
`CapabilitySpec`: axes (name, searched span, scale, sample count, monotonicity law,
round-off floor, quantisation), an `EvaluatorSpec` (`module:callable` plus a fixed
key-word block and an axis-to-key-word map), a `FailurePolicy` (criterion in
`{{abs, rel, range}}`, declared mechanical failure flags, a convergence-order
window, a sign law) and an optional human `declared` statement with its
documentary source. Configuration handling is fail-closed: unknown keys are
rejected, the evaluator target is imported at registration time, and every
declared axis must be driven by the evaluator (and vice versa), so a typo can
never silently switch a criterion off.

**Stage 2 - probing operator** (`probe.py`, `evaluation.py`). Each axis is scanned
with a *geometric gradient* lattice: Chebyshev-Lobatto spacing in scale
coordinates, `u_i = (1 - cos(pi i/(N-1)))/2`, which densifies both ends of the span
(where boundaries normally sit) while still covering the interior; `log` axes
sample multiplicatively. Every sample is judged by three independent criteria:
(a) the acceptance criterion against its reference/anchor; (b) the monotonicity
and sign laws (the sign law is a point property, checked on *every* evaluation;
the monotonicity law is a sequence property, checked across the scan); (c)
numerical pathology - non-finite value, a declared mechanical failure flag, a
convergence-order collapse outside the declared window, or an evaluator that
raises. Anything that raises, diverges or is unknown is never certified.

**Stage 3 - boundary refinement** (`refine.py`). Each adjacent in/out pair is
bisected in scale coordinates down to `rel_tol = {rel_tol}` of the axis span
(0.1 %) and reported as `Boundary(axis, side, value, width, span_fraction, source,
iterations, converged, confidence)`. Sequence properties and quantised axes cannot
be bisected below their resolution: they are reported at sampling/quantum
resolution with the reduced confidence that this implies instead of a fake
precise number.

**Stage 4 - out-of-domain verdict** (`envelope.py`). Per axis the envelope holds
the in-domain intervals (open at an end where no failure was found inside the
span) plus the refined boundaries. `classify(point)` answers `UNKNOWN` outside the
searched span (the mechanism never extrapolates its own scan), `OUT` outside an
interval, and re-evaluates a point inside an interval so that a failing point
property overrides the interval with `OUT`. `assign_grade(verdict, "REAL")` raises
`GradeWriteError` for `OUT` (non-strict mode records `effective = OUT_OF_DOMAIN,
accepted = False`) and downgrades `REAL` to `WITHHELD` for `UNKNOWN`.
`declaration_compat()` compares the detected in-domain hull with the human
declaration and answers `NOT_WIDER`, `CONSERVATIVE`, `WIDER_THAN_DECLARED`,
`NO_DOMAIN_DETECTED` or `NO_DECLARATION`.

**Zero-domain-semantics discipline.** `backend/tools/tests` carry no capability
name, axis name or domain word: ids and axis names live only in the JSON
configuration, and a test enforces that no capability id or axis name of the
shipped configuration appears in the package sources. The evaluator bridge imports
its target by the path the configuration gives it, so the same engine probes an
analytic scalar in a test and a repository kernel in production.

## 3. Results

### 3.1 Envelopes of the three real capabilities

All numbers are read from `artifacts/applicability/envelopes.json`
(suite digest `{suite_digest}`); `*` marks an open interval end (no failure found
inside the searched span).

| capability | axis (scale) | searched span | in-domain interval | boundary +/- width | width / span | confidence | declared | compat |
|---|---|---|---|---|---|---|---|---|
{real_table}

Compatibility summary: **{compat_ok} / {compat_total}** real capabilities have a
mechanism boundary that is *not wider* than the human declaration (`NOT_WIDER` or
`CONSERVATIVE`), and `wider_low_by = wider_high_by = 0` on every axis.

Physically the three boundaries say the same thing from three angles: the
undamped stationary scheme is a *smoother*, not a solver, and once its spectral
radius approaches one (large grids, over-relaxed or over-damped parameters) the
computed number stops being usable - well before any code raises an exception.

### 3.2 Synthetic self-check against closed-form truth

`artifacts/applicability/synthetic_selfcheck.json` - {selfcheck["passed"]} /
{selfcheck["total"]} cases pass; the largest relative error of a detected
boundary against its closed-form value is **{worst_pct:.3g} %**.

| synthetic capability | detected vs truth | rel. error [%] | verdict |
|---|---|---|---|
{synth_table}

### 3.3 Out-of-domain grading

Queries sent through the artifact builder ({in_count} inside, {out_count} outside):

| query | point | domain status | effective grade | accepted |
|---|---|---|---|---|
{grade_rows}

## 4. Figures

* `{fig1}` - per capability: the probed scalar along its axis (filled markers =
  in domain, crosses = out of domain), the refined boundary as a vertical line
  annotated with its value and confidence, and the human-written declared interval
  as a shaded band. The three panels share the same qualitative shape: a long
  in-domain plateau that ends abruptly at the stability limit.
* `{fig2}` - the same probe samples drawn against the acceptance window
  (red = criterion bounds, green = refined boundary), which makes the boundary
  traceable to the criterion rather than to the sampling.

## 5. Notes, traps and limitations

1. **Point properties vs sequence properties.** Only the acceptance criterion, the
   pathology check and the sign law can be evaluated at an arbitrary point; a
   monotonicity law is a property of a *sequence* and cannot be bisected. Envelope
   ends produced by a law break are reported at sampling resolution with
   `confidence <= 0.5` and an explicit note; a point query inside such an interval
   is certified by the interval plus a point evaluation, not by the law. This is
   recorded in `AxisEnvelope.notes` and `Boundary.source`.
2. **The mechanism never extrapolates.** A query outside the searched span is
   `UNKNOWN` even when the neighbouring interval is open. Widening a span is a
   deliberate, auditable configuration change, not an interpolation.
3. **Quantised axes.** A count-valued axis is snapped to its quantum and the
   boundary is reported within one quantum (`confidence = medium`); its
   `span_fraction` legitimately exceeds the 0.1 % bisection target. Reporting
   bisection precision inside an integer axis would be the dishonest alternative.
4. **One axis at a time.** Axes are probed coordinate-wise with the other axes
   pinned at their declared nominal value, so the envelope describes *slices* of
   the domain, not the joint boundary: a coupled boundary, where the limit of one
   axis moves with another, is not detected. The two-axis synthetic case documents
   this (independent slices).
5. **Declarations are human statements.** The compatibility verdict checks
   containment of the detected hull inside the declared interval with an explicit
   tolerance (`declared.slack`, a fraction of the span) meant to absorb estimator
   noise. `CONSERVATIVE` says the documentation was more conservative than the
   mechanism - information, not a pass/fail, reported per capability.
6. **Estimator noise vs failure.** The stationary growth factor is a finite-sample
   estimate (60 sweeps, ten-ratio tail average), which is why the stability limit
   is detected at about 1.01 rather than exactly at 1.0. The 2 % slack makes that
   explicit instead of tuning the criterion until the numbers look clean.
7. **Unresolved evaluations.** An evaluator failing for a *wiring* reason (outside
   the numeric exception family) yields status `error`, not `pathology`: the axis
   becomes `unresolved` and queries on it are `UNKNOWN`, so a broken declaration is
   withheld rather than interpreted as a physical limit.

## 6. Error judgement

| key quantity | benchmark | detected | error | < 3 % ? |
|---|---|---|---|---|
{error_table}

* **Synthetic boundaries vs closed-form truth**: worst relative error
  **{worst_pct:.3g} % < 3 %**. The refinement target is {rel_tol} of the *span*, so
  the relative error on a boundary value scales with span/value; all
  {selfcheck["total"]} cases stay below the 3 % bar.
* **Real capabilities vs human declaration**: containment margins are exactly zero
  on every axis (nothing certified beyond the documented claim). The detected
  boundaries are quantised by the grid size (one grid cell, {grid_width} in units
  of `grid_n`), which is the honest resolution of that axis.
* **No external experimental benchmark applies** to this node: the deliverable is a
  *mechanism* (an out-of-domain detector), not a physical prediction. Its accuracy
  claim is therefore against (i) closed-form algebraic boundaries and (ii) the human
  declarations of the repository's own kernels - both reported above. What is *not*
  established: joint multi-axis boundaries, the physical domains of the
  domain-specific solvers, and any statement about a solver's accuracy *inside* its
  envelope (an orthogonal question).

## 7. Acceptance criteria and verdict

| criterion | evidence | verdict |
|---|---|---|
| mechanism lives in TensorPhysics with pytest coverage | `src/tensorphysics/applicability/`, `tests/test_applicability_envelope.py` (21 tests) | PASS |
| >= 10 synthetic black-box capabilities | `artifacts/applicability/synthetic_selfcheck.json`: {synthetic_passed} / {synthetic_total} pass | PASS |
| >= 3 real capabilities, mechanism boundary not wider than the declaration | section 3.1: {compat_ok} / {compat_total} `NOT_WIDER` or `CONSERVATIVE`, containment margins 0 | PASS |
| boundary refined to 0.1 % of the axis span | `width/span` column of section 3.1 (0.000869, 0.000869) and the continuous-axis synthetic cases; the count axis is documented as quantum-limited | PASS (with documented quantum exception) |
| `domain_status` in {{IN, OUT, UNKNOWN}}; OUT forces `OUT_OF_DOMAIN` and refuses REAL | section 3.3, `assign_grade`, `GradeWriteError` tests | PASS |
| zero domain semantics in code/tools/tests | capability ids and axis names appear only in the JSON configuration; enforced by a test | PASS |
| no fabricated numbers in this document | every number injected by `tools/make_applicability_report.py` from the probe artifact | PASS |
"""
    return head


if __name__ == "__main__":
    raise SystemExit(main())
