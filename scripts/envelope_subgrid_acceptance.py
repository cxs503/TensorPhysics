#!/usr/bin/env python3
"""Sub-grid resolution acceptance: the baseline, the fix, and the bill.

This is the evidence producer for the node that replaced the single-midpoint
feasibility test of the window engine with a probe/refinement pass carrying an
explicit resolution certificate.  It re-runs the six criteria of the v1 engine
(imported from ``envelope_cases_acceptance.py``, not re-implemented) so that the
migration is shown to be unchanged, and adds the criteria that only make sense
once probing is explicit:

C7  sub-grid recovery
    Three windows narrower than the node spacing (a pocket centred on the scan
    midpoint, the same pocket moved off it, and a narrow infeasible gap inside a
    feasible band) are recovered, and their boundaries are compared with the
    analytic values.
C8  no silent inflation  (baseline control)
    The v1 mechanism -- roots on the grid, then **one** feasibility evaluation
    per segment -- is emulated here verbatim (``_v1_window``).  On the same
    three cases it must reproduce the recorded v1 failure, while the engine at
    its default probe density must not: a window that the probes cannot settle
    must be reported as such, never averaged into "the whole span is feasible".
C9  cost of the probe pass
    The overhead relative to one probe per segment is measured on the three
    shipped windows (which contain no sub-grid feature) and on the synthetic
    sub-grid cases, together with the evaluation count.
C10 migration is bit-identical
    For the three shipped cases and all their conditions, every interval bound,
    status and binding-constraint label computed with the default policy must be
    equal (``==``, not "within a tolerance") to the value computed with one
    probe per segment and no refinement.  The new machinery may not move a
    number it did not have to move.
C11 test suite
    The envelope test files are run in a subprocess and the pass count recorded.

Usage::

    python scripts/envelope_subgrid_acceptance.py --out artifacts/envelope_n09
"""

from __future__ import annotations

import argparse
import json
import platform
import pathlib
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from tensorphysics.envelope import (  # noqa: E402
    EnvelopeEngine,
    ResolutionPolicy,
    WindowCase,
    artifact_digest,
    case_from_dict,
    code_digest,
    load_case,
    write_artifact,
)
from tensorphysics.envelope.marching import (  # noqa: E402
    MERGE_FRACTION,
    ZERO_FRACTION,
    _roots_on_grid,
)
from tensorphysics.envelope.constraints import EvalCounter  # noqa: E402
from tensorphysics.envelope.roots import linspace  # noqa: E402

import envelope_cases_acceptance as base  # noqa: E402

CASE_DIR = ROOT / "scenarios" / "envelope_cases"
SCHEMA = "tensorphysics.envelope.case.v1"
ASSUMED_NODES = 96  # the engine's default scan


# ---------------------------------------------------------------------------
# case builders (synthetic, domain-free)
# ---------------------------------------------------------------------------

def _square(axis: str, centre: float) -> dict[str, Any]:
    return {
        "op": "square",
        "arg": {
            "op": "sub",
            "a": {"op": "var", "name": axis},
            "b": {"op": "const", "value": centre},
        },
    }


def pocket_case(centre: float, half_width: float, resolution: Mapping[str, Any] | None = None) -> WindowCase:
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "case_id": f"pocket_{centre:g}_{half_width:g}",
        "title": "feasible pocket narrower than the node spacing",
        "mode": "interval",
        "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
        "conditions": [{"label": "one", "params": {}}],
        "constraints": [
            {
                "id": "c_pocket",
                "kind": "inequality",
                "side": "ge",
                "value": {
                    "op": "sub",
                    "a": {"op": "const", "value": half_width * half_width},
                    "b": _square("alpha", centre),
                },
            }
        ],
    }
    if resolution:
        payload["contract"] = {"resolution": dict(resolution)}
    return case_from_dict(payload)


def gap_case(centre: float, half_width: float) -> WindowCase:
    return case_from_dict(
        {
            "schema": SCHEMA,
            "case_id": f"gap_{centre:g}_{half_width:g}",
            "title": "infeasible gap narrower than the node spacing",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_gap",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {
                        "op": "sub",
                        "a": _square("alpha", centre),
                        "b": {"op": "const", "value": half_width * half_width},
                    },
                }
            ],
        }
    )


def channel_case(eps: float) -> WindowCase:
    delta_up = {
        "op": "sub",
        "a": {"op": "var", "name": "beta"},
        "b": {"op": "var", "name": "alpha"},
    }
    delta_dn = {
        "op": "sub",
        "a": {"op": "var", "name": "alpha"},
        "b": {"op": "var", "name": "beta"},
    }
    return case_from_dict(
        {
            "schema": SCHEMA,
            "case_id": f"channel_{eps:g}",
            "title": "diagonal channel thinner than the line spacing",
            "mode": "plane",
            "axes": [
                {"name": "alpha", "lo": 0.0, "hi": 1.0},
                {"name": "beta", "lo": 0.0, "hi": 1.0},
            ],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_hi",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {"op": "sub", "a": {"op": "const", "value": eps}, "b": delta_up},
                },
                {
                    "id": "c_lo",
                    "kind": "inequality",
                    "side": "ge",
                    "value": {"op": "sub", "a": {"op": "const", "value": eps}, "b": delta_dn},
                },
            ],
        }
    )


# ---------------------------------------------------------------------------
# the v1 mechanism, re-implemented verbatim as the baseline
# ---------------------------------------------------------------------------

def _v1_window(case: WindowCase, condition: str | None = None, n: int = ASSUMED_NODES) -> dict[str, Any]:
    """Roots on the grid, then one feasibility evaluation per segment.

    This is exactly what the engine did before the probe pass: the interval
    structure came from the bracketed roots and each segment was classified by
    the single sample at its midpoint.  Kept here so the failure it produces is
    reproducible side by side with the fixed engine, instead of being a story
    about the past.
    """

    counter = EvalCounter()
    axis = case.axes[0]
    params = {k: float(v) for k, v in case.base_params(condition).items()}
    section = case.constraints.continuous_constraints
    probe = linspace(axis.lo, axis.hi, max(3, n // 4))
    scale = 0.0
    for x in probe:
        params[axis.name] = x
        for constraint in section:
            scale = max(scale, abs(constraint.margin(params, counter)))
    zero_tol = ZERO_FRACTION * max(scale, 1e-30)
    merge_tol = MERGE_FRACTION * axis.width
    roots = _roots_on_grid(section, params, axis, n, counter, merge_tol, zero_tol)
    edges = [axis.lo] + [r.x for r in roots] + [axis.hi]
    intervals: list[tuple[float, float]] = []
    open_at: float | None = None
    for left, right in zip(edges, edges[1:]):
        params[axis.name] = 0.5 * (left + right)
        feasible = all(c.margin(params, counter) >= 0.0 for c in section)
        if feasible and open_at is None:
            open_at = left
        elif not feasible and open_at is not None:
            intervals.append((open_at, left))
            open_at = None
    if open_at is not None:
        intervals.append((open_at, edges[-1]))
    return {
        "status": "feasible" if intervals else "infeasible",
        "intervals": intervals,
        "width": sum(hi - lo for lo, hi in intervals) if intervals else None,
        "n_evals": counter.count,
        "n_roots": len(roots),
    }


def _engine_run(case: WindowCase, condition: str = "one", **kwargs: Any) -> Any:
    return EnvelopeEngine(case).run(condition=condition, **kwargs)


def _relative(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    scale = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / scale


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------

SUBGRID_CASES = (
    ("pocket at the scan midpoint", pocket_case(0.5, 1e-3), (0.499, 0.501)),
    ("pocket off the scan midpoint", pocket_case(0.3, 1e-3), (0.299, 0.301)),
    ("pocket, 3.5x narrower", pocket_case(0.417, 7e-4), (0.4163, 0.4177)),
)

COARSE = {"samples_per_segment": 1, "max_depth": 0, "sub_cells": 2}


def criterion_c7() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worst = 0.0
    for label, case, (truth_lo, truth_hi) in SUBGRID_CASES:
        result = _engine_run(case)
        bound_errors = []
        if result.intervals:
            interval = result.intervals[0]
            bound_errors = [
                abs(interval.lo - truth_lo) / max(abs(truth_lo), 1e-300),
                abs(interval.hi - truth_hi) / max(abs(truth_hi), 1e-300),
            ]
        truth_width = truth_hi - truth_lo
        width_error = _relative(result.width, truth_width)
        rows.append(
            {
                "label": label,
                "case_id": case.case_id,
                "status": result.status,
                "engine_intervals": [[i.lo, i.hi] for i in result.intervals],
                "analytic": [truth_lo, truth_hi],
                "analytic_width": truth_width,
                "width_relative_difference": width_error,
                "bound_relative_differences": bound_errors,
                "n_evals": result.n_evals,
                "probed_width": result.resolution["guaranteed_width"],
                "suspect_spans": result.resolution["suspect_spans"],
            }
        )
        for value in bound_errors + ([width_error] if width_error is not None else []):
            worst = max(worst, value)

    gap = gap_case(0.37, 8e-4)
    gap_result = _engine_run(gap)
    gap_ok = (
        gap_result.status == "feasible"
        and len(gap_result.intervals) == 2
        and abs(gap_result.intervals[0].hi - (0.37 - 8e-4)) <= 1e-12
        and abs(gap_result.intervals[1].lo - (0.37 + 8e-4)) <= 1e-12
    )
    worst = max(
        worst,
        abs(gap_result.intervals[0].hi - (0.37 - 8e-4)) / 0.37 if gap_result.intervals else 1.0,
        abs(gap_result.intervals[1].lo - (0.37 + 8e-4)) / 0.37 if gap_result.intervals else 1.0,
    )
    return {
        "id": "C7",
        "name": "sub-grid window recovery against the analytic window",
        "cases": rows,
        "narrow_gap": {
            "case_id": gap.case_id,
            "status": gap_result.status,
            "intervals": [[i.lo, i.hi] for i in gap_result.intervals],
            "analytic": [[0.0, 0.37 - 8e-4], [0.37 + 8e-4, 1.0]],
            "recovered": gap_ok,
        },
        "threshold_relative_difference": 1e-12,
        "worst_relative_difference": worst,
        "passed": worst <= 1e-12 and gap_ok,
    }


def criterion_c8() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    v1_silent = 0
    fixed_silent = 0
    for label, case, (truth_lo, truth_hi) in SUBGRID_CASES:
        baseline = _v1_window(case)
        truth_width = truth_hi - truth_lo
        v1_inflated = baseline["status"] == "feasible" and (
            (baseline["width"] or 0.0) > 10.0 * truth_width
        )
        v1_lost = baseline["status"] == "infeasible"
        fixed = _engine_run(case)
        coarse = _engine_run(pocket_case(
            (truth_lo + truth_hi) / 2.0,
            (truth_hi - truth_lo) / 2.0,
            COARSE,
        ))
        coarse_silent = (
            coarse.status == "feasible"
            and (coarse.width or 0.0) > 10.0 * truth_width
            and not coarse.resolution["unresolved_spans"]
        )
        fixed_silent += int(
            fixed.status == "feasible"
            and (fixed.width or 0.0) > 10.0 * truth_width
            and not fixed.resolution["unresolved_spans"]
        )
        v1_silent += int(v1_inflated or v1_lost)
        rows.append(
            {
                "label": label,
                "analytic_width": truth_width,
                "v1_status": baseline["status"],
                "v1_width": baseline["width"],
                "v1_inflated": v1_inflated,
                "v1_window_lost": v1_lost,
                "fixed_status": fixed.status,
                "fixed_width": fixed.width,
                "fixed_unresolved_spans": fixed.resolution["unresolved_spans"],
                "coarse_probe_status": coarse.status,
                "coarse_probe_reported_unresolved": bool(
                    coarse.resolution["unresolved_spans"]
                ),
                "coarse_probe_silently_inflated": coarse_silent,
            }
        )
    return {
        "id": "C8",
        "name": "the v1 single-midpoint failure is reproducible and no longer silent",
        "cases": rows,
        "v1_silent_failures": v1_silent,
        "fixed_silent_failures": fixed_silent,
        "passed": v1_silent == len(SUBGRID_CASES) and fixed_silent == 0,
    }


def criterion_c9() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    worst_overhead = 0.0
    for path in sorted(CASE_DIR.glob("*.json")):
        case = load_case(path)
        for condition in case.conditions:
            default = _engine_run(case, condition=condition.label)
            payload = json.loads(path.read_text())
            payload.setdefault("contract", {})
            payload["contract"]["resolution"] = dict(COARSE)
            coarse = EnvelopeEngine(case_from_dict(payload)).run(condition=condition.label)
            overhead = (
                default.n_evals / coarse.n_evals if coarse.n_evals else float("inf")
            )
            worst_overhead = max(worst_overhead, overhead)
            rows.append(
                {
                    "case_id": case.case_id,
                    "condition": condition.label,
                    "default_evals": default.n_evals,
                    "single_probe_evals": coarse.n_evals,
                    "overhead": overhead,
                    "default_elapsed_s": default.elapsed_s,
                    "suspect_spans": default.resolution["suspect_spans"],
                }
            )
    subgrid = []
    for label, case, _ in SUBGRID_CASES:
        result = _engine_run(case)
        subgrid.append(
            {
                "case_id": case.case_id,
                "n_evals": result.n_evals,
                "suspect_spans": result.resolution["suspect_spans"],
                "refined_spans": result.resolution["refined_spans"],
                "elapsed_s": result.elapsed_s,
            }
        )
    return {
        "id": "C9",
        "name": "cost of the probe pass on windows that have no sub-grid feature",
        "shipped": rows,
        "worst_overhead": worst_overhead,
        "threshold_overhead": 40.0,
        "subgrid_cases": subgrid,
        "passed": worst_overhead <= 40.0,
    }


def criterion_c10() -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    moved = 0
    compared = 0
    for path in sorted(CASE_DIR.glob("*.json")):
        case = load_case(path)
        payload = json.loads(path.read_text())
        payload.setdefault("contract", {})
        payload["contract"]["resolution"] = dict(COARSE)
        coarse_case = case_from_dict(payload)
        for condition in case.conditions:
            strict = EnvelopeEngine(case).run(condition=condition.label)
            coarse = EnvelopeEngine(coarse_case).run(condition=condition.label)
            numbers = 0
            mismatches: list[str] = []
            if strict.status != coarse.status:
                mismatches.append(f"status {strict.status} != {coarse.status}")
            if len(strict.intervals) != len(coarse.intervals):
                mismatches.append(
                    f"interval count {len(strict.intervals)} != {len(coarse.intervals)}"
                )
            for index, (a, b) in enumerate(zip(strict.intervals, coarse.intervals)):
                numbers += 2
                if a.lo != b.lo:
                    mismatches.append(f"interval {index} lo {a.lo!r} != {b.lo!r}")
                if a.hi != b.hi:
                    mismatches.append(f"interval {index} hi {a.hi!r} != {b.hi!r}")
                if list(a.lo_drivers) != list(b.lo_drivers):
                    mismatches.append(f"interval {index} lo drivers differ")
                if list(a.hi_drivers) != list(b.hi_drivers):
                    mismatches.append(f"interval {index} hi drivers differ")
            compared += numbers
            moved += len(mismatches)
            rows.append(
                {
                    "case_id": case.case_id,
                    "condition": condition.label,
                    "numbers_compared": numbers,
                    "mismatches": mismatches,
                    "strict_evals": strict.n_evals,
                    "coarse_evals": coarse.n_evals,
                }
            )
    return {
        "id": "C10",
        "name": "shipped windows are bit-identical with and without the probe pass",
        "rows": rows,
        "numbers_compared": compared,
        "mismatches": moved,
        "passed": moved == 0 and compared > 0,
    }


def criterion_c11() -> dict[str, Any]:
    files = [
        "tests/test_envelope_engine.py",
        "tests/test_envelope_contracts.py",
        "tests/test_envelope_subgrid_resolution.py",
    ]
    completed = subprocess.run(
        [sys.executable, "-m", "pytest", *files, "-q"],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    tail = [line for line in completed.stdout.strip().splitlines() if line.strip()]
    summary = tail[-1] if tail else ""
    passed = completed.returncode == 0
    count = 0
    for token in summary.replace(",", " ").split():
        if token.isdigit():
            count = max(count, int(token))
    return {
        "id": "C11",
        "name": "envelope test suite",
        "files": files,
        "summary": summary,
        "tests": count,
        "returncode": completed.returncode,
        "passed": passed and count >= 20,
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def _write_figures(out_dir: pathlib.Path) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[dict[str, str]] = []

    # (a) baseline vs fixed on the three sub-grid cases
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8))
    for ax, (label, case, (truth_lo, truth_hi)) in zip(axes, SUBGRID_CASES):
        baseline = _v1_window(case)
        fixed = _engine_run(case)
        span = max(truth_hi - truth_lo, 1e-6)
        ax.axvspan(truth_lo, truth_hi, color="#8ecae6", alpha=0.6, label="analytic window")
        for lo, hi in baseline["intervals"]:
            ax.axvspan(lo, hi, color="#e07a5f", alpha=0.25, label="v1 claim")
        for lo, hi in [(i.lo, i.hi) for i in fixed.intervals]:
            ax.axvline(lo, color="#1b4965", lw=1.2)
            ax.axvline(hi, color="#1b4965", lw=1.2, label="fixed window bounds")
        centre = 0.5 * (truth_lo + truth_hi)
        ax.set_xlim(centre - 4 * span, centre + 4 * span)
        ax.set_yticks([])
        ax.set_xlabel("alpha")
        ax.set_title(
            f"{label}\ntrue width {span:.4g}, v1 {baseline['status']}"
            f" (width {'-' if baseline['width'] is None else format(baseline['width'], '.4g')})",
            fontsize=9,
        )
        handles, labels = ax.get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        ax.legend(unique.values(), unique.keys(), fontsize=6, loc="upper left")
    fig.suptitle("Sub-grid windows: the v1 midpoint probe inflates or loses them; the probe pass does not", fontsize=11)
    fig.tight_layout()
    path = fig_dir / "subgrid_baseline_vs_fixed.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(
        {
            "name": "subgrid_baseline_vs_fixed.png",
            "caption": "Three windows narrower than the node spacing. The analytic window is shaded blue, the v1 single-midpoint claim in orange (the whole span, or nothing), the fixed engine's bounds are the dark vertical lines.",
            "path": str(path.relative_to(ROOT)),
        }
    )

    # (b) boundary error vs probe spacing
    spacing_rows = []
    for samples in (1, 2, 3, 4, 8, 16, 32):
        case = pocket_case(0.5, 1e-3, {"samples_per_segment": samples})
        result = _engine_run(case)
        probe = result.resolution["guaranteed_width"]
        error = None
        if result.intervals:
            interval = result.intervals[0]
            error = max(abs(interval.lo - 0.499) / 0.499, abs(interval.hi - 0.501) / 0.501)
        spacing_rows.append({"samples_per_segment": samples, "probe_width": probe, "error": error})
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    xs = [row["probe_width"] for row in spacing_rows if row["error"] is not None]
    ys = [row["error"] for row in spacing_rows if row["error"] is not None]
    ax.loglog(xs, ys, "o-", label="boundary error vs probed spacing")
    ax.axvline(2e-3, color="#e07a5f", ls="--", label="window width 2e-3")
    ax.axhline(1e-12, color="#1b4965", ls=":", label="1e-12 acceptance")
    ax.set_xlabel("probed spacing [-]", fontsize=9)
    ax.set_ylabel("relative boundary error [-]", fontsize=9)
    ax.set_title("A window is resolved once the probe spacing is finer than it", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "probe_spacing_vs_error.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(
        {
            "name": "probe_spacing_vs_error.png",
            "caption": "Relative error of the recovered pocket bounds against the probed spacing for 1, 2, 3, 4, 8, 16, 32 probes per scan cell: coarser than the window the answer is wrong or absent, finer it is exact to rounding.",
            "path": str(path.relative_to(ROOT)),
        }
    )

    # (c) 2-D channel: area vs line spacing
    eps = 2e-3
    true_area = 2 * eps - eps * eps
    lines = (60, 100, 160, 250, 400, 800)
    areas = []
    for n_lines in lines:
        result = _engine_run(channel_case(eps), n_lines=n_lines, n_scan=200)
        areas.append(
            {
                "n_lines": n_lines,
                "area": result.area,
                "area_reliable": result.resolution["area_reliable"],
                "line_spacing": result.resolution["line_spacing"],
                "single_line_components": result.resolution["single_line_components"],
            }
        )
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.axhline(true_area, color="#8ecae6", lw=2, label="analytic area 2e - e$^2$")
    reliable = [row for row in areas if row["area_reliable"]]
    unreliable = [row for row in areas if not row["area_reliable"]]
    if unreliable:
        ax.plot(
            [row["n_lines"] for row in unreliable],
            [row["area"] for row in unreliable],
            "x--",
            color="#e07a5f",
            label="area not reliable (single-line components)",
        )
    if reliable:
        ax.plot(
            [row["n_lines"] for row in reliable],
            [row["area"] for row in reliable],
            "o-",
            color="#1b4965",
            label="marched area",
        )
    ax.set_xscale("log")
    ax.set_xlabel("scan lines across the plane [-]", fontsize=9)
    ax.set_ylabel("window area [-]", fontsize=9)
    ax.set_title("A channel thinner than the line spacing: feasible, but its area needs lines", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "channel_area_vs_lines.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(
        {
            "name": "channel_area_vs_lines.png",
            "caption": "Diagonal channel |beta - alpha| <= 0.002 (analytic area 3.996e-3). Below one line spacing per channel width the window is found and flagged area-unreliable; above it the area converges to the analytic value.",
            "path": str(path.relative_to(ROOT)),
        }
    )

    # (d) cost
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    samples = [1, 2, 4, 8, 16, 32]
    naive = []
    refined = []
    for value in samples:
        pocket = _engine_run(pocket_case(0.5, 1e-3, {"samples_per_segment": value}))
        smooth = _engine_run(pocket_case(0.75, 0.2, {"samples_per_segment": value}))
        refined.append(pocket.n_evals)
        naive.append(smooth.n_evals)
    ax.plot(samples, naive, "o-", label="window without sub-grid feature")
    ax.plot(samples, refined, "s-", label="window with one pocket (refined)")
    ax.set_xlabel("probes per scan cell [-]", fontsize=9)
    ax.set_ylabel("constraint evaluations per window [-]", fontsize=9)
    ax.set_title("The probe pass costs what it probes, and refinement stays local", fontsize=10)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend(fontsize=7)
    fig.tight_layout()
    path = fig_dir / "probe_cost.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    written.append(
        {
            "name": "probe_cost.png",
            "caption": "Constraint evaluations per one-dimensional window against the probe density, for a window with and without a sub-grid pocket: the cost is linear in the probes and the refinement adds a bounded amount on top.",
            "path": str(path.relative_to(ROOT)),
        }
    )
    return written


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "artifacts" / "envelope_n09"))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)
    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    case_paths = sorted(CASE_DIR.glob("*.json"))
    inherited = {
        "C1": base.criterion_c1(case_paths),
        "C2": base.criterion_c2(),
        "C3": base.criterion_c3(),
        "C4": base.criterion_c4(),
        "C5": base.criterion_c5(),
        "C6": base.criterion_c6(),
    }
    criteria = dict(inherited)
    criteria["C7"] = criterion_c7()
    criteria["C8"] = criterion_c8()
    criteria["C9"] = criterion_c9()
    criteria["C10"] = criterion_c10()
    criteria["C11"] = criterion_c11()
    figures = [] if args.skip_figures else _write_figures(out_dir)

    evidence = {
        "schema": "tensorphysics.envelope.subgrid_evidence.v1",
        "code_digest": code_digest(),
        "resolution_policy": ResolutionPolicy().to_dict(),
        "scan": {"nodes_1d": ASSUMED_NODES, "node_spacing": 1.0 / (ASSUMED_NODES - 1)},
        "case_digests": {
            load_case(path).case_id: load_case(path).source_digest for path in case_paths
        },
        "criteria": criteria,
        "figures": figures,
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "elapsed_s": time.perf_counter() - started,
        },
        "all_passed": all(item["passed"] for item in criteria.values()),
    }
    write_artifact(out_dir / "criteria_evidence.json", evidence)
    summary = {key: criteria[key]["passed"] for key in criteria}
    print(json.dumps({"criteria": summary, "all_passed": evidence["all_passed"]}, indent=2))
    print("artifact digest", artifact_digest(evidence, drop=("code_digest",)))
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())