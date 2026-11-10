#!/usr/bin/env python3
"""Envelope-engine acceptance run: every criterion, one artifact.

This script is the evidence producer for the operating-window engine
(:mod:`tensorphysics.envelope`).  It is deliberately *outside* the package:
the domain content it consumes (three shipped window cases) is ordinary
configuration data, and the independent reference implementations it carries
exist only to grade the engine.

Criteria and how each is established
------------------------------------
C1  migration consistency
    A closed-form per-window path -- the pattern the engine replaces: each
    constraint is declared as a polynomial (or reciprocal) in the window axis
    with coefficients given as expressions of the condition parameters; its
    roots are solved analytically, the span is partitioned at those roots, and
    feasibility is decided by one evaluation per sub-interval.  For 3 cases x 3
    conditions the engine's interval bounds and width must agree with this path
    to a relative difference <= 1e-12.
C2  boundary order
    Synthetic band with two parabolic analytic boundaries: the interpolated
    boundary error is measured at the chord midpoints under grid refinement and
    the observed order must be >= 1.9.  The tip-dominated disc case is reported
    alongside as a documented limitation, not as the criterion.
C3  contract negative control
    A window whose margin deliberately bends back is declared monotone; the
    checker must report a violation.  The same geometry with the declaration
    removed must pass -- the pair shows the checker reacts to the *declaration
    mismatch*, not to nonlinearity as such.
C4  infeasible semantics
    An empty window must return ``infeasible`` with a nearest-feasible residual
    and no boundary/area number.
C5  zero domain semantics
    The package sources and the test files must contain no domain vocabulary;
    the scan list is declared below and the hits are counted.
C6  performance and area
    2-D window run at ~5e4 constraint evaluations on one core (the run records
    wall time and the evaluation count); the marched area must agree with a
    brute-force grid scan to <= 1e-3 relative difference.

Usage::

    python scripts/envelope_cases_acceptance.py --out artifacts/envelope
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import platform
import sys
import time
from typing import Any, Mapping, Sequence

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.envelope import (  # noqa: E402
    EnvelopeEngine,
    WindowCase,
    artifact_digest,
    check_infeasible_semantics,
    code_digest,
    evaluate_expression,
    load_case,
    order_study,
    write_artifact,
)
from tensorphysics.envelope import benchmarks as synth  # noqa: E402

CASE_DIR = ROOT / "scenarios" / "envelope_cases"
OUT_DIR_DEFAULT = ROOT / "artifacts" / "envelope"

# C5: the vocabulary that must not appear in the package sources or in the tests
DOMAIN_VOCABULARY = (
    "stranding", "stranded", "grounding", "tide", "tidal", "draught", "draft",
    "ballast", "pump", "ship", "ships", "vessel", "hull", "keel", "quay",
    "berth", "manifold", "hose", "mooring", "propeller", "seakeeping", "heave",
    "pitch", "roll", "ice", "naval", "marine", "floating", "cargo", "tanker",
    "sts", "lng", "metacentric", "buoyancy", "seawater", "harbour", "harbor",
)


# ---------------------------------------------------------------------------
# independent closed-form reference path (C1)
# ---------------------------------------------------------------------------

def _closed_form_roots(
    form: str, coefficients: Mapping[str, float], lo: float, hi: float
) -> list[float]:
    """Real roots in ``[lo, hi]`` of a declared form, solved analytically."""

    span = hi - lo
    tol = 1e-12 * span
    roots: list[float] = []
    if form == "linear":
        a, b = coefficients["a"], coefficients["b"]
        if a != 0.0:
            roots.append(-b / a)
    elif form == "quadratic":
        a, b, c = coefficients["a"], coefficients["b"], coefficients["c"]
        scale = max(abs(a), abs(b), abs(c), 1e-300)
        if abs(a) <= 1e-14 * scale:
            if b != 0.0:
                roots.append(-c / b)
        else:
            discriminant = b * b - 4.0 * a * c
            if discriminant >= 0.0:
                root = math.sqrt(discriminant)
                # numerically stable form: avoid cancellation in the small root
                q = -0.5 * (b + math.copysign(root, b))
                if q != 0.0:
                    roots.extend((q / a, c / q))
                else:
                    roots.append(0.0)
    elif form == "reciprocal":
        a, b = coefficients["a"], coefficients["b"]
        if b != 0.0:
            roots.append(-a / b)
    else:
        raise ValueError(f"unsupported reference form {form!r}")
    return [r for r in roots if lo - tol <= r <= hi + tol]


def _legacy_window(case: WindowCase, condition: str) -> dict[str, Any]:
    """Prior path: analytic roots + midpoint feasibility, no numerical search."""

    axis = case.axes[0]
    base = case.base_params(condition)
    references = case.extensions["legacy_reference"]
    section = case.constraints.continuous_constraints
    lo, hi = axis.lo, axis.hi
    span = hi - lo
    merge = 1e-12 * span

    found: list[tuple[float, str]] = []
    for constraint in section:
        spec = references[constraint.id]
        coefficients = {
            key: float(evaluate_expression(spec[key], base))
            for key in ("a", "b", "c")
            if key in spec
        }
        for root in _closed_form_roots(spec["form"], coefficients, lo, hi):
            found.append((min(max(root, lo), hi), constraint.id))
    found.sort()
    merged: list[tuple[float, list[str]]] = []
    for root, cid in found:
        if merged and abs(root - merged[-1][0]) <= merge:
            merged[-1][1].append(cid)
        else:
            merged.append((root, [cid]))

    edges: list[tuple[float, tuple[str, ...]]] = [(lo, ())]
    edges += [(root, tuple(dict.fromkeys(ids))) for root, ids in merged]
    edges.append((hi, ()))

    params = dict(base)
    intervals: list[dict[str, Any]] = []
    opened: tuple[float, tuple[str, ...]] | None = None
    for index in range(len(edges) - 1):
        left, right = edges[index], edges[index + 1]
        if right[0] <= left[0]:
            continue
        params[axis.name] = 0.5 * (left[0] + right[0])
        feasible = all(c.margin(params) >= 0.0 for c in section)
        if feasible and opened is None:
            opened = left
        elif not feasible and opened is not None:
            intervals.append(
                {"lo": opened[0], "hi": left[0], "lo_drivers": opened[1], "hi_drivers": left[1]}
            )
            opened = None
    if opened is not None:
        intervals.append(
            {"lo": opened[0], "hi": hi, "lo_drivers": opened[1], "hi_drivers": ()}
        )

    settings: list[float] = []
    for constraint in case.constraints.discrete_constraints:
        lattice = constraint.lattice
        assert lattice is not None
        for interval in intervals:
            for value in lattice.candidates(interval["lo"], interval["hi"]):
                params[axis.name] = value
                if all(c.margin(params) >= 0.0 for c in case.constraints):
                    settings.append(value)
    return {"intervals": intervals, "settings": sorted(settings)}


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------

def _relative(a: float, b: float) -> float:
    denominator = max(abs(a), abs(b), 1e-300)
    return abs(a - b) / denominator


def criterion_c1(case_paths: Sequence[pathlib.Path]) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    worst = 0.0
    for path in case_paths:
        case = load_case(path)
        for condition in case.conditions:
            engine_result = EnvelopeEngine(case).run(condition=condition.label, n_scan=128)
            legacy = _legacy_window(case, condition.label)
            if len(engine_result.intervals) != len(legacy["intervals"]):
                raise SystemExit(
                    f"C1 structural mismatch for {case.case_id}/{condition.label}: "
                    f"engine {len(engine_result.intervals)} intervals, "
                    f"legacy {len(legacy['intervals'])}"
                )
            for index, (engine_interval, legacy_interval) in enumerate(
                zip(engine_result.intervals, legacy["intervals"])
            ):
                for bound in ("lo", "hi"):
                    engine_value = getattr(engine_interval, bound)
                    legacy_value = legacy_interval[bound]
                    difference = _relative(engine_value, legacy_value)
                    worst = max(worst, difference)
                    comparisons.append(
                        {
                            "case_id": case.case_id,
                            "condition": condition.label,
                            "interval": index,
                            "quantity": f"window_{bound}",
                            "engine": engine_value,
                            "legacy": legacy_value,
                            "relative_difference": difference,
                        }
                    )
                difference = _relative(engine_interval.width, legacy_interval["hi"] - legacy_interval["lo"])
                worst = max(worst, difference)
                comparisons.append(
                    {
                        "case_id": case.case_id,
                        "condition": condition.label,
                        "interval": index,
                        "quantity": "window_width",
                        "engine": engine_interval.width,
                        "legacy": legacy_interval["hi"] - legacy_interval["lo"],
                        "relative_difference": difference,
                    }
                )
                if list(engine_interval.lo_drivers) != list(legacy_interval["lo_drivers"]):
                    raise SystemExit(
                        f"C1 active-set mismatch at the lower bound of "
                        f"{case.case_id}/{condition.label}"
                    )
                if list(engine_interval.hi_drivers) != list(legacy_interval["hi_drivers"]):
                    raise SystemExit(
                        f"C1 active-set mismatch at the upper bound of "
                        f"{case.case_id}/{condition.label}"
                    )
            engine_settings = (
                []
                if engine_result.discrete is None
                else sorted(
                    value
                    for run in engine_result.discrete.runs
                    for value in _settings_in_run(engine_result, run)
                )
            )
            if len(engine_settings) != len(legacy["settings"]) or any(
                abs(a - b) > 1e-9 * max(1.0, abs(b))
                for a, b in zip(engine_settings, legacy["settings"])
            ):
                raise SystemExit(
                    f"C1 discrete-setting mismatch for {case.case_id}/{condition.label}"
                )
    return {
        "id": "C1",
        "name": "migration consistency vs the closed-form per-window path",
        "passed": worst <= 1e-12,
        "threshold_relative_difference": 1e-12,
        "worst_relative_difference": worst,
        "n_comparisons": len(comparisons),
        "comparisons": comparisons,
    }


def _settings_in_run(result: Any, run: Any) -> list[float]:
    if run.count <= 0:
        return []
    step = run.step if run.step is not None else 0.0
    return [run.lo + index * step for index in range(run.count)]


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------

def criterion_c2() -> dict[str, Any]:
    band = synth.parabolic_band_case()
    studies = []
    for resolutions in ((20, 40, 80), (40, 80, 160), (80, 160, 320)):
        study = order_study(
            band,
            resolutions=resolutions,
            n_scan=96,
            analytic=synth.analytic_band(),
        )
        studies.append(study)
    observed = [order for study in studies for order in study.orders]
    disc_studies = []
    for resolutions in ((40, 80, 160),):
        disc_studies.append(
            order_study(
                synth.disc_case(),
                resolutions=resolutions,
                n_scan=96,
                analytic=synth.analytic_disc(),
            )
        )
    return {
        "id": "C2",
        "name": "boundary reconstruction order (interpolated boundary vs analytic)",
        "passed": bool(observed) and min(observed) >= 1.9,
        "threshold_min_order": 1.9,
        "min_observed_order": min(observed) if observed else None,
        "studies": [
            {
                "case_id": band.case_id,
                "resolutions": list(study.resolutions),
                "step_sizes": list(study.step_sizes),
                "errors": list(study.errors),
                "orders": list(study.orders),
            }
            for study in studies
        ],
        "limitation_closed_boundary_with_vertical_tangents": [
            {
                "case_id": "synth_disc",
                "resolutions": list(study.resolutions),
                "step_sizes": list(study.step_sizes),
                "errors": list(study.errors),
                "orders": list(study.orders),
                "note": (
                    "near a vertical tangent the chord error is O(h^1/2), so the "
                    "pointwise maximum is not a second-order quantity; the area of "
                    "the same object still converges (see C6)"
                ),
            }
            for study in disc_studies
        ],
    }


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------

def criterion_c3() -> dict[str, Any]:
    folded = synth.folded_case()
    folded_result = EnvelopeEngine(folded).run(n_scan=96)
    undecorated = synth.case_from_dict(
        {
            "case_id": "synth_folded_undeclared",
            "mode": "interval",
            "axes": [{"name": "alpha", "lo": 0.0, "hi": 1.0}],
            "conditions": [{"label": "one", "params": {}}],
            "constraints": [
                {
                    "id": "c_band",
                    "kind": "inequality",
                    "side": "ge",
                    "value": folded.constraints.by_id("c_band").expression,
                }
            ],
            "contract": {"expect": {"intervals": 1}},
        }
    )
    control_result = EnvelopeEngine(undecorated).run(n_scan=96)
    return {
        "id": "C3",
        "name": "contract checker negative control: declared monotone, actually folded",
        "passed": (not folded_result.contracts.ok)
        and "monotone:c_band:alpha" in folded_result.contracts.failed
        and control_result.contracts.ok,
        "violation_reported": not folded_result.contracts.ok,
        "failed_checks": list(folded_result.contracts.failed),
        "violation_detail": [
            outcome.to_dict()
            for outcome in folded_result.contracts.outcomes
            if not outcome.passed
        ],
        "control_without_declaration": {
            "contracts_ok": control_result.contracts.ok,
            "failed_checks": list(control_result.contracts.failed),
            "window": [[i.lo, i.hi] for i in control_result.intervals],
        },
    }


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------

def criterion_c4() -> dict[str, Any]:
    case = synth.empty_case()
    result = EnvelopeEngine(case).run(n_scan=96)
    payload = result.to_dict()
    semantics = check_infeasible_semantics(
        payload["status"],
        result.intervals,
        result.components,
        result.area,
        result.nearest,
    )
    return {
        "id": "C4",
        "name": "infeasible semantics: empty window, nearest feasible point, no extrapolation",
        "passed": (
            payload["status"] == "infeasible"
            and not payload["intervals"]
            and result.area is None
            and result.nearest is not None
            and result.nearest.residual > 0.0
            and not result.nearest.found
            and semantics.passed
        ),
        "status": payload["status"],
        "intervals": payload["intervals"],
        "area": payload["area"],
        "nearest": payload["nearest"],
        "semantics_check": semantics.to_dict(),
    }


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------

def criterion_c5() -> dict[str, Any]:
    package = ROOT / "src" / "tensorphysics" / "envelope"
    tests = ROOT / "tests"
    targets = sorted(package.glob("*.py")) + sorted(tests.glob("test_envelope*.py"))
    import re

    patterns = {
        word: re.compile(r"(?<![a-z])" + word + r"(?![a-z])") for word in DOMAIN_VOCABULARY
    }
    hits: list[dict[str, Any]] = []
    scanned_lines = 0
    for path in targets:
        text = path.read_text(encoding="utf-8").lower()
        lines = text.splitlines()
        scanned_lines += len(lines)
        for lineno, line in enumerate(lines, start=1):
            for word, pattern in patterns.items():
                if pattern.search(line):
                    hits.append(
                        {
                            "file": str(path.relative_to(ROOT)),
                            "line": lineno,
                            "word": word,
                            "text": line.strip()[:160],
                        }
                    )
    return {
        "id": "C5",
        "name": "zero domain semantics in package sources and tests",
        "passed": not hits,
        "vocabulary": list(DOMAIN_VOCABULARY),
        "files_scanned": [str(p.relative_to(ROOT)) for p in targets],
        "lines_scanned": scanned_lines,
        "hits": hits,
    }


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------

def _vector_margin(constraint: Any, params: Mapping[str, Any]) -> Any:
    """Vectorised margin, used only by the brute-force cross-check."""

    value = evaluate_expression(constraint.expression, params)
    if constraint.kind.value == "inequality":
        return constraint.side * value
    if constraint.kind.value == "equality":
        return constraint.tolerance - abs(value - constraint.target)
    raise ValueError("vector brute force supports continuous constraints only")


def _brute_area(case: WindowCase, condition: str, n: int, chunk: int = 96) -> tuple[float, int]:
    import numpy as np  # local import: only the cross-check needs the array backend

    axes = case.axes[0], case.axes[1]
    base = case.base_params(condition)
    section = case.constraints.continuous_constraints
    u = np.linspace(axes[0].lo, axes[0].hi, n)
    v = np.linspace(axes[1].lo, axes[1].hi, n)
    cell = (u[1] - u[0]) * (v[1] - v[0])
    feasible = 0
    for start in range(0, n, chunk):
        block = u[start : start + chunk]
        grid_u, grid_v = np.meshgrid(block, v, indexing="ij")
        params = dict(base)
        params[axes[0].name] = grid_u
        params[axes[1].name] = grid_v
        mask = np.ones_like(grid_u, dtype=bool)
        for constraint in section:
            mask &= _vector_margin(constraint, params) >= 0.0
        feasible += int(mask.sum())
    return feasible * cell, feasible


def criterion_c6() -> dict[str, Any]:
    case = synth.disc_case()
    engine = EnvelopeEngine(case)
    perf = engine.run(n_lines=160, n_scan=320, n_grid_check=0)
    exact = math.pi * case.constants["radius"] ** 2

    brute: list[dict[str, Any]] = []
    for n in (512, 1024, 2048):
        started = time.perf_counter()
        area, cells = _brute_area(case, "one", n)
        brute.append(
            {
                "n_per_axis": n,
                "area": area,
                "feasible_cells": cells,
                "queries": n * n,
                "elapsed_s": time.perf_counter() - started,
                "relative_difference_vs_exact": abs(area - exact) / exact,
            }
        )
    coarse, middle, fine = brute
    extrapolated = 2.0 * middle["area"] - coarse["area"]
    extrapolated_fine = 2.0 * fine["area"] - middle["area"]
    reference = extrapolated_fine
    relative = abs(perf.area - reference) / reference
    # a second, independent reference: the closed-form area of the synthetic disc
    relative_exact = abs(perf.area - exact) / exact
    return {
        "id": "C6",
        "name": "cost per window and marched area vs brute-force grid scan",
        "passed": (
            relative <= 1e-3
            and perf.n_evals >= 50000
            and perf.elapsed_s <= 60.0
            and perf.elapsed_s * 50000.0 / perf.n_evals <= 60.0
        ),
        "projected_seconds_for_50000_evaluations": perf.elapsed_s * 50000.0
        / max(perf.n_evals, 1),
        "threshold_relative_difference": 1e-3,
        "relative_difference_vs_extrapolated_brute": relative,
        "relative_difference_vs_closed_form_area": relative_exact,
        "engine": {
            "case_id": case.case_id,
            "n_lines": 160,
            "n_scan": 320,
            "n_evals": perf.n_evals,
            "elapsed_s": perf.elapsed_s,
            "evals_per_second": perf.n_evals / perf.elapsed_s if perf.elapsed_s else None,
            "area": perf.area,
            "area_method": perf.components[0].area_method if perf.components else None,
            "components": len(perf.components),
        },
        "brute_force": brute,
        "brute_richardson": {
            "coarse_to_middle": extrapolated,
            "middle_to_fine": extrapolated_fine,
            "two_level_consistency": abs(extrapolated_fine - extrapolated) / extrapolated_fine,
        },
        "closed_form_area": exact,
        "machine": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "platform": platform.platform(),
        },
        "note": (
            "the raw cell count is first order in the cell size, so its own bias at "
            "2048 lines is larger than the acceptance threshold; the reference is the "
            "same brute force linearly extrapolated over two grid levels, whose "
            "two-level consistency is reported above"
        ),
    }


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------

def _case_results(case_paths: Sequence[pathlib.Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in case_paths:
        case = load_case(path)
        for condition in case.conditions:
            started = time.perf_counter()
            result = EnvelopeEngine(case).run(condition=condition.label, n_scan=128)
            rows.append(
                {
                    "case_id": case.case_id,
                    "title": case.title,
                    "condition": condition.label,
                    "condition_params": dict(condition.params),
                    "source_digest": case.source_digest,
                    "status": result.status,
                    "n_evals": result.n_evals,
                    "elapsed_s": result.elapsed_s,
                    "contracts_ok": result.contracts.ok,
                    "failed_checks": list(result.contracts.failed),
                    "intervals": [i.to_dict() for i in result.intervals],
                    "discrete": None if result.discrete is None else result.discrete.to_dict(),
                    "nearest": None if result.nearest is None else result.nearest.to_dict(),
                    "active_sets": result.active_sets,
                    "contracts": result.contracts.to_dict(),
                }
            )
            del started
    return rows


def _write_figures(out_dir: pathlib.Path, case_paths: Sequence[pathlib.Path]) -> list[dict[str, str]]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    figures: list[dict[str, str]] = []

    # --- figure 1: the three shipped windows -------------------------------
    fig, axes_plot = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, path in zip(axes_plot, case_paths):
        case = load_case(path)
        axis = case.axes[0]
        grid = np.linspace(axis.lo, axis.hi, 400)
        colours = plt.cm.viridis(np.linspace(0.1, 0.85, len(case.conditions)))
        for colour, condition in zip(colours, case.conditions):
            params = case.base_params(condition.label)
            params[axis.name] = grid
            for constraint in case.constraints.continuous_constraints:
                ax.plot(
                    grid,
                    _vector_margin(constraint, params),
                    color=colour,
                    lw=1.1,
                    ls="-" if constraint.kind.value == "inequality" else "--",
                    label=f"{condition.label} / {constraint.id}",
                )
            result = EnvelopeEngine(case).run(condition=condition.label, n_scan=128)
            for interval in result.intervals:
                ax.axvspan(interval.lo, interval.hi, color=colour, alpha=0.13)
                ax.axvline(interval.lo, color=colour, lw=1.6)
                ax.axvline(interval.hi, color=colour, lw=1.6)
            if result.discrete is not None:
                for run in result.discrete.runs:
                    step = run.step if run.step is not None else 0.0
                    values = [run.lo + index * step for index in range(run.count)]
                    if values:
                        level = -0.03 * max(
                            abs(_vector_margin(case.constraints.continuous_constraints[0], params)).max(),
                            1.0,
                        )
                        ax.plot(values, [level] * len(values), "|", color=colour, ms=12)
        ax.axhline(0.0, color="k", lw=1.0, ls=":")
        ax.set_title(case.case_id, fontsize=10)
        ax.set_xlabel(axis.name)
        ax.set_ylabel("normalised constraint margin [-]")
        ax.legend(fontsize=6.5, loc="lower left", ncol=1, framealpha=0.85)
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Operating windows of the three shipped cases: margins, admissible bands "
        "(shaded) and attainable discrete settings (ticks)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path_png = figures_dir / "envelope_windows.png"
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    figures.append(
        {
            "name": "envelope_windows.png",
            "caption": (
                "Constraint margins along the window axis for the three shipped cases "
                "(three load conditions each). Shaded bands are the detected windows, "
                "vertical lines their boundaries, ticks the attainable discrete "
                "settings on the declared lattice, dotted line the feasibility "
                "threshold."
            ),
            "path": str(path_png.relative_to(ROOT)),
        }
    )

    # --- figure 2: marching on a plane, order and area ---------------------
    fig, (ax_map, ax_order, ax_area) = plt.subplots(1, 3, figsize=(16.5, 5.0))
    disc = synth.disc_case()
    disc_result = EnvelopeEngine(disc).run(n_lines=120, n_scan=120, n_grid_check=40)
    for component in disc_result.components:
        loop = list(component.points) + [component.points[0]]
        ax_map.plot([p[0] for p in loop], [p[1] for p in loop], "-", lw=1.4)
        ax_map.fill(
            [p[0] for p in loop], [p[1] for p in loop], alpha=0.18, label="marched region"
        )
    theta = np.linspace(0.0, 2.0 * math.pi, 400)
    radius = disc.constants["radius"]
    ax_map.plot(
        disc.constants["centre_a"] + radius * np.cos(theta),
        disc.constants["centre_b"] + radius * np.sin(theta),
        "k--",
        lw=1.0,
        label="analytic boundary",
    )
    if disc_result.grid_check is not None:
        mask = np.array(disc_result.grid_check.feasible)
        ax_map.imshow(
            mask.T,
            origin="lower",
            interpolation="nearest",
            extent=(
                disc_result.grid_check.u_values[0],
                disc_result.grid_check.u_values[-1],
                disc_result.grid_check.v_values[0],
                disc_result.grid_check.v_values[-1],
            ),
            cmap="Greys",
            alpha=0.25,
        )
    ax_map.set_title("marched boundary vs analytic (disc, two-component test)", fontsize=10)
    ax_map.set_xlabel("alpha")
    ax_map.set_ylabel("beta")
    ax_map.legend(fontsize=8)
    ax_map.grid(alpha=0.25)

    band_study = order_study(
        synth.parabolic_band_case(),
        resolutions=(20, 40, 80, 160, 320),
        n_scan=96,
        analytic=synth.analytic_band(),
    )
    ax_order.loglog(band_study.step_sizes, band_study.errors, "o-", label="measured")
    reference = band_study.errors[0] * (np.array(band_study.step_sizes) / band_study.step_sizes[0]) ** 2
    ax_order.loglog(band_study.step_sizes, reference, "k--", label="O(h^2) reference")
    ax_order.set_xlabel("line spacing h [-]")
    ax_order.set_ylabel("max chord-midpoint error [-]")
    ax_order.set_title(
        "boundary order: observed " + ", ".join(f"{o:.2f}" for o in band_study.orders),
        fontsize=10,
    )
    ax_order.legend(fontsize=8)
    ax_order.grid(alpha=0.25, which="both")

    areas = []
    for n_lines in (40, 60, 80, 120, 160, 240, 320):
        result = EnvelopeEngine(disc).run(n_lines=n_lines, n_scan=96, run_contracts=False)
        areas.append((n_lines, result.area))
    exact = math.pi * disc.constants["radius"] ** 2
    ax_area.semilogx(
        [a[0] for a in areas],
        [abs(a[1] - exact) / exact for a in areas],
        "o-",
        label="marched area error",
    )
    ax_area.axhline(1e-3, color="r", ls="--", label="1e-3 acceptance")
    ax_area.set_xlabel("number of scan lines [-]")
    ax_area.set_ylabel("relative area error [-]")
    ax_area.set_title("area of a closed window vs closed form", fontsize=10)
    ax_area.legend(fontsize=8)
    ax_area.grid(alpha=0.25, which="both")
    fig.tight_layout()
    path_png = figures_dir / "envelope_marching.png"
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    figures.append(
        {
            "name": "envelope_marching.png",
            "caption": (
                "Left: marched boundary of the synthetic disc (two-component case "
                "reported separately in the text) against the analytic circle and "
                "the coarse feasibility mask. Middle: boundary reconstruction order "
                "for the parabolic band under refinement, with the O(h^2) reference "
                "line. Right: relative area error of the same closed window versus "
                "the number of scan lines, against the 1e-3 acceptance line."
            ),
            "path": str(path_png.relative_to(ROOT)),
        }
    )

    # --- figure 3: contract controls --------------------------------------
    fig, (ax_fold, ax_empty) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    folded = synth.folded_case()
    grid = np.linspace(0.0, 1.0, 400)
    params = folded.base_params("one")
    params["alpha"] = grid
    const = folded.constraints.by_id("c_band")
    ax_fold.plot(grid, _vector_margin(const, params), label="declared monotone (+1)")
    ax_fold.axhline(0.0, color="k", ls=":")
    result = EnvelopeEngine(folded).run(n_scan=96)
    for interval in result.intervals:
        ax_fold.axvspan(interval.lo, interval.hi, alpha=0.15, color="tab:orange")
    ax_fold.annotate(
        "fold: margin decreases after alpha = 0.5\nchecker verdict: "
        + ("VIOLATION" if not result.contracts.ok else "pass"),
        xy=(0.5, 0.0),
        xytext=(0.55, 0.004),
        fontsize=8.5,
        arrowprops={"arrowstyle": "->"},
    )
    ax_fold.set_xlabel("alpha")
    ax_fold.set_ylabel("margin [-]")
    ax_fold.set_title("C3 negative control: folded boundary declared monotone", fontsize=10)
    ax_fold.legend(fontsize=8)
    ax_fold.grid(alpha=0.25)

    empty = synth.empty_case()
    params = empty.base_params("one")
    params["alpha"] = grid
    never = empty.constraints.by_id("c_never")
    ax_empty.plot(grid, _vector_margin(never, params), label="only constraint")
    ax_empty.axhline(0.0, color="k", ls=":", label="feasibility threshold")
    empty_result = EnvelopeEngine(empty).run(n_scan=96)
    nearest = empty_result.nearest
    if nearest is not None:
        ax_empty.axvline(nearest.params["alpha"], color="tab:red", ls="--")
        ax_empty.annotate(
            f"nearest searched point\nresidual = {nearest.residual:.3f}",
            xy=(nearest.params["alpha"], nearest.residual),
            xytext=(0.55, 0.2),
            fontsize=8.5,
            arrowprops={"arrowstyle": "->"},
        )
    ax_empty.set_xlabel("alpha")
    ax_empty.set_ylabel("margin [-]")
    ax_empty.set_title("C4 empty window: status infeasible, no extrapolated number", fontsize=10)
    ax_empty.legend(fontsize=8)
    ax_empty.grid(alpha=0.25)
    fig.tight_layout()
    path_png = figures_dir / "envelope_contracts.png"
    fig.savefig(path_png, dpi=150)
    plt.close(fig)
    figures.append(
        {
            "name": "envelope_contracts.png",
            "caption": (
                "Left: the deliberate negative control -- a margin that bends back "
                "although it is declared monotone; the checker reports the violation. "
                "Right: the empty-window control -- the engine returns 'infeasible' "
                "with the nearest examined point and its residual instead of "
                "extrapolating a boundary."
            ),
            "path": str(path_png.relative_to(ROOT)),
        }
    )
    return figures


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(OUT_DIR_DEFAULT))
    parser.add_argument("--skip-figures", action="store_true")
    args = parser.parse_args(argv)

    out_dir = pathlib.Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    case_paths = sorted(CASE_DIR.glob("*.json"))

    started = time.perf_counter()
    criteria = {
        "C1": criterion_c1(case_paths),
        "C2": criterion_c2(),
        "C3": criterion_c3(),
        "C4": criterion_c4(),
        "C5": criterion_c5(),
        "C6": criterion_c6(),
    }
    case_rows = _case_results(case_paths)
    figures = [] if args.skip_figures else _write_figures(out_dir, case_paths)

    evidence = {
        "schema": "tensorphysics.envelope.evidence.v1",
        "code_digest": code_digest(),
        "case_digests": {load_case(p).case_id: load_case(p).source_digest for p in case_paths},
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
    write_artifact(
        out_dir / "case_results.json",
        {
            "schema": "tensorphysics.envelope.case_results.v1",
            "code_digest": code_digest(),
            "cases": case_rows,
        },
    )
    summary = {key: criteria[key]["passed"] for key in criteria}
    print(json.dumps({"criteria": summary, "all_passed": evidence["all_passed"]}, indent=2))
    print("artifact digest", artifact_digest(evidence, drop=("code_digest",)))
    return 0 if evidence["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())