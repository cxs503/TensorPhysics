"""Closed-form self-check of the envelope mechanism (synthetic capabilities).

Every case below is pure algebra with a boundary that can be written down in
closed form, so the probing operator can be graded against the truth rather than
against itself.  The cases together exercise:

* the three failure signals -- acceptance criterion, monotonicity / sign law,
  numerical pathology (raising evaluator, declared mechanical flag, order
  collapse);
* the two criterion forms -- a two-sided window (``abs``) and a one-sided
  range (``range``);
* the edge semantics -- no failure inside the span (open interval), no in-domain
  sample at all (empty domain), a quantised axis, and an axis pair.

The suite is shipped (not test-only) so the report, the figures and the tests all
read the same numbers: ``python -m tensorphysics.applicability.benchmarks``
prints the table.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from .capability import CapabilityRegistry
from .probe import ProbeReport, probe_capability

__all__ = ["SyntheticCase", "SYNTHETIC_CASES", "synthetic_registry", "run_selfcheck"]

METRICS = "tensorphysics.applicability.metrics:"


def _axis(name: str, low: float, high: float, **kw: Any) -> dict:
    entry = {"name": name, "low": low, "high": high}
    entry.update(kw)
    return entry


@dataclass(frozen=True)
class SyntheticCase:
    """One synthetic capability plus its closed-form truth."""

    capability: dict
    truth: dict
    note: str = ""

    @property
    def capability_id(self) -> str:
        return self.capability["capability_id"]

    @property
    def axis_name(self) -> str:
        return self.capability["axes"][0]["name"]


def _capability(capability_id: str, target: str, kwargs: dict, axis_kwargs: dict,
                axes: list[dict], criterion: dict, **policy: Any) -> dict:
    entry = {
        "capability_id": capability_id,
        "evaluator": {"target": METRICS + target, "kwargs": kwargs,
                      "axis_kwargs": axis_kwargs},
        "axes": axes,
        "policy": {"criterion": criterion,
                   "pathology": policy.get("pathology") or {}},
    }
    if policy.get("order_window") is not None:
        entry["policy"]["order_window"] = policy["order_window"]
    if policy.get("sign"):
        entry["policy"]["sign"] = policy["sign"]
    return entry


SYNTHETIC_CASES: tuple[SyntheticCase, ...] = (
    SyntheticCase(
        _capability("syn-square-window", "power_law",
                    {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0}, {"x": "x"},
                    [_axis("x", 0.1, 5.0, samples=13, monotone="increasing")],
                    {"kind": "abs", "reference": 0.0, "tolerance": 1.0}),
        {"high": 1.0},
        "value = x**2 in a |v| <= 1 window: domain ends exactly at x = 1",
    ),
    SyntheticCase(
        _capability("syn-reciprocal-limit", "reciprocal", {"amplitude": 1.0}, {"x": "x"},
                    [_axis("x", 0.05, 2.0, samples=13, scale="log", monotone="decreasing")],
                    {"kind": "abs", "reference": 0.0, "tolerance": 2.0}),
        {"low": 0.5},
        "value = 1/x in a |v| <= 2 window: domain starts exactly at x = 0.5",
    ),
    SyntheticCase(
        _capability("syn-exponential-tolerance", "exponential_deviation",
                    {"amplitude": 1.0, "rate": 2.0, "target": 0.0}, {"t": "x"},
                    [_axis("t", 0.0, 3.0, samples=15, monotone="decreasing")],
                    {"kind": "abs", "reference": 0.0, "tolerance": 0.05}),
        {"low": math.log(20.0) / 2.0},
        "value = exp(-2t) in a 5% window: domain starts at t = ln(20)/2",
    ),
    SyntheticCase(
        _capability("syn-divergence-window", "hyperbolic_excess",
                    {"critical": 2.0, "amplitude": 1.0, "epsilon": 1e-9}, {"x": "x"},
                    [_axis("x", 0.1, 3.0, samples=13)],
                    {"kind": "range", "lower": 0.0, "upper": 10.0}),
        {"high": 20.0 / 11.0},
        "value = x/(2-x) must stay in [0, 10]: domain ends at x = 20/11 (before the pole)",
    ),
    SyntheticCase(
        _capability("syn-mechanical-flag", "flag_gate",
                    {"limit": 3.0, "signal": "converged"}, {"z": "x"},
                    [_axis("z", 0.5, 6.0, samples=13)],
                    {"kind": "range", "lower": 0.0, "upper": 1e9},
                    pathology={"flags": {"converged": False}}),
        {"high": 3.0},
        "a declared mechanical failure signal ends the domain at z = 3 even though the value passes",
    ),
    SyntheticCase(
        _capability("syn-order-window", "order_report", {"key": "observed_order"}, {"p": "x"},
                    [_axis("p", 0.2, 8.0, samples=15)],
                    {"kind": "range", "lower": 0.0, "upper": 1e6},
                    order_window={"key": "observed_order", "low": 0.5, "high": 6.0}),
        {"low": 0.5, "high": 6.0},
        "convergence-order collapse on both ends of the declared order window",
    ),
    SyntheticCase(
        _capability("syn-no-failure", "power_law",
                    {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0}, {"x": "x"},
                    [_axis("x", 0.1, 0.9, samples=9)],
                    {"kind": "abs", "reference": 0.0, "tolerance": 100.0}),
        {"open": (True, True), "boundaries": 0},
        "no failure inside the span: both ends open, nothing is claimed outside",
    ),
    SyntheticCase(
        _capability("syn-empty-domain", "power_law",
                    {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0}, {"x": "x"},
                    [_axis("x", 0.2, 2.0, samples=9)],
                    {"kind": "abs", "reference": 0.0, "tolerance": 0.0}),
        {"intervals": 0},
        "zero tolerance and a strictly positive value: the domain is empty everywhere",
    ),
    SyntheticCase(
        _capability("syn-monotone-break", "power_law",
                    {"amplitude": 1.0, "exponent": 1.0, "offset": 0.0}, {"x": "x"},
                    [_axis("x", 0.1, 2.0, samples=5, monotone="decreasing",
                           monotone_slack=0.0)],
                    {"kind": "range", "lower": -1e9, "upper": 1e9}),
        {"samples_in": 1, "law_boundary": 1},
        "rising scalar under a declared decreasing law: only the first sample survives",
    ),
    SyntheticCase(
        _capability("syn-sign-law", "power_law",
                    {"amplitude": 1.0, "exponent": 1.0, "offset": 0.0}, {"x": "x"},
                    [_axis("x", 0.1, 2.0, samples=5)],
                    {"kind": "range", "lower": -1e9, "upper": 1e9}, sign=-1),
        {"samples_in": 0},
        "declared non-positive measure with a positive scalar: nothing is certified",
    ),
    SyntheticCase(
        _capability("syn-two-axis-product", "bilinear", {"amplitude": 1.0},
                    {"x": "x", "y": "y"},
                    [_axis("x", 0.1, 4.0, samples=11, nominal=2.0),
                     _axis("y", 0.5, 8.0, samples=11, nominal=2.0)],
                    {"kind": "abs", "reference": 0.0, "tolerance": 4.0}),
        {"x": 2.0, "y": 2.0},
        "value = x*y <= 4 with the other axis at its nominal: both boundaries at 2",
    ),
    SyntheticCase(
        _capability("syn-singular-exception", "hyperbolic_excess",
                    {"critical": 2.0, "amplitude": 1.0, "epsilon": 1e-9}, {"x": "x"},
                    [_axis("x", 0.5, 3.5, samples=5)],
                    {"kind": "range", "lower": 0.0, "upper": 10.0}),
        {"pathology": "ZeroDivisionError"},
        "an evaluator raising at the singularity is a pathology, never a certified sample",
    ),
    SyntheticCase(
        _capability("syn-quantised-axis", "power_law",
                    {"amplitude": 1.0, "exponent": 2.0, "offset": 0.0}, {"n": "x"},
                    [_axis("n", 2.0, 40.0, samples=13, scale="log", quantum=1.0)],
                    {"kind": "abs", "reference": 0.0, "tolerance": 100.0}),
        {"high": 10.0, "quantum": 1.0},
        "a discrete (quantised) axis resolves its boundary only to one quantum",
    ),
)


def synthetic_registry(cases: tuple[SyntheticCase, ...] = SYNTHETIC_CASES) -> CapabilityRegistry:
    """Registry holding every synthetic case (all ids are data, not code)."""
    return CapabilityRegistry.from_config(
        {"schema_version": 1, "capabilities": [case.capability for case in cases]})


def _boundary_value(report: ProbeReport, axis_name: str, side: str) -> float | None:
    for boundary in report.envelope.axes[axis_name].boundaries:
        if boundary.side == side:
            return boundary.value
    return None


def run_selfcheck(rel_tol: float = 1e-3) -> dict[str, Any]:
    """Probe every synthetic case and compare the detected boundary with the truth."""
    registry = synthetic_registry()
    rows: list[dict[str, Any]] = []
    for case in SYNTHETIC_CASES:
        report = probe_capability(registry.get(case.capability_id), rel_tol=rel_tol)
        axis_spec = registry.get(case.capability_id).axes[0]
        axis_name = axis_spec.name
        envelope = report.envelope.axes[axis_name]
        truth = case.truth
        row: dict[str, Any] = {"capability_id": case.capability_id, "note": case.note,
                               "axis": axis_name,
                               "searched_span": [axis_spec.low, axis_spec.high],
                               "span": axis_spec.width,
                               "samples_in": envelope.samples_in,
                               "samples_total": envelope.samples_total,
                               "intervals": [iv.to_dict() for iv in envelope.intervals],
                               "boundaries": [b.to_dict() for b in envelope.boundaries],
                               "confidence": envelope.confidence, "checks": {}}
        checks = row["checks"]
        passed = True
        if "high" in truth or "low" in truth:
            for side in ("low", "high"):
                if side not in truth:
                    continue
                detected = _boundary_value(report, axis_name, side)
                expected = float(truth[side])
                ok = detected is not None and abs(detected - expected) <= rel_tol * axis_spec.width \
                    or (detected is not None and abs(detected - expected) <= 1e-12 * max(1.0, abs(expected)))
                rel_err = (abs(detected - expected) / abs(expected)) if detected is not None and expected else None
                checks[side] = {"detected": detected, "truth": expected,
                                "abs_error": None if detected is None else abs(detected - expected),
                                "rel_error": rel_err, "passed": bool(ok)}
                passed = passed and bool(ok)
        if "x" in truth and "y" in truth:      # two-axis case
            for axis_name_two in ("x", "y"):
                detected = _boundary_value(report, axis_name_two, "high")
                expected = float(truth[axis_name_two])
                ok = detected is not None and abs(detected - expected) <= rel_tol * registry.get(
                    case.capability_id).axis(axis_name_two).width
                checks[axis_name_two] = {"detected": detected, "truth": expected,
                                         "rel_error": None if not detected else abs(detected - expected) / expected,
                                         "passed": bool(ok)}
                passed = passed and bool(ok)
        if "open" in truth:
            interval = envelope.intervals[0] if envelope.intervals else None
            ok = (interval is not None and (interval.low_open, interval.high_open) == truth["open"]
                  and len(envelope.boundaries) == truth.get("boundaries", 0))
            checks["open"] = {"detected": None if interval is None else
                              [interval.low_open, interval.high_open],
                              "truth": list(truth["open"]), "passed": bool(ok)}
            passed = passed and bool(ok)
        if "intervals" in truth:
            ok = len(envelope.intervals) == truth["intervals"]
            checks["intervals"] = {"detected": len(envelope.intervals),
                                   "truth": truth["intervals"], "passed": bool(ok)}
            passed = passed and bool(ok)
        if "samples_in" in truth:
            ok = envelope.samples_in == truth["samples_in"]
            checks["samples_in"] = {"detected": envelope.samples_in,
                                    "truth": truth["samples_in"], "passed": bool(ok)}
            passed = passed and bool(ok)
        if "law_boundary" in truth:
            count = sum(1 for b in envelope.boundaries if b.source == "law")
            ok = count == truth["law_boundary"]
            checks["law_boundary"] = {"detected": count, "truth": truth["law_boundary"],
                                      "passed": bool(ok)}
            passed = passed and bool(ok)
        if "pathology" in truth:
            hits = [ev for scan in report.scans.values() for ev in scan.evaluations
                    if ev.status == "pathology"
                    and any(truth["pathology"] in reason for reason in ev.reasons)]
            ok = bool(hits)
            checks["pathology"] = {"detected": len(hits), "truth": truth["pathology"],
                                   "passed": bool(ok)}
            passed = passed and bool(ok)
        if "quantum" in truth:
            ok = bool(envelope.boundaries) and all(
                abs(b.width - truth["quantum"]) <= 1e-9 for b in envelope.boundaries)
            checks["quantum"] = {"detected": None if not envelope.boundaries else
                                 envelope.boundaries[0].width,
                                 "truth": truth["quantum"], "passed": bool(ok)}
            passed = passed and bool(ok)
        row["passed"] = passed
        rows.append(row)
    errors = [c["rel_error"] for row in rows for c in row["checks"].values()
              if isinstance(c, dict) and isinstance(c.get("rel_error"), (int, float))]
    return {
        "cases": rows,
        "total": len(rows),
        "passed": sum(1 for row in rows if row["passed"]),
        "max_relative_error": max(errors) if errors else None,
        "rel_tol": rel_tol,
    }


def main() -> int:  # pragma: no cover - CLI sugar
    result = run_selfcheck()
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] == result["total"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())