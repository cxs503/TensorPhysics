"""KPI view over criterion pre-registration + drift audits.

Turns a list of :class:`~tensorphysics.campaign.criterion_prereg.drift_audit.DriftReport`
into a single machine-readable scorecard:

* ``prereg_coverage`` -- share of audited nodes whose frozen manifest verifies
  as ``registered`` (pre-freeze actually happened);
* ``audit_pass_rate`` -- share of audits returning ``PASS``;
* ``drift_rate`` -- drifting criteria / compared criteria;
* ``class_histogram`` / ``reason_histogram`` -- which movement dominates, so a
  systematic loosening trend is visible;
* ``fail_closed_incidents`` -- ``unregistered`` + ``tampered`` occurrences
  (these are *not* drift, they are integrity failures and are counted apart).

Zero domain semantics: everything is derived from report fields.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

from .drift_audit import DriftClass, DriftReport, Verdict
from .criterion_manifest import REGISTERED


def _as_dict(report: DriftReport | Mapping[str, Any]) -> dict[str, Any]:
    return report.to_dict() if isinstance(report, DriftReport) else dict(report)


def kpi_view(reports: Iterable[DriftReport | Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate audits into a KPI scorecard."""
    rows = [_as_dict(r) for r in reports]
    n_audits = len(rows)
    states = Counter(r.get("state", "") for r in rows)
    verdicts = Counter(r.get("verdict", "") for r in rows)
    classes = Counter()
    reasons = Counter()
    n_items = 0
    n_drift = 0
    nodes: dict[str, list[str]] = {}
    for r in rows:
        n_items += int(r.get("n_items", 0))
        n_drift += int(r.get("n_drift", 0))
        for item in r.get("items", []):
            classes[item.get("class", "")] += 1
            reasons[item.get("reason", "")] += 1
        node = str(r.get("node_id") or "")
        if node:
            nodes.setdefault(node, []).append(str(r.get("verdict", "")))

    for cls in DriftClass:
        classes.setdefault(cls.value, 0)
    n_pass = verdicts.get(Verdict.PASS.value, 0)
    n_registered = states.get(REGISTERED, 0)
    fail_closed = sum(v for k, v in states.items() if k and k != REGISTERED)
    return {
        "schema": "criterion-prereg-kpi/1",
        "n_audits": n_audits,
        "n_registered": n_registered,
        "prereg_coverage": (n_registered / n_audits) if n_audits else 0.0,
        "n_pass": n_pass,
        "audit_pass_rate": (n_pass / n_audits) if n_audits else 0.0,
        "n_items": n_items,
        "n_drift": n_drift,
        "drift_rate": (n_drift / n_items) if n_items else 0.0,
        "class_histogram": dict(sorted(classes.items())),
        "reason_histogram": dict(sorted(reasons.items())),
        "state_histogram": dict(sorted(states.items())),
        "verdict_histogram": dict(sorted(verdicts.items())),
        "fail_closed_incidents": fail_closed,
        "fail_closed_rate": (fail_closed / n_audits) if n_audits else 0.0,
        "nodes": {k: {"n": len(v), "n_fail": sum(1 for x in v if x == Verdict.FAIL.value)} for k, v in sorted(nodes.items())},
    }


__all__ = ["kpi_view"]