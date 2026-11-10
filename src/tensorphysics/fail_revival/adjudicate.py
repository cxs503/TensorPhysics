"""Three-way adjudication of a revival run.

A revived measurement gets exactly one of three verdicts:

``PASS``
    comparable (inputs, caliber and code version all reproduce or are covered by
    a reproduction control) *and* the measured deviation is inside the
    registered tolerance taken from the FAIL record itself.
``STILL_FAIL``
    comparable but still outside tolerance -- or outside tolerance and not
    decidable because the registry holds no tolerance (fail-closed: a PASS can
    never be earned without a registered tolerance).
``NOT_COMPARABLE``
    something that matters moved: the input digest, the numerical caliber, or
    the code version without a reproduction control.  The change list is
    mandatory and the outcome stays a *member of the FAIL family* -- it is never
    laundered into a PASS.

Tolerances are read from the registered FAIL record (``allowed``), never from
the revival request, so a revival cannot relax its own bar.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import Adjudication, FailRecord, RevivalTask

ADJUDICATION_SCHEMA = "revival_adjudication.v1"
_NUMBER = None


@dataclass(frozen=True)
class RevivalOutcome:
    """What a revival run reports back (measurement + provenance of the run)."""

    inputs_sha256: str
    caliber_id: str
    code_version: str
    measured: float | None = None
    reproduced_baseline: bool = False
    notes: str = ""
    schema: str = "revival_outcome.v1"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "inputs_sha256": self.inputs_sha256,
            "caliber_id": self.caliber_id,
            "code_version": self.code_version,
            "measured": self.measured,
            "reproduced_baseline": self.reproduced_baseline,
            "notes": self.notes,
        }


def _strip(text: str) -> str:
    return text.replace("*", "").replace("_", "").replace("`", "").strip()


def parse_tolerance(allowed: str | None) -> tuple[str, float] | None:
    """``"<3%"`` -> ``("<", 3.0)``; returns ``None`` when no tolerance is held."""
    if not allowed:
        return None
    import re

    m = re.match(r"^\s*([<≤>≥]=?)\s*([0-9.]+)\s*%?\s*$", _strip(allowed))
    if not m:
        return None
    return m.group(1), float(m.group(2))


def parse_number(text: str | None) -> float | None:
    """First signed number in a cell (``"+219.8%"`` -> ``219.8``)."""
    if text is None:
        return None
    import re

    m = re.search(r"[-+]?[0-9]+(?:\.[0-9]+)?(?:[eE][-+]?[0-9]+)?", _strip(text))
    return float(m.group(0)) if m else None


def _within(measured: float, op: str, tol: float) -> bool:
    v = abs(measured)
    if op.startswith("<") and "=" not in op:
        return v < tol
    if op.startswith("≤") or op.startswith("<="):
        return v <= tol
    if op.startswith(">"):
        return v > tol
    return False


def adjudicate(task: RevivalTask, record: FailRecord, outcome: RevivalOutcome, *,
               baseline_inputs_sha256: str | None = None,
               baseline_caliber: str | None = None,
               baseline_code_version: str | None = None) -> Adjudication:
    """Decide PASS / STILL_FAIL / NOT_COMPARABLE with an explicit change list."""
    changes: list[dict[str, Any]] = []
    reasons: list[str] = []

    baseline_inputs = baseline_inputs_sha256 or task.input_sha256
    if outcome.inputs_sha256 != baseline_inputs:
        changes.append({"field": "inputs", "baseline": baseline_inputs,
                        "revival": outcome.inputs_sha256})
    if baseline_caliber is not None and outcome.caliber_id != baseline_caliber:
        changes.append({"field": "caliber", "baseline": baseline_caliber,
                        "revival": outcome.caliber_id})
    if baseline_code_version is not None and outcome.code_version != baseline_code_version:
        entry = {"field": "code_version", "baseline": baseline_code_version,
                 "revival": outcome.code_version}
        if outcome.reproduced_baseline:
            entry["scoped_by_reproduction"] = True
            reasons.append("code_version_changed_but_baseline_reproduced")
        else:
            changes.append(entry)

    if changes:
        return Adjudication(
            task_key=task.task_key, fail_key=task.fail_key,
            criterion_sha256=task.criterion_sha256, verdict="NOT_COMPARABLE",
            changes=tuple(changes), family_member=True, reasons=tuple(reasons),
        )

    tol = parse_tolerance(record.allowed)
    measured = outcome.measured
    if tol is None:
        return Adjudication(
            task_key=task.task_key, fail_key=task.fail_key,
            criterion_sha256=task.criterion_sha256, verdict="STILL_FAIL",
            changes=(), family_member=True,
            reasons=("tolerance_absent_in_registry",) + tuple(reasons),
        )
    if measured is None:
        return Adjudication(
            task_key=task.task_key, fail_key=task.fail_key,
            criterion_sha256=task.criterion_sha256, verdict="STILL_FAIL",
            changes=(), family_member=True,
            reasons=("no_numeric_measurement",) + tuple(reasons),
        )
    op, value = tol
    ok = _within(measured, op, value)
    verdict = "PASS" if ok else "STILL_FAIL"
    reasons.append("measured=%s tolerance=%s%s" % (measured, op, value))
    return Adjudication(
        task_key=task.task_key, fail_key=task.fail_key,
        criterion_sha256=task.criterion_sha256, verdict=verdict,
        changes=(), family_member=not ok, reasons=tuple(reasons),
    )


def family_status(adjudications: list[Adjudication]) -> dict[str, Any]:
    """A FAIL stays in the family unless a comparable PASS closes it."""
    members = [a for a in adjudications if a.verdict != "PASS"]
    return {
        "adjudications": len(adjudications),
        "family_members": len(members),
        "not_comparable": sum(1 for a in adjudications if a.verdict == "NOT_COMPARABLE"),
        "still_fail": sum(1 for a in adjudications if a.verdict == "STILL_FAIL"),
        "pass": sum(1 for a in adjudications if a.verdict == "PASS"),
        "family_keys": sorted({a.fail_key for a in members}),
    }