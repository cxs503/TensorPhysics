"""Negative-control loader for the criterion-degeneracy classifier.

The control set is a data file, not code: ``data/negative_controls.json`` holds

* 12 **degenerate** criteria (4 per type T1/T2/T3) that MUST be detected, each with the
  numeric witness that makes it degenerate, and
* 12 **benign** criteria that MUST NOT be flagged — including the tricky legitimate cases
  (1e-6/1e-9 consistency, a zero reference whose measurement is also zero, tiny magnitudes
  with a well-posed ratio, and a one-sided engineering bound).

The two assertions are separate on purpose: a false *detection* (benign flagged) and a false
*verdict* (benign judged FAIL/WITHHELD) are different failures and are reported separately.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from .adaptive import evaluate
from .degeneracy import classify
from .model import Criterion, Degeneracy, Verdict
from .parser import criterion_from_dict

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data",
                         "negative_controls.json")


@dataclass
class ControlResult:
    id: str
    group: str            # "degenerate" | "benign"
    expected: str         # expected degeneracy label
    detected: str
    verdict: str
    detected_ok: bool
    verdict_ok: bool
    rationale: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "group": self.group, "expected": self.expected,
                "detected": self.detected, "verdict": self.verdict,
                "detected_ok": self.detected_ok, "verdict_ok": self.verdict_ok,
                "rationale": self.rationale}


def load_controls(path: str = DATA_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def run_controls(path: str = DATA_PATH) -> dict[str, Any]:
    """Run the whole control set; return per-sample results and the two acceptance rates."""
    data = load_controls(path)
    results: list[ControlResult] = []
    for group in ("degenerate", "benign"):
        for item in data.get(group, []):
            row = criterion_from_dict(item)
            dg = classify(row.criterion)
            ev = evaluate(row.criterion, degeneracy=dg)
            expected = item["expect"]
            detected = dg.kind.value
            detected_ok = (detected == expected) if group == "degenerate" \
                else (dg.kind is Degeneracy.NONE)
            expect_verdict = item.get("expect_verdict")
            verdict_ok = (ev.verdict.value == expect_verdict) if expect_verdict \
                else (ev.verdict is not Verdict.PASS)
            results.append(ControlResult(id=item["id"], group=group, expected=expected,
                                         detected=detected, verdict=ev.verdict.value,
                                         detected_ok=bool(detected_ok),
                                         verdict_ok=bool(verdict_ok),
                                         rationale=item.get("rationale", "")))
    deg = [r for r in results if r.group == "degenerate"]
    ben = [r for r in results if r.group == "benign"]
    return {
        "n_degenerate": len(deg),
        "n_benign": len(ben),
        "degenerate_detected": sum(1 for r in deg if r.detected_ok),
        "benign_false_positive": sum(1 for r in ben if not r.detected_ok),
        "benign_verdict_ok": sum(1 for r in ben if r.verdict_ok),
        "degenerate_verdict_ok": sum(1 for r in deg if r.verdict_ok),
        "results": results,
    }


def controls_report(path: str = DATA_PATH) -> dict[str, Any]:
    out = run_controls(path)
    return {k: v for k, v in out.items() if k != "results"} | {
        "results": [r.as_dict() for r in out["results"]]}