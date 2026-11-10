"""Pre-registration manifest handling and the machine judge for its criteria.

The manifest fixes, *before* any production run, (i) the number of seeds and
ensembles, (ii) the physical cases, and (iii) the numeric thresholds of every
acceptance criterion.  ``validate`` then compares a run's results against that
manifest and refuses to certify it when

* a pre-registered criterion has no measured value (``MISSING_METRIC``),
* the results claim a criterion that was never registered
  (``UNREGISTERED_CRITERION`` -- stops post-hoc cherry-picking), or
* the manifest hash recorded in the results does not match the manifest handed
  to the validator (``MANIFEST_MISMATCH`` -- the thresholds were edited after
  the run).

Only ``PASS`` certifies the node; ``INVALID`` and ``FAIL`` are both failures.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import ConfigError, Manifest, manifest_hash

OPS = {
    ">=": lambda a, b: a >= b,
    "<=": lambda a, b: a <= b,
    ">": lambda a, b: a > b,
    "<": lambda a, b: a < b,
    "==": lambda a, b: a == b,
}


def _lookup(results: Mapping[str, Any], path: str) -> Any:
    node: Any = results
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise KeyError(path)
        node = node[part]
    return node


@dataclass(frozen=True)
class CriterionCheck:
    criterion_id: str
    metric: str
    op: str
    threshold: float
    value: Any
    passed: bool
    note: str = ""

    def to_dict(self) -> dict:
        value = self.value
        if isinstance(value, (int, float)):
            value = float(value)
        return {"id": self.criterion_id, "metric": self.metric, "op": self.op,
                "threshold": float(self.threshold), "value": value,
                "passed": bool(self.passed), "note": self.note}


def build_criteria(manifest: Manifest) -> tuple:
    criteria = []
    for entry in manifest.criteria:
        for key in ("id", "metric", "op", "threshold"):
            if key not in entry:
                raise ConfigError(f"criterion entry missing {key!r}: {entry}")
        if entry["op"] not in OPS:
            raise ConfigError(f"unknown operator {entry['op']!r} in criterion {entry['id']}")
        criteria.append(entry)
    return tuple(criteria)


def validate(manifest: Manifest, results: Mapping[str, Any]) -> dict:
    """Judge ``results`` against the pre-registered ``manifest``."""
    criteria = build_criteria(manifest)
    registered = {entry["id"] for entry in criteria}
    checks: list[CriterionCheck] = []
    problems: list[dict] = []

    expected_hash = manifest_hash(manifest.payload)
    reported_hash = results.get("manifest_sha256")
    hash_ok = reported_hash == expected_hash
    if not hash_ok:
        problems.append({"code": "MANIFEST_MISMATCH",
                         "detail": f"results={reported_hash!r} manifest={expected_hash!r}"})

    claimed: Sequence[str] = results.get("criterion_ids", sorted(registered))
    for cid in claimed:
        if cid not in registered:
            problems.append({"code": "UNREGISTERED_CRITERION", "detail": str(cid)})

    for entry in criteria:
        try:
            value = _lookup(results, entry["metric"])
        except KeyError:
            problems.append({"code": "MISSING_METRIC",
                             "detail": f"{entry['id']}: {entry['metric']}"})
            checks.append(CriterionCheck(entry["id"], entry["metric"], entry["op"],
                                         float(entry["threshold"]), None, False,
                                         note="metric not present in results"))
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append({"code": "NON_NUMERIC_METRIC",
                             "detail": f"{entry['id']}: {entry['metric']}={value!r}"})
            checks.append(CriterionCheck(entry["id"], entry["metric"], entry["op"],
                                         float(entry["threshold"]), value, False,
                                         note="metric is not numeric"))
            continue
        passed = bool(OPS[entry["op"]](float(value), float(entry["threshold"])))
        checks.append(CriterionCheck(entry["id"], entry["metric"], entry["op"],
                                     float(entry["threshold"]), float(value), passed,
                                     note=str(entry.get("description", ""))))

    all_pass = all(check.passed for check in checks) and not problems
    verdict = "PASS" if all_pass else ("INVALID" if problems else "FAIL")
    return {
        "node": manifest.node,
        "manifest_source": manifest.source,
        "manifest_sha256": expected_hash,
        "reported_manifest_sha256": reported_hash,
        "problems": problems,
        "checks": [check.to_dict() for check in checks],
        "all_pass": bool(all_pass),
        "verdict": verdict,
    }


def summarize(verdict: Mapping[str, Any]) -> str:
    lines = [f"{c['id']}: {c['metric']} = {c['value']} {c['op']} {c['threshold']} "
             f"-> {'PASS' if c['passed'] else 'FAIL'}" for c in verdict["checks"]]
    header = f"{verdict['node']} verdict = {verdict['verdict']} " \
             f"(manifest {verdict['manifest_sha256'][:12]})"
    if verdict["problems"]:
        header += " | problems: " + ", ".join(p["code"] for p in verdict["problems"])
    return "\n".join([header] + lines)


def write_verdict(verdict: Mapping[str, Any], path) -> None:
    from pathlib import Path

    Path(path).write_text(json.dumps(verdict, indent=2, sort_keys=True) + "\n",
                          encoding="utf-8")


__all__ = ["CriterionCheck", "build_criteria", "summarize", "validate", "write_verdict"]