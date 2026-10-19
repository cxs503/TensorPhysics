"""Deterministic, fail-closed policy for bounded scientific research loops.

The policy does not decide whether a scientific claim is true in general. It
checks whether a declared set of machine-verifiable gates passed and returns a
bounded next action. Domain-specific validators must produce the checks.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

from .research_contracts import ResearchContractError, ValidationReport

ACTIONS = frozenset({
    "accept_validated_result",
    "run_next_bounded_experiment",
    "stop_and_repair",
    "stop_budget_exhausted",
    "escalate_to_stronger_reasoner",
})


@dataclass(frozen=True)
class ResearchDecision:
    """Auditable policy output; every decision records its reasons."""

    action: str
    run_id: str
    reasons: tuple[str, ...]
    passed_required_checks: tuple[str, ...]
    missing_required_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.action not in ACTIONS:
            raise ResearchContractError(f"unsupported research action: {self.action}")
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ResearchContractError("run_id must be a non-empty string")
        if not self.reasons or any(not isinstance(x, str) or not x.strip() for x in self.reasons):
            raise ResearchContractError("decision reasons must be non-empty strings")
        if self.schema_version != 1:
            raise ResearchContractError("unsupported ResearchDecision schema_version")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def decide_next_action(
    report: ValidationReport,
    *,
    required_checks: Iterable[str],
    runs_completed: int,
    max_runs: int,
    budget_exhausted: bool = False,
    ambiguous_or_novel_failure: bool = False,
) -> ResearchDecision:
    """Choose a bounded next action from independently produced validation.

    Safety properties:
    - A process exit or an empty check list cannot count as scientific success.
    - Every required check must be present and pass before a result is accepted.
    - Missing/failed checks fail closed.
    - An ambiguous or novel failure is escalated, not blindly retried.
    - Retry decisions never exceed the declared run budget.
    """
    required = tuple(dict.fromkeys(
        x.strip() for x in required_checks
        if isinstance(x, str) and x.strip()
    ))
    if not required:
        raise ResearchContractError("required_checks must contain at least one non-empty check name")
    if isinstance(runs_completed, bool) or not isinstance(runs_completed, int) or runs_completed < 0:
        raise ResearchContractError("runs_completed must be a non-negative integer")
    if isinstance(max_runs, bool) or not isinstance(max_runs, int) or max_runs < 1:
        raise ResearchContractError("max_runs must be a positive integer")

    check_map: dict[str, bool] = {}
    for check in report.checks:
        name = check.get("name")
        if isinstance(name, str) and name.strip():
            # Duplicate names are ambiguous: treat the gate as failed.
            normalized = name.strip()
            passed = check.get("passed") is True
            check_map[normalized] = check_map.get(normalized, True) and passed

    missing = tuple(name for name in required if name not in check_map)
    failed = tuple(name for name in required if name in check_map and not check_map[name])
    passed = tuple(name for name in required if name in check_map and check_map[name])

    if report.status != "passed" or missing or failed:
        reasons = []
        if report.status != "passed":
            reasons.append(f"validation report status is {report.status!r}, not 'passed'")
        if missing:
            reasons.append("required checks missing: " + ", ".join(missing))
        if failed:
            reasons.append("required checks failed: " + ", ".join(failed))
        if ambiguous_or_novel_failure:
            return ResearchDecision(
                "escalate_to_stronger_reasoner", report.run_id, tuple(reasons) + (
                    "failure is ambiguous or novel; do not retry without diagnosis",
                ), passed, missing, failed,
            )
        if budget_exhausted or runs_completed >= max_runs:
            return ResearchDecision(
                "stop_budget_exhausted", report.run_id, tuple(reasons) + (
                    "declared run budget is exhausted",
                ), passed, missing, failed,
            )
        return ResearchDecision(
            "stop_and_repair", report.run_id, tuple(reasons) + (
                "repair the failed or missing gate before another run",
            ), passed, missing, failed,
        )

    if budget_exhausted or runs_completed >= max_runs:
        return ResearchDecision(
            "accept_validated_result", report.run_id,
            ("all required independent checks passed", "no further runs permitted or needed by this policy"),
            passed, missing, failed,
        )

    return ResearchDecision(
        "run_next_bounded_experiment", report.run_id,
        ("all required independent checks passed", "additional runs remain within the declared budget"),
        passed, missing, failed,
    )
