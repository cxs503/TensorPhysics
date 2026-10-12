"""Versioned research-loop contracts shared by agents and solver workflows.

These contracts intentionally describe intent and evidence, not solver internals.
Schema version 1 is JSON-compatible and rejects non-finite or malformed values.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math
from typing import Any, Mapping


class ResearchContractError(ValueError):
    """A research goal, experiment plan, or validation report is invalid."""


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ResearchContractError(f"{name} must be a non-empty string")
    return value.strip()


def _json_safe(value: Any, name: str) -> None:
    try:
        json.dumps(value, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ResearchContractError(f"{name} must be finite JSON-compatible data: {exc}") from exc


@dataclass(frozen=True)
class ResearchGoal:
    """Human-owned objective and explicit acceptance/compute boundaries."""

    goal_id: str
    objective: str
    acceptance_criteria: tuple[str, ...]
    constraints: Mapping[str, Any] = field(default_factory=dict)
    budget: Mapping[str, Any] = field(default_factory=dict)
    approval_required_for: tuple[str, ...] = ("external_compute", "destructive_action")
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.goal_id, "goal_id")
        _nonempty(self.objective, "objective")
        if not self.acceptance_criteria or any(not isinstance(x, str) or not x.strip() for x in self.acceptance_criteria):
            raise ResearchContractError("acceptance_criteria must contain non-empty strings")
        if self.schema_version != 1:
            raise ResearchContractError("unsupported ResearchGoal schema_version")
        _json_safe(self.to_dict(), "ResearchGoal")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentPlan:
    """Bounded, reproducible plan for one solver execution."""

    plan_id: str
    goal_id: str
    solver: str
    parameters: Mapping[str, Any]
    validation_checks: tuple[str, ...]
    max_runs: int = 1
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("plan_id", "goal_id", "solver"):
            _nonempty(getattr(self, name), name)
        if not self.validation_checks or any(not isinstance(x, str) or not x.strip() for x in self.validation_checks):
            raise ResearchContractError("validation_checks must contain non-empty strings")
        if not isinstance(self.max_runs, int) or isinstance(self.max_runs, bool) or self.max_runs < 1:
            raise ResearchContractError("max_runs must be a positive integer")
        if self.schema_version != 1:
            raise ResearchContractError("unsupported ExperimentPlan schema_version")
        _json_safe(self.to_dict(), "ExperimentPlan")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ValidationReport:
    """Independent checks and evidence produced after a solver returns."""

    run_id: str
    status: str
    checks: tuple[Mapping[str, Any], ...]
    summary: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        if self.status not in {"passed", "failed", "incomplete"}:
            raise ResearchContractError("status must be passed, failed, or incomplete")
        if not self.checks:
            raise ResearchContractError("checks must contain at least one check")
        if not isinstance(self.summary, str) or not self.summary.strip():
            raise ResearchContractError("summary must be a non-empty string")
        for index, check in enumerate(self.checks):
            if not isinstance(check, Mapping) or not isinstance(check.get("name"), str) or not check["name"].strip():
                raise ResearchContractError(f"checks[{index}] requires a name")
            if not isinstance(check.get("passed"), bool):
                raise ResearchContractError(f"checks[{index}].passed must be boolean")
        if self.status == "passed" and not all(check["passed"] for check in self.checks):
            raise ResearchContractError("a passed report cannot contain failed checks")
        if self.schema_version != 1:
            raise ResearchContractError("unsupported ValidationReport schema_version")
        _json_safe(self.to_dict(), "ValidationReport")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ResearchMemoryRecord:
    """Compact evidence-backed record for a future research iteration."""

    goal_id: str
    run_id: str
    finding: str
    evidence_uris: tuple[str, ...]
    next_action: str
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name in ("goal_id", "run_id", "finding", "next_action"):
            _nonempty(getattr(self, name), name)
        if any(not isinstance(uri, str) or not uri.strip() for uri in self.evidence_uris):
            raise ResearchContractError("evidence_uris must contain non-empty URI strings")
        if self.schema_version != 1:
            raise ResearchContractError("unsupported ResearchMemoryRecord schema_version")
        _json_safe(self.to_dict(), "ResearchMemoryRecord")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
