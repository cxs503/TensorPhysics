"""Typed records of the honest-FAIL revival registry.

Every object that travels through the registry is canonicalised and hashed, so
that a later revival run can be *refuted* against the bytes that produced it.
Nothing here knows an engineering domain: fields are structural, values are
data supplied by extraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
import hashlib
import json
import re
from typing import Any, Iterable, Mapping, Sequence

RECORD_SCHEMA = "fail_record.v1"
BLOCKER_SCHEMA = "fail_blocker.v1"
TASK_SCHEMA = "revival_task.v1"
LINEAGE_SCHEMA = "revival_lineage.v1"
VERDICTS = ("PASS", "STILL_FAIL", "NOT_COMPARABLE")
BLOCKER_KINDS = ("capability_slug", "upstream_mission", "numerical_caliber_change",
                 "undeclared")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON bytes (sorted keys, no NaN) used for every digest."""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_text(canonical_json(obj))


def normalise_text(text: str) -> str:
    """Whitespace-normalise while keeping the wording byte-identical else."""
    return re.sub(r"\s+", " ", text).strip()


def is_hex64(value: str | None) -> bool:
    return bool(value and _HEX64.match(value))


@dataclass(frozen=True)
class FailRecord:
    """One honest-FAIL observation lifted out of a node acceptance artifact."""

    fail_id: str
    mission_id: int
    plan_id: int | None
    report_id: int | None
    report_sha256: str
    metric: str
    verdict: str
    criterion_text: str
    criterion_sha256: str
    measured: str | None
    reference: str | None
    allowed: str | None
    error: str | None
    attribution: str
    attribution_basis: str
    cells: tuple[str, ...]
    table_header: tuple[str, ...]
    line_no: int
    verdict_source: str
    allowed_source: str = "column"
    schema: str = RECORD_SCHEMA

    def key(self) -> str:
        """Stable identity: mission + criterion digest + source line."""
        return "%d:%s:%d" % (self.mission_id, self.criterion_sha256, self.line_no)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["cells"] = list(self.cells)
        d["table_header"] = list(self.table_header)
        d["key"] = self.key()
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "FailRecord":
        return cls(
            fail_id=d["fail_id"],
            mission_id=int(d["mission_id"]),
            plan_id=None if d.get("plan_id") is None else int(d["plan_id"]),
            report_id=None if d.get("report_id") is None else int(d["report_id"]),
            report_sha256=d["report_sha256"],
            metric=d["metric"],
            verdict=d["verdict"],
            criterion_text=d["criterion_text"],
            criterion_sha256=d["criterion_sha256"],
            measured=d.get("measured"),
            reference=d.get("reference"),
            allowed=d.get("allowed"),
            error=d.get("error"),
            attribution=d["attribution"],
            attribution_basis=d["attribution_basis"],
            cells=tuple(d.get("cells") or ()),
            table_header=tuple(d.get("table_header") or ()),
            line_no=int(d["line_no"]),
            verdict_source=d.get("verdict_source", "unknown"),
            allowed_source=d.get("allowed_source", "column"),
            schema=d.get("schema", RECORD_SCHEMA),
        )


@dataclass(frozen=True)
class Blocker:
    """A declared obstruction that keeps a FAIL from being re-measured."""

    blocker_id: str
    kind: str
    fail_keys: tuple[str, ...]
    note: str = ""
    schema: str = BLOCKER_SCHEMA

    def __post_init__(self) -> None:
        if self.kind not in BLOCKER_KINDS:
            raise ValueError("unknown blocker kind: %r" % (self.kind,))

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["fail_keys"] = list(self.fail_keys)
        return d

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Blocker":
        return cls(
            blocker_id=d["blocker_id"],
            kind=d["kind"],
            fail_keys=tuple(d.get("fail_keys") or ()),
            note=d.get("note", ""),
            schema=d.get("schema", BLOCKER_SCHEMA),
        )


@dataclass(frozen=True)
class RevivalRequest:
    """A request to re-run one registered FAIL.

    The body is deliberately minimal: the criterion digest and the input digest
    only.  There is no field for a threshold, a tolerance, a metric name or a
    verdict -- the zero-rewrite gate rejects anything else.
    """

    fail_key: str
    criterion_sha256: str
    input_sha256: str
    registry_criterion_sha256: str | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    ALLOWED = ("fail_key", "criterion_sha256", "input_sha256", "registry_criterion_sha256")

    def forbidden_fields(self) -> tuple[str, ...]:
        return tuple(sorted(k for k in self.extra if k not in self.ALLOWED))


def task_key(criterion_sha256: str, input_sha256: str) -> str:
    """Idempotency key: one revival task per (criterion digest, input digest)."""
    return sha256_text("revival.v1|" + criterion_sha256 + "|" + input_sha256)


def sort_records(records: Iterable[FailRecord]) -> list[FailRecord]:
    return sorted(records, key=lambda r: (r.mission_id, r.line_no, r.criterion_sha256))


@dataclass(frozen=True)
class RevivalTask:
    task_key: str
    fail_key: str
    criterion_sha256: str
    input_sha256: str
    blocker_id: str
    blocker_kind: str
    mission_id: int
    cycle: int
    schema: str = TASK_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Adjudication:
    """Three-way verdict of a revival run plus the retained FAIL-family status."""

    task_key: str
    fail_key: str
    criterion_sha256: str
    verdict: str
    changes: tuple[dict[str, Any], ...] = ()
    family_member: bool = True
    reasons: tuple[str, ...] = ()
    schema: str = "revival_adjudication.v1"

    def __post_init__(self) -> None:
        if self.verdict not in VERDICTS:
            raise ValueError("unknown verdict: %r" % (self.verdict,))
        if self.verdict == "NOT_COMPARABLE" and not self.changes:
            raise ValueError("NOT_COMPARABLE requires an explicit change list")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["changes"] = [dict(c) for c in self.changes]
        d["reasons"] = list(self.reasons)
        return d


def summarize(records: Sequence[FailRecord]) -> dict[str, Any]:
    by_mission: dict[int, int] = {}
    by_attr: dict[str, int] = {}
    for r in records:
        by_mission[r.mission_id] = by_mission.get(r.mission_id, 0) + 1
        by_attr[r.attribution] = by_attr.get(r.attribution, 0) + 1
    return {
        "count": len(records),
        "by_mission": {str(k): v for k, v in sorted(by_mission.items())},
        "by_attribution": dict(sorted(by_attr.items())),
        "missions": sorted(by_mission),
    }