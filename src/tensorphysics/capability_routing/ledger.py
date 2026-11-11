"""Capability ledger: grade index over (capability, solver) pairs.

The ledger is *data*.  It is loaded from a JSON mapping (file or dict) and
validated fail-closed at load time, so a malformed ledger can never silently
widen what the router is allowed to run:

* unknown keys are rejected (a misspelled ``evidance`` cannot default to REAL);
* the schema version must match exactly;
* ``solver_id`` / ``capability_id`` must be non-empty strings;
* the same (capability, solver) pair may appear only once;
* every grade string must parse into :class:`EvidenceLevel`.

Lookups are pre-indexed at load time (one dict probe per query) so that the
evidence gate can be inserted in front of routing within the latency budget.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from .evidence import EvidenceLevel, UnknownEvidenceLevel

__all__ = [
    "LedgerError",
    "SolverRecord",
    "CapabilityLedger",
    "SCHEMA_VERSION",
    "ROW_KEYS",
]

SCHEMA_VERSION = 1
ROW_KEYS = ("capability_id", "solver_id", "evidence", "available", "priority", "note")
CONFIG_KEYS = ("schema_version", "rows")


class LedgerError(ValueError):
    """The ledger payload is not interpretable; routing must not start."""


@dataclass(frozen=True)
class SolverRecord:
    """One implementation of one capability, with its evidence grade."""

    capability_id: str
    solver_id: str
    level: EvidenceLevel
    available: bool = True
    priority: int = 0
    note: str = ""

    @property
    def key(self) -> tuple[str, str]:
        return (self.capability_id, self.solver_id)

    def as_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "solver_id": self.solver_id,
            "evidence": self.level.value,
            "available": self.available,
            "priority": self.priority,
            "note": self.note,
        }


def _require_text(value, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise LedgerError(f"field {field!r} must be a non-empty string, got {value!r}")
    return value.strip()


class CapabilityLedger:
    """Immutable, pre-indexed evidence index over capability implementations."""

    def __init__(self, rows: Iterable[SolverRecord]) -> None:
        records: dict[tuple[str, str], SolverRecord] = {}
        by_capability: dict[str, list[SolverRecord]] = {}
        for record in rows:
            if record.key in records:
                raise LedgerError(f"duplicate ledger row for {record.key}")
            records[record.key] = record
            by_capability.setdefault(record.capability_id, []).append(record)
        if not records:
            raise LedgerError("ledger is empty: routing would have no evidence to gate on")
        self._records = records
        self._by_capability = {
            cap: tuple(sorted(items, key=lambda r: (-r.level.rank, r.priority, r.solver_id)))
            for cap, items in by_capability.items()
        }
        canonical = json.dumps(
            sorted([k[0], k[1], r.level.value, r.available, r.priority]
                   for k, r in records.items()),
            separators=(",", ":"),
        )
        self._digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()

    # ---------------------------------------------------------------- loading
    @classmethod
    def from_rows(cls, rows: Iterable[Mapping]) -> "CapabilityLedger":
        parsed: list[SolverRecord] = []
        for index, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise LedgerError(f"ledger row #{index} must be a mapping")
            unknown = set(row) - set(ROW_KEYS)
            if unknown:
                raise LedgerError(
                    f"ledger row #{index} has unknown keys {sorted(unknown)}; "
                    f"allowed: {list(ROW_KEYS)}"
                )
            try:
                level = EvidenceLevel.parse(row.get("evidence"))
            except UnknownEvidenceLevel as exc:
                raise LedgerError(f"ledger row #{index}: {exc}") from None
            available = row.get("available", True)
            if not isinstance(available, bool):
                raise LedgerError(f"ledger row #{index}: 'available' must be a bool")
            priority = row.get("priority", 0)
            if isinstance(priority, bool) or not isinstance(priority, int):
                raise LedgerError(f"ledger row #{index}: 'priority' must be an int")
            note = row.get("note", "")
            if not isinstance(note, str):
                raise LedgerError(f"ledger row #{index}: 'note' must be a string")
            parsed.append(
                SolverRecord(
                    capability_id=_require_text(row.get("capability_id"),
                                                f"rows[{index}].capability_id"),
                    solver_id=_require_text(row.get("solver_id"), f"rows[{index}].solver_id"),
                    level=level,
                    available=available,
                    priority=priority,
                    note=note,
                )
            )
        return cls(parsed)

    @classmethod
    def from_config(cls, config: Mapping) -> "CapabilityLedger":
        if not isinstance(config, Mapping):
            raise LedgerError("ledger config must be a mapping")
        unknown = set(config) - set(CONFIG_KEYS)
        if unknown:
            raise LedgerError(
                f"ledger config has unknown keys {sorted(unknown)}; allowed: {list(CONFIG_KEYS)}"
            )
        version = config.get("schema_version")
        if version != SCHEMA_VERSION:
            raise LedgerError(
                f"unsupported ledger schema_version {version!r}; expected {SCHEMA_VERSION}"
            )
        rows = config.get("rows")
        if not isinstance(rows, list) or not rows:
            raise LedgerError("ledger config 'rows' must be a non-empty list")
        return cls.from_rows(rows)

    @classmethod
    def from_path(cls, path: "str | Path") -> "CapabilityLedger":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_config(payload)

    # ---------------------------------------------------------------- queries
    def capabilities(self) -> tuple[str, ...]:
        return tuple(sorted(self._by_capability))

    def records(self) -> tuple[SolverRecord, ...]:
        return tuple(self._records[key] for key in sorted(self._records))

    def record(self, capability_id: str, solver_id: str) -> "SolverRecord | None":
        """Constant-time graded lookup; ``None`` means "no evidence on file"."""
        return self._records.get((capability_id, solver_id))

    def candidates(self, capability_id: str) -> tuple[SolverRecord, ...]:
        """All implementations of a capability, best evidence first."""
        return self._by_capability.get(capability_id, ())

    def levels(self, capability_id: str) -> tuple[tuple[str, str], ...]:
        """``((solver_id, grade), ...)`` in evidence order (audit-friendly)."""
        return tuple((r.solver_id, r.level.value) for r in self.candidates(capability_id))

    def ranked(self, capability_id: str,
               unavailable: Iterable[str] = ()) -> tuple[SolverRecord, ...]:
        """Evidence-ordered candidates minus the ones declared unavailable."""
        excluded = {unavailable} if isinstance(unavailable, str) else set(unavailable)
        return tuple(r for r in self.candidates(capability_id) if r.solver_id not in excluded)

    def as_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "rows": [r.as_dict() for r in self.records()],
        }

    def fingerprint(self) -> str:
        """Stable content digest (sha256): pins a decision to its ledger version."""
        return self._digest

    def __len__(self) -> int:
        return len(self._records)

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"CapabilityLedger(rows={len(self._records)}, "
            f"capabilities={len(self._by_capability)})"
        )