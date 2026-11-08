"""Evidence ledger for the P27-W2 campaign (criterion d).

Every campaign node must be traceable: a ``result.json`` (or an explicit
honest-FAIL record), at least one artifact path, and the commit hash of
the branch that produced it.  The ledger is a pure aggregation over
per-node evidence entries so it can be unit-tested without I/O.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EvidenceEntry:
    """Per-node evidence record."""

    node_id: int
    mission_id: int | None
    name: str
    status: str
    result_json: str | None = None
    artifacts: list[str] = field(default_factory=list)
    commit: str | None = None
    notes: str = ""

    @property
    def complete(self) -> bool:
        return (
            self.result_json is not None
            and bool(self.artifacts)
            and bool(self.commit)
        )

    def missing_fields(self) -> list[str]:
        missing = []
        if self.result_json is None:
            missing.append("result_json")
        if not self.artifacts:
            missing.append("artifacts")
        if not self.commit:
            missing.append("commit")
        return missing

    def as_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "mission_id": self.mission_id,
            "name": self.name,
            "status": self.status,
            "result_json": self.result_json,
            "artifacts": list(self.artifacts),
            "commit": self.commit,
            "complete": self.complete,
            "missing_fields": self.missing_fields(),
            "notes": self.notes,
        }


@dataclass
class EvidenceLedger:
    """Aggregated evidence ledger over all campaign nodes."""

    entries: list[EvidenceEntry] = field(default_factory=list)

    @property
    def completeness_rate(self) -> float:
        if not self.entries:
            return 0.0
        return sum(1 for e in self.entries if e.complete) / len(self.entries)

    @property
    def complete(self) -> bool:
        return bool(self.entries) and all(e.complete for e in self.entries)

    def incomplete(self) -> list[EvidenceEntry]:
        return [e for e in self.entries if not e.complete]

    def as_dict(self) -> dict:
        return {
            "total_nodes": len(self.entries),
            "complete_nodes": sum(1 for e in self.entries if e.complete),
            "completeness_rate": self.completeness_rate,
            "complete": self.complete,
            "entries": [e.as_dict() for e in self.entries],
        }


def build_evidence_ledger(entries: list[EvidenceEntry]) -> EvidenceLedger:
    """Build a ledger; rejects duplicate node ids."""
    seen: set[int] = set()
    for e in entries:
        if e.node_id in seen:
            raise ValueError(f"duplicate evidence entry for node {e.node_id}")
        seen.add(e.node_id)
    return EvidenceLedger(entries=list(entries))


def entry_from_mission(node: dict, mission: dict | None,
                       commit: str | None = None) -> EvidenceEntry:
    """Build an :class:`EvidenceEntry` from a campaign node + mission dict.

    ``mission`` may be ``None`` (mission not yet claimed); the entry is
    then incomplete by construction, which the gate reports honestly.
    """
    result = (mission or {}).get("result") or {}
    ev = result.get("evidence") or {}
    result_json = (
        result.get("result_json")
        or result.get("report_path")
        or (f"mission:{(mission or {}).get('id')}" if result else None)
    )
    artifacts = list(result.get("artifacts") or [])
    if not artifacts:
        artifacts = list(ev.get("on_disk_changes") or [])
    if not artifacts:
        artifacts = list(ev.get("changed_files") or [])
    if not artifacts and result.get("artifacts_dir"):
        artifacts = [str(result["artifacts_dir"])]
    notes = ""
    if not mission:
        notes = "mission not yet claimed"
    elif not result:
        notes = "mission in flight; no result yet"
    return EvidenceEntry(
        node_id=int(node["id"]),
        mission_id=mission.get("id") if mission else node.get("mission_id"),
        name=str(node.get("name", "")),
        status=str(node.get("status", "queued")),
        result_json=result_json,
        artifacts=artifacts,
        commit=commit,
        notes=notes,
    )


def ledger_from_bundle(bundle: dict[str, Any]) -> EvidenceLedger:
    """Build a ledger from a JSON-serializable bundle.

    ``bundle`` maps ``str(node_id)`` to a dict with optional keys
    ``mission_id``, ``name``, ``status``, ``result_json``, ``artifacts``,
    ``commit``, ``notes``.
    """
    entries = [
        EvidenceEntry(
            node_id=int(k),
            mission_id=v.get("mission_id"),
            name=str(v.get("name", "")),
            status=str(v.get("status", "queued")),
            result_json=v.get("result_json"),
            artifacts=list(v.get("artifacts") or []),
            commit=v.get("commit"),
            notes=str(v.get("notes", "")),
        )
        for k, v in sorted(bundle.items(), key=lambda kv: int(kv[0]))
    ]
    return build_evidence_ledger(entries)
