"""Arbitration interface: close conflicts with replayable, hash-chained records.

A conflict is *open* until an arbitration record with decision
``adopt_left``/``adopt_right``/``harmonize``/``dismiss`` closes it.
``escalate`` keeps it open but records the escalation.  The ledger is an
append-only JSONL hash chain: replaying it reproduces the exact same state
digest, and any tampered/forged record breaks the chain (fail-closed).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from .conflict import Conflict
from .model import canonical_json

GENESIS = "0" * 64
CLOSING_DECISIONS = frozenset({"adopt_left", "adopt_right", "harmonize", "dismiss"})
ALL_DECISIONS = CLOSING_DECISIONS | {"escalate"}


class ArbitrationError(ValueError):
    """Invalid arbitration record or broken chain (fail-closed)."""


@dataclass(frozen=True)
class ArbitrationRecord:
    record_id: str
    conflict_id: str
    decision: str
    arbiter: str
    rationale: str
    chosen_threshold: Optional[float]
    chosen_unit: Optional[str]
    timestamp: str
    prev_hash: str
    record_hash: str = ""

    def _body(self) -> dict[str, Any]:
        data = asdict(self)
        data.pop("record_hash", None)
        return data

    def compute_hash(self) -> str:
        return hashlib.sha256(canonical_json(self._body()).encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def create(cls, conflict_id: str, decision: str, arbiter: str, rationale: str,
               timestamp: str, prev_hash: str = GENESIS,
               chosen_threshold: Optional[float] = None,
               chosen_unit: Optional[str] = None) -> "ArbitrationRecord":
        if decision not in ALL_DECISIONS:
            raise ArbitrationError(f"unknown decision {decision!r}")
        if not rationale or not rationale.strip():
            raise ArbitrationError("rationale is mandatory (fail-closed)")
        if not arbiter or not arbiter.strip():
            raise ArbitrationError("arbiter is mandatory")
        if not timestamp:
            raise ArbitrationError("timestamp is mandatory for replayability")
        if decision == "harmonize" and chosen_threshold is None:
            raise ArbitrationError("harmonize requires chosen_threshold")
        payload = "|".join([conflict_id, decision, arbiter, timestamp,
                            "" if chosen_threshold is None else f"{chosen_threshold:.12g}",
                            chosen_unit or "", prev_hash])
        record_id = "arb_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        record = cls(record_id=record_id, conflict_id=conflict_id, decision=decision,
                     arbiter=arbiter, rationale=rationale.strip(),
                     chosen_threshold=chosen_threshold, chosen_unit=chosen_unit,
                     timestamp=timestamp, prev_hash=prev_hash)
        return cls(**{**record.to_dict(), "record_hash": record.compute_hash()})

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ArbitrationRecord":
        return cls(**data)


class ArbitrationLedger:
    """Append-only arbitration log with hash-chain integrity."""

    def __init__(self, records: Optional[Iterable[ArbitrationRecord]] = None,
                 known_conflicts: Optional[Iterable[str]] = None,
                 path: Optional[str | Path] = None) -> None:
        self._records: list[ArbitrationRecord] = []
        self._known = set(known_conflicts or [])
        self.path = Path(path) if path else None
        if records:
            for rec in records:
                self.append(rec)
        if self.path and self.path.exists():
            for rec in self.load_jsonl(self.path):
                self.append(rec)

    # ---- integrity ------------------------------------------------------
    def _tail(self) -> str:
        return self._records[-1].record_hash if self._records else GENESIS

    def append(self, record: ArbitrationRecord) -> ArbitrationRecord:
        if record.decision not in ALL_DECISIONS:
            raise ArbitrationError(f"unknown decision {record.decision!r}")
        if not record.rationale or not record.rationale.strip():
            raise ArbitrationError("rationale is mandatory (fail-closed)")
        if self._known and record.conflict_id not in self._known:
            raise ArbitrationError(f"unknown conflict_id {record.conflict_id!r}")
        if any(r.record_id == record.record_id for r in self._records):
            raise ArbitrationError(f"duplicate record_id {record.record_id!r}")
        if record.prev_hash != self._tail():
            raise ArbitrationError("prev_hash does not extend the current chain")
        if record.record_hash != record.compute_hash():
            raise ArbitrationError("record_hash mismatch (tampered record)")
        self._records.append(record)
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return record

    def arbitrate(self, conflict: Conflict, decision: str, arbiter: str,
                  rationale: str, timestamp: str,
                  chosen_threshold: Optional[float] = None,
                  chosen_unit: Optional[str] = None) -> ArbitrationRecord:
        return self.append(ArbitrationRecord.create(
            conflict_id=conflict.conflict_id, decision=decision, arbiter=arbiter,
            rationale=rationale, timestamp=timestamp, prev_hash=self._tail(),
            chosen_threshold=chosen_threshold, chosen_unit=chosen_unit,
        ))

    # ---- state ----------------------------------------------------------
    def state(self) -> dict[str, dict[str, Any]]:
        out: dict[str, dict[str, Any]] = {}
        for rec in self._records:
            entry = out.setdefault(rec.conflict_id, {
                "status": "open", "decision": None, "chosen_threshold": None,
                "chosen_unit": None, "records": [],
            })
            entry["records"].append(rec.record_id)
            if rec.decision == "escalate":
                entry["status"] = "escalated"
                entry["escalated_to"] = rec.arbiter
                continue
            entry.update({
                "status": "closed",
                "decision": rec.decision,
                "chosen_threshold": rec.chosen_threshold,
                "chosen_unit": rec.chosen_unit,
                "closed_by": rec.arbiter,
                "closed_at": rec.timestamp,
            })
        return out

    def open_conflicts(self, conflicts: Iterable[Conflict]) -> list[Conflict]:
        state = self.state()
        return [c for c in conflicts if state.get(c.conflict_id, {}).get("status") != "closed"]

    def replay_digest(self) -> str:
        return hashlib.sha256(canonical_json(self.state()).encode("utf-8")).hexdigest()

    def verify_chain(self) -> bool:
        prev = GENESIS
        for rec in self._records:
            if rec.prev_hash != prev or rec.record_hash != rec.compute_hash():
                return False
            prev = rec.record_hash
        return True

    # ---- IO -------------------------------------------------------------
    def to_jsonl(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self._records:
                fh.write(json.dumps(rec.to_dict(), ensure_ascii=False, sort_keys=True) + "\n")
        return path

    @staticmethod
    def load_jsonl(path: str | Path) -> list[ArbitrationRecord]:
        records: list[ArbitrationRecord] = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(ArbitrationRecord.from_dict(json.loads(line)))
        return records

    @property
    def records(self) -> list[ArbitrationRecord]:
        return list(self._records)
