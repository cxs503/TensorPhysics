"""Anti-whitewash audit: a verdict may not outlive the criterion it claims.

The failure mode this guards against is *number laundering*: a revival run that
reports PASS while its criterion digest no longer matches the registry (an edited
criterion, a relaxed tolerance, a substituted input).  The audit is fail-closed
and has three effects, all observable:

1. :class:`WhitewashRefused` is raised -- the caller cannot record a PASS;
2. an ``OVERCLAIM`` row is written to the audit ledger with *both* digests, so
   the mismatch is auditable later;
3. no lineage edge is written: the FAIL keeps its previous verdict.

The reverse direction is checked too: an ``OK`` ledger row is written for every
admissible adjudication, so a missing row is itself evidence that a verdict was
never audited.
"""
from __future__ import annotations

from typing import Any, Mapping

from .lineage import connect, now_iso
from .model import Adjudication, FailRecord, RevivalTask, sha256_obj

OK = "OK"
OVERCLAIM = "OVERCLAIM"
REFUSED = "REFUSED"


class WhitewashRefused(RuntimeError):
    """Raised when a verdict is not backed by the registered criterion digest."""

    def __init__(self, reason: str, adjudication: Adjudication, record: FailRecord | None):
        self.reason = reason
        self.adjudication = adjudication
        self.record = record
        super().__init__("revival verdict refused (%s) for fail_key=%s"
                         % (reason, adjudication.fail_key))


def audit_adjudication(conn, adjudication: Adjudication, record: FailRecord | None,
                       task: RevivalTask, *, at: str | None = None) -> dict[str, Any]:
    """Audit one adjudication against the registry; raise on a whitewash attempt."""
    at = at or now_iso()
    if record is None:
        status, reason = OVERCLAIM, "criterion_not_in_registry"
        registry_sha = None
    elif adjudication.criterion_sha256 != record.criterion_sha256:
        status, reason = OVERCLAIM, "criterion_digest_mismatch"
        registry_sha = record.criterion_sha256
    elif adjudication.task_key != task.task_key:
        status, reason = OVERCLAIM, "task_key_mismatch"
        registry_sha = record.criterion_sha256
    else:
        status, reason = OK, "registry_match"
        registry_sha = record.criterion_sha256

    audit_id = sha256_obj([task.task_key, adjudication.verdict, at])[:32]
    conn.execute(
        "INSERT OR REPLACE INTO revival_audit VALUES (?,?,?,?,?,?,?,?,?)",
        (
            audit_id, status, adjudication.task_key, adjudication.fail_key,
            adjudication.criterion_sha256, registry_sha, reason, at,
            _payload(adjudication),
        ),
    )
    conn.commit()
    row = {"audit_id": audit_id, "status": status, "reason": reason,
           "fail_key": adjudication.fail_key, "task_key": adjudication.task_key,
           "criterion_sha256": adjudication.criterion_sha256,
           "registry_sha256": registry_sha, "recorded_at": at}
    if status != OK:
        raise WhitewashRefused(reason, adjudication, record)
    return row


def _payload(adjudication: Adjudication) -> str:
    import json

    return json.dumps(adjudication.to_dict(), ensure_ascii=False, sort_keys=True)


def audited_adjudicate(conn, adjudication: Adjudication, record: FailRecord | None,
                       task: RevivalTask, *, at: str | None = None) -> dict[str, Any]:
    """Audit, then write the lineage edge only if the audit passed."""
    from .lineage import record_edge

    row = audit_adjudication(conn, adjudication, record, task, at=at)
    record_edge(conn, adjudication, task, old_verdict="FAIL",
                mission_id=record.mission_id if record else None, at=at)
    return row


def overclaim_count(conn) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM revival_audit WHERE status = ?", (OVERCLAIM,)).fetchone()[0])