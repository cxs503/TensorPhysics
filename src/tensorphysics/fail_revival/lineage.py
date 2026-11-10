"""Revival lineage: every verdict change is one queryable edge.

Storage is a single SQLite file (``artifacts/fail_revival/lineage.sqlite3``) with
two tables:

``revival_lineage``   one row per adjudicated revival: old verdict -> new
                      verdict, triggering blocker, criterion digest, input
                      digest, timestamp, and the change list;
``revival_audit``     the anti-whitewash ledger: one row per adjudication that
                      was refused (``OVERCLAIM``) or accepted (``OK``).

Both are written through parameterised SQL; a single query answers "what is the
lineage of this FAIL / this criterion digest / this task key".
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import sqlite3
from typing import Any, Iterable, Mapping

from .model import Adjudication, RevivalTask, sha256_obj

LINEAGE_SCHEMA = "revival_lineage.v1"
DDL = (
    """CREATE TABLE IF NOT EXISTS revival_lineage (
        edge_id TEXT PRIMARY KEY,
        schema TEXT NOT NULL,
        task_key TEXT NOT NULL,
        fail_key TEXT NOT NULL,
        criterion_sha256 TEXT NOT NULL,
        input_sha256 TEXT NOT NULL,
        old_verdict TEXT NOT NULL,
        new_verdict TEXT NOT NULL,
        blocker_id TEXT NOT NULL,
        blocker_kind TEXT NOT NULL,
        mission_id INTEGER,
        family_member INTEGER NOT NULL,
        changes_json TEXT NOT NULL,
        reasons_json TEXT NOT NULL,
        adjudicated_at TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_lineage_fail ON revival_lineage(fail_key)""",
    """CREATE INDEX IF NOT EXISTS idx_lineage_criterion ON revival_lineage(criterion_sha256)""",
    """CREATE TABLE IF NOT EXISTS revival_audit (
        audit_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        task_key TEXT NOT NULL,
        fail_key TEXT NOT NULL,
        criterion_sha256 TEXT NOT NULL,
        registry_sha256 TEXT,
        reason TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        payload_json TEXT NOT NULL
    )""",
    """CREATE INDEX IF NOT EXISTS idx_audit_status ON revival_audit(status)""",
)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path | str) -> sqlite3.Connection:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    for stmt in DDL:
        conn.execute(stmt)
    conn.commit()
    return conn


def record_edge(conn: sqlite3.Connection, adjudication: Adjudication, task: RevivalTask,
                *, old_verdict: str = "FAIL", mission_id: int | None = None,
                at: str | None = None) -> str:
    edge_id = sha256_obj([task.task_key, adjudication.verdict, at or ""])[:32]
    conn.execute(
        "INSERT OR REPLACE INTO revival_lineage VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            edge_id, LINEAGE_SCHEMA, task.task_key, task.fail_key,
            adjudication.criterion_sha256, task.input_sha256, old_verdict,
            adjudication.verdict, task.blocker_id, task.blocker_kind, mission_id,
            1 if adjudication.family_member else 0,
            json.dumps([dict(c) for c in adjudication.changes], ensure_ascii=False, sort_keys=True),
            json.dumps(list(adjudication.reasons), ensure_ascii=False),
            at or now_iso(),
        ),
    )
    conn.commit()
    return edge_id


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    d = dict(row)
    d["changes"] = json.loads(d.pop("changes_json") or "[]")
    d["reasons"] = json.loads(d.pop("reasons_json") or "[]")
    d["family_member"] = bool(d["family_member"])
    return d


def lineage_of(conn: sqlite3.Connection, fail_key: str) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM revival_lineage WHERE fail_key = ? ORDER BY adjudicated_at, edge_id", (fail_key,)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def edge_of_task(conn: sqlite3.Connection, task_key: str) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM revival_lineage WHERE task_key = ? ORDER BY adjudicated_at, edge_id", (task_key,)
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def lineage_of_criterion(conn: sqlite3.Connection, criterion_sha256: str) -> list[dict[str, Any]]:
    cur = conn.execute(
        "SELECT * FROM revival_lineage WHERE criterion_sha256 = ? ORDER BY adjudicated_at, edge_id",
        (criterion_sha256,),
    )
    return [_row_to_dict(r) for r in cur.fetchall()]


def ledger(conn: sqlite3.Connection, status: str | None = None) -> list[dict[str, Any]]:
    if status is None:
        cur = conn.execute("SELECT * FROM revival_audit ORDER BY recorded_at, audit_id")
    else:
        cur = conn.execute(
            "SELECT * FROM revival_audit WHERE status = ? ORDER BY recorded_at, audit_id", (status,)
        )
    out = []
    for r in cur.fetchall():
        d = dict(r)
        d["payload"] = json.loads(d.pop("payload_json") or "{}")
        out.append(d)
    return out


def edge_count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM revival_lineage").fetchone()[0])