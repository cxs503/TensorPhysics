"""Durable task ledger and replaceable Agent execution contracts.

This module is deliberately solver- and vendor-neutral. It records task state;
it does not execute arbitrary shell commands or decide scientific correctness.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import json
from pathlib import Path
import sqlite3
from threading import RLock
from typing import Any, Mapping, Protocol


class ResearchRuntimeError(RuntimeError):
    """Base error for durable research-task operations."""


class TaskNotFoundError(ResearchRuntimeError):
    """The requested task does not exist."""


class InvalidTaskTransition(ResearchRuntimeError):
    """A requested state transition is not allowed."""


class TaskStatus(str, Enum):
    QUEUED = "queued"
    CLAIMED = "claimed"
    RUNNING = "running"
    VERIFYING = "verifying"
    WAITING_INPUT = "waiting_input"
    RETRY_WAIT = "retry_wait"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


_ALLOWED: dict[TaskStatus, set[TaskStatus]] = {
    TaskStatus.QUEUED: {TaskStatus.CLAIMED, TaskStatus.CANCELLED},
    TaskStatus.CLAIMED: {TaskStatus.RUNNING, TaskStatus.QUEUED, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.RUNNING: {TaskStatus.VERIFYING, TaskStatus.WAITING_INPUT, TaskStatus.RETRY_WAIT, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.VERIFYING: {TaskStatus.SUCCEEDED, TaskStatus.RETRY_WAIT, TaskStatus.FAILED, TaskStatus.CANCELLED},
    TaskStatus.WAITING_INPUT: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.RETRY_WAIT: {TaskStatus.QUEUED, TaskStatus.CANCELLED},
    TaskStatus.SUCCEEDED: set(),
    TaskStatus.FAILED: set(),
    TaskStatus.CANCELLED: set(),
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _json(value: Any, field: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite JSON-compatible data: {exc}") from exc


@dataclass(frozen=True)
class ResearchTask:
    task_id: str
    goal_id: str
    plan: Mapping[str, Any]
    status: TaskStatus
    attempt: int
    max_attempts: int
    created_at: str
    updated_at: str
    result: Mapping[str, Any] | None = None
    error: str | None = None


@dataclass(frozen=True)
class TaskEvent:
    event_id: int
    task_id: str
    from_status: TaskStatus | None
    to_status: TaskStatus
    event_type: str
    payload: Mapping[str, Any]
    created_at: str


@dataclass(frozen=True)
class AgentRequest:
    """Vendor-neutral request passed to a Codex/Claude/other Agent adapter."""

    task_id: str
    goal: Mapping[str, Any]
    plan: Mapping[str, Any]
    workspace: str
    timeout_seconds: float
    allowed_tools: tuple[str, ...] = ()


@dataclass(frozen=True)
class AgentResult:
    """Structured Agent output; it is not scientific acceptance evidence."""

    outcome: str
    summary: str
    artifact_uris: tuple[str, ...] = ()
    metadata: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"completed", "needs_input", "failed"}:
            raise ValueError("outcome must be completed, needs_input, or failed")
        _text(self.summary, "summary")
        if any(not isinstance(uri, str) or not uri.strip() for uri in self.artifact_uris):
            raise ValueError("artifact_uris must contain non-empty strings")
        _json(dict(self.metadata or {}), "metadata")


class AgentBackend(Protocol):
    """Adapter contract for a replaceable Agent execution engine."""

    @property
    def name(self) -> str: ...

    def execute(self, request: AgentRequest) -> AgentResult:
        """Execute within the supplied workspace and return structured output."""
        ...


class ResearchTaskStore:
    """SQLite-backed task state machine with an append-only transition log.

    Use a filesystem path for restart durability. State changes use SQLite
    transactions and an expected-state check so competing workers cannot
    silently apply stale transitions.
    """

    def __init__(self, database: str | Path) -> None:
        self.database = str(database)
        if self.database != ":memory:":
            Path(self.database).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._db = sqlite3.connect(self.database, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA foreign_keys = ON")
        if self.database != ":memory:":
            self._db.execute("PRAGMA journal_mode = WAL")
            self._db.execute("PRAGMA synchronous = FULL")
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS research_tasks (
                task_id TEXT PRIMARY KEY,
                goal_id TEXT NOT NULL,
                plan_json TEXT NOT NULL,
                status TEXT NOT NULL,
                attempt INTEGER NOT NULL DEFAULT 0,
                max_attempts INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                result_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS research_task_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL REFERENCES research_tasks(task_id),
                from_status TEXT,
                to_status TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_research_task_events_task
                ON research_task_events(task_id, event_id);
            CREATE INDEX IF NOT EXISTS idx_research_tasks_status
                ON research_tasks(status, created_at);
            """
        )

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def __enter__(self) -> "ResearchTaskStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def create_task(
        self,
        task_id: str,
        goal_id: str,
        plan: Mapping[str, Any],
        *,
        max_attempts: int = 3,
    ) -> ResearchTask:
        task_id = _text(task_id, "task_id")
        goal_id = _text(goal_id, "goal_id")
        if not isinstance(plan, Mapping):
            raise ValueError("plan must be a mapping")
        if not isinstance(max_attempts, int) or isinstance(max_attempts, bool) or max_attempts < 1:
            raise ValueError("max_attempts must be a positive integer")
        plan_json = _json(dict(plan), "plan")
        now = _now()
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                self._db.execute(
                    """INSERT INTO research_tasks
                    (task_id, goal_id, plan_json, status, attempt, max_attempts, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 0, ?, ?, ?)""",
                    (task_id, goal_id, plan_json, TaskStatus.QUEUED.value, max_attempts, now, now),
                )
                self._event(task_id, None, TaskStatus.QUEUED, "task_created", {"goal_id": goal_id})
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get_task(task_id)

    def get_task(self, task_id: str) -> ResearchTask:
        with self._lock:
            row = self._db.execute(
                "SELECT * FROM research_tasks WHERE task_id = ?", (_text(task_id, "task_id"),)
            ).fetchone()
        if row is None:
            raise TaskNotFoundError(f"task not found: {task_id}")
        return self._to_task(row)

    def list_tasks(self, *, status: TaskStatus | None = None, limit: int = 100) -> list[ResearchTask]:
        if not isinstance(limit, int) or isinstance(limit, bool) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._lock:
            if status is None:
                rows = self._db.execute(
                    "SELECT * FROM research_tasks ORDER BY created_at, task_id LIMIT ?", (limit,)
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM research_tasks WHERE status = ? ORDER BY created_at, task_id LIMIT ?",
                    (TaskStatus(status).value, limit),
                ).fetchall()
        return [self._to_task(row) for row in rows]

    def transition(
        self,
        task_id: str,
        *,
        expected_status: TaskStatus,
        new_status: TaskStatus,
        event_type: str,
        payload: Mapping[str, Any] | None = None,
        result: Mapping[str, Any] | None = None,
        error: str | None = None,
    ) -> ResearchTask:
        task_id = _text(task_id, "task_id")
        event_type = _text(event_type, "event_type")
        expected_status = TaskStatus(expected_status)
        new_status = TaskStatus(new_status)
        payload_json = _json(dict(payload or {}), "payload")
        result_json = None if result is None else _json(dict(result), "result")
        if error is not None and not isinstance(error, str):
            raise ValueError("error must be a string or None")
        if new_status not in _ALLOWED[expected_status]:
            raise InvalidTaskTransition(f"transition {expected_status.value} -> {new_status.value} is not allowed")
        now = _now()
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                row = self._db.execute(
                    "SELECT * FROM research_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    raise TaskNotFoundError(f"task not found: {task_id}")
                actual = TaskStatus(row["status"])
                if actual is not expected_status:
                    raise InvalidTaskTransition(
                        f"task {task_id} is {actual.value}; expected {expected_status.value}"
                    )
                attempt = row["attempt"] + (1 if expected_status is TaskStatus.CLAIMED and new_status is TaskStatus.RUNNING else 0)
                if new_status is TaskStatus.RETRY_WAIT and row["attempt"] >= row["max_attempts"]:
                    raise InvalidTaskTransition("retry budget exhausted; transition to failed instead")
                self._db.execute(
                    """UPDATE research_tasks SET status = ?, attempt = ?, updated_at = ?,
                    result_json = COALESCE(?, result_json), error = ? WHERE task_id = ?""",
                    (new_status.value, attempt, now, result_json, error, task_id),
                )
                self._event(task_id, actual, new_status, event_type, json.loads(payload_json), now=now)
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return self.get_task(task_id)

    def recover_interrupted(self, *, reason: str = "worker_interrupted") -> list[ResearchTask]:
        """Move non-terminal in-flight tasks to retry_wait or failed after restart."""
        reason = _text(reason, "reason")
        recovered: list[str] = []
        now = _now()
        with self._lock:
            try:
                self._db.execute("BEGIN IMMEDIATE")
                rows = self._db.execute(
                    "SELECT * FROM research_tasks WHERE status IN (?, ?, ?)",
                    (TaskStatus.CLAIMED.value, TaskStatus.RUNNING.value, TaskStatus.VERIFYING.value),
                ).fetchall()
                for row in rows:
                    old = TaskStatus(row["status"])
                    new = TaskStatus.RETRY_WAIT if row["attempt"] < row["max_attempts"] else TaskStatus.FAILED
                    self._db.execute(
                        "UPDATE research_tasks SET status = ?, updated_at = ?, error = ? WHERE task_id = ?",
                        (new.value, now, reason, row["task_id"]),
                    )
                    self._event(row["task_id"], old, new, "task_recovered", {"reason": reason}, now=now)
                    recovered.append(row["task_id"])
                self._db.commit()
            except Exception:
                self._db.rollback()
                raise
        return [self.get_task(task_id) for task_id in recovered]

    def list_events(self, task_id: str) -> list[TaskEvent]:
        task_id = _text(task_id, "task_id")
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM research_task_events WHERE task_id = ? ORDER BY event_id", (task_id,)
            ).fetchall()
        return [
            TaskEvent(
                event_id=row["event_id"],
                task_id=row["task_id"],
                from_status=TaskStatus(row["from_status"]) if row["from_status"] else None,
                to_status=TaskStatus(row["to_status"]),
                event_type=row["event_type"],
                payload=json.loads(row["payload_json"]),
                created_at=row["created_at"],
            )
            for row in rows
        ]

    def _event(
        self,
        task_id: str,
        from_status: TaskStatus | None,
        to_status: TaskStatus,
        event_type: str,
        payload: Mapping[str, Any],
        *,
        now: str | None = None,
    ) -> None:
        self._db.execute(
            """INSERT INTO research_task_events
            (task_id, from_status, to_status, event_type, payload_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (task_id, from_status.value if from_status else None, to_status.value,
             event_type, _json(dict(payload), "event payload"), now or _now()),
        )

    @staticmethod
    def _to_task(row: sqlite3.Row) -> ResearchTask:
        return ResearchTask(
            task_id=row["task_id"],
            goal_id=row["goal_id"],
            plan=json.loads(row["plan_json"]),
            status=TaskStatus(row["status"]),
            attempt=row["attempt"],
            max_attempts=row["max_attempts"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            result=json.loads(row["result_json"]) if row["result_json"] else None,
            error=row["error"],
        )
