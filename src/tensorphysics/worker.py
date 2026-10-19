"""Long-running, restart-safe worker for the durable TensorPhysics task ledger.

Configure an Agent CLI and an independent validator via JSON argv environment
variables. No shell is used. The worker never marks a run successful without a
validator returning a JSON object with status="passed".
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import signal
import sys
import time
from typing import Any, Mapping, Sequence

from .agent_subprocess import SubprocessAgentBackend
from .research_runner import ResearchTaskRunner
from .research_runtime import (
    AgentRequest, AgentResult, ResearchTask, ResearchTaskStore, TaskStatus,
)

LOG = logging.getLogger("tensorphysics.worker")


class JsonCommandValidator:
    """Invoke a separate validator command and parse its JSON report.

    The validator receives one JSON document on stdin containing the task,
    goal, plan, Agent summary, artifact URIs, and Agent metadata. It must emit
    a JSON object on stdout. Its process exit code must be zero.
    """

    def __init__(self, argv: Sequence[str], *, timeout_seconds: float = 300.0) -> None:
        if not argv or any(not isinstance(part, str) or not part for part in argv):
            raise ValueError("validator argv must be a non-empty string sequence")
        if timeout_seconds <= 0:
            raise ValueError("validator timeout must be positive")
        self.argv = tuple(argv)
        self.timeout_seconds = timeout_seconds

    def validate(self, request: AgentRequest, result: AgentResult) -> Mapping[str, Any]:
        import subprocess

        payload = {
            "schema_version": 1,
            "task_id": request.task_id,
            "goal": dict(request.goal),
            "plan": dict(request.plan),
            "agent_result": {
                "outcome": result.outcome,
                "summary": result.summary,
                "artifact_uris": list(result.artifact_uris),
                "metadata": dict(result.metadata or {}),
            },
        }
        completed = subprocess.run(
            list(self.argv),
            input=json.dumps(payload, ensure_ascii=False, allow_nan=False),
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self.timeout_seconds,
            check=False,
            shell=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"validator exited {completed.returncode}: {completed.stderr[-4000:]}"
            )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise ValueError(f"validator stdout must be one JSON object: {exc}") from exc
        if not isinstance(report, dict):
            raise ValueError("validator report must be a JSON object")
        if report.get("status") not in {"passed", "failed", "incomplete"}:
            raise ValueError("validator report status must be passed, failed, or incomplete")
        checks = report.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("validator report must include a non-empty checks list")
        if report["status"] == "passed" and any(
            not isinstance(check, dict) or check.get("passed") is not True for check in checks
        ):
            raise ValueError("validator cannot pass when any check is absent or failed")
        return report


def _argv_from_env(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        raise RuntimeError(f"{name} is required; configure a real Agent/validator command")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{name} must be a JSON array of argv strings") from exc
    if not isinstance(value, list) or not value or any(not isinstance(x, str) or not x for x in value):
        raise RuntimeError(f"{name} must be a non-empty JSON array of non-empty strings")
    return value


def _goal_for(task: ResearchTask) -> dict[str, Any]:
    goal = task.plan.get("goal")
    if isinstance(goal, dict) and goal:
        return goal
    objective = task.plan.get("objective")
    if isinstance(objective, str) and objective.strip():
        return {"goal_id": task.goal_id, "objective": objective.strip()}
    raise ValueError(
        f"task {task.task_id} plan must include a non-empty 'goal' object or 'objective' string"
    )


def process_one(
    store: ResearchTaskStore,
    runner: ResearchTaskRunner,
    workspace_root: Path,
    *,
    timeout_seconds: float,
    allowed_tools: tuple[str, ...],
) -> bool:
    """Run one queued task, returning whether work was claimed."""
    tasks = store.list_tasks(status=TaskStatus.QUEUED, limit=1)
    if not tasks:
        return False
    task = tasks[0]
    workspace = (workspace_root / task.task_id).resolve()
    root = workspace_root.resolve()
    if workspace != root and root not in workspace.parents:
        LOG.error("Refusing workspace path outside workspace root for task %s", task.task_id)
        store.transition(
            task.task_id, expected_status=TaskStatus.QUEUED, new_status=TaskStatus.CANCELLED,
            event_type="unsafe_workspace_rejected", payload={"workspace": str(workspace)},
        )
        return True
    workspace.mkdir(parents=True, exist_ok=True)
    try:
        result = runner.run_once(
            task.task_id, goal=_goal_for(task), workspace=str(workspace),
            timeout_seconds=timeout_seconds, allowed_tools=allowed_tools,
        )
        LOG.info("task=%s status=%s attempt=%s", result.task_id, result.status.value, result.attempt)
    except Exception:
        LOG.exception("Failed to process task %s", task.task_id)
    return True


def requeue_retryable(store: ResearchTaskStore) -> int:
    """Requeue retryable attempts without exceeding each task's budget."""
    count = 0
    for task in store.list_tasks(status=TaskStatus.RETRY_WAIT, limit=10000):
        if task.attempt < task.max_attempts:
            store.transition(
                task.task_id, expected_status=TaskStatus.RETRY_WAIT,
                new_status=TaskStatus.QUEUED, event_type="retry_requeued",
                payload={"attempt": task.attempt, "max_attempts": task.max_attempts},
            )
            count += 1
    return count


def worker_loop(
    database: str,
    *,
    workspace_root: str,
    poll_seconds: float = 5.0,
    timeout_seconds: float = 3600.0,
    validator_timeout_seconds: float = 300.0,
    allowed_tools: tuple[str, ...] = (),
    once: bool = False,
) -> int:
    if poll_seconds <= 0 or timeout_seconds <= 0 or validator_timeout_seconds <= 0:
        raise ValueError("poll and timeout values must be positive")
    root = Path(workspace_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    agent = SubprocessAgentBackend(
        os.environ.get("TENSORPHYSICS_AGENT_NAME", "configured-agent-cli"),
        _argv_from_env("TENSORPHYSICS_AGENT_ARGV_JSON"),
        artifact_dir=root / "_agent_logs",
        inherit_env=os.environ.get("TENSORPHYSICS_AGENT_INHERIT_ENV", "true").lower() in {"1", "true", "yes"},
    )
    validator = JsonCommandValidator(
        _argv_from_env("TENSORPHYSICS_VALIDATOR_ARGV_JSON"),
        timeout_seconds=validator_timeout_seconds,
    )
    stop = False

    def request_stop(signum: int, _frame: Any) -> None:
        nonlocal stop
        LOG.info("Received signal %s; stopping after current task", signum)
        stop = True

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    with ResearchTaskStore(database) as store:
        recovered = store.recover_interrupted()
        for task in recovered:
            LOG.warning("Recovered interrupted task=%s new_status=%s", task.task_id, task.status.value)
        requeue_retryable(store)
        runner = ResearchTaskRunner(store, agent, validator)
        LOG.info("worker_started database=%s workspace_root=%s poll_seconds=%s", database, root, poll_seconds)
        while not stop:
            requeue_retryable(store)
            did_work = process_one(
                store, runner, root, timeout_seconds=timeout_seconds, allowed_tools=allowed_tools,
            )
            if once:
                return 0
            if not did_work:
                time.sleep(poll_seconds)
    LOG.info("worker_stopped")
    return 0


def submit_task(database: str, task_id: str, goal_id: str, plan_path: str, max_attempts: int) -> int:
    with Path(plan_path).open("r", encoding="utf-8") as stream:
        plan = json.load(stream)
    if not isinstance(plan, dict):
        raise ValueError("plan JSON file must contain an object")
    with ResearchTaskStore(database) as store:
        task = store.create_task(task_id, goal_id, plan, max_attempts=max_attempts)
    print(json.dumps({
        "task_id": task.task_id, "goal_id": task.goal_id,
        "status": task.status.value, "created_at": task.created_at,
    }, ensure_ascii=False))
    return 0


def task_status(database: str, task_id: str) -> int:
    with ResearchTaskStore(database) as store:
        task = store.get_task(task_id)
        events = store.list_events(task_id)
    print(json.dumps({
        "task_id": task.task_id, "goal_id": task.goal_id, "status": task.status.value,
        "attempt": task.attempt, "max_attempts": task.max_attempts,
        "updated_at": task.updated_at, "result": task.result, "error": task.error,
        "events": [
            {"event_id": e.event_id, "type": e.event_type, "from": e.from_status.value if e.from_status else None,
             "to": e.to_status.value, "created_at": e.created_at, "payload": dict(e.payload)}
            for e in events
        ],
    }, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Persistent TensorPhysics research worker")
    sub = parser.add_subparsers(dest="command", required=True)
    worker = sub.add_parser("worker", help="run the persistent queue consumer")
    worker.add_argument("--database", default=os.environ.get("TENSORPHYSICS_DATABASE", "state/research.sqlite3"))
    worker.add_argument("--workspace-root", default=os.environ.get("TENSORPHYSICS_WORKSPACE_ROOT", "workspaces"))
    worker.add_argument("--poll-seconds", type=float, default=float(os.environ.get("TENSORPHYSICS_POLL_SECONDS", "5")))
    worker.add_argument("--timeout-seconds", type=float, default=float(os.environ.get("TENSORPHYSICS_TASK_TIMEOUT_SECONDS", "3600")))
    worker.add_argument("--validator-timeout-seconds", type=float, default=float(os.environ.get("TENSORPHYSICS_VALIDATOR_TIMEOUT_SECONDS", "300")))
    worker.add_argument("--allowed-tools", default=os.environ.get("TENSORPHYSICS_ALLOWED_TOOLS", ""))
    worker.add_argument("--once", action="store_true", help="process at most one task and exit")
    submit = sub.add_parser("submit", help="enqueue a task from a JSON plan file")
    submit.add_argument("--database", default=os.environ.get("TENSORPHYSICS_DATABASE", "state/research.sqlite3"))
    submit.add_argument("--task-id", required=True)
    submit.add_argument("--goal-id", required=True)
    submit.add_argument("--plan", required=True, help="path to JSON plan")
    submit.add_argument("--max-attempts", type=int, default=3)
    status = sub.add_parser("status", help="show task state and event history")
    status.add_argument("--database", default=os.environ.get("TENSORPHYSICS_DATABASE", "state/research.sqlite3"))
    status.add_argument("--task-id", required=True)
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    if args.command == "worker":
        return worker_loop(
            args.database, workspace_root=args.workspace_root, poll_seconds=args.poll_seconds,
            timeout_seconds=args.timeout_seconds,
            validator_timeout_seconds=args.validator_timeout_seconds,
            allowed_tools=tuple(x.strip() for x in args.allowed_tools.split(",") if x.strip()),
            once=args.once,
        )
    if args.command == "submit":
        return submit_task(args.database, args.task_id, args.goal_id, args.plan, args.max_attempts)
    return task_status(args.database, args.task_id)


if __name__ == "__main__":
    sys.exit(main())
