"""Single-attempt research task runner with independent validation."""
from __future__ import annotations

from typing import Any, Mapping, Protocol

from .research_runtime import (
    AgentBackend, AgentRequest, AgentResult, ResearchTask, ResearchTaskStore,
    TaskStatus,
)


class ResearchValidator(Protocol):
    """Independent validator contract; must not rely only on Agent self-report."""

    def validate(self, request: AgentRequest, result: AgentResult) -> Mapping[str, Any]: ...


class ResearchTaskRunner:
    """Run one bounded attempt, persist every stage, and require validation.

    The caller owns scheduling and requeueing RETRY_WAIT tasks. This method does
    not loop indefinitely: one call invokes the Agent at most once.
    """

    def __init__(self, store: ResearchTaskStore, agent: AgentBackend, validator: ResearchValidator):
        self.store = store
        self.agent = agent
        self.validator = validator

    def run_once(
        self,
        task_id: str,
        *,
        goal: Mapping[str, Any],
        workspace: str,
        timeout_seconds: float = 3600,
        allowed_tools: tuple[str, ...] = (),
    ) -> ResearchTask:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        task = self.store.get_task(task_id)
        if task.status is not TaskStatus.QUEUED:
            raise ValueError(f"task must be queued before run_once; got {task.status.value}")
        self.store.transition(
            task_id, expected_status=TaskStatus.QUEUED, new_status=TaskStatus.CLAIMED,
            event_type="worker_claimed", payload={"agent": self.agent.name},
        )
        task = self.store.transition(
            task_id, expected_status=TaskStatus.CLAIMED, new_status=TaskStatus.RUNNING,
            event_type="agent_started", payload={"agent": self.agent.name},
        )
        request = AgentRequest(
            task_id=task.task_id, goal=dict(goal), plan=task.plan, workspace=workspace,
            timeout_seconds=timeout_seconds, allowed_tools=tuple(allowed_tools),
        )
        try:
            agent_result = self.agent.execute(request)
        except Exception as exc:
            return self._retry_or_fail(task, f"agent execution error: {type(exc).__name__}: {exc}")

        result_payload = {
            "agent": {
                "name": self.agent.name,
                "outcome": agent_result.outcome,
                "summary": agent_result.summary,
                "artifact_uris": list(agent_result.artifact_uris),
                "metadata": dict(agent_result.metadata or {}),
            }
        }
        if agent_result.outcome == "needs_input":
            return self.store.transition(
                task_id, expected_status=TaskStatus.RUNNING, new_status=TaskStatus.WAITING_INPUT,
                event_type="agent_needs_input", payload={"summary": agent_result.summary},
                result=result_payload,
            )
        if agent_result.outcome == "failed":
            return self._retry_or_fail(task, agent_result.summary, result=result_payload)

        task = self.store.transition(
            task_id, expected_status=TaskStatus.RUNNING, new_status=TaskStatus.VERIFYING,
            event_type="agent_completed", payload={"summary": agent_result.summary},
            result=result_payload,
        )
        try:
            report = dict(self.validator.validate(request, agent_result))
            status = report.get("status")
            if status not in {"passed", "failed", "incomplete"}:
                raise ValueError("validator report status must be passed, failed, or incomplete")
            report_payload = {**result_payload, "validation": report}
            if status == "passed":
                return self.store.transition(
                    task_id, expected_status=TaskStatus.VERIFYING, new_status=TaskStatus.SUCCEEDED,
                    event_type="validation_passed", payload={"validation_status": status},
                    result=report_payload,
                )
            return self._retry_or_fail(
                task, f"independent validation status: {status}", result=report_payload,
                expected_status=TaskStatus.VERIFYING,
            )
        except Exception as exc:
            return self._retry_or_fail(
                task, f"validator error: {type(exc).__name__}: {exc}", result=result_payload,
                expected_status=TaskStatus.VERIFYING,
            )

    def _retry_or_fail(
        self,
        task: ResearchTask,
        error: str,
        *,
        result: Mapping[str, Any] | None = None,
        expected_status: TaskStatus = TaskStatus.RUNNING,
    ) -> ResearchTask:
        target = TaskStatus.RETRY_WAIT if task.attempt < task.max_attempts else TaskStatus.FAILED
        return self.store.transition(
            task.task_id, expected_status=expected_status, new_status=target,
            event_type="attempt_failed" if target is TaskStatus.RETRY_WAIT else "task_failed",
            payload={"error": error, "attempt": task.attempt},
            result=result, error=error,
        )
