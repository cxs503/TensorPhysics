"""Platform run-registration wiring tests (R3, boundary ruling 2026-11-02).

research_runner registers every TERMINAL outcome (succeeded / failed) to
TensorPlatform and appends the outcome record to the task's append-only
event stream.  Registration is reporting, never gating: env-absent -> honest
skip, transport error -> failed-with-reason, task state never changes.
"""
import pytest

from tensorphysics.platform_reporting import register_run_outcome
from tensorphysics.platform_registry import PlatformRegistryError
from tensorphysics.research_runtime import ResearchTaskStore, TaskStatus
from tensorphysics.research_runner import ResearchTaskRunner


class FakeAgent:
    name = "fake-agent"

    def __init__(self, result):
        self.result = result
        self.calls = 0

    def execute(self, request):
        self.calls += 1
        return self.result


class FakeValidator:
    def __init__(self, report):
        self.report = report
        self.calls = 0

    def validate(self, request, result):
        self.calls += 1
        return self.report


class AgentResult:
    def __init__(self, outcome, summary, artifact_uris=()):
        self.outcome = outcome
        self.summary = summary
        self.artifact_uris = tuple(artifact_uris)
        self.metadata = {}


class FakeRegistry:
    def __init__(self, response=None, error=None):
        self.calls = []
        self.response = response or {"run_id": "ok", "duplicate": False}
        self.error = error

    def register(self, manifest):
        self.calls.append(manifest)
        if self.error is not None:
            raise self.error
        return self.response


def _events(store, task_id):
    import json
    rows = store._db.execute(
        "SELECT event_type, payload_json FROM research_task_events "
        "WHERE task_id = ? ORDER BY event_id", (task_id,)).fetchall()
    return [(r["event_type"], json.loads(r["payload_json"])) for r in rows]


class TestRegisterRunOutcomeUnit:
    def test_registered(self):
        reg = FakeRegistry(response={"run_id": "t1", "duplicate": False})
        out = register_run_outcome(
            run_id="t1", status="succeeded", validation_status="passed",
            manifest={"schema_version": 1, "goal_id": "g"}, registry=reg)
        assert out["outcome"] == "registered"
        assert out["response"]["run_id"] == "t1"
        assert reg.calls[0]["run_id"] == "t1"
        assert reg.calls[0]["status"] == "succeeded"

    def test_failed_on_transport_error(self):
        reg = FakeRegistry(error=PlatformRegistryError("connection refused"))
        out = register_run_outcome(
            run_id="t2", status="failed", validation_status="not_run",
            manifest={}, registry=reg)
        assert out["outcome"] == "failed"
        assert "connection refused" in out["reason"]

    def test_failed_on_invalid_payload_is_not_swallowed_silently(self):
        reg = FakeRegistry(error=ValueError("manifest.status must be one of: ..."))
        out = register_run_outcome(
            run_id="t3", status="succeeded", validation_status="passed",
            manifest={}, registry=reg)
        assert out["outcome"] == "failed"
        assert "manifest.status" in out["reason"]


class TestRunnerRegistersTerminalOutcomes:
    def test_success_registers_and_logs_event(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r1", "goal-r1", {"solver": "TensorDEM"})
        reg = FakeRegistry()
        agent = FakeAgent(AgentResult("completed", "DEM run finished", ("file:///h.csv",)))
        validator = FakeValidator({"status": "passed", "checks": []})
        task = ResearchTaskRunner(store, agent, validator, platform_registry=reg).run_once(
            "task-r1", goal={"objective": "validate"}, workspace="/tmp/run")
        assert task.status is TaskStatus.SUCCEEDED
        assert reg.calls and reg.calls[0]["run_id"] == "task-r1"
        assert reg.calls[0]["status"] == "succeeded"
        assert reg.calls[0]["validation_status"] == "passed"
        assert reg.calls[0]["goal_id"] == "goal-r1"
        assert reg.calls[0]["artifact_uris"] == ["file:///h.csv"]
        evs = _events(store, "task-r1")
        reg_ev = [p for t, p in evs if t == "platform_registration"]
        assert reg_ev and reg_ev[0]["outcome"] == "registered"

    def test_env_absent_skips_honestly(self, monkeypatch):
        for var in ("TENSORPLATFORM_URL", "TENSORPLATFORM_API_TOKEN", "TENSORPLATFORM_ORGANIZATION_ID"):
            monkeypatch.delenv(var, raising=False)
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r2", "goal-r2", {"solver": "TensorDEM"})
        agent = FakeAgent(AgentResult("completed", "done"))
        validator = FakeValidator({"status": "passed"})
        task = ResearchTaskRunner(store, agent, validator).run_once(
            "task-r2", goal={"objective": "v"}, workspace="/tmp/run")
        assert task.status is TaskStatus.SUCCEEDED
        evs = _events(store, "task-r2")
        reg_ev = [p for t, p in evs if t == "platform_registration"]
        assert reg_ev and reg_ev[0]["outcome"] == "skipped"
        assert "TENSORPLATFORM_URL" in reg_ev[0]["reason"]

    def test_final_failure_registers_failed_with_validation_status(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r3", "goal-r3", {"solver": "TensorDEM"}, max_attempts=1)
        reg = FakeRegistry()
        agent = FakeAgent(AgentResult("completed", "bad output"))
        validator = FakeValidator({"status": "failed", "checks": []})
        task = ResearchTaskRunner(store, agent, validator, platform_registry=reg).run_once(
            "task-r3", goal={"objective": "v"}, workspace="/tmp/run")
        assert task.status is TaskStatus.FAILED
        assert reg.calls[0]["status"] == "failed"
        assert reg.calls[0]["validation_status"] == "failed"
        evs = _events(store, "task-r3")
        assert any(t == "platform_registration" and p["outcome"] == "registered" for t, p in evs)

    def test_retry_wait_does_not_register(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r4", "goal-r4", {"solver": "TensorDEM"}, max_attempts=2)
        reg = FakeRegistry()
        agent = FakeAgent(AgentResult("completed", "bad"))
        validator = FakeValidator({"status": "failed"})
        task = ResearchTaskRunner(store, agent, validator, platform_registry=reg).run_once(
            "task-r4", goal={"objective": "v"}, workspace="/tmp/run")
        assert task.status is TaskStatus.RETRY_WAIT
        assert reg.calls == []  # non-terminal: nothing registered
        assert all(t != "platform_registration" for t, _ in _events(store, "task-r4"))

    def test_agent_failure_registers_not_run(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r5", "goal-r5", {"solver": "TensorDEM"}, max_attempts=1)
        reg = FakeRegistry()
        agent = FakeAgent(AgentResult("failed", "crashed"))
        validator = FakeValidator({"status": "passed"})
        task = ResearchTaskRunner(store, agent, validator, platform_registry=reg).run_once(
            "task-r5", goal={"objective": "v"}, workspace="/tmp/run")
        assert task.status is TaskStatus.FAILED
        assert reg.calls[0]["validation_status"] == "not_run"

    def test_waiting_input_does_not_register(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-r6", "goal-r6", {"solver": "TensorDEM"})
        reg = FakeRegistry()
        agent = FakeAgent(AgentResult("needs_input", "need stiffness"))
        validator = FakeValidator({"status": "passed"})
        task = ResearchTaskRunner(store, agent, validator, platform_registry=reg).run_once(
            "task-r6", goal={"objective": "v"}, workspace="/tmp/run")
        assert task.status is TaskStatus.WAITING_INPUT
        assert reg.calls == []


class TestAppendEventPrimitive:
    def test_append_event_requires_existing_task(self):
        store = ResearchTaskStore(":memory:")
        with pytest.raises(Exception):
            store.append_event("nope", "platform_registration", {})

    def test_append_event_does_not_change_status(self):
        store = ResearchTaskStore(":memory:")
        store.create_task("task-a", "goal-a", {"solver": "X"})
        before = store.get_task("task-a").status
        store.append_event("task-a", "platform_registration", {"outcome": "skipped"})
        assert store.get_task("task-a").status is before  # append-only, no state change
        assert _events(store, "task-a")[-1][0] == "platform_registration"
