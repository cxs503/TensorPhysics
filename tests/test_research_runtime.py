import pytest

from tensorphysics.research_runtime import (
    AgentRequest,
    AgentResult,
    InvalidTaskTransition,
    ResearchTaskStore,
    TaskStatus,
)


def test_task_lifecycle_is_persisted_and_audited(tmp_path):
    db = tmp_path / "research.sqlite3"
    store = ResearchTaskStore(db)
    task = store.create_task("task-1", "goal-1", {"solver": "TensorDEM"}, max_attempts=2)
    assert task.status is TaskStatus.QUEUED
    task = store.transition("task-1", expected_status=TaskStatus.QUEUED,
                            new_status=TaskStatus.CLAIMED, event_type="worker_claimed")
    task = store.transition("task-1", expected_status=TaskStatus.CLAIMED,
                            new_status=TaskStatus.RUNNING, event_type="agent_started")
    assert task.attempt == 1
    task = store.transition("task-1", expected_status=TaskStatus.RUNNING,
                            new_status=TaskStatus.VERIFYING, event_type="solver_finished")
    task = store.transition("task-1", expected_status=TaskStatus.VERIFYING,
                            new_status=TaskStatus.SUCCEEDED, event_type="validation_passed",
                            result={"validation": "passed"})
    assert task.result == {"validation": "passed"}
    assert [event.event_type for event in store.list_events("task-1")] == [
        "task_created", "worker_claimed", "agent_started", "solver_finished", "validation_passed"
    ]
    store.close()

    reopened = ResearchTaskStore(db)
    assert reopened.get_task("task-1").status is TaskStatus.SUCCEEDED
    reopened.close()


def test_stale_transition_is_rejected():
    store = ResearchTaskStore(":memory:")
    store.create_task("task-1", "goal-1", {"x": 1})
    store.transition("task-1", expected_status=TaskStatus.QUEUED,
                     new_status=TaskStatus.CLAIMED, event_type="claim")
    with pytest.raises(InvalidTaskTransition, match="expected queued"):
        store.transition("task-1", expected_status=TaskStatus.QUEUED,
                         new_status=TaskStatus.CANCELLED, event_type="stale_cancel")


def test_invalid_transition_is_rejected():
    store = ResearchTaskStore(":memory:")
    store.create_task("task-1", "goal-1", {"x": 1})
    with pytest.raises(InvalidTaskTransition, match="not allowed"):
        store.transition("task-1", expected_status=TaskStatus.QUEUED,
                         new_status=TaskStatus.SUCCEEDED, event_type="skip_validation")


def test_retry_budget_and_recovery(tmp_path):
    store = ResearchTaskStore(tmp_path / "recover.sqlite3")
    store.create_task("task-1", "goal-1", {"x": 1}, max_attempts=1)
    store.transition("task-1", expected_status=TaskStatus.QUEUED,
                     new_status=TaskStatus.CLAIMED, event_type="claim")
    store.transition("task-1", expected_status=TaskStatus.CLAIMED,
                     new_status=TaskStatus.RUNNING, event_type="start")
    recovered = store.recover_interrupted()
    assert len(recovered) == 1
    assert recovered[0].status is TaskStatus.FAILED
    assert recovered[0].error == "worker_interrupted"


def test_retryable_interruption_returns_to_retry_wait(tmp_path):
    store = ResearchTaskStore(tmp_path / "retry.sqlite3")
    store.create_task("task-1", "goal-1", {"x": 1}, max_attempts=2)
    store.transition("task-1", expected_status=TaskStatus.QUEUED,
                     new_status=TaskStatus.CLAIMED, event_type="claim")
    store.transition("task-1", expected_status=TaskStatus.CLAIMED,
                     new_status=TaskStatus.RUNNING, event_type="start")
    task = store.recover_interrupted()[0]
    assert task.status is TaskStatus.RETRY_WAIT
    task = store.transition("task-1", expected_status=TaskStatus.RETRY_WAIT,
                             new_status=TaskStatus.QUEUED, event_type="retry_queued")
    assert task.status is TaskStatus.QUEUED


def test_payloads_must_be_finite_json():
    store = ResearchTaskStore(":memory:")
    with pytest.raises(ValueError, match="finite JSON-compatible"):
        store.create_task("task-1", "goal-1", {"bad": float("nan")})


def test_agent_result_is_structured_and_not_free_form_status():
    result = AgentResult(outcome="completed", summary="Prepared a candidate patch",
                         artifact_uris=("file:///workspace/patch.diff",))
    assert result.outcome == "completed"
    with pytest.raises(ValueError, match="outcome"):
        AgentResult(outcome="scientifically_correct", summary="trust me")


def test_agent_request_contract_is_vendor_neutral():
    request = AgentRequest(task_id="task-1", goal={"objective": "test"},
                           plan={"solver": "TensorDEM"}, workspace="/tmp/task-1",
                           timeout_seconds=120, allowed_tools=("read_file", "run_tests"))
    assert request.task_id == "task-1"
    assert request.allowed_tools == ("read_file", "run_tests")
