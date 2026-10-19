import json
import os
import sys

from tensorphysics.worker import JsonCommandValidator, _goal_for
from tensorphysics.research_runtime import AgentRequest, AgentResult, ResearchTask, TaskStatus


def test_goal_is_taken_from_persisted_plan():
    task = ResearchTask(
        task_id="t1", goal_id="g1", plan={"objective": "run a benchmark"},
        status=TaskStatus.QUEUED, attempt=0, max_attempts=1,
        created_at="now", updated_at="now",
    )
    assert _goal_for(task) == {"goal_id": "g1", "objective": "run a benchmark"}


def test_validator_accepts_passed_report_only_with_all_checks_true(tmp_path):
    code = (
        "import json,sys; payload=json.load(sys.stdin); "
        "assert payload['task_id']=='task-1'; "
        "print(json.dumps({'status':'passed','summary':'fixture only',"
        "'checks':[{'name':'fixture_contract','passed':True}]}))"
    )
    validator = JsonCommandValidator([sys.executable, "-c", code])
    request = AgentRequest(
        task_id="task-1", goal={"objective": "test"}, plan={"solver": "TensorDEM"},
        workspace=str(tmp_path), timeout_seconds=10,
    )
    report = validator.validate(request, AgentResult("completed", "finished"))
    assert report["status"] == "passed"


def test_validator_rejects_pass_with_failed_check(tmp_path):
    code = (
        "import json; print(json.dumps({'status':'passed','checks':"
        "[{'name':'bad','passed':False}]}))"
    )
    validator = JsonCommandValidator([sys.executable, "-c", code])
    request = AgentRequest("task-1", {}, {}, str(tmp_path), 10)
    import pytest
    with pytest.raises(ValueError, match="cannot pass"):
        validator.validate(request, AgentResult("completed", "finished"))


def test_missing_agent_configuration_fails_closed(monkeypatch):
    from tensorphysics.worker import _argv_from_env
    monkeypatch.delenv("TENSORPHYSICS_AGENT_ARGV_JSON", raising=False)
    import pytest
    with pytest.raises(RuntimeError, match="required"):
        _argv_from_env("TENSORPHYSICS_AGENT_ARGV_JSON")
