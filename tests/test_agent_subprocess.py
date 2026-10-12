import json
import sys

from tensorphysics.agent_subprocess import SubprocessAgentBackend
from tensorphysics.research_runtime import AgentRequest


def request(workspace, timeout=5, tools=("read_files", "run_tests")):
    return AgentRequest(
        task_id="task/with unsafe chars",
        goal={"objective": "check a small benchmark"},
        plan={"solver": "TensorDEM", "case": "bending"},
        workspace=str(workspace),
        timeout_seconds=timeout,
        allowed_tools=tools,
    )


def test_subprocess_backend_runs_without_shell_and_passes_structured_prompt(tmp_path):
    backend = SubprocessAgentBackend(
        "python-test",
        [sys.executable, "-c", "import sys; print(sys.argv[1])", "{prompt}"],
        inherit_env=False,
    )
    result = backend.execute(request(tmp_path))
    assert result.outcome == "completed"
    payload = json.loads(result.summary)
    assert payload["goal"]["objective"] == "check a small benchmark"
    assert payload["allowed_tools"] == ["read_files", "run_tests"]
    assert result.metadata["shell"] is False
    assert result.metadata["returncode"] == 0


def test_nonzero_exit_is_a_failure_not_success(tmp_path):
    backend = SubprocessAgentBackend(
        "failing-cli",
        [sys.executable, "-c", "import sys; print('bad', file=sys.stderr); sys.exit(7)"],
        inherit_env=False,
    )
    result = backend.execute(request(tmp_path))
    assert result.outcome == "failed"
    assert result.metadata["returncode"] == 7
    assert "bad" in result.summary


def test_timeout_is_bounded_and_reported(tmp_path):
    backend = SubprocessAgentBackend(
        "slow-cli",
        [sys.executable, "-c", "import time; time.sleep(2)"],
        inherit_env=False,
    )
    result = backend.execute(request(tmp_path, timeout=0.1))
    assert result.outcome == "failed"
    assert result.metadata["timed_out"] is True


def test_logs_are_persisted_as_artifacts_and_task_id_is_sanitized(tmp_path):
    backend = SubprocessAgentBackend(
        "logged-cli",
        [sys.executable, "-c", "print('hello'); import sys; print('note', file=sys.stderr)"],
        inherit_env=False,
        artifact_dir=tmp_path / "artifacts",
    )
    result = backend.execute(request(tmp_path))
    assert result.outcome == "completed"
    assert len(result.artifact_uris) == 2
    assert all(uri.startswith("file://") for uri in result.artifact_uris)
    assert list((tmp_path / "artifacts").iterdir())[0].name == "task_with_unsafe_chars"


def test_workspace_must_exist(tmp_path):
    backend = SubprocessAgentBackend("cli", [sys.executable, "-c", "print('ok')"])
    result = backend.execute(request(tmp_path / "missing"))
    assert result.outcome == "failed"
    assert "existing directory" in result.summary


def test_invalid_configuration_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="argv"):
        SubprocessAgentBackend("empty", [])
    with pytest.raises(ValueError, match="max_output_chars"):
        SubprocessAgentBackend("small", ["agent"], max_output_chars=1)


def test_missing_placeholder_is_reported(tmp_path):
    backend = SubprocessAgentBackend("cli", ["agent", "{unknown}"], inherit_env=False)
    import pytest

    with pytest.raises(ValueError, match="placeholder"):
        backend.execute(request(tmp_path))
