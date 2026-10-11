import json
import pytest
from tensorphysics import (
    CheckpointError, CheckpointStore, SimulationConfig, SolverState,
)


def make_state(name="TensorLBM", step=4, time=0.4):
    return SolverState(
        solver_name=name, step=step, time=time,
        payload={"density_min": 0.98, "iterations": step},
    )


def test_checkpoint_round_trip(tmp_path):
    path = tmp_path / "run" / "checkpoint.json"
    config = SimulationConfig(dt=0.1, end_time=2.0, max_steps=50)
    CheckpointStore.save(
        path, config=config, step=4, time=0.4,
        states={"TensorLBM": make_state()},
    )
    loaded = CheckpointStore.load(path)
    assert loaded.schema_version == 1
    assert loaded.step == 4
    assert loaded.time == pytest.approx(0.4)
    assert loaded.config == config
    assert loaded.solvers["TensorLBM"].payload["iterations"] == 4


def test_checkpoint_rejects_mismatched_clock(tmp_path):
    with pytest.raises(ValueError, match="snapshot step"):
        CheckpointStore.save(
            tmp_path / "bad.json", config=SimulationConfig(0.1, 1.0),
            step=5, time=0.5, states={"TensorLBM": make_state()},
        )


def test_checkpoint_rejects_non_json_payload_without_leaving_temp_file(tmp_path):
    path = tmp_path / "bad.json"
    state = SolverState("TensorLBM", 0, 0.0, {"array": object()})
    with pytest.raises(CheckpointError, match="failed to write"):
        CheckpointStore.save(
            path, config=SimulationConfig(0.1, 1.0), step=0, time=0.0,
            states={"TensorLBM": state},
        )
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_checkpoint_rejects_unknown_schema_version(tmp_path):
    path = tmp_path / "unknown.json"
    path.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
    with pytest.raises(CheckpointError, match="unsupported checkpoint schema"):
        CheckpointStore.load(path)


def test_checkpoint_rejects_corrupt_json(tmp_path):
    path = tmp_path / "corrupt.json"
    path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(CheckpointError, match="failed to read"):
        CheckpointStore.load(path)
