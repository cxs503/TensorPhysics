"""Tests for the solver-adapter fan-out and its hash-pinned merge."""
from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.uq.adapter import (
    CallableAdapter,
    SampleHashMismatch,
    cluster_shard_command,
    fanout_evaluate,
)
from tensorphysics.uq.sample_spec import Marginal, SampleSpec
from tensorphysics.uq.sampler import sample


@pytest.fixture()
def spec() -> SampleSpec:
    return SampleSpec(
        (Marginal.uniform("a", 0.0, 1.0), Marginal.uniform("b", 0.0, 4.0))
    )


def _model(points: np.ndarray) -> np.ndarray:
    return 3.0 * points[:, 0] + points[:, 1]


def test_sharding_preserves_order_and_values(spec: SampleSpec) -> None:
    ss = sample(spec, 103, seed=7)
    adapter = CallableAdapter(_model, name="linear")
    whole = fanout_evaluate(adapter, ss, shards=1)
    for shards in (2, 3, 5, 200):
        part = fanout_evaluate(adapter, ss, shards=shards)
        assert np.array_equal(part.values, whole.values)
        assert part.sample_hash == ss.hash == whole.sample_hash
        assert part.shards == min(shards, ss.n)
        assert part.invalid_indices == ()


def test_row_wise_adapter_matches_vectorised(spec: SampleSpec) -> None:
    ss = sample(spec, 40, seed=8)
    vec = CallableAdapter(_model, names=spec.names)
    row = CallableAdapter(
        lambda params: 3.0 * params["a"] + params["b"], names=spec.names, vectorized=False
    )
    assert np.allclose(vec.evaluate(ss.points), row.evaluate(ss.points))


def test_hash_mismatch_is_rejected(spec: SampleSpec) -> None:
    ss = sample(spec, 32, seed=9)
    adapter = CallableAdapter(_model)

    def stale_runner(adapter_, points, idx):
        return {"values": _model(points), "sample_hash": "0" * 64}

    with pytest.raises(SampleHashMismatch) as excinfo:
        fanout_evaluate(adapter, ss, shards=2, runner=stale_runner)
    assert excinfo.value.expected == ss.hash
    assert excinfo.value.shard == 0


def test_runner_reporting_the_right_hash_is_accepted(spec: SampleSpec) -> None:
    ss = sample(spec, 32, seed=9)
    adapter = CallableAdapter(_model)

    def runner(adapter_, points, idx):
        return {"values": _model(points), "sample_hash": ss.hash}

    out = fanout_evaluate(adapter, ss, shards=4, runner=runner)
    assert np.allclose(out.values, _model(ss.points))


def test_failed_rows_are_recorded_not_dropped(spec: SampleSpec) -> None:
    ss = sample(spec, 50, seed=10)

    def flaky(points: np.ndarray) -> np.ndarray:
        values = _model(points)
        values[points[:, 0] > 0.9] = np.nan
        return values

    out = fanout_evaluate(CallableAdapter(flaky), ss, shards=2)
    expected_bad = tuple(int(i) for i in np.flatnonzero(ss.points[:, 0] > 0.9))
    assert out.invalid_indices == expected_bad
    assert out.has_invalid
    assert out.n_valid == ss.n - len(expected_bad)
    assert np.isnan(out.values[list(expected_bad)]).all()


def test_exception_raises_only_the_bad_rows(spec: SampleSpec) -> None:
    ss = sample(spec, 20, seed=11)

    def raising(points: np.ndarray) -> np.ndarray:
        if np.any(points[:, 1] > 3.5):
            raise RuntimeError("geometry outside the validated envelope")
        return _model(points)

    out = fanout_evaluate(CallableAdapter(raising), ss)
    assert out.has_invalid
    assert out.error_messages, "the failure reason must be captured"
    assert "geometry outside the validated envelope" in " ".join(out.error_messages)
    assert out.as_dict()["n_valid"] == out.n_valid


def test_adapter_result_size_is_validated(spec: SampleSpec) -> None:
    ss = sample(spec, 10, seed=12)
    with pytest.raises(ValueError):
        fanout_evaluate(CallableAdapter(lambda p: np.zeros(3), vectorized=True), ss)
    with pytest.raises(TypeError):
        fanout_evaluate(lambda p: p, "not a sample set")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        fanout_evaluate(CallableAdapter(_model), ss, shards=0)


def test_shard_count_cannot_exceed_samples(spec: SampleSpec) -> None:
    ss = sample(spec, 5, seed=13)
    out = fanout_evaluate(CallableAdapter(_model), ss, shards=50)
    assert out.shards == 5
    assert np.allclose(out.values, _model(ss.points))


def test_cluster_command_renders_run_cluster_template() -> None:
    cmd = cluster_shard_command(repo="TensorPhysics", script="scripts/uq_shard_worker.py")
    assert cmd == "python scripts/uq_shard_worker.py --shard {i}/{n}"
    cmd2 = cluster_shard_command(
        repo="TensorPhysics",
        script="scripts/uq_shard_worker.py",
        extra_args=("--job uq-ishigami",),
    )
    assert cmd2.endswith("--shard {i}/{n} --job uq-ishigami")