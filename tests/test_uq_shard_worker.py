"""Tests for the cluster shard worker (two subprocesses + a hash-pinned merge)."""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np

from tensorphysics.uq.adapter import CallableAdapter, fanout_evaluate
from tensorphysics.uq.benchmarks import holtrop_type_benchmark
from tensorphysics.uq.sampler import sample

ROOT = pathlib.Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "uq_shard_worker.py"


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(WORKER), *args], capture_output=True, text=True
    )


def test_shards_rebuild_the_design_and_merge_bitwise(tmp_path: pathlib.Path) -> None:
    bench = holtrop_type_benchmark()
    design = sample(bench.spec, 64, seed=20240)
    outs = []
    for index in range(2):
        out = tmp_path / f"shard_{index}.npz"
        completed = _run(
            "--benchmark", "holtrop", "--n", str(design.n), "--seed", str(design.seed),
            "--shard", f"{index}/2", "--out", str(out),
        )
        assert completed.returncode == 0, completed.stderr
        meta = json.loads(completed.stdout)
        assert meta["sample_hash"] == design.hash
        assert meta["n_nonfinite"] == 0
        outs.append(out)

    merged_path = tmp_path / "merged.npz"
    merged = _run("--merge", *map(str, outs), "--out", str(merged_path))
    assert merged.returncode == 0, merged.stderr
    payload = np.load(merged_path)
    reference = fanout_evaluate(CallableAdapter(bench.resistance_kn), design)
    assert str(payload["sample_hash"]) == design.hash
    assert np.array_equal(payload["values"], reference.values)


def test_stale_shard_is_refused(tmp_path: pathlib.Path) -> None:
    bench = holtrop_type_benchmark()
    design = sample(bench.spec, 32, seed=7)
    good = tmp_path / "good.npz"
    assert _run(
        "--benchmark", "holtrop", "--n", "32", "--seed", "7", "--shard", "0/1",
        "--out", str(good),
    ).returncode == 0
    stale = tmp_path / "stale.npz"
    np.savez(
        stale,
        values=np.zeros(32),
        indices=np.arange(32),
        sample_hash=np.array("0" * 64),
        spec_hash=np.array(design.spec.fingerprint()),
        n=np.array(32),
    )
    bad = _run("--merge", str(good), str(stale), "--out", str(tmp_path / "bad.npz"))
    assert bad.returncode != 0
    assert "hash mismatch" in bad.stderr


def test_worker_cli_guards() -> None:
    assert _run("--benchmark", "nope", "--n", "8").returncode != 0
    bad = _run("--benchmark", "holtrop", "--n", "8", "--shard", "3/2")
    assert bad.returncode != 0
    assert "outside" in bad.stderr


def test_ishigami_and_gfunction_variants() -> None:
    for name, n, seed in (("ishigami", 32, 1), ("gfunction", 32, 2)):
        out_name = f"/tmp/uq_shard_{name}.npz"
        completed = _run(
            "--benchmark", name, "--n", str(n), "--seed", str(seed),
            "--shard", "0/1", "--out", out_name,
        )
        assert completed.returncode == 0, completed.stderr
        payload = np.load(out_name)
        assert np.all(np.isfinite(payload["values"]))
        assert payload["values"].size == n