"""Launcher and transport-backend tests (MPI-style runtime semantics)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.fem_parallel.comm import MPIComm, SerialComm, SharedMemoryComm
from tensorphysics.fem_parallel.runtime import available_backends, mpi_available, run


def test_serial_backend_runs_in_process():
    def worker(comm):
        assert isinstance(comm, SerialComm)
        assert comm.size == 1
        return {"rank": comm.rank, "sum": comm.sum_scalar(2.5)}

    res = run(worker, 1)
    assert res == [{"rank": 0, "sum": 2.5}]


def test_shared_memory_backend_gives_distinct_ranks():
    def worker(comm):
        assert isinstance(comm, SharedMemoryComm)
        return {"rank": comm.rank, "size": comm.size, "total": comm.sum_scalar(1.0)}

    res = run(worker, 4, barrier_timeout=300)
    assert sorted(r["rank"] for r in res) == [0, 1, 2, 3]
    assert {r["size"] for r in res} == {4}
    assert all(r["total"] == 4.0 for r in res)


def test_worker_exception_propagates_to_the_parent():
    def worker(comm):
        raise RuntimeError(f"rank {comm.rank} fails on purpose")

    with pytest.raises(RuntimeError, match="fails on purpose"):
        run(worker, 2, barrier_timeout=120)


def test_backend_discovery_is_honest_about_missing_mpi():
    backs = available_backends()
    assert "serial" in backs and "shmem" in backs
    if not mpi_available():
        assert "mpi" not in backs
        with pytest.raises(ImportError):
            MPIComm()


def test_exchange_and_reduction_agree_with_serial_semantics():
    """The shared-memory transport must reproduce the reference collectives."""
    def worker(comm):
        vec = np.arange(4, dtype=float) + comm.rank
        total = comm.allreduce_sum(vec)
        mx = comm.allreduce_max(np.array([float(comm.rank)]))
        peers = [(comm.rank + 1) % comm.size, (comm.rank - 1) % comm.size]
        sends = {p: np.full(3, float(comm.rank)) for p in peers}
        recv = comm.exchange_phase(sends, {p: 3 for p in peers})
        return {"rank": comm.rank, "total": total.tolist(), "max": float(mx[0]),
                "recv": {int(k): v.tolist() for k, v in recv.items()}}

    size = 4
    res = run(worker, size, barrier_timeout=300)
    ref = np.arange(4, dtype=float) * size + sum(range(size))
    for r in res:
        assert np.allclose(r["total"], ref)
        assert r["max"] == size - 1
        for peer, values in r["recv"].items():
            assert np.allclose(values, float(peer))


def test_slot_overflow_is_reported_not_silent():
    def worker(comm):
        n = 100_000  # far beyond the arena chunk for this rank count
        comm.exchange_phase({(comm.rank + 1) % comm.size: np.zeros(n)},
                            {(comm.rank - 1) % comm.size: n})
        return {"rank": comm.rank}

    with pytest.raises(RuntimeError, match="arena allows|enlarge slot_floats"):
        run(worker, 2, slot_floats=1024, barrier_timeout=120)
