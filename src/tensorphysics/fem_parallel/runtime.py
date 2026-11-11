"""Rank launcher: start ``nranks`` worker processes and hand each one a ``Comm``.

The launcher is the equivalent of ``mpirun -np N`` for the shared-memory
backend, and a thin pass-through when a real MPI runtime started the process.

Usage (library)::

    from tensorphysics.fem_parallel.runtime import run

    results = run(worker, nranks=8, slot_floats=1 << 16, kwargs={"n": 1000})
    # worker(comm, n=1000) is executed by every rank; one dict per rank is
    # returned to the parent process.

Usage (command line)::

    python -m tensorphysics.fem_parallel.runtime --nranks 8 bench.json
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
import time
import traceback
from typing import Any, Callable, Dict, List, Optional

from .comm import MPIComm, SerialComm, SharedMemoryComm

__all__ = ["run", "available_backends", "mpi_available", "main"]

DEFAULT_SLOT_FLOATS = 1 << 16


def mpi_available() -> bool:
    """True when ``mpi4py`` can be imported *and* the process runs under MPI."""
    try:
        import mpi4py  # noqa: F401 - availability probe only
    except Exception:
        return False
    # Common MPI launchers export at least one of these.
    return any(k in os.environ for k in ("OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_RANK"))


def available_backends() -> List[str]:
    backs = ["serial", "shmem"]
    if mpi_available():
        backs.append("mpi")
    return backs


def _resolve_backend(nranks: int, backend: str) -> str:
    backend = (backend or "auto").lower()
    if backend == "auto":
        if mpi_available():
            return "mpi"
        return "serial" if nranks <= 1 else "shmem"
    if backend not in available_backends():  # pragma: no cover - operator input
        raise ValueError(
            f"backend {backend!r} unavailable; choose from {available_backends()}"
        )
    if backend == "serial" and nranks > 1:  # pragma: no cover - operator input
        raise ValueError("serial backend only supports nranks=1")
    return backend


def run(
    worker: Callable[..., Dict[str, Any]],
    nranks: int,
    backend: str = "auto",
    slot_floats: int = DEFAULT_SLOT_FLOATS,
    kwargs: Optional[Dict[str, Any]] = None,
    barrier_timeout: float = 600.0,
) -> List[Dict[str, Any]]:
    """Execute ``worker(comm, **kwargs)`` on ``nranks`` ranks.

    Parameters
    ----------
    worker:
        Picklable module-level callable; receives the ``Comm`` as first argument.
    nranks:
        Number of ranks (processes).  ``1`` runs in-process.
    backend:
        ``"auto"`` (MPI if launched under mpirun, else shared memory),
        ``"serial"``, ``"shmem"`` or ``"mpi"``.
    slot_floats:
        Per-rank shared-memory arena size (in float64 words), only used by the
        ``shmem`` backend.  Must exceed ``nranks * max_chunk_floats``.
    """
    kwargs = dict(kwargs or {})
    resolved = _resolve_backend(nranks, backend)

    if resolved == "serial" or nranks == 1:
        comm = SerialComm()
        return [worker(comm, **kwargs)]

    if resolved == "mpi":
        comm = MPIComm()
        return [worker(comm, **kwargs)]

    return _run_shmem(worker, nranks, slot_floats, kwargs, barrier_timeout)


def _run_shmem(
    worker, nranks: int, slot_floats: int, kwargs, barrier_timeout: float = 600.0
) -> List[Dict[str, Any]]:
    """Launch ``nranks`` processes over a shared-memory arena.

    Results travel back through one pipe per rank, so the launcher never touches
    the filesystem (a full ``/tmp`` on a shared host would otherwise kill the
    run) and no temporary files are left behind.
    """
    import multiprocessing as mp
    from multiprocessing import shared_memory

    ctx = mp.get_context("fork")
    slot_floats = int(max(slot_floats, 256 * nranks))
    shm = shared_memory.SharedMemory(create=True, size=8 * slot_floats * nranks)
    barrier = ctx.Barrier(int(nranks))
    # int64 generation flags for the lock-free SpinBarrier (2 generations/rank)
    flags = shared_memory.SharedMemory(create=True, size=16 * int(nranks))
    procs, pipes = [], []
    payload = (worker, kwargs)
    t0 = time.perf_counter()
    for r in range(nranks):
        parent_conn, child_conn = ctx.Pipe(duplex=False)
        p = ctx.Process(
            target=_rank_entry,
            args=(payload, r, nranks, shm, barrier, slot_floats, barrier_timeout,
                  child_conn, flags),
            daemon=False,
        )
        p.start()
        child_conn.close()
        procs.append(p)
        pipes.append(parent_conn)

    from multiprocessing.connection import wait

    results: List[Dict[str, Any]] = []
    failure: Optional[str] = None
    pending = {conn: r for r, conn in enumerate(pipes)}
    while pending:
        ready = wait(list(pending), timeout=0.5)
        if not ready:
            # a rank died without reporting -> the others would block in a
            # collective forever, so stop the run instead of waiting it out
            dead = [p for p in procs if (not p.is_alive()) and p.exitcode not in (0, None)]
            if dead:
                failure = failure or (
                    f"worker pid {dead[0].pid} exited with {dead[0].exitcode} "
                    "before reporting a result"
                )
                break
            continue
        for conn in ready:
            r = pending.pop(conn)
            try:
                msg = conn.recv()
            except EOFError:  # pragma: no cover - rank died before sending
                msg = {"rank": r, "__error__": "rank exited without sending a result"}
            finally:
                conn.close()
            if isinstance(msg, dict) and msg.get("__error__"):
                failure = failure or f"rank {r} failed:\n{msg['__error__']}"
                break
            flat = dict(msg.get("result") or {})
            flat.setdefault("rank", msg.get("rank", r))
            results.append(flat)
        if failure:
            break
    if failure:
        for p in procs:
            if p.is_alive():  # pragma: no cover - only on a failed run
                p.terminate()
    for p in procs:
        p.join(timeout=30)
        if p.is_alive():  # pragma: no cover - pathological
            p.kill()
            p.join()
    spawn_and_run = time.perf_counter() - t0
    shm.close()
    shm.unlink()
    flags.close()
    flags.unlink()
    if failure is None:
        for p in procs:
            if p.exitcode not in (0, None):  # pragma: no cover - defensive
                failure = f"worker pid {p.pid} exited with {p.exitcode}"
    if failure:
        raise RuntimeError(failure)
    for res in results:
        res.setdefault("launch_wall", spawn_and_run)
    return results


def _rank_entry(payload, rank: int, nranks: int, shm, barrier, slot_floats,
                timeout: float, conn, flags=None) -> None:
    worker, kwargs = payload
    try:
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
            os.environ.setdefault(var, "1")
        comm = SharedMemoryComm(rank, nranks, shm, barrier, slot_floats, timeout,
                                flags=flags)
        result = worker(comm, **kwargs)
        payload_out: Dict[str, Any] = {"rank": rank, "result": result}
    except Exception:  # noqa: BLE001 - propagate to parent
        payload_out = {"rank": rank, "__error__": traceback.format_exc()}
    try:
        conn.send(payload_out)
    finally:
        conn.close()


def main(argv: Optional[List[str]] = None) -> int:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nranks", type=int, default=1)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--slot-floats", type=int, default=DEFAULT_SLOT_FLOATS)
    parser.add_argument("worker", help="'module:function' worker entry point")
    parser.add_argument("kwargs_json", nargs="?", default="{}")
    args = parser.parse_args(argv)
    import json

    mod, _, fn = args.worker.partition(":")
    worker = getattr(importlib.import_module(mod), fn or "main")
    kwargs = json.loads(args.kwargs_json)
    t0 = time.perf_counter()
    results = run(worker, args.nranks, args.backend, args.slot_floats, kwargs)
    print(f"# {args.nranks} ranks, {time.perf_counter() - t0:.2f}s")
    print(json.dumps(results[0], indent=2, default=float))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
