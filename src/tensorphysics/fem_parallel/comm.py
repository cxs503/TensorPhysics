"""Message-passing transport abstraction for the distributed FEM kernels.

The parallel finite-element infrastructure in :mod:`tensorphysics.fem_parallel`
is written against a *minimal* MPI-like API (rank/size, neighbour exchange,
global reduction, barrier).  Three interchangeable implementations are
provided:

``SerialComm``
    Single rank, zero-cost collectives (correctness reference).
``SharedMemoryComm``
    ``nranks`` independent OS processes launched by
    :mod:`tensorphysics.fem_parallel.runtime`, talking through a POSIX shared
    memory arena.  Every rank owns one slot of the arena; a neighbour exchange
    is "write my slot -> barrier -> read the peers' slots", which is the same
    copy-based transport MPI uses for intra-node neighbours, so the measured
    communication time includes real memcpy traffic and real synchronisation
    cost.  This is what makes it possible to measure strong/weak scaling and
    the communication fraction on a host without an MPI installation.
``MPIComm``
    Genuine ``mpi4py`` backend (``MPI.COMM_WORLD``).  Selected automatically
    when ``mpi4py`` can be imported and the process was started by ``mpirun``
    (see :func:`tensorphysics.fem_parallel.runtime.run`).

All three expose the same three primitives used by the solver:

``exchange_phase(sends, recv_lengths)``
    Simultaneous neighbour all-to-all: hand the array destined for each peer to
    that peer's receive buffer.  Ordering inside a chunk is caller-defined and
    must be symmetric (see :class:`tensorphysics.fem_parallel.operator.HaloPlan`).
``allreduce_sum(arr)`` / ``allreduce_max(arr)``
    Element-wise global reduction.
``barrier()``
    Global synchronisation.

Timing counters (:attr:`Comm.stats`) accumulate the wall time spent inside the
collectives, so the communication fraction ``T_comm / T_total`` required by the
campaign acceptance criteria is measured rather than assumed.
"""

from __future__ import annotations

import os
import time
from typing import Dict, Mapping, Tuple

import numpy as np

__all__ = [
    "Comm",
    "SpinBarrier",
    "SerialComm",
    "SharedMemoryComm",
    "MPIComm",
    "make_timer_stats",
]


def make_timer_stats() -> Dict[str, float]:
    """Fresh communication-accounting dictionary."""
    return {
        "exchange_time": 0.0,
        "allreduce_time": 0.0,
        "barrier_time": 0.0,
        "exchange_calls": 0,
        "allreduce_calls": 0,
        "barrier_calls": 0,
        "exchange_bytes": 0,
    }


class Comm:
    """MPI-like transport base class."""

    rank: int = 0
    size: int = 1
    name: str = "base"

    def __init__(self) -> None:
        self.stats = make_timer_stats()

    # -- introspection ----------------------------------------------------
    @property
    def is_root(self) -> bool:
        return self.rank == 0

    def reset_stats(self) -> None:
        self.stats = make_timer_stats()

    # -- collectives (implemented by the backends) -------------------------
    def exchange_phase(
        self,
        sends: Mapping[int, np.ndarray],
        recv_lengths: Mapping[int, int],
    ) -> Dict[int, np.ndarray]:  # pragma: no cover - interface
        raise NotImplementedError

    def allreduce_sum(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def allreduce_max(self, values: np.ndarray) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def exchange_and_reduce(self, sends, recv_lengths, values):
        """Exchange the ghost layer **and** reduce a small vector in one latency.

        Default implementation keeps the two collectives separate (correct but
        two latencies); backends that own their transport override it.  Callers
        must pass values whose local partial sums are already computed, i.e. the
        reduction must not depend on the exchanged data.
        """
        recv = self.exchange_phase(sends, recv_lengths)
        return recv, self.allreduce_sum(np.asarray(values, dtype=np.float64))

    def allreduce_sum_pair(self, first: float, second: float) -> Tuple[float, float]:
        """Reduce two scalars in a **single** collective.

        Conjugate gradients needs ``(r, z)`` and ``(r, r)`` in the same step;
        sending them as one two-element vector halves the number of global
        reductions per iteration, which dominates the cost at high rank counts
        (one latency instead of two).
        """
        out = self.allreduce_sum(np.asarray([float(first), float(second)], dtype=np.float64))
        return float(out[0]), float(out[1])

    def barrier(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    # -- helpers -----------------------------------------------------------
    def sum_scalar(self, value: float) -> float:
        out = self.allreduce_sum(np.asarray([float(value)], dtype=np.float64))
        return float(out[0])

    def max_scalar(self, value: float) -> float:
        out = self.allreduce_max(np.asarray([float(value)], dtype=np.float64))
        return float(out[0])

    def allgather_scalars(self, value: float) -> np.ndarray:
        """Gather one scalar per rank (via a length-``size`` vector)."""
        buf = np.zeros(self.size, dtype=np.float64)
        contribution = np.zeros(self.size, dtype=np.float64)
        contribution[self.rank] = float(value)
        buf = self.allreduce_sum(contribution)
        return buf

    def comm_time(self) -> float:
        """Total time inside the collectives.

        ``barrier_time`` is a *subset* of ``exchange_time``/``allreduce_time``
        (each collective is timed around its own barrier), so it is reported
        separately and not added here -- otherwise the barrier latency would be
        counted twice.
        """
        return float(self.stats["exchange_time"] + self.stats["allreduce_time"])


class SerialComm(Comm):
    """Single-rank transport: collectives are the identity."""

    name = "serial"

    def __init__(self, rank: int = 0, size: int = 1) -> None:
        super().__init__()
        self.rank = rank
        self.size = size

    def exchange_phase(self, sends, recv_lengths):  # noqa: D401 - trivial
        t0 = time.perf_counter()
        out = {
            int(p): np.zeros(int(n), dtype=np.float64)
            for p, n in recv_lengths.items()
        }
        self.stats["exchange_calls"] += 1
        self.stats["exchange_bytes"] += int(sum(v.size for v in sends.values()) * 8)
        self.stats["exchange_time"] += time.perf_counter() - t0
        return out

    def allreduce_sum(self, values):
        t0 = time.perf_counter()
        out = np.array(values, dtype=np.float64, copy=True)
        self.stats["allreduce_calls"] += 1
        self.stats["allreduce_time"] += time.perf_counter() - t0
        return out

    def allreduce_max(self, values):
        t0 = time.perf_counter()
        out = np.array(values, dtype=np.float64, copy=True)
        self.stats["allreduce_calls"] += 1
        self.stats["allreduce_time"] += time.perf_counter() - t0
        return out

    def barrier(self) -> None:
        self.stats["barrier_calls"] += 1


class SpinBarrier:
    """Sense-reversal barrier over a shared ``int64`` flag array (no lock).

    ``multiprocessing.Barrier`` is built on a lock + condition variable: every
    collective wakes ``size`` processes through the kernel, which costs tens of
    milliseconds once the ranks exceed the free cores.  Here each rank publishes
    its generation number with one aligned 8-byte store and spin-waits (with
    ``sched_yield``) until every peer has published the same generation; the
    latency is then pinned to the *slowest peer*, not to a centralised lock, and
    the metadata stays O(1) per rank.

    Two flag generations are enough: a rank can never run more than one
    collective ahead of its peers, so the slot it writes is never one that a
    peer is still reading.
    """

    def __init__(self, flags, size: int, rank: int, spin_rounds: int = 2000) -> None:
        self._buf = flags
        self._arr = np.frombuffer(flags.buf, dtype=np.int64, count=2 * int(size))
        self.size = int(size)
        self.rank = int(rank)
        self.gen = 0
        self.spin_rounds = int(spin_rounds)
        self.stats = {"barrier_calls": 0, "barrier_time": 0.0}

    def wait(self, timeout: float = 600.0) -> float:
        t0 = time.perf_counter()
        self.gen += 1
        g = self.gen
        base = ((g - 1) % 2) * self.size
        self._arr[base + self.rank] = g
        spins = 0
        while True:
            vals = self._arr[base : base + self.size]
            if bool((vals >= g).all()):
                break
            spins += 1
            if spins % self.spin_rounds == 0:
                os.sched_yield()
                if time.perf_counter() - t0 > timeout:  # pragma: no cover - lost peer
                    raise RuntimeError(
                        f"spin barrier timeout after {timeout}s on rank {self.rank}"
                    )
        dt = time.perf_counter() - t0
        self.stats["barrier_calls"] += 1
        self.stats["barrier_time"] += dt
        return dt


class SharedMemoryComm(Comm):
    """Process-level transport over a POSIX shared-memory arena.

    Arena layout: ``size`` slots of ``slot_floats`` float64 values each; rank
    ``r`` owns slot ``r``.  Every slot is split in two generations (double
    buffering) so a collective needs a single barrier: the generation is flipped
    on each call, hence no rank can overwrite a buffer that a slower peer is
    still reading.  Inside a generation the slot is cut into ``size`` chunks of
    ``chunk = (slot_floats // 2) // size`` floats: rank ``r`` writes the data
    destined for peer ``q`` at ``q * chunk`` and rank ``q`` reads it at
    ``r * chunk``.  The fixed chunking keeps the protocol metadata-free, so the
    measured exchange cost is memcpy bandwidth + barrier latency, not
    bookkeeping.
    """

    name = "shmem"

    def __init__(
        self, rank: int, size: int, arena, barrier, slot_floats: int, timeout: float = 600.0,
        flags=None, spin_rounds: int = 2000,
    ) -> None:
        super().__init__()
        self.rank = int(rank)
        self.size = int(size)
        self._arena = arena  # multiprocessing.shared_memory.SharedMemory
        self._flat = np.frombuffer(
            self._arena.buf, dtype=np.float64, count=slot_floats * size
        )
        self._slots = self._flat.reshape(size, slot_floats)
        self._barrier = barrier
        self._slot_floats = int(slot_floats)
        self._timeout = float(timeout)
        # spin barrier (preferred): one int64 flag array shared by all ranks
        self._spin = (
            SpinBarrier(flags, size, rank, spin_rounds=spin_rounds)
            if flags is not None
            else None
        )
        # double buffering: generation alternates on every collective, so only
        # one barrier per collective is needed (a rank can never overwrite the
        # buffer a slower peer is still reading).
        self._half = int(slot_floats) // 2
        # keep a small tail free for the fused reduction vector (see
        # ``exchange_and_reduce``), otherwise a size that divides the half
        # exactly would leave no room for it
        self._red_reserve = 8
        self._chunk = max(1, (self._half - self._red_reserve) // int(size))
        self._gen = 0
        # tail of each slot (after the peer chunks) holds the reduction vector
        self._red_width = int(self._half - self._chunk * int(size))
        if self._chunk < 1:  # pragma: no cover - defensive
            raise ValueError(
                f"shared-memory arena too small: slot_floats={slot_floats} "
                f"for {size} ranks"
            )

    # -- primitives --------------------------------------------------------
    def _wait(self) -> float:
        if self._spin is not None:
            dt = self._spin.wait(timeout=self._timeout)
            self.stats["barrier_time"] += dt
            self.stats["barrier_calls"] += 1
            return dt
        t0 = time.perf_counter()
        # A timeout turns a lost neighbour (crashed rank, mismatched collective
        # count) into a BrokenBarrierError instead of an unkillable hang.
        self._barrier.wait(timeout=self._timeout)
        dt = time.perf_counter() - t0
        self.stats["barrier_time"] += dt
        self.stats["barrier_calls"] += 1
        return dt

    def _next_base(self) -> int:
        gen = self._gen
        self._gen ^= 1
        return gen * self._half

    def exchange_phase(self, sends, recv_lengths):
        t0 = time.perf_counter()
        base = self._next_base()
        own = self._slots[self.rank]
        moved = 0
        for peer, arr in sends.items():
            peer = int(peer)
            arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
            n = arr.size
            if n > self._chunk:  # pragma: no cover - guard
                raise RuntimeError(
                    f"rank {self.rank}: chunk for peer {peer} is {n} floats but "
                    f"arena allows {self._chunk}; enlarge slot_floats"
                )
            off = base + peer * self._chunk
            own[off : off + n] = arr
            moved += n
        self._wait()
        out = {}
        my_off = base + self.rank * self._chunk
        for peer, n in recv_lengths.items():
            peer = int(peer)
            n = int(n)
            out[peer] = np.array(
                self._slots[peer][my_off : my_off + n], dtype=np.float64, copy=True
            )
            moved += n
        self.stats["exchange_time"] += time.perf_counter() - t0
        self.stats["exchange_calls"] += 1
        self.stats["exchange_bytes"] += moved * 8
        return out

    def allreduce_sum(self, values):
        return self._reduce(values, op="sum")

    def allreduce_max(self, values):
        return self._reduce(values, op="max")

    def exchange_and_reduce(self, sends, recv_lengths, values):
        """Ghost exchange + small-vector reduction behind **one** barrier.

        The exchange data live in the per-peer chunk region of the rank slot,
        the reduction contribution in the tail region of the same slot; both are
        published (and consumed) inside one generation, so the two collectives
        cost one barrier instead of two -- the whole point of fusing them.
        """
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
        n = int(arr.size)
        if n > self._red_width:  # pragma: no cover - guard
            raise RuntimeError(f"reduction vector {n} exceeds tail width {self._red_width}")
        base = self._next_base()
        own = self._slots[self.rank]
        moved = 0
        for peer, chunk in sends.items():
            peer = int(peer)
            chunk = np.ascontiguousarray(chunk, dtype=np.float64).ravel()
            if chunk.size > self._chunk:  # pragma: no cover - guard
                raise RuntimeError(
                    f"rank {self.rank}: chunk for peer {peer} is {chunk.size} floats but "
                    f"arena allows {self._chunk}; enlarge slot_floats"
                )
            off = base + peer * self._chunk
            own[off : off + chunk.size] = chunk
            moved += chunk.size
        red_off = base + self._chunk * self.size
        own[red_off : red_off + n] = arr
        self._wait()
        out = {}
        my_off = base + self.rank * self._chunk
        for peer, length in recv_lengths.items():
            peer = int(peer)
            length = int(length)
            out[peer] = np.array(
                self._slots[peer][my_off : my_off + length], dtype=np.float64, copy=True
            )
            moved += length
        acc = np.array(self._slots[0][red_off : red_off + n], dtype=np.float64, copy=True)
        for r in range(1, self.size):
            acc += self._slots[r][red_off : red_off + n]
        self.stats["exchange_time"] += time.perf_counter() - t0
        self.stats["exchange_calls"] += 1
        self.stats["exchange_bytes"] += moved * 8
        return out, acc

    def _reduce(self, values, op: str):
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
        n = arr.size
        if n > self._chunk:  # pragma: no cover - guard
            # element-wise reduction does not need more than chunk floats; a
            # larger vector is split into chunks (deterministic order).
            out = np.empty(n, dtype=np.float64)
            for start in range(0, n, self._chunk):
                stop = min(start + self._chunk, n)
                out[start:stop] = self._reduce(arr[start:stop], op)
            self.stats["allreduce_time"] += time.perf_counter() - t0
            return out
        base = self._next_base()
        self._slots[self.rank][base : base + n] = arr
        self._wait()
        acc = np.array(self._slots[0][base : base + n], dtype=np.float64, copy=True)
        for r in range(1, self.size):
            peer = self._slots[r][base : base + n]
            if op == "sum":
                acc += peer
            else:
                np.maximum(acc, peer, out=acc)
        self.stats["allreduce_time"] += time.perf_counter() - t0
        self.stats["allreduce_calls"] += 1
        return acc

    def barrier(self) -> None:
        self._wait()


class MPIComm(Comm):
    """Genuine MPI transport (``mpi4py``), used when launched under ``mpirun``."""

    name = "mpi"

    def __init__(self, mpi_comm=None) -> None:
        from mpi4py import MPI  # local import: optional dependency

        super().__init__()
        self._MPI = MPI
        self._comm = mpi_comm if mpi_comm is not None else MPI.COMM_WORLD
        self.rank = int(self._comm.Get_rank())
        self.size = int(self._comm.Get_size())

    def barrier(self) -> None:
        t0 = time.perf_counter()
        self._comm.Barrier()
        self.stats["barrier_time"] += time.perf_counter() - t0
        self.stats["barrier_calls"] += 1

    def exchange_phase(self, sends, recv_lengths):
        t0 = time.perf_counter()
        MPI = self._MPI
        recv_buf = {
            int(p): np.empty(int(n), dtype=np.float64) for p, n in recv_lengths.items()
        }
        reqs = []
        for peer, arr in sends.items():
            arr = np.ascontiguousarray(arr, dtype=np.float64).ravel()
            reqs.append(self._comm.Isend(arr, dest=int(peer), tag=11))
        for peer, buf in recv_buf.items():
            reqs.append(self._comm.Irecv(buf, source=int(peer), tag=11))
        MPI.Request.Waitall(reqs)
        moved = int(sum(v.size for v in sends.values()) + sum(v.size for v in recv_buf.values()))
        self.stats["exchange_time"] += time.perf_counter() - t0
        self.stats["exchange_calls"] += 1
        self.stats["exchange_bytes"] += moved * 8
        return recv_buf

    def allreduce_sum(self, values):
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
        out = np.empty_like(arr)
        self._comm.Allreduce(arr, out, op=self._MPI.SUM)
        self.stats["allreduce_time"] += time.perf_counter() - t0
        self.stats["allreduce_calls"] += 1
        return out

    def allreduce_max(self, values):
        t0 = time.perf_counter()
        arr = np.ascontiguousarray(np.asarray(values, dtype=np.float64).ravel())
        out = np.empty_like(arr)
        self._comm.Allreduce(arr, out, op=self._MPI.MAX)
        self.stats["allreduce_time"] += time.perf_counter() - t0
        self.stats["allreduce_calls"] += 1
        return out
