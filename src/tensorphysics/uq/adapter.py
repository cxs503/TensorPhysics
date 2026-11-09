"""Solver adapter fan-out with hash-pinned merging.

The UQ workflow never imports a solver kernel: it calls a **black box**
``points (n, d) -> values (n,)``.  This module

* normalises such a black box (:class:`CallableAdapter`),
* evaluates a :class:`~tensorphysics.uq.sampler.SampleSet` in ``shards``
  independent chunks (:func:`fanout_evaluate`),
* merges shard results **by the sample-set hash** (:class:`SampleResults`),
  refusing a merge when a shard reports a different design, and
* records every failed evaluation as ``NaN`` plus an error message instead of
  silently dropping it (a dropped sample biases the surrogate; a NaN sample is
  detected by :mod:`tensorphysics.uq.verdict` and forces ``WITHHELD``).

Sharding is the hook used to dispatch a UQ campaign to a compute cluster: a
``runner`` is any callable ``runner(adapter, points, indices) -> values`` (or
``-> {"values": ..., "sample_hash": ...}``).  The default runner evaluates
in-process; :func:`cluster_shard_command` renders the ``run_cluster`` command
template for the same sample set so that the wiring is auditable rather than
implicit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence

import numpy as np

from .sampler import SampleSet

__all__ = [
    "AdapterContractError",
    "CallableAdapter",
    "SampleHashMismatch",
    "SampleResults",
    "SolverAdapter",
    "cluster_shard_command",
    "fanout_evaluate",
]


class AdapterContractError(ValueError):
    """The adapter violated the ``points (n, d) -> values (n,)`` contract.

    Deliberately *not* treated as a bad sample: a wrong result length means the
    adapter itself is broken, so it must surface as an error instead of being
    degraded into a per-row fallback.
    """


class SampleHashMismatch(RuntimeError):
    """Raised when a shard returns results for a different sample set."""

    def __init__(self, expected: str, received: str, shard: int) -> None:
        super().__init__(
            f"shard {shard} returned results for sample set {received[:12]}..., "
            f"expected {expected[:12]}..."
        )
        self.expected = expected
        self.received = received
        self.shard = shard


class SolverAdapter(Protocol):
    """Structural protocol for a black-box solver used by the UQ workflow."""

    def evaluate(self, points: np.ndarray) -> np.ndarray:  # pragma: no cover - protocol
        ...


class CallableAdapter:
    """Wrap a plain callable as a solver adapter.

    Parameters
    ----------
    fn:
        ``fn(points) -> values`` where ``points`` has shape ``(n, d)``.  With
        ``vectorized=True`` (default) one call receives the whole block; with
        ``vectorized=False`` the callable receives a single parameter dict and
        is applied row by row.
    names:
        Parameter names (used for row-wise calls and error messages).
    name:
        Human readable solver label recorded in the evidence.
    """

    def __init__(
        self,
        fn: Callable[..., Any],
        *,
        names: Sequence[str] | None = None,
        name: str = "callable",
        vectorized: bool = True,
    ) -> None:
        if not callable(fn):
            raise TypeError("fn must be callable")
        self._fn = fn
        self._names = tuple(names) if names is not None else None
        self.name = name
        self.vectorized = vectorized

    def evaluate(self, points: np.ndarray) -> np.ndarray:
        pts = np.asarray(points, dtype=float)
        if pts.ndim != 2:
            raise AdapterContractError(f"points must be 2-D, got shape {pts.shape}")
        if self.vectorized:
            values = np.asarray(self._fn(pts), dtype=float).reshape(-1)
        else:
            if self._names is None:
                raise AdapterContractError("row-wise adapters require parameter names")
            out = np.empty(pts.shape[0], dtype=float)
            for i, row in enumerate(pts):
                out[i] = float(self._fn(dict(zip(self._names, row.tolist()))))
            values = out
        if values.shape[0] != pts.shape[0]:
            raise AdapterContractError(
                f"adapter returned {values.shape[0]} values for {pts.shape[0]} points"
            )
        return values


@dataclass(frozen=True)
class SampleResults:
    """Evaluated samples with provenance and failure bookkeeping.

    ``values`` contains ``NaN`` for samples the adapter failed on; the indices
    and messages are kept so that the verdict can name the cause instead of
    reporting a generic failure.
    """

    values: np.ndarray = field(repr=False)
    sample_hash: str
    invalid_indices: tuple[int, ...] = ()
    error_messages: tuple[str, ...] = ()
    shards: int = 1

    @property
    def n(self) -> int:
        return int(self.values.shape[0])

    @property
    def valid(self) -> np.ndarray:
        return np.isfinite(self.values)

    @property
    def n_valid(self) -> int:
        return int(np.count_nonzero(self.valid))

    @property
    def has_invalid(self) -> bool:
        return self.n_valid != self.n

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_hash": self.sample_hash,
            "n": self.n,
            "n_valid": self.n_valid,
            "shards": int(self.shards),
            "invalid_indices": list(self.invalid_indices),
            "error_messages": list(self.error_messages),
        }


def _call_adapter(
    adapter: SolverAdapter, points: np.ndarray
) -> tuple[np.ndarray, tuple[int, ...], tuple[str, ...]]:
    """Evaluate with per-row failure capture (a failure becomes NaN)."""
    try:
        raw = adapter.evaluate(points)
    except AdapterContractError:
        raise
    except Exception as exc:  # noqa: BLE001 - recorded, never swallowed silently
        # Fall back to row-by-row so that one bad row does not lose the block.
        values = np.full(points.shape[0], np.nan)
        bad: list[int] = []
        messages: list[str] = []
        for i, row in enumerate(points):
            try:
                row_adapter_values = adapter.evaluate(row.reshape(1, -1))
                value = float(np.asarray(row_adapter_values).reshape(-1)[0])
                values[i] = value
                if not np.isfinite(value):
                    bad.append(i)
            except Exception as row_exc:  # noqa: BLE001
                bad.append(i)
                messages.append(f"row {i}: {type(row_exc).__name__}: {row_exc}")
        if not messages:
            messages.append(f"block evaluation failed: {type(exc).__name__}: {exc}")
        return values, tuple(bad), tuple(messages)
    # a block whose *size* is wrong is a contract violation, not a bad sample:
    # report it instead of silently degrading to row-wise evaluation.
    values = np.asarray(raw, dtype=float).reshape(-1)
    if values.shape[0] != points.shape[0]:
        raise AdapterContractError(
            f"adapter returned {values.shape[0]} values for {points.shape[0]} points"
        )
    bad = np.flatnonzero(~np.isfinite(values))
    return values, tuple(int(i) for i in bad), ()


def fanout_evaluate(
    adapter: SolverAdapter,
    samples: SampleSet,
    *,
    shards: int = 1,
    runner: Callable[[SolverAdapter, np.ndarray, np.ndarray], Any] | None = None,
) -> SampleResults:
    """Evaluate ``samples`` in ``shards`` chunks and merge by sample hash.

    The merge reassembles results strictly by the *original row index*, and
    (when the runner reports one) verifies that every shard was computed for
    the same sample-set hash; a mismatch raises :class:`SampleHashMismatch`
    instead of producing a silently mis-paired surrogate.
    """
    if not isinstance(samples, SampleSet):
        raise TypeError(f"samples must be a SampleSet, got {type(samples)!r}")
    if not isinstance(shards, int) or shards < 1:
        raise ValueError(f"shards must be a positive integer, got {shards!r}")
    n = samples.n
    shards = min(shards, n)
    values = np.full(n, np.nan)
    invalid: list[int] = []
    messages: list[str] = []
    bounds = np.linspace(0, n, shards + 1).astype(int)
    for s in range(shards):
        lo, hi = int(bounds[s]), int(bounds[s + 1])
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        pts = samples.points[idx]
        if runner is None:
            block, bad, msgs = _call_adapter(adapter, pts)
        else:
            out = runner(adapter, pts, idx)
            if isinstance(out, Mapping):
                reported = out.get("sample_hash")
                if reported is not None and reported != samples.hash:
                    raise SampleHashMismatch(samples.hash, str(reported), s)
                block = np.asarray(out["values"], dtype=float).reshape(-1)
                bad = tuple(int(i) for i in np.flatnonzero(~np.isfinite(block)))
                msgs = tuple(out.get("error_messages", ()))
            else:
                block = np.asarray(out, dtype=float).reshape(-1)
                bad = tuple(int(i) for i in np.flatnonzero(~np.isfinite(block)))
                msgs = ()
            if block.shape[0] != idx.shape[0]:
                raise AdapterContractError(
                    f"shard {s} returned {block.shape[0]} values for {idx.shape[0]} samples"
                )
        values[idx] = block
        invalid.extend(int(lo + i) for i in bad)
        messages.extend(msgs)
    return SampleResults(
        values=values,
        sample_hash=samples.hash,
        invalid_indices=tuple(sorted(invalid)),
        error_messages=tuple(messages),
        shards=int(shards),
    )


def cluster_shard_command(
    *,
    repo: str,
    script: str,
    shard_placeholder: bool = True,
    extra_args: Sequence[str] = (),
) -> str:
    """Render the ``run_cluster`` ``--cmd-tpl`` for a sharded UQ evaluation.

    Example
    -------
    >>> cluster_shard_command(repo="TensorPhysics", script="scripts/uq_shard_worker.py")
    'python scripts/uq_shard_worker.py --shard {i}/{n}'

    The worker script is expected to (a) rebuild the sample set from the
    pinned ``(spec, rule, seed, n)``, (b) select the shard's rows, (c) write
    results tagged with the sample-set hash.  Merging happens through
    :func:`fanout_evaluate`'s hash check, so a stale or mis-ordered shard is
    rejected rather than silently averaged in.
    """
    args = " ".join(extra_args)
    inner = f"python {script}"
    if shard_placeholder:
        inner += " --shard {i}/{n}"
    if args:
        inner += " " + args
    return inner