"""Solver adapters: the one interface a model must implement to be UQ'd.

A UQ study only ever sees the model through this interface, which keeps the
polynomial-chaos machinery non-intrusive (the solver is never modified) and
makes the *cost* of the evidence explicit: every adapter counts evaluations and
every per-sample exception becomes a recorded ``NaN`` instead of a crash, so the
gate can decide ``WITHHELD`` rather than a study silently averaging over the
samples that happened to succeed.

Two adapters cover the cases met in practice:

``CallableAdapter``
    A plain ``f(X) -> y`` with ``X`` of shape ``(n, d)``; use ``batch=False`` for
    a scalar-in/scalar-out solver (it is looped, and the loop is counted).
``RecordedAdapter``
    Wraps a table of already-computed evaluations (e.g. a remote/cluster
    campaign) and interpolates nothing: a missing sample is an error, because a
    silent interpolation would fabricate evidence.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Mapping, Sequence

import numpy as np

__all__ = [
    "AdapterFailure",
    "CallableAdapter",
    "RecordedAdapter",
    "SolverAdapter",
    "as_adapter",
]


class AdapterFailure(RuntimeError):
    """Raised when an adapter cannot produce a finite value for a sample."""


@dataclass
class AdapterStats:
    n_evaluations: int = 0
    n_failures: int = 0
    failures: list = field(default_factory=list)

    def record_failure(self, message: str) -> None:
        self.n_failures += 1
        if len(self.failures) < 20:
            self.failures.append(message)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "n_evaluations": int(self.n_evaluations),
            "n_failures": int(self.n_failures),
            "failure_samples": list(self.failures),
        }


class SolverAdapter:
    """Base class: named scalar quantity of interest driven by ``parameter_names``."""

    name: str = "adapter"
    quantity: str = "qoi"
    unit: str = ""
    parameter_names: Sequence[str] = ()

    def evaluate(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    @property
    def stats(self) -> AdapterStats:
        raise NotImplementedError

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "quantity": self.quantity,
            "unit": self.unit,
            "parameter_names": list(self.parameter_names),
            "stats": self.stats.as_dict() if hasattr(self, "stats") else {},
        }


class CallableAdapter(SolverAdapter):
    """Adapt a python callable to the solver-adapter contract."""

    def __init__(
        self,
        fn: Callable[[np.ndarray], Any],
        parameter_names: Sequence[str],
        *,
        name: str = "callable",
        quantity: str = "qoi",
        unit: str = "",
        batch: bool = True,
        chunk: int = 8192,
        strict: bool = False,
    ) -> None:
        self._fn = fn
        self.parameter_names = list(parameter_names)
        self.name = name
        self.quantity = quantity
        self.unit = unit
        self.batch = bool(batch)
        self.chunk = int(chunk)
        self.strict = bool(strict)
        self._stats = AdapterStats()

    @property
    def stats(self) -> AdapterStats:
        return self._stats

    def _call(self, x: np.ndarray) -> np.ndarray:
        out = self._fn(x) if self.batch else np.array([self._fn(row) for row in x], dtype=float)
        arr = np.asarray(out, dtype=float).ravel()
        if arr.size != x.shape[0]:
            raise AdapterFailure(
                f"adapter {self.name!r} returned {arr.size} values for {x.shape[0]} samples"
            )
        return arr

    def _call_per_row(self, x: np.ndarray) -> np.ndarray:
        """Scalar-in/scalar-out path: one failure must not poison its neighbours."""
        out = np.full(x.shape[0], np.nan)
        for k, row in enumerate(x):
            try:
                value = float(self._fn(row))
            except Exception as exc:  # noqa: BLE001 - deliberate: convert to NaN evidence
                if self.strict:
                    raise AdapterFailure(f"adapter {self.name!r} failed: {exc}") from exc
                self._stats.record_failure(f"sample {k}: {exc}")
                continue
            out[k] = value
        return out

    def _unused(self, arr, x):  # pragma: no cover - kept for signature symmetry
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size != x.shape[0]:
            raise AdapterFailure(
                f"adapter {self.name!r} returned {arr.size} values for {x.shape[0]} samples"
            )
        return arr

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[1] != len(self.parameter_names):
            raise AdapterFailure(
                f"adapter {self.name!r} expects {len(self.parameter_names)} parameters, got {X.shape[1]}"
            )
        out = np.empty(X.shape[0], dtype=float)
        if not self.batch:
            self._stats.n_evaluations += int(X.shape[0])
            return self._call_per_row(X)
        for start in range(0, X.shape[0], self.chunk):
            block = X[start : start + self.chunk]
            try:
                vals = self._call(block)
            except Exception as exc:  # noqa: BLE001 - deliberate: convert to NaN evidence
                if self.strict:
                    raise AdapterFailure(f"adapter {self.name!r} failed: {exc}") from exc
                self._stats.record_failure(f"rows {start}..{start + block.shape[0]}: {exc}")
                vals = np.full(block.shape[0], np.nan)
            self._stats.n_evaluations += int(block.shape[0])
            out[start : start + block.shape[0]] = vals
        return out


class RecordedAdapter(SolverAdapter):
    """Exact lookup of pre-computed evaluations (no interpolation, by design)."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        parameter_names: Sequence[str],
        *,
        name: str = "recorded",
        quantity: str = "qoi",
        unit: str = "",
        rtol: float = 1e-12,
    ) -> None:
        self._X = np.ascontiguousarray(np.asarray(X, dtype=float))
        self._y = np.asarray(y, dtype=float).ravel()
        if self._X.shape[0] != self._y.size:
            raise ValueError("X and y disagree on the number of samples")
        if self._X.shape[1] != len(parameter_names):
            raise ValueError("parameter_names does not match X")
        self.parameter_names = list(parameter_names)
        self.name = name
        self.quantity = quantity
        self.unit = unit
        self.rtol = float(rtol)
        self._stats = AdapterStats()

    @property
    def stats(self) -> AdapterStats:
        return self._stats

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, dtype=float))
        scale = np.maximum(np.abs(self._X), 1.0)
        out = np.full(X.shape[0], np.nan)
        for k, row in enumerate(X):
            hit = np.all(np.abs(self._X - row) <= self.rtol * scale, axis=1)
            idx = np.flatnonzero(hit)
            if idx.size:
                out[k] = self._y[idx[0]]
            else:
                self._stats.record_failure(f"sample {k} was never evaluated")
            self._stats.n_evaluations += 1
        return out


def as_adapter(model_or_callable, parameter_names: Sequence[str] | None = None, **kwargs) -> SolverAdapter:
    """Coerce a ``SolverAdapter`` or a callable into a ``SolverAdapter``."""
    if isinstance(model_or_callable, SolverAdapter):
        return model_or_callable
    if parameter_names is None:
        raise ValueError("parameter_names is required when adapting a bare callable")
    return CallableAdapter(model_or_callable, parameter_names, **kwargs)