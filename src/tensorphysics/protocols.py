"""Stable contracts between the orchestrator and specialist solvers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

@dataclass(frozen=True)
class SolverContext:
    """Context shared with a solver for one lifecycle operation."""
    step: int
    time: float
    dt: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

@dataclass
class SolverState:
    """Backend-neutral snapshot envelope; serialization is adapter-owned."""
    solver_name: str
    step: int
    time: float
    payload: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

@runtime_checkable
class Solver(Protocol):
    @property
    def name(self) -> str: ...
    def initialize(self, context: SolverContext) -> None: ...
    def advance(self, context: SolverContext) -> None: ...
    def finalize(self, context: SolverContext) -> None: ...
    def snapshot(self, context: SolverContext) -> SolverState: ...
