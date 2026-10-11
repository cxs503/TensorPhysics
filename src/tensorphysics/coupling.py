"""Explicit, deterministic phases for solver-to-solver coupling callbacks."""
from dataclasses import dataclass
from enum import Enum
from typing import Callable

from .fields import FieldStore
from .protocols import SolverContext


class CouplingError(RuntimeError):
    """A coupling callback registration or execution failed."""


class CouplingPhase(str, Enum):
    PRE_STEP = "pre_step"
    AFTER_SOLVER = "after_solver"
    POST_STEP = "post_step"


@dataclass(frozen=True)
class CouplingContext:
    """Context passed to a coupling operator."""

    solver: SolverContext
    fields: FieldStore
    solver_name: str | None = None


CouplingCallback = Callable[[CouplingContext], None]


@dataclass(frozen=True)
class _Registration:
    phase: CouplingPhase
    callback: CouplingCallback
    solver_name: str | None
    priority: int
    sequence: int


class CouplingManager:
    """Runs coupling callbacks in stable priority/registration order.

    AFTER_SOLVER callbacks run immediately after the named solver advances. This
    allows a downstream solver later in registration order to consume fresh data
    in the same global step. Callbacks are intentionally synchronous; distributed
    exchange and rollback policies are not implied.
    """

    def __init__(self, fields: FieldStore | None = None) -> None:
        self.fields = fields or FieldStore()
        self._registrations: list[_Registration] = []
        self._sequence = 0

    def register(self, phase: CouplingPhase, callback: CouplingCallback, *,
                 solver_name: str | None = None, priority: int = 0) -> None:
        if not isinstance(phase, CouplingPhase):
            raise TypeError("phase must be a CouplingPhase")
        if not callable(callback):
            raise TypeError("callback must be callable")
        if phase is CouplingPhase.AFTER_SOLVER:
            if not isinstance(solver_name, str) or not solver_name.strip():
                raise ValueError("solver_name is required for AFTER_SOLVER callbacks")
        elif solver_name is not None:
            raise ValueError("solver_name is only valid for AFTER_SOLVER callbacks")
        if not isinstance(priority, int) or isinstance(priority, bool):
            raise TypeError("priority must be an integer")
        self._registrations.append(
            _Registration(phase, callback, solver_name, priority, self._sequence)
        )
        self._sequence += 1

    def execute(self, phase: CouplingPhase, context: CouplingContext, *,
                solver_name: str | None = None) -> None:
        if phase is CouplingPhase.AFTER_SOLVER and not solver_name:
            raise ValueError("solver_name is required when executing AFTER_SOLVER")
        selected = [
            item for item in self._registrations
            if item.phase is phase and (
                phase is not CouplingPhase.AFTER_SOLVER or item.solver_name == solver_name
            )
        ]
        for item in sorted(selected, key=lambda entry: (entry.priority, entry.sequence)):
            item.callback(context)
