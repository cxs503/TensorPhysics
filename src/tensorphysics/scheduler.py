"""Deterministic lifecycle scheduler for independent physics solvers."""
from enum import Enum
from typing import Iterable
from .config import SimulationConfig
from .diagnostics import Diagnostics
from .protocols import Solver, SolverContext


class SimulationError(RuntimeError):
    """A solver lifecycle operation failed."""


class SimulationState(str, Enum):
    CREATED = "created"
    INITIALIZED = "initialized"
    RUNNING = "running"
    FINISHED = "finished"
    LIMIT_REACHED = "limit_reached"
    FAILED = "failed"
    CLOSED = "closed"


class Simulation:
    """Synchronous orchestration; coupling policies remain separate components."""

    def __init__(self, config: SimulationConfig, solvers: Iterable[Solver] = (),
                 diagnostics: Diagnostics | None = None) -> None:
        self.config = config
        self.diagnostics = diagnostics or Diagnostics()
        self.solvers: list[Solver] = []
        self.state = SimulationState.CREATED
        self.step = 0
        self.time = 0.0
        self._initialized: list[Solver] = []
        for solver in solvers:
            self.register(solver)

    def register(self, solver: Solver) -> None:
        if self.state is not SimulationState.CREATED:
            raise SimulationError("solvers can only be registered before initialize")
        if not isinstance(solver.name, str) or not solver.name.strip():
            raise ValueError("solver.name must be a non-empty string")
        if any(item.name == solver.name for item in self.solvers):
            raise ValueError(f"duplicate solver name: {solver.name}")
        for method in ("initialize", "advance", "finalize", "snapshot"):
            if not callable(getattr(solver, method, None)):
                raise TypeError(f"solver {solver.name!r} lacks callable {method}()")
        self.solvers.append(solver)

    def _context(self, dt: float | None = None) -> SolverContext:
        return SolverContext(self.step, self.time,
                             self.config.dt if dt is None else dt,
                             {"simulation_state": self.state.value})

    def initialize(self) -> None:
        if self.state is not SimulationState.CREATED:
            raise SimulationError(f"cannot initialize from state {self.state.value}")
        try:
            for solver in self.solvers:
                solver.initialize(self._context())
                self._initialized.append(solver)
                self.diagnostics.emit("solver_initialized", self.step, self.time,
                                      {"solver": solver.name})
            self.state = SimulationState.INITIALIZED
            self.diagnostics.emit("simulation_initialized", self.step, self.time,
                                  {"solver_count": len(self.solvers)})
        except Exception as exc:
            self.state = SimulationState.FAILED
            cleanup_errors = self._finalize_initialized()
            message = f"initialization failed: {exc}"
            if cleanup_errors:
                message += "; cleanup errors: " + "; ".join(cleanup_errors)
            raise SimulationError(message) from exc

    def run(self, steps: int | None = None) -> None:
        if steps is not None and (not isinstance(steps, int) or isinstance(steps, bool)):
            raise TypeError("steps must be an integer or None")
        if steps is not None and steps < 0:
            raise ValueError("steps must be non-negative")
        if self.state is SimulationState.CREATED:
            self.initialize()
        if self.state is not SimulationState.INITIALIZED:
            raise SimulationError(f"cannot run from state {self.state.value}")
        remaining = self.config.planned_steps - self.step
        target = remaining if steps is None else min(steps, remaining)
        self.state = SimulationState.RUNNING
        try:
            for _ in range(target):
                next_dt = min(self.config.dt, max(0.0, self.config.end_time - self.time))
                if next_dt <= self.config.time_tolerance:
                    break
                context = self._context(next_dt)
                for solver in self.solvers:
                    solver.advance(context)
                self.time += next_dt
                self.step += 1
                self.diagnostics.emit("step_completed", self.step, self.time, {"dt": next_dt})
            if self.time + self.config.time_tolerance >= self.config.end_time:
                self.state = SimulationState.FINISHED
            elif self.step >= self.config.planned_steps:
                self.state = SimulationState.LIMIT_REACHED
            else:
                self.state = SimulationState.INITIALIZED
        except Exception as exc:
            self.state = SimulationState.FAILED
            self.diagnostics.emit("solver_failed", self.step, self.time, {"error": repr(exc)})
            raise SimulationError(f"advance failed at step {self.step}, time {self.time}: {exc}") from exc

    def snapshots(self):
        if self.state in (SimulationState.CREATED, SimulationState.CLOSED):
            raise SimulationError(f"cannot snapshot from state {self.state.value}")
        context = self._context()
        return {solver.name: solver.snapshot(context) for solver in self.solvers}

    def _finalize_initialized(self) -> list[str]:
        errors = []
        for solver in reversed(self._initialized):
            try:
                solver.finalize(self._context())
            except Exception as exc:
                errors.append(f"{solver.name}: {exc}")
        self._initialized.clear()
        return errors

    def close(self) -> None:
        if self.state is SimulationState.CLOSED:
            return
        errors = self._finalize_initialized()
        self.state = SimulationState.CLOSED
        self.diagnostics.emit("simulation_closed", self.step, self.time, {"finalize_errors": errors})
        if errors:
            raise SimulationError("finalization errors: " + "; ".join(errors))

    def __enter__(self):
        self.initialize()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False
