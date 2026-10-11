"""Adapter for TensorDEM's 2-D IceDEM stepping API.

TensorDEM remains an optional dependency: callers construct an IceDEM instance
and pass it here, so importing TensorPhysics core never imports PyTorch.
"""
from __future__ import annotations

from math import isclose, isfinite
from typing import Any, Callable

from ..protocols import SolverContext


class TensorDEMAdapterError(RuntimeError):
    """The native DEM model cannot honor the orchestration step contract."""


LoadProvider = Callable[[SolverContext, Any], Any | None]


class TensorDEMAdapter:
    """Adapt a native tensordem.IceDEM object to the TensorPhysics Solver API.

    The DEM's native time step must divide each global increment into an integer
    number of substeps. Loads are requested at each DEM substep start and held
    constant for that substep. This adapter is for the 2-D IceDEM API, whose
    step accepts both external_forces and load_time.

    snapshot intentionally returns orchestration metadata only; full restart
    requires a native solver state/restore contract and is not implied here.
    If a later substep fails, earlier native substeps are not rolled back; the
    failed simulation must not be resumed from that partially advanced model.
    """

    def __init__(self, model: Any, *, name: str = "TensorDEM",
                 load_provider: LoadProvider | None = None,
                 time_tolerance: float = 1e-10) -> None:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("name must be a non-empty string")
        if not isfinite(time_tolerance) or time_tolerance < 0:
            raise ValueError("time_tolerance must be finite and non-negative")
        if not callable(getattr(model, "step", None)):
            raise TypeError("DEM model lacks callable step()")
        if not hasattr(model, "dt") or not hasattr(model, "time"):
            raise TypeError("DEM model must expose dt and time")
        self.model = model
        self.name = name
        self.load_provider = load_provider
        self.time_tolerance = time_tolerance
        self._substeps = 0

    def initialize(self, context: SolverContext) -> None:
        dt = self.model.dt
        if not isinstance(dt, (int, float)) or not isfinite(dt) or dt <= 0:
            raise TensorDEMAdapterError("native DEM dt must be finite and positive")
        if not isclose(float(self.model.time), context.time,
                       rel_tol=0.0, abs_tol=self.time_tolerance):
            raise TensorDEMAdapterError(
                f"native DEM time {self.model.time} does not match simulation time {context.time}"
            )

    def advance(self, context: SolverContext) -> None:
        native_dt = float(self.model.dt)
        if context.dt <= 0 or not isfinite(context.dt):
            raise TensorDEMAdapterError("global dt must be finite and positive")
        ratio = context.dt / native_dt
        n_substeps = round(ratio)
        if n_substeps < 1 or not isclose(
            ratio, n_substeps, rel_tol=1e-9, abs_tol=1e-9
        ):
            raise TensorDEMAdapterError(
                f"global dt {context.dt} must be an integer multiple of native DEM dt {native_dt}"
            )
        expected_time = context.time
        if not isclose(float(self.model.time), expected_time,
                       rel_tol=0.0, abs_tol=self.time_tolerance):
            raise TensorDEMAdapterError(
                f"native DEM time {self.model.time} drifted from scheduler time {expected_time}"
            )

        for index in range(n_substeps):
            sub_time = context.time + index * native_dt
            sub_context = SolverContext(
                step=context.step, time=sub_time, dt=native_dt,
                metadata={**context.metadata, "substep_index": index,
                          "substeps_in_global_step": n_substeps},
            )
            loads = None if self.load_provider is None else self.load_provider(sub_context, self.model)
            self.model.step(external_forces=loads, load_time=sub_time)
            self._substeps += 1

        expected_end = context.time + context.dt
        if not isclose(float(self.model.time), expected_end,
                       rel_tol=0.0, abs_tol=self.time_tolerance * max(1.0, abs(expected_end))):
            raise TensorDEMAdapterError(
                f"native DEM ended at {self.model.time}; expected {expected_end}"
            )

    def finalize(self, context: SolverContext) -> None:
        """No native resource teardown is required by the current IceDEM API."""

    def snapshot(self, context: SolverContext) -> dict[str, Any]:
        positions = getattr(self.model, "positions", None)
        shape = getattr(positions, "shape", None)
        return {
            "adapter": "tensorphysics.adapters.tensordem.TensorDEMAdapter",
            "native_time": float(self.model.time),
            "native_dt": float(self.model.dt),
            "native_step_count": int(getattr(self.model, "step_count", self._substeps)),
            "adapter_substeps": self._substeps,
            "particle_count": int(shape[0]) if shape is not None and len(shape) >= 1 else None,
            "restartable": False,
        }
