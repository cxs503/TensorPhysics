"""Stability-limited solver rates and the multiphysics time-step coordinator.

Three coordination strategies are provided, all driven by the *current* stability
limits of the participating solvers:

``synchronized``
    one common step ``dt = safety * min_i dt_max_i``; every solver advances once
    per macro step.

``subcycling``
    macro step is the *coarse* limit ``dt = safety * max_i dt_max_i`` and the
    faster solvers take an integer number of sub-steps inside it.
    ``n_i = ceil(dt_macro / (safety * dt_max_i))`` and the effective step
    ``dt_macro / n_i`` is guaranteed below that solver's limit.

``supercycling``
    macro step is the *fine* limit ``dt = safety * min_i dt_max_i`` and the slow
    solvers are updated once every ``m_i`` macro steps.

Stability limits are functions of the resolved state, which is what makes the
controller stability-aware rather than a fixed decimation ratio:

* lattice-Boltzmann / transport CFL:  ``dt <= CFL dx / (|u| + c_s)``
* explicit structural dynamics:       ``dt <= factor * 2 / omega_max``
* structural accuracy (Newmark):      ``dt <= 1 / (ppc f_max)``
* electrical circuit:                 ``dt <= factor L / R``
* gear mesh resolution:               ``dt <= 2 pi / (Z Omega ppc)``

Interface bookkeeping: the exchanged quantity is the torque ``T(t)`` from the
ice/flow side and the speed ``omega(t)`` from the structural side.  The physical
conservation statement at the interface is that the work ``int T omega dt`` and
the impulse ``int T dt`` delivered over a macro step must be the same for both
sides.  A reconstruction (hold or linear interpolation of the coarse samples)
breaks that equality; :func:`exchange_work_residual` measures how much, against a
high-resolution reference of the same integral.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "SolverRate",
    "CoordinationPlan",
    "TimeStepCoordinator",
    "AdaptiveTimeStep",
    "cfl_step",
    "structural_explicit_step",
    "structural_accuracy_step",
    "circuit_step",
    "exchange_step",
    "mesh_step",
    "exchange_work_residual",
    "CoarseTorqueReconstruction",
    "upwind_step",
]

STRATEGIES = ("synchronized", "subcycling", "supercycling")


# ---------------------------------------------------------------------------
# stability limits
# ---------------------------------------------------------------------------
def cfl_step(dx: float, speed: float, cfl: float = 0.5) -> float:
    """Transport/CEL limit ``dt = cfl dx / speed`` with a positive floor on speed."""
    if not (dx > 0.0):
        raise ValueError("dx must be positive")
    if not (0.0 < cfl <= 1.0):
        raise ValueError("cfl must lie in (0, 1]")
    return cfl * dx / max(abs(speed), 1.0e-12)


def structural_explicit_step(f_max: float, factor: float = 0.5) -> float:
    """Central-difference stability limit ``dt <= factor * 2 / omega_max``."""
    if not (f_max > 0.0):
        raise ValueError("f_max must be positive")
    return factor * 2.0 / (2.0 * math.pi * f_max)


def structural_accuracy_step(f_max: float, points_per_period: int = 20) -> float:
    """Accuracy-driven step resolving the highest structural mode."""
    if not (f_max > 0.0):
        raise ValueError("f_max must be positive")
    if points_per_period < 2:
        raise ValueError("at least 2 points per period are needed to represent a mode")
    return 1.0 / (points_per_period * f_max)


def circuit_step(inductance: float, resistance: float, factor: float = 0.2) -> float:
    """Electrical step ``dt = factor L / R`` (explicit current update)."""
    if not (inductance > 0.0 and resistance > 0.0):
        raise ValueError("inductance and resistance must be positive")
    return factor * inductance / resistance


def exchange_step(f_max_interface: float, points_per_period: int = 20) -> float:
    """Step that resolves the highest frequency present in an exchanged signal.

    The exchange budget is a *stability/accuracy* limit like any other solver
    limit: a linearly reconstructed interface signal of frequency ``f`` carries a
    relative waveform error of order ``(2 pi f dt)**2 / 8``, so resolving it with
    ``points_per_period`` points keeps that error below ~1 %.
    """
    if not (f_max_interface > 0.0):
        raise ValueError("f_max_interface must be positive")
    if points_per_period < 2:
        raise ValueError("need at least 2 points per period")
    return 1.0 / (points_per_period * f_max_interface)


def mesh_step(teeth: int, omega: float, points_per_mesh: int = 20) -> float:
    """Step that resolves the tooth-passing period of a gear pair."""
    if teeth < 3 or points_per_mesh < 2:
        raise ValueError("need teeth >= 3 and points_per_mesh >= 2")
    if not (omega > 0.0):
        raise ValueError("omega must be positive")
    return 2.0 * math.pi / (teeth * omega * points_per_mesh)


@dataclass(frozen=True)
class SolverRate:
    """A participating solver: a name plus a state-dependent stability limit."""

    name: str
    limit: Callable[[Mapping[str, float]], float]

    def dt_max(self, state: Optional[Mapping[str, float]] = None) -> float:
        value = float(self.limit(state or {}))
        if not (value > 0.0 and math.isfinite(value)):
            raise ValueError(f"solver {self.name!r}: non-positive or non-finite stability limit {value!r}")
        return value


@dataclass
class CoordinationPlan:
    """Result of a coordination decision."""

    strategy: str
    dt_macro: float
    dt_solver: Dict[str, float]
    substeps: Dict[str, int]
    supersteps: Dict[str, int]
    limits: Dict[str, float]

    def check(self, safety: float = 1.0) -> None:
        """Assert every solver step respects its stability limit and ratios are integral."""
        for name, dt in self.dt_solver.items():
            if dt > self.limits[name] * (1.0 + 1e-12):
                raise AssertionError(
                    f"{name}: dt={dt!r} exceeds stability limit {self.limits[name]!r}")
        for name, n in self.substeps.items():
            if n < 1:
                raise AssertionError(f"{name}: substeps must be >= 1")
            if self.strategy == "supercycling":
                expected = self.dt_macro * self.supersteps[name]
            else:
                expected = self.dt_macro / n
            if abs(expected - self.dt_solver[name]) > 1e-12 * max(self.dt_macro,
                                                                  abs(expected)):
                raise AssertionError(f"{name}: coordinated step {expected!r} != "
                                     f"dt_solver {self.dt_solver[name]!r}")
        if self.strategy not in STRATEGIES:
            raise AssertionError(f"unknown strategy {self.strategy!r}")

    @property
    def max_ratio(self) -> int:
        return max(self.substeps.values()) if self.substeps else 1

    def as_dict(self) -> Dict[str, object]:
        return {"strategy": self.strategy, "dt_macro": self.dt_macro,
                "dt_solver": dict(self.dt_solver), "substeps": dict(self.substeps),
                "supersteps": dict(self.supersteps), "limits": dict(self.limits)}


class TimeStepCoordinator:
    """Chooses a macro step and the per-solver sub/super-cycle counts."""

    def __init__(self, rates: Sequence[SolverRate], strategy: str = "subcycling",
                 safety: float = 0.9, max_ratio: int = 64) -> None:
        if strategy not in STRATEGIES:
            raise ValueError(f"strategy must be one of {STRATEGIES}")
        if not (0.0 < safety <= 1.0):
            raise ValueError("safety must lie in (0, 1]")
        if max_ratio < 1:
            raise ValueError("max_ratio must be >= 1")
        if not rates:
            raise ValueError("at least one solver rate is required")
        names = [r.name for r in rates]
        if len(set(names)) != len(names):
            raise ValueError("duplicate solver names")
        self.rates = list(rates)
        self.strategy = strategy
        self.safety = safety
        self.max_ratio = max_ratio

    def limits(self, state: Optional[Mapping[str, float]] = None) -> Dict[str, float]:
        return {r.name: r.dt_max(state) for r in self.rates}

    def plan(self, state: Optional[Mapping[str, float]] = None) -> CoordinationPlan:
        limits = self.limits(state)
        s = self.safety
        if self.strategy == "synchronized":
            dt = s * min(limits.values())
            dt_solver = {name: dt for name in limits}
            substeps = {name: 1 for name in limits}
            supersteps = {name: 1 for name in limits}
        elif self.strategy == "subcycling":
            dt = s * max(limits.values())
            substeps = {name: max(1, int(math.ceil(dt / (s * lim))))
                        for name, lim in limits.items()}
            if max(substeps.values()) > self.max_ratio:
                # shrink the macro step until every sub-step ratio is admissible
                scale = self.max_ratio / max(substeps.values())
                dt = dt * scale
                substeps = {name: max(1, int(math.ceil(dt / (s * lim))))
                            for name, lim in limits.items()}
                if max(substeps.values()) > self.max_ratio:
                    raise ValueError("stability limits are too disparate for max_ratio")
            dt_solver = {name: dt / substeps[name] for name in limits}
            supersteps = {name: 1 for name in limits}
        else:  # supercycling
            dt = s * min(limits.values())
            dt_solver = {}
            substeps = {}
            supersteps = {}
            for name, lim in limits.items():
                m = max(1, int(math.floor(s * lim / dt)))
                supersteps[name] = m
                substeps[name] = 1
                dt_solver[name] = dt if m == 1 else dt * m
                # the slow solver must not exceed its own limit either
                if dt_solver[name] > lim:
                    supersteps[name] = max(1, m - 1)
                    dt_solver[name] = dt * supersteps[name]
        plan = CoordinationPlan(self.strategy, dt, dt_solver, substeps, supersteps, limits)
        plan.check()
        return plan


class AdaptiveTimeStep:
    """Stability-aware adaptive controller with growth/shrink limits.

    ``update`` never returns a step larger than the coordinator's stability
    proposal and never grows by more than ``growth`` per call, which prevents the
    step from jumping after a transient stability alert.
    """

    def __init__(self, coordinator: TimeStepCoordinator, growth: float = 1.5,
                 shrink: float = 0.5) -> None:
        if growth <= 1.0:
            raise ValueError("growth must exceed 1")
        if not (0.0 < shrink < 1.0):
            raise ValueError("shrink must lie in (0, 1)")
        self.coordinator = coordinator
        self.growth = growth
        self.shrink = shrink
        self.history: list[float] = []

    def update(self, state: Optional[Mapping[str, float]] = None,
               previous_dt: Optional[float] = None,
               stability_alert: bool = False) -> float:
        dt = self.coordinator.plan(state).dt_macro
        if previous_dt is not None:
            if stability_alert:
                dt = min(dt, previous_dt * self.shrink)
            else:
                dt = min(dt, previous_dt * self.growth)
        self.history.append(dt)
        return dt


# ---------------------------------------------------------------------------
# interface exchange / conservation
# ---------------------------------------------------------------------------
class CoarseTorqueReconstruction:
    """Reconstruct the flow-side torque inside one macro step from coarse samples."""

    def __init__(self, torque_fn: Callable[[float], float], dt_macro: float,
                 mode: str = "linear") -> None:
        if mode not in ("linear", "hold"):
            raise ValueError("mode must be 'linear' or 'hold'")
        self.torque_fn = torque_fn
        self.dt_macro = dt_macro
        self.mode = mode
        self.samples: Dict[int, Tuple[float, float]] = {}

    def _samples_for(self, macro_index: int) -> Tuple[float, float]:
        if macro_index not in self.samples:
            t0 = macro_index * self.dt_macro
            self.samples[macro_index] = (float(self.torque_fn(t0)),
                                         float(self.torque_fn(t0 + self.dt_macro)))
        return self.samples[macro_index]

    def value(self, t: float) -> float:
        m = int(math.floor(t / self.dt_macro + 1e-12))
        t0 = m * self.dt_macro
        v0, v1 = self._samples_for(m)
        frac = (t - t0) / self.dt_macro
        frac = min(max(frac, 0.0), 1.0)
        if self.mode == "linear":
            return v0 + (v1 - v0) * frac
        return v0

    def __call__(self, t: float, theta=None, omega=None):
        """Chain-compatible torque callback: puts the value on node 0."""
        if omega is None:
            return self.value(t)
        out = np.zeros_like(np.asarray(omega, dtype=float))
        out[0] = self.value(t)
        return out


def exchange_work_residual(torque_fn: Callable[[float], float],
                           omega_fn: Callable[[float], float],
                           dt_macro: float, n_sub: int, mode: str = "linear",
                           n_ref: int = 2000) -> Dict[str, float]:
    """Interface work/impulse mismatch of a sub-cycled exchange.

    Reference: composite-Simpson evaluation of ``int_0^dt T omega dt`` and
    ``int_0^dt T dt`` on ``n_ref`` points of one macro step, i.e. the limit a
    refined run would reach.  Scheme: the fine solver samples the exchanged
    torque through a hold or linear reconstruction of the two coarse samples
    (``t=0`` and ``t=dt_macro``) at ``n_sub`` sub-instants.

    Returns ``work_ref``, ``work_scheme``, ``work_residual``, ``impulse_residual``.
    """
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1")
    if n_ref < n_sub:
        raise ValueError("n_ref must be at least n_sub")
    t_ref = np.linspace(0.0, dt_macro, n_ref + 1)
    f_ref = np.array([torque_fn(t) * omega_fn(t) for t in t_ref])
    work_ref = float(np.trapezoid(f_ref, t_ref)) if hasattr(np, "trapezoid") \
        else float(np.trapz(f_ref, t_ref))
    tq_ref = np.array([torque_fn(t) for t in t_ref])
    trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    imp_ref = float(trapezoid(tq_ref, t_ref))
    imp_scale = float(trapezoid(np.abs(tq_ref), t_ref))

    recon = CoarseTorqueReconstruction(torque_fn, dt_macro, mode)
    dt_sub = dt_macro / n_sub
    t_sub = np.arange(n_sub + 1) * dt_sub
    work_scheme = 0.0
    imp_scheme = 0.0
    for k in range(n_sub):
        ta, tb = t_sub[k], t_sub[k + 1]
        tm = 0.5 * (ta + tb)
        work_scheme += recon.value(tm) * omega_fn(tm) * dt_sub
        imp_scheme += recon.value(tm) * dt_sub
    scale = max(abs(work_ref), 1e-30)
    return {"work_ref": work_ref, "work_scheme": work_scheme,
            "work_residual": abs(work_scheme - work_ref) / scale,
            "impulse_residual": abs(imp_scheme - imp_ref) / max(imp_scale, 1e-30),
            "impulse_scale": imp_scale}

def upwind_step(field: np.ndarray, velocity: float, dx: float, dt: float) -> np.ndarray:
    """One first-order upwind step of ``du/dt + v du/dx = 0`` on a periodic 1-D grid.

    Demonstration kernel used *only* to exercise the CFL branch of the
    coordinator: it is stable for ``|v| dt / dx <= 1`` and diverges for larger
    steps, exactly like the transport solver whose rate the coordinator limits.
    """
    cfl = abs(velocity) * dt / dx
    shifted = np.roll(field, 1) if velocity >= 0.0 else np.roll(field, -1)
    return field - cfl * (field - shifted)
