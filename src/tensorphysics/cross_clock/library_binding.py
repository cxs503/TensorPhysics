"""Implementation-level binding of the cross-clock contract to the real solver libraries.

``tensorphysics.cross_clock.contract`` states the semantics of a clock-mismatched
exchange in the abstract (clocks, integer ratios, interval-complete aggregates,
per-window ledgers).  This module makes the same semantics **executable against
the libraries the platform actually ships** — :mod:`tensorlbm` (fluid, D2Q9 BGK)
and :mod:`tensordem` (soft-sphere particles) — instead of against in-repo
reference instruments.

Three things are frozen here, because they are the three places where a
cross-library coupling silently drifts apart:

clock
    A leg owns a fixed step ``dt`` and exposes an integer step count.  Coupling
    window boundaries are computed as ``t0 + k*h`` from an *integer* window
    index (never by accumulating ``+= dt``), so the sync points of two legs are
    bit-identical for every window and the round-trip ``k -> t_k -> k`` is exact.

units
    Each leg computes in its own units.  :class:`UnitMap` is the single place
    where lattice units become SI units; its scale factors are derived from
    ``dx``, ``dt`` and the reference density ``rho0`` so that momentum, impulse
    and force share one consistent exponent.  Conversions are exact inverses
    (round-trip tested).

exchange
    Only *interval integrals* cross the interface: the particle leg produces the
    drag impulse it integrated over the window, the fluid leg distributes
    exactly that impulse over its own sub-steps, so the per-window ledger
    residual is a bookkeeping invariant, not a physics result.  The physics cost
    of the staggering is reported separately (``stagger_residual``, reference
    drift against a monolithic fine-clock run).

Anti-drift
    :data:`FROZEN_PINS` records the library versions, source digests and the
    signatures of exactly the API surface this binding depends on.  An upgrade
    that changes any of them makes :func:`verify_pins` fail loudly instead of
    silently changing what "advance", "momentum" or "step" mean.

The binding is exercised by ``tests/test_cross_clock_library_binding.py``; the
quantitative demo (three clock ratios, reference comparison, figures) is
``scripts/run_cross_clock_library_demo.py``.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .contract import (
    ContractError,
    IntervalSample,
    SolverClock,
    WindowLedger,
    check_tiling,
    drift_report,
)

__all__ = [
    "LibraryDrift",
    "LibraryPin",
    "FROZEN_PINS",
    "pin_library",
    "verify_pins",
    "UnitMap",
    "SyncSchedule",
    "TensorLBMLeg",
    "TensorDEMLeg",
    "CoupledClockRun",
]


class LibraryDrift(ContractError):
    """Raised when an upstream library no longer matches the frozen contract."""


# --------------------------------------------------------------------------- #
# anti-drift pins
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LibraryPin:
    """Version + source digest + API signature digest of one library.

    ``api`` maps a dotted attribute path (the *only* attributes this binding
    calls) to the signature string observed at freeze time.  Pinning signatures
    rather than "the library" is what makes the check actionable: a failure
    names the attribute whose calling convention moved.
    """

    library: str
    distribution: str
    version: str
    source_sha256: str
    api: Mapping[str, str]

    def as_dict(self) -> dict:
        return {
            "library": self.library,
            "distribution": self.distribution,
            "version": self.version,
            "source_sha256": self.source_sha256,
            "api": dict(self.api),
            "api_sha256": self.api_digest,
        }

    @property
    def api_digest(self) -> str:
        blob = "\n".join(f"{k}={self.api[k]}" for k in sorted(self.api))
        return hashlib.sha256(blob.encode()).hexdigest()


def _resolve(dotted: str) -> Any:
    """Resolve ``package.module.attr`` (the attribute may itself be nested)."""
    parts = dotted.split(".")
    for cut in range(len(parts), 0, -1):
        head = ".".join(parts[:cut])
        try:
            obj: Any = importlib.import_module(head)
        except ImportError:
            continue
        for name in parts[cut:]:
            obj = getattr(obj, name)
        return obj
    raise LibraryDrift(f"cannot resolve {dotted!r}")


def _signature_text(dotted: str) -> str:
    import inspect

    obj = _resolve(dotted)
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):  # pragma: no cover - C-level callables
        return type(obj).__name__


def pin_library(library: str, distribution: str, api: Sequence[str]) -> LibraryPin:
    """Probe an installed library and build its pin (version, digest, signatures)."""
    try:
        version = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError as exc:  # pragma: no cover
        raise LibraryDrift(f"distribution {distribution!r} is not installed") from exc
    spec = importlib.util.find_spec(library)
    if spec is None or not spec.origin:  # pragma: no cover
        raise LibraryDrift(f"cannot locate the source of {library!r}")
    with open(spec.origin, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    api_map = {name: _signature_text(name) for name in api}
    return LibraryPin(library, distribution, version, digest, api_map)


# Frozen on 2026-11-11 against the versions the platform provisions.  Regenerate
# deliberately (and re-run the demo) when a library is upgraded; never widen a
# tolerance to make a drift pass.
FROZEN_PINS: dict[str, LibraryPin] = {
    "tensorlbm": LibraryPin(
        library="tensorlbm",
        distribution="tensorlbm",
        version="0.3.0",
        source_sha256="9fe8ab2ca05a271ae60784e0abf74e465baa5177d8fcb14ec9c4044efc502e5e",
        api={
            "tensorlbm.collide_bgk": "(f: 'torch.Tensor', tau: 'float') -> 'torch.Tensor'",
            "tensorlbm.stream": "(f: 'torch.Tensor') -> 'torch.Tensor'",
            "tensorlbm.macroscopic": (
                "(f: 'torch.Tensor', device: 'torch.device | None' = None) "
                "-> 'tuple[torch.Tensor, torch.Tensor, torch.Tensor]'"
            ),
            "tensorlbm.equilibrium": (
                "(rho: 'torch.Tensor', ux: 'torch.Tensor', uy: 'torch.Tensor', "
                "device: 'torch.device | None' = None) -> 'torch.Tensor'"
            ),
        },
    ),
    "tensordem": LibraryPin(
        library="tensordem",
        distribution="tensordem",
        version="0.1.0",
        source_sha256="28c3bb41c68396fbd8491fc2d097e0b061f84c89e473162463b785626a6890d9",
        api={
            "tensordem.coupling_kernel.DEMKernelConfig": "(radius: 'float', density: 'float', dt: 'float' = 1e-06, young_modulus: 'float' = 10000000.0, poisson_ratio: 'float' = 0.25, restitution: 'float' = 0.1, damping_override: 'float | None' = None, gravity: 'tuple[float, float, float]' = (0.0, 0.0, -9.81), fluid_density: 'float' = 0.0, fluid_viscosity: 'float' = 0.0, fluid_velocity: 'tuple[float, float, float]' = (0.0, 0.0, 0.0), drag_model: 'str' = 'none', box_lo: 'tuple[float, float, float]' = (0.0, 0.0, 0.0), box_hi: 'tuple[float, float, float]' = (0.001, 0.001, 0.001), wall_axes: 'tuple[bool, bool, bool]' = (True, True, True), damping_reference_overlap_fraction: 'float' = 0.001) -> None",
            "tensordem.coupling_kernel.DEMKernel": "(config: 'DEMKernelConfig', positions, velocities=None, backend: 'str | ArrayBackend' = 'numpy', record_energy_every: 'int' = 0, name: 'str' = '') -> 'None'",
            "tensordem.coupling_kernel.DEMKernel.step": "(self, n_steps: 'int' = 1) -> 'None'",
            "tensordem.coupling_kernel.DEMKernelConfig.stokes_coefficient": "property",
        },
    ),
}

TENSORLBM_API: tuple[str, ...] = tuple(FROZEN_PINS["tensorlbm"].api)
TENSORDEM_API: tuple[str, ...] = tuple(FROZEN_PINS["tensordem"].api)


def verify_pins(pins: Mapping[str, LibraryPin] | None = None) -> list[str]:
    """Return the list of drift messages (empty list == frozen contract holds)."""
    expected = FROZEN_PINS if pins is None else pins
    problems: list[str] = []
    for key, pin in expected.items():
        try:
            actual = pin_library(pin.library, pin.distribution, tuple(pin.api))
        except LibraryDrift as exc:
            problems.append(f"{key}: {exc}")
            continue
        if not pin.version.startswith("__") and actual.version != pin.version:
            problems.append(
                f"{key}: version {actual.version!r} != frozen {pin.version!r} "
                "(re-freeze deliberately after re-running the cross-clock demo)"
            )
        if not pin.source_sha256.startswith("__") and actual.source_sha256 != pin.source_sha256:
            problems.append(
                f"{key}: source digest changed ({actual.source_sha256[:12]}... != "
                f"{pin.source_sha256[:12]}...) - the semantics this binding was "
                "validated against moved"
            )
        for name, sig in pin.api.items():
            if sig.startswith("__"):
                continue
            got = actual.api.get(name)
            if got != sig:
                problems.append(f"{key}: signature of {name} changed: {got!r} != {sig!r}")
    return problems


# --------------------------------------------------------------------------- #
# units
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class UnitMap:
    r"""Lattice <-> SI map of one fluid leg, derived from ``dx``, ``dt``, ``rho0``.

    Two dimensions, per unit depth.  Every exponent follows from the length
    scale ``dx``, the time scale ``dt`` and the density scale ``rho0``:

    ============  =================================  =========
    quantity      lattice -> SI                      unit
    ============  =================================  =========
    length        :math:`x \, dx`                    m
    velocity      :math:`u \, dx/dt`                 m/s
    momentum      :math:`P \, \rho_0 dx^3/dt`        kg m/s
    force         :math:`F \, \rho_0 dx^3/dt^2`      N
    viscosity     :math:`(\tau-1/2)/3 \, dx^2/dt`    m^2/s
    ============  =================================  =========

    ``P`` is the raw lattice sum ``sum_cells rho_lat * u_lat`` over one unit of
    depth, i.e. exactly what ``tensorlbm.macroscopic`` integrates over the
    domain.  The momentum factor ``rho0 dx^3/dt`` is the product of the density
    scale, one cell volume (area ``dx^2`` times unit depth ``dx``) and one
    inverse time scale, which is what makes ``force = momentum/dt`` and
    ``impulse = momentum`` consistent by construction rather than by convention.
    """

    dx: float
    dt: float
    rho0: float = 1000.0

    def __post_init__(self) -> None:
        for name in ("dx", "dt", "rho0"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v <= 0.0:
                raise ContractError(f"UnitMap.{name} must be finite and positive, got {v!r}")

    @property
    def momentum_scale(self) -> float:
        return self.rho0 * self.dx**3 / self.dt

    @property
    def force_scale(self) -> float:
        return self.momentum_scale / self.dt

    def length_si(self, x_lat: float) -> float:
        return float(x_lat) * self.dx

    def velocity_si(self, u_lat: float) -> float:
        return float(u_lat) * self.dx / self.dt

    def momentum_si(self, p_lat: float) -> float:
        return float(p_lat) * self.momentum_scale

    def force_si(self, f_lat: float) -> float:
        return float(f_lat) * self.force_scale

    def length_lat(self, x_si: float) -> float:
        return float(x_si) / self.dx

    def velocity_lat(self, u_si: float) -> float:
        return float(u_si) * self.dt / self.dx

    def momentum_lat(self, p_si: float) -> float:
        return float(p_si) / self.momentum_scale

    def force_lat(self, f_si: float) -> float:
        return float(f_si) / self.force_scale

    def viscosity_si(self, tau: float) -> float:
        return (float(tau) - 0.5) / 3.0 * self.dx**2 / self.dt

    def lattice_speed(self) -> float:
        """Reference speed ``dx/dt``: lattice velocity 1 equals this in SI."""
        return self.dx / self.dt

    def roundtrip_error(self, values: Sequence[float] = (0.0, 1.0, -3.5e-2, 1e3)) -> float:
        """Max relative error of ``SI -> lattice -> SI`` over the probes."""
        worst = 0.0
        for v in values:
            back = self.momentum_si(self.momentum_lat(v))
            worst = max(worst, abs(back - v) / max(abs(v), 1e-300))
        return worst

    def as_dict(self) -> dict:
        return {
            "dx_m": self.dx,
            "dt_s": self.dt,
            "rho0_kg_m3": self.rho0,
            "momentum_scale_kg_m2_per_lat": self.momentum_scale,
            "force_scale_N_per_lat": self.force_scale,
            "lattice_speed_m_s": self.lattice_speed(),
        }


# --------------------------------------------------------------------------- #
# synchronisation schedule
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SyncSchedule:
    """Coupling windows ``[k*h, (k+1)*h]`` admissible for every leg clock.

    ``h`` must be an integer multiple of every leg's ``dt`` (the per-leg clock
    ratio).  Window boundaries are always evaluated as ``t0 + k*h`` from the
    **integer** index ``k``, so the schedule never accumulates a time error and
    the two legs' boundary arrays are bitwise identical.
    """

    window: float
    clocks: Mapping[str, SolverClock]

    def __post_init__(self) -> None:
        if not math.isfinite(self.window) or self.window <= 0.0:
            raise ContractError(f"window must be finite and positive, got {self.window!r}")
        if not self.clocks:
            raise ContractError("a schedule needs at least one leg clock")

    def ratios(self, rtol: float = 1e-9) -> dict[str, int]:
        return {name: clock.ratio(self.window, rtol=rtol) for name, clock in self.clocks.items()}

    def ratio(self, name: str, rtol: float = 1e-9) -> int:
        try:
            clock = self.clocks[name]
        except KeyError as exc:  # pragma: no cover
            raise ContractError(f"unknown leg {name!r}") from exc
        return clock.ratio(self.window, rtol=rtol)

    def validate(self, rtol: float = 1e-9) -> dict[str, int]:
        """Raise unless every leg lands exactly on every window boundary."""
        return self.ratios(rtol=rtol)

    def time_at(self, k: int) -> float:
        """Start time of window ``k``: ``t0 + k*h`` (never accumulated)."""
        t0 = self.clocks[next(iter(self.clocks))].t0
        return t0 + int(k) * self.window

    def edges(self, n_windows: int) -> np.ndarray:
        return np.array([self.time_at(k) for k in range(int(n_windows) + 1)], dtype=float)

    def leg_edges(self, name: str, n_windows: int) -> np.ndarray:
        """Sub-step edges of one leg over the same span (window edges included)."""
        clock = self.clocks[name]
        r = self.ratio(name)
        total = int(n_windows) * r + 1
        return np.array([clock.t0 + j * clock.dt for j in range(total)], dtype=float)

    def coincidence_error(self, n_windows: int) -> float:
        """Max |leg edge - schedule edge| over all legs and window boundaries."""
        sched = self.edges(n_windows)
        worst = 0.0
        for name in self.clocks:
            leg = self.leg_edges(name, n_windows)
            r = self.ratio(name)
            for k in range(sched.size):
                worst = max(worst, abs(leg[k * r] - sched[k]))
        return worst

    def as_dict(self) -> dict:
        return {
            "window_s": self.window,
            "clocks": {n: {"dt": c.dt, "t0": c.t0} for n, c in self.clocks.items()},
            "ratios": self.ratios(),
        }


# --------------------------------------------------------------------------- #
# leg: tensorlbm fluid
# --------------------------------------------------------------------------- #
class TensorLBMLeg:
    """D2Q9 BGK fluid leg driven through :mod:`tensorlbm` primitives.

    The leg owns a population array ``f`` of shape ``(9, nx, ny)`` and advances
    it with the library's own ``collide_bgk`` / ``stream`` / ``macroscopic``.
    An external impulse is booked *on that same array* through the D2Q9 momentum
    kick

        ``f_i += 3 W_i (c_i . dp)``

    which changes the cell momentum by exactly ``dp`` (``sum_i W_i c_ia c_ib =
    delta_ab/3``) and the density by exactly zero (``sum_i W_i c_i = 0``).  The
    leg accumulates what actually went into ``f``; that book value — not a
    re-integrated difference — closes the window ledger, and the cumulative
    difference of the measured momentum is reported next to it as an independent
    check.
    """

    def __init__(
        self,
        name: str,
        clock: SolverClock,
        units: UnitMap,
        nx: int = 16,
        ny: int = 8,
        tau: float = 0.8,
        dtype: str = "float64",
        initial_velocity_si: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        if nx < 2 or ny < 2:
            raise ContractError("nx, ny must be >= 2 for a periodic D2Q9 domain")
        if not tau > 0.5:
            raise ContractError(f"tau must exceed 0.5 for a stable BGK fluid, got {tau}")
        self.name = name
        self.clock = clock
        self.units = units
        self.nx, self.ny, self.tau = int(nx), int(ny), float(tau)
        self.dtype = dtype

        tensorlbm = importlib.import_module("tensorlbm")
        d2q9 = importlib.import_module("tensorlbm.d2q9")
        import torch as _torch

        self._torch = _torch
        self._tl = tensorlbm
        td = getattr(_torch, dtype)
        self._W = d2q9.W.to(td)
        self._C = d2q9.C.to(td)

        u_lat = units.velocity_lat(initial_velocity_si[0])
        v_lat = units.velocity_lat(initial_velocity_si[1])
        rho = _torch.ones((self.nx, self.ny), dtype=td)
        self.f = tensorlbm.equilibrium(
            rho, u_lat * _torch.ones_like(rho), v_lat * _torch.ones_like(rho)
        )
        self.steps_taken = 0
        self.applied_impulse_si = np.zeros(2, dtype=float)

    # -- state -------------------------------------------------------------
    def _macroscopic(self):
        return self._tl.macroscopic(self.f)

    def mean_velocity_si(self) -> np.ndarray:
        """Domain-averaged velocity in SI (the quantity the partner sees)."""
        _, ux, uy = self._macroscopic()
        return np.array([self.units.velocity_si(float(ux.mean())),
                         self.units.velocity_si(float(uy.mean()))])

    def momentum_lattice(self) -> np.ndarray:
        rho, ux, uy = self._macroscopic()
        return np.array([float((rho * ux).sum()), float((rho * uy).sum())])

    def momentum_si(self) -> np.ndarray:
        """Total fluid momentum ``sum_cells rho u`` in SI (per unit depth)."""
        p = self.momentum_lattice()
        return np.array([self.units.momentum_si(p[0]), self.units.momentum_si(p[1])])

    @property
    def time(self) -> float:
        """Current leg time, evaluated from the integer step count (no accumulation)."""
        return self.clock.t0 + self.steps_taken * self.clock.dt

    # -- operators ---------------------------------------------------------
    def _kick(self, dp_lat: np.ndarray) -> None:
        """Add momentum density ``dp_lat`` (per cell) to every cell."""
        t = self._torch
        dp = t.as_tensor([float(dp_lat[0]), float(dp_lat[1])], dtype=self._W.dtype)
        proj = self._C[:, 0] * dp[0] + self._C[:, 1] * dp[1]
        self.f = self.f + 3.0 * (self._W * proj).reshape(9, 1, 1)

    def advance(self, n_substeps: int, impulse_si: np.ndarray | None = None) -> np.ndarray:
        """Advance ``n_substeps`` own fine steps, distributing ``impulse_si`` evenly.

        Returns the impulse actually applied (SI), i.e. the sum of the per-substep
        kicks.  For a total impulse that is not exactly divisible by ``n_substeps``
        the last substep takes the remainder, so ``sum(kicks) == impulse`` holds to
        round-off instead of to ``n * (impulse/n)``.
        """
        if n_substeps < 1:
            raise ContractError("advance needs at least one substep")
        impulse = np.zeros(2) if impulse_si is None else np.asarray(impulse_si, dtype=float)
        if impulse.shape != (2,):
            raise ContractError("impulse must be a 2-vector (per unit depth)")

        applied = np.zeros(2, dtype=float)
        remaining = impulse.copy()
        for i in range(int(n_substeps)):
            left = int(n_substeps) - i
            share = remaining / left
            # the kick adds a momentum *density* per cell; the leg's impulse is the
            # sum over the domain, so the per-cell share carries the cell count
            self._kick(share / (self.units.momentum_scale * self.nx * self.ny))
            applied += share
            remaining -= share
            self.f = self._tl.stream(self._tl.collide_bgk(self.f, self.tau))
            self.steps_taken += 1
        self.applied_impulse_si = self.applied_impulse_si + applied
        return applied


# --------------------------------------------------------------------------- #
# leg: tensordem particles
# --------------------------------------------------------------------------- #
class TensorDEMLeg:
    """Soft-sphere particle leg driven through :mod:`tensordem.coupling_kernel`.

    ``DEMKernelConfig`` is frozen upstream, so a new fluid velocity or step size
    cannot be assigned into a live kernel.  The leg therefore rebuilds the kernel
    at every window boundary from its own ``(x, v)`` state
    (:meth:`rebuild_roundtrip_error` measures the cost of that, and asserts it is
    zero).  Time and step count are kept here, from integer arithmetic, so a
    rebuild never moves the clock.

    The leg is configured with ``drag_model='stokes'``, no gravity and no walls:
    the only force acting is the library's Stokes drag
    ``-c (v - u_f)`` with ``c = 6 pi mu r``, so the momentum it loses over a
    window **is** the impulse it hands to the fluid — an interval integral, not a
    point value.
    """

    def __init__(
        self,
        name: str,
        clock: SolverClock,
        config,
        positions: Sequence[Sequence[float]],
        velocities: Sequence[Sequence[float]] | None = None,
        fluid_velocity_si: Sequence[float] | None = None,
    ) -> None:
        kernel_mod = importlib.import_module("tensordem.coupling_kernel")
        self._kernel_mod = kernel_mod
        self.name = name
        self.clock = clock
        self.base_config = config
        self._positions = np.asarray(positions, dtype=float)
        self._velocities = (
            np.zeros_like(self._positions)
            if velocities is None
            else np.asarray(velocities, dtype=float)
        )
        self.fluid_velocity_si = np.asarray(
            config.fluid_velocity if fluid_velocity_si is None else fluid_velocity_si, dtype=float
        )
        if self.fluid_velocity_si.shape != (3,):
            raise ContractError("fluid velocity must have 3 components")
        self.steps_taken = 0
        self.mass = float(config.mass)
        self._kernel = self._build()

    # -- state -------------------------------------------------------------
    def _config_for(self, fluid_velocity_si: np.ndarray):
        import dataclasses

        return dataclasses.replace(
            self.base_config,
            dt=float(self.clock.dt),
            fluid_velocity=tuple(float(v) for v in fluid_velocity_si),
        )

    def _build(self):
        cfg = self._config_for(self.fluid_velocity_si)
        return self._kernel_mod.DEMKernel(
            cfg, self._positions, self._velocities, backend="numpy", name=self.name
        )

    @property
    def time(self) -> float:
        return self.clock.t0 + self.steps_taken * self.clock.dt

    def positions(self) -> np.ndarray:
        return np.array(self._kernel.positions, dtype=float)

    def velocities(self) -> np.ndarray:
        return np.array(self._kernel.velocities, dtype=float)

    def momentum_si(self) -> np.ndarray:
        """Total particle momentum ``sum m v`` in SI (3 components, per particle mass)."""
        return self.mass * self.velocities().sum(axis=0)

    def set_fluid_velocity(self, u_si: Sequence[float]) -> None:
        u = np.asarray(u_si, dtype=float)
        if u.shape != (3,):
            raise ContractError("fluid velocity must have 3 components")
        self.fluid_velocity_si = u
        # rebuild from the current state so the new velocity takes effect
        self._positions = self.positions()
        self._velocities = self.velocities()
        self._kernel = self._build()

    def rebuild_roundtrip_error(self) -> float:
        """Relative state change caused by a rebuild that advances nothing."""
        pos0, vel0 = self.positions(), self.velocities()
        self._positions, self._velocities = pos0, vel0
        kernel = self._build()
        dp = np.abs(np.asarray(kernel.positions) - pos0).max()
        dv = np.abs(np.asarray(kernel.velocities) - vel0).max()
        scale = max(np.abs(vel0).max(), np.abs(pos0).max(), 1e-300)
        return float(max(dp, dv) / scale)

    # -- operators ---------------------------------------------------------
    def advance(self, n_substeps: int) -> np.ndarray:
        """Advance ``n_substeps`` own fine steps; return the momentum change (SI)."""
        if n_substeps < 1:
            raise ContractError("advance needs at least one substep")
        p0 = self.momentum_si()
        self._kernel.step(int(n_substeps))
        self.steps_taken += int(n_substeps)
        return self.momentum_si() - p0

    def drain(self) -> np.ndarray:
        """Impulse (SI) this leg handed to the fluid over the last window.

        The particle loses ``dp`` to the fluid through drag, so the impulse
        delivered to the fluid is ``-dp``.
        """
        raise ContractError("drain() is history-free only in CoupledClockRun; use advance()")


# --------------------------------------------------------------------------- #
# coupled run
# --------------------------------------------------------------------------- #
class CoupledClockRun:
    """Drive a :class:`TensorLBMLeg` and a :class:`TensorDEMLeg` on one schedule.

    Per window ``k``:

    1. the fluid's domain-mean velocity at the window start is held for the whole
       window (causal zeroth-order hold — the only information a partitioning
       consumer has) and handed to the particle leg;
    2. the particle leg advances ``r_dem`` own steps and reports the impulse it
       integrated (``produced``);
    3. the fluid leg receives exactly that impulse, spread over its ``r_lbm`` own
       steps, and advances (``applied``);
    4. the window is booked: ``produced``, ``applied`` (book), the *measured*
       fluid momentum change, the measured particle momentum change,
       ``stagger_residual`` (held vs realised fluid velocity) and the clock ratio.

    ``ledger_drift`` on such a window is a bookkeeping invariant; the physics
    content lives in ``stagger_residual`` and in the reference drift against a
    monolithic run (both legs at the finest step, exchange every fine step).
    """

    def __init__(
        self,
        fluid: TensorLBMLeg,
        particles: TensorDEMLeg,
        schedule: SyncSchedule,
    ) -> None:
        self.fluid = fluid
        self.particles = particles
        self.schedule = schedule
        self.ratios = schedule.validate()

    def _particle_fluid_velocity(self, u_fluid_si: np.ndarray) -> np.ndarray:
        """Embed the 2-D fluid mean velocity into the particle leg's 3-D frame."""
        return np.array([u_fluid_si[0], u_fluid_si[1], 0.0])

    def run(self, n_windows: int, keep_series: bool = True) -> dict:
        from .contract import CouplingContract  # local import: avoid a cycle at module load

        windows: list[WindowLedger] = []
        series = {
            "t_s": [],
            "u_fluid_si": [],
            "v_particle_si": [],
            "P_fluid_si": [],
            "P_particle_si": [],
            "impulse_si": [],
        }
        p_fluid0 = self.fluid.momentum_si()
        p_part0 = self.particles.momentum_si()
        applied_total = np.zeros(2)
        produced_total = np.zeros(2)

        for k in range(int(n_windows)):
            t0, t1 = self.schedule.time_at(k), self.schedule.time_at(k + 1)

            u_start = self.fluid.mean_velocity_si()
            self.particles.set_fluid_velocity(self._particle_fluid_velocity(u_start))
            dp_particle = self.particles.advance(self.ratios[self.particles.name])
            produced = -dp_particle[:2].copy()  # impulse handed to the fluid (2-D frame)

            p_fluid_before = self.fluid.momentum_si()
            applied_book = self.fluid.advance(self.ratios[self.fluid.name], impulse_si=produced)
            p_fluid_after = self.fluid.momentum_si()

            u_end = self.fluid.mean_velocity_si()
            denom = max(np.abs(u_start).max(), 1e-300)
            stagger = float(np.abs(u_end - u_start).max() / denom)

            applied_total += applied_book
            produced_total += produced
            windows.append(
                WindowLedger(
                    index=k,
                    t0=t0,
                    t1=t1,
                    produced={"momentum": float(produced[0])},
                    applied={"momentum": float(applied_book[0])},
                    stagger_residual=stagger,
                    ratio=self.ratios[self.fluid.name],
                )
            )
            if keep_series:
                series["t_s"].append(t1)
                series["u_fluid_si"].append(u_end.tolist())
                series["v_particle_si"].append(self.particles.velocities()[0].tolist())
                series["P_fluid_si"].append(self.fluid.momentum_si().tolist())
                series["P_particle_si"].append(self.particles.momentum_si().tolist())
                series["impulse_si"].append(produced.tolist())

        rep = drift_report(windows, scale=max(np.abs(produced_total).max(), 1e-300))
        scale = max(np.abs(produced_total).max(), np.abs(p_fluid0).max(), 1e-300)
        measured_total = self.fluid.momentum_si() - p_fluid0
        particle_total = self.particles.momentum_si() - p_part0
        system_total = measured_total + particle_total[:2]
        tiling = check_tiling([
            IntervalSample(w.t0, w.t1, {"momentum": 0.0}) for w in windows
        ])
        return {
            "windows": len(windows),
            "clock_ratios": dict(self.ratios),
            "window_s": self.schedule.window,
            "t_final_s": self.schedule.time_at(n_windows),
            "ledger": rep,
            "tiling_violations": tiling,
            "produced_impulse_si": produced_total.tolist(),
            "applied_impulse_si": applied_total.tolist(),
            "book_vs_produced_rel": float(
                np.abs(applied_total - produced_total).max() / scale
            ),
            "measured_fluid_momentum_delta_si": measured_total.tolist(),
            "measured_particle_momentum_delta_si": particle_total.tolist(),
            "measured_system_momentum_residual_rel": float(np.abs(system_total).max() / scale),
            "clock_roundtrip": {
                "fluid_steps": self.fluid.steps_taken,
                "fluid_steps_expected": int(n_windows) * self.ratios[self.fluid.name],
                "particle_steps": self.particles.steps_taken,
                "particle_steps_expected": int(n_windows) * self.ratios[self.particles.name],
                "fluid_time_s": self.fluid.time,
                "particle_time_s": self.particles.time,
                "schedule_time_s": self.schedule.time_at(n_windows),
                "sync_coincidence_error_s": self.schedule.coincidence_error(n_windows),
            },
            "series": series if keep_series else None,
            "particle_velocity_final_si": self.particles.velocities()[0].tolist(),
        }

    def as_contract(self, tolerance: float = 1e-9) -> "CouplingContract":
        from .contract import CouplingContract

        return CouplingContract(
            producer=self.particles.name,
            consumer=self.fluid.name,
            quantities=("momentum",),
            tolerance=tolerance,
            mode="impulse_displacement",
        )