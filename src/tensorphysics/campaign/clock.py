"""Explicit master/sub coupling-clock semantics (M12 disposition).

Ice-region campaigns couple solvers with very different stable step sizes: a
rigid-body / hydrodynamic master (ice-load envelope, contact stiffness) advances
with ``dt_master`` while a lattice-Boltzmann or explicit contact sub-solver
needs ``dt_sub = dt_master / ratio`` for stability.  Disposition M12 requires
that the *ratio* and the *exchange cadence* are explicit contract fields rather
than implicit loop nesting, because an implicit cadence makes the information
age of a coupled force non-reproducible across restarts and across libraries.

Two design rules are enforced here:

1. ``ratio`` must be an integer within a documented floating-point window, and
   ``exchange_every`` (master ticks between exchanges) is mandatory -- there is
   no default, so a caller cannot silently inherit a cadence.
2. Time is carried on an integer *sub-step* lattice.  Exchange instants are
   therefore exact integers (``master_tick * ratio``) and no drift can
   accumulate over a campaign, unlike repeated ``t += dt`` accumulation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

from .schema import SchemaError
from .schema_defs import validate_artifact

#: Relative window for accepting ``dt_master / dt_sub`` as an integer ratio.
RATIO_RTOL = 1e-12

DISPOSITION = "M12"
TIME_BASE = "integer_sub_step"


class ClockContractError(ValueError):
    """The coupling-clock contract is inconsistent or violates M12 semantics."""


def _positive_finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ClockContractError(f"{name} must be a real number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ClockContractError(f"{name} must be finite and > 0, got {value!r}")
    return number


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ClockContractError(f"{name} must be a non-empty string")
    return value.strip()


@dataclass(frozen=True)
class ExchangeEvent:
    """One coupled exchange instant on the integer sub-step lattice."""

    index: int
    master_tick: int
    sub_step: int
    time_s: float

    def __post_init__(self) -> None:
        if self.index < 1:
            raise ClockContractError("exchange index counts from 1")
        if self.master_tick < 1 or self.sub_step < 1:
            raise ClockContractError("exchange ticks must be positive integers")
        if not math.isfinite(self.time_s) or self.time_s <= 0.0:
            raise ClockContractError("exchange time must be finite and > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "master_tick": self.master_tick,
            "sub_step": self.sub_step,
            "time_s": self.time_s,
        }


@dataclass(frozen=True)
class CouplingClockSpec:
    """M12 coupling-clock contract: explicit ratio and explicit cadence."""

    master_name: str
    sub_name: str
    master_dt_s: float
    sub_dt_s: float
    exchange_every: int
    disposition: str = DISPOSITION
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.master_name, "master_name")
        _nonempty(self.sub_name, "sub_name")
        if self.master_name == self.sub_name:
            raise ClockContractError("master and sub solvers must differ")
        master_dt = _positive_finite(self.master_dt_s, "master_dt_s")
        sub_dt = _positive_finite(self.sub_dt_s, "sub_dt_s")
        if isinstance(self.exchange_every, bool) or not isinstance(self.exchange_every, int):
            raise ClockContractError("exchange_every must be an integer")
        if self.exchange_every < 1:
            raise ClockContractError("exchange_every must be >= 1 (cadence is mandatory)")
        if self.disposition != DISPOSITION:
            raise ClockContractError(f"only disposition {DISPOSITION!r} is supported")
        if self.schema_version != 1:
            raise ClockContractError("unsupported CouplingClockSpec schema_version")

        estimate = master_dt / sub_dt
        ratio = int(round(estimate))
        if ratio < 1:
            raise ClockContractError(
                f"sub clock must be finer than the master clock (ratio={estimate!r})"
            )
        mismatch = abs(master_dt - ratio * sub_dt)
        if mismatch > RATIO_RTOL * max(master_dt, ratio * sub_dt):
            raise ClockContractError(
                "master_dt/sub_dt is not an integer ratio within "
                f"{RATIO_RTOL:g}: {master_dt!r}/{sub_dt!r} ~ {estimate!r}"
            )

    # -- derived contract quantities -------------------------------------------------
    @property
    def ratio(self) -> int:
        """Exact integer sub-steps per master step."""
        return int(round(self.master_dt_s / self.sub_dt_s))

    @property
    def exchange_interval_s(self) -> float:
        """Physical time between two coupled exchanges."""
        return self.exchange_every * self.master_dt_s

    @property
    def sub_steps_per_exchange(self) -> int:
        """Exact number of sub-steps executed between two exchanges."""
        return self.exchange_every * self.ratio

    @property
    def max_information_age_s(self) -> float:
        """Upper bound on how stale a coupled quantity may be (disposition M12)."""
        return self.exchange_interval_s

    # -- serialisation --------------------------------------------------------------
    def to_dict(self, *, master_tick: int = 0) -> dict[str, Any]:
        """Contract artifact (validated against ``coupling_clock.v1``)."""
        if isinstance(master_tick, bool) or not isinstance(master_tick, int) or master_tick < 0:
            raise ClockContractError("master_tick must be a non-negative integer")
        payload = {
            "schema_name": "coupling_clock",
            "schema_version": self.schema_version,
            "disposition": self.disposition,
            "master": {"name": self.master_name, "dt_s": self.master_dt_s},
            "sub": {"name": self.sub_name, "dt_s": self.sub_dt_s},
            "time_base": TIME_BASE,
            "ratio": self.ratio,
            "exchange_every": self.exchange_every,
            "exchange_interval_s": self.exchange_interval_s,
            "sub_steps_per_exchange": self.sub_steps_per_exchange,
            "max_information_age_s": self.max_information_age_s,
            "ticks": {"master": master_tick, "sub_step": master_tick * self.ratio},
        }
        validate_artifact("coupling_clock.v1", payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "CouplingClockSpec":
        """Consumer-side constructor: schema gate first, then semantics gate."""
        try:
            validate_artifact("coupling_clock.v1", payload)
        except SchemaError as exc:
            raise ClockContractError(f"coupling clock artifact rejected: {exc}") from exc
        spec = cls(
            master_name=payload["master"]["name"],
            sub_name=payload["sub"]["name"],
            master_dt_s=float(payload["master"]["dt_s"]),
            sub_dt_s=float(payload["sub"]["dt_s"]),
            exchange_every=int(payload["exchange_every"]),
            disposition=payload["disposition"],
            schema_version=payload["schema_version"],
        )
        _cross_check_derived(payload, spec)
        return spec


def _cross_check_derived(payload: Any, spec: "CouplingClockSpec") -> None:
    """Reject artifacts whose declared derived fields disagree with recomputation.

    The contract ships both the primitives (dts, cadence) and the derived
    quantities (ratio, interval, sub-steps per exchange).  A consumer that only
    trusted the primitives could silently accept a tampered cadence, so the
    derived fields are recomputed and compared at machine precision.
    """
    intervals = {
        "ratio": (payload["ratio"], spec.ratio, 0),
        "exchange_every": (payload["exchange_every"], spec.exchange_every, 0),
        "sub_steps_per_exchange": (
            payload["sub_steps_per_exchange"],
            spec.sub_steps_per_exchange,
            0,
        ),
    }
    for name, (declared, recomputed, _) in intervals.items():
        if declared != recomputed:
            raise ClockContractError(
                f"derived field {name!r} declares {declared!r} but recomputes to "
                f"{recomputed!r}; the artifact is internally inconsistent"
            )
    for name, declared, recomputed in (
        ("exchange_interval_s", float(payload["exchange_interval_s"]), spec.exchange_interval_s),
        ("max_information_age_s", float(payload["max_information_age_s"]), spec.max_information_age_s),
    ):
        window = 4.0 * math.ulp(max(abs(recomputed), 1e-300))
        if abs(declared - recomputed) > window:
            raise ClockContractError(
                f"derived field {name!r} declares {declared!r} but recomputes to "
                f"{recomputed!r}; the artifact is internally inconsistent"
            )
    ticks = payload["ticks"]
    if ticks["sub_step"] != ticks["master"] * spec.ratio:
        raise ClockContractError(
            "tick state is not on the integer lattice: sub_step != master * ratio"
        )

class CouplingClock:
    """Stateful driver of a :class:`CouplingClockSpec` on the integer lattice."""

    def __init__(self, spec: CouplingClockSpec) -> None:
        if not isinstance(spec, CouplingClockSpec):
            raise ClockContractError("spec must be a CouplingClockSpec")
        self._spec = spec
        self._master_tick = 0
        self._events: list[ExchangeEvent] = []

    @property
    def spec(self) -> CouplingClockSpec:
        return self._spec

    @property
    def master_tick(self) -> int:
        return self._master_tick

    @property
    def sub_step(self) -> int:
        """Absolute integer sub-step position (drift-free)."""
        return self._master_tick * self._spec.ratio

    @property
    def time_s(self) -> float:
        """Physical time, rounded once from the integer tick (no accumulation)."""
        return self._master_tick * self._spec.master_dt_s

    @property
    def events(self) -> tuple[ExchangeEvent, ...]:
        return tuple(self._events)

    def advance_master(self) -> tuple[ExchangeEvent, ...]:
        """Advance one master tick; return the exchange events due at its end."""
        tick = self._master_tick + 1
        self._master_tick = tick
        if tick % self._spec.exchange_every != 0:
            return ()
        event = ExchangeEvent(
            index=tick // self._spec.exchange_every,
            master_tick=tick,
            sub_step=tick * self._spec.ratio,
            time_s=tick * self._spec.master_dt_s,
        )
        self._events.append(event)
        return (event,)

    def run(self, master_ticks: int) -> tuple[ExchangeEvent, ...]:
        """Advance ``master_ticks`` master steps and return the due exchanges."""
        if isinstance(master_ticks, bool) or not isinstance(master_ticks, int) or master_ticks < 0:
            raise ClockContractError("master_ticks must be a non-negative integer")
        emitted: list[ExchangeEvent] = []
        for _ in range(master_ticks):
            emitted.extend(self.advance_master())
        return tuple(emitted)

    def reset(self) -> None:
        self._master_tick = 0
        self._events.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract": self._spec.to_dict(master_tick=self._master_tick),
            "events": [event.to_dict() for event in self._events],
        }


def expected_exchange_count(spec: CouplingClockSpec, master_ticks: int) -> int:
    """Closed-form exchange count for a run of ``master_ticks``."""
    if master_ticks < 0:
        raise ClockContractError("master_ticks must be >= 0")
    return master_ticks // spec.exchange_every


def verify_alignment(
    spec: CouplingClockSpec, events: Iterable[ExchangeEvent], *, ulps: float = 4.0
) -> dict[str, Any]:
    """Check exchange events against the exact integer lattice.

    Returns residuals that must be zero: the sub-step index is an exact integer,
    the exchange time equals one rounding of ``master_tick * master_dt``, and the
    cadence step is constant.
    """
    events = tuple(events)
    sub_step_residual = 0
    time_residual_s = 0.0
    cadence_residual_s = 0.0
    previous_tick: int | None = None
    for event in events:
        sub_step_residual = max(
            sub_step_residual, abs(event.sub_step - event.master_tick * spec.ratio)
        )
        time_residual_s = max(
            time_residual_s,
            abs(event.time_s - event.master_tick * spec.master_dt_s),
        )
        if previous_tick is not None:
            cadence_residual_s = max(
                cadence_residual_s,
                abs((event.master_tick - previous_tick) * spec.master_dt_s
                    - spec.exchange_interval_s),
            )
        previous_tick = event.master_tick
    tolerance_s = ulps * math.ulp(max(spec.exchange_interval_s, 1e-300))
    return {
        "event_count": len(events),
        "sub_step_residual": sub_step_residual,
        "time_residual_s": time_residual_s,
        "cadence_residual_s": cadence_residual_s,
        "tolerance_s": tolerance_s,
        "aligned": sub_step_residual == 0
        and time_residual_s <= tolerance_s
        and cadence_residual_s <= tolerance_s,
    }


def clock_drift(spec: CouplingClockSpec, master_ticks: int) -> dict[str, Any]:
    """Compare the integer lattice against naive float accumulation.

    The canonical clock performs exactly zero accumulated additions, so its
    drift is identically zero.  Naively issuing ``master_ticks * ratio``
    increments of ``dt_sub`` accumulates rounding error; the size of that error
    is the quantitative argument for the integer time base.
    """
    if isinstance(master_ticks, bool) or not isinstance(master_ticks, int) or master_ticks < 0:
        raise ClockContractError("master_ticks must be a non-negative integer")
    canonical_s = master_ticks * spec.master_dt_s
    naive_s = 0.0
    for _ in range(master_ticks * spec.ratio):
        naive_s += spec.sub_dt_s
    drift_s = naive_s - canonical_s
    scale = max(abs(canonical_s), 1e-300)
    return {
        "master_ticks": master_ticks,
        "sub_steps": master_ticks * spec.ratio,
        "canonical_time_s": canonical_s,
        "naive_time_s": naive_s,
        "canonical_drift_s": 0.0,
        "naive_drift_s": drift_s,
        "naive_relative_drift": abs(drift_s) / scale,
    }