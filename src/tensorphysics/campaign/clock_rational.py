"""Rational master/sub coupling clocks with GCD-aligned exchange cadence (M12).

Background
----------
The v1 contract in :mod:`tensorphysics.campaign.clock` fixes the M12 fields
(``ratio``, ``exchange_every``) but only on an **integer** sub-step lattice:
``dt_master / dt_sub`` must be an integer within ``RATIO_RTOL``.  Real coupled
campaigns do not always satisfy that: a structure step of 3 ms coupled to an
explicit contact step of 2 ms gives ``ratio = 3/2``, and the stable step of a
lattice-Boltzmann sub-solver is set by the collision frequency, not by the
master.  The v1 module therefore rejected a legitimate physical configuration.

This module lifts the restriction to **arbitrary rational ratios** while keeping
the two properties that made the v1 lattice attractive: exactness (no float
accumulation, so no drift over a campaign) and an explicit cadence (M12: no
implicit loop nesting).

Construction
------------
Two rational step sizes are placed on their **greatest common divisor** lattice.
With ``dt_m = a*g`` and ``dt_s = b*g`` (``gcd(a, b) = 1``), every instant that is
reachable by *both* clocks is a multiple of the least common multiple
``lcm(dt_m, dt_s) = dt_m*dt_s/g = a*b*g``.  That LCM is the only cadence on which
an exchange can be both causal and complete, so the exchange cadence is *derived*
from the two clocks rather than guessed; an explicit integer ``exchange_periods``
coarsens it by an integer factor (``1`` = exchange at every common period).
Exchanges therefore always land on integer master ticks and integer sub steps:
no interpolation of the clock itself is ever needed.

Time base is the integer **atom** index: ``t = atom_index * g`` computed as an
exact :class:`fractions.Fraction`.  ``float`` times are a single rounding of that
exact value for reporting only.

Causality
---------
An exchange at ``t_k`` may consume only quantities produced at ``t <= t_k``.
:class:`CausalBuffer` and :func:`assert_causal_hold` make the negative case
machine-visible: asking a consumer for data whose earliest available sample lies
in the future raises :class:`CausalityViolation` instead of silently
extrapolating.  That keeps the information age bounded by
``exchange_interval_s`` (M12 disposition) and is what forbids in-interval linear
interpolation of the partner signal (see :mod:`tensorphysics.campaign.multirate`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from fractions import Fraction
from typing import Any, Iterable, Sequence

from .clock import DISPOSITION
from .schema import SchemaError, payload_digest, validate

#: Time base label for the rational-atom lattice (distinct from ``integer_sub_step``).
TIME_BASE = "rational_atom"

#: Relative tolerance accepted when rationalising an irrational-looking float pair.
DEFAULT_RATIO_RTOL = 1e-9


class RationalClockError(ValueError):
    """The rational coupling-clock contract is inconsistent (M12)."""


class CausalityViolation(RationalClockError):
    """A consumer asked for coupled data that does not exist yet (future).

    Raised instead of extrapolating: an exchange may only read the past, so a
    future sample is a contract violation, not an accuracy inconvenience.
    """


# ---------------------------------------------------------------------------
# exact rational helpers
# ---------------------------------------------------------------------------
def as_fraction(value: Any, name: str) -> Fraction:
    """Return ``value`` as an exact :class:`Fraction` (> 0).

    ``float`` input is interpreted through ``repr`` (its shortest decimal
    representation), so ``0.004`` becomes ``1/250`` rather than the binary
    expansion of the double.  This is the documented "decimal intent" rule: a
    user writing ``dt=0.004`` means 4 ms, not 4.0000000000000002... ms.
    """
    if isinstance(value, bool):
        raise RationalClockError(f"{name} must be a number, not bool")
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise RationalClockError(f"{name} must be finite, got {value!r}")
        frac = Fraction(repr(value))
    elif isinstance(value, str):
        try:
            frac = Fraction(value)
        except (ValueError, ZeroDivisionError) as exc:
            raise RationalClockError(f"{name} is not a rational literal: {value!r}") from exc
    else:
        raise RationalClockError(f"{name} must be int/float/str/Fraction, got {type(value).__name__}")
    if frac <= 0:
        raise RationalClockError(f"{name} must be > 0, got {value!r}")
    return frac


def as_time(value: Any, name: str) -> Fraction:
    """Exact time stamp (``>= 0`` allowed: campaigns start at ``t = 0``)."""
    if isinstance(value, Fraction):
        frac = value
    elif isinstance(value, bool):
        raise RationalClockError(f"{name} must be a number, not bool")
    elif isinstance(value, int):
        frac = Fraction(value)
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise RationalClockError(f"{name} must be finite, got {value!r}")
        frac = Fraction(repr(value))
    elif isinstance(value, str):
        frac = Fraction(value)
    else:
        raise RationalClockError(f"{name} must be int/float/str/Fraction, got {type(value).__name__}")
    if frac < 0:
        raise RationalClockError(f"{name} must be >= 0, got {value!r}")
    return frac


def fraction_gcd(a: Fraction, b: Fraction) -> Fraction:
    """Greatest common divisor on the rationals: ``gcd(an, bn) / lcm(ad, bd)``.

    Both numbers are integer multiples of the result, and the result is the
    largest rational with that property.
    """
    an, ad = a.numerator, a.denominator
    bn, bd = b.numerator, b.denominator
    numerator = math.gcd(an * bd, bn * ad)
    denominator = ad * bd
    return Fraction(numerator, denominator)


def fraction_lcm(a: Fraction, b: Fraction) -> Fraction:
    """Least common multiple on the rationals: ``a*b / gcd(a, b)``."""
    return a * b / fraction_gcd(a, b)


def rationalise(value: Any, *, max_denominator: int = 10**6) -> Fraction:
    """Best rational approximation of ``value`` with a bounded denominator."""
    if isinstance(value, Fraction):
        return value
    raw = Fraction(repr(value)) if isinstance(value, float) else Fraction(value)
    return raw.limit_denominator(max_denominator)


# ---------------------------------------------------------------------------
# rational clock contract
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RationalExchangeEvent:
    """One coupled exchange instant on the atom lattice."""

    index: int
    atom_index: int
    master_tick: int
    sub_step: int
    time_exact: Fraction

    def __post_init__(self) -> None:
        if self.index < 1 or self.master_tick < 1 or self.sub_step < 1 or self.atom_index < 1:
            raise RationalClockError("exchange counters start at 1")
        if not isinstance(self.time_exact, Fraction) or self.time_exact <= 0:
            raise RationalClockError("exchange time must be a positive exact Fraction")

    @property
    def time_s(self) -> float:
        """Single rounding of the exact instant (reporting only)."""
        return float(self.time_exact)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "atom_index": self.atom_index,
            "master_tick": self.master_tick,
            "sub_step": self.sub_step,
            "time_s": self.time_s,
            "time_exact": f"{self.time_exact.numerator}/{self.time_exact.denominator}",
        }


@dataclass(frozen=True)
class RationalClockSpec:
    """Master/sub clocks with an arbitrary rational ratio and a GCD-derived cadence.

    ``exchange_periods`` is the number of *common periods* (base LCM) between two
    exchanges.  It has no default: M12 forbids inheriting a cadence implicitly.
    """

    master_name: str
    sub_name: str
    master_dt_s: Any
    sub_dt_s: Any
    exchange_periods: int
    disposition: str = DISPOSITION
    ratio_rtol: float = DEFAULT_RATIO_RTOL
    schema_version: int = 1

    def __post_init__(self) -> None:
        for name, value in (("master_name", self.master_name), ("sub_name", self.sub_name)):
            if not isinstance(value, str) or not value:
                raise RationalClockError(f"{name} must be a non-empty string")
        if self.master_name == self.sub_name:
            raise RationalClockError("master and sub solvers must differ")
        master = as_fraction(self.master_dt_s, "master_dt_s")
        sub = as_fraction(self.sub_dt_s, "sub_dt_s")
        if sub > master:
            raise RationalClockError(
                f"sub clock must not be coarser than the master clock ({float(sub)!r} > "
                f"{float(master)!r}); ratio={float(master / sub)!r} < 1"
            )
        if isinstance(self.exchange_periods, bool) or not isinstance(self.exchange_periods, int):
            raise RationalClockError("exchange_periods must be an integer (M12: no implicit cadence)")
        if self.exchange_periods < 1:
            raise RationalClockError("exchange_periods must be >= 1")
        if self.disposition != DISPOSITION:
            raise RationalClockError(f"only disposition {DISPOSITION!r} is supported")
        if self.schema_version != 1:
            raise RationalClockError("unsupported RationalClockSpec schema_version")
        if not (0.0 < float(self.ratio_rtol) < 1.0):
            raise RationalClockError("ratio_rtol must be in (0, 1)")
        # store canonical exact values (frozen dataclass -> object.__setattr__)
        object.__setattr__(self, "master_dt_s", master)
        object.__setattr__(self, "sub_dt_s", sub)

    # -- derived exact quantities ---------------------------------------------------
    @property
    def master_dt(self) -> Fraction:
        return self.master_dt_s

    @property
    def sub_dt(self) -> Fraction:
        return self.sub_dt_s

    @property
    def atom_s(self) -> Fraction:
        """GCD lattice spacing: the largest rational dividing both step sizes."""
        return fraction_gcd(self.master_dt, self.sub_dt)

    @property
    def master_atoms(self) -> int:
        """Integer atoms per master step (exact by construction)."""
        return int(self.master_dt / self.atom_s)

    @property
    def sub_atoms(self) -> int:
        """Integer atoms per sub step (exact by construction)."""
        return int(self.sub_dt / self.atom_s)

    @property
    def coprime(self) -> bool:
        """True when the atom coarsening is maximal (``gcd(a, b) == 1``)."""
        return math.gcd(self.master_atoms, self.sub_atoms) == 1

    @property
    def ratio(self) -> Fraction:
        """Sub steps per master step (may be fractional: e.g. ``3/2``)."""
        return self.master_dt / self.sub_dt

    @property
    def ratio_is_integer(self) -> bool:
        return self.ratio.denominator == 1

    @property
    def base_exchange_s(self) -> Fraction:
        """Smallest period resolvable by both clocks (the LCM)."""
        return self.master_dt * self.sub_dt / self.atom_s

    @property
    def exchange_interval_s(self) -> Fraction:
        """Physical time between two coupled exchanges."""
        return self.base_exchange_s * self.exchange_periods

    @property
    def master_ticks_per_exchange(self) -> int:
        return self.sub_atoms * self.exchange_periods

    @property
    def sub_steps_per_exchange(self) -> int:
        return self.master_atoms * self.exchange_periods

    @property
    def atoms_per_exchange(self) -> int:
        return self.master_atoms * self.sub_atoms * self.exchange_periods

    @property
    def max_information_age_s(self) -> Fraction:
        """Upper bound on staleness of a coupled quantity (M12 disposition)."""
        return self.exchange_interval_s

    @property
    def natural_frequency_ratio(self) -> Fraction:
        """Master/sub step ratio expressed as an exact rational."""
        return self.ratio

    # -- construction helpers -------------------------------------------------------
    @classmethod
    def from_float_dts(
        cls,
        master_name: str,
        sub_name: str,
        master_dt_s: float,
        sub_dt_s: float,
        *,
        exchange_periods: int,
        max_denominator: int = 10**6,
        rel_tol: float = DEFAULT_RATIO_RTOL,
        disposition: str = DISPOSITION,
    ) -> "RationalClockSpec":
        """Rationalise a float step pair, failing loudly if the fit is too coarse.

        ``Fraction(0.004)`` is an exact 53-bit rational with a ~10^17
        denominator, which is useless as a clock lattice.  Here each ``dt`` is
        replaced by its best rational approximation with a bounded denominator
        and the *declared* relative error is checked against ``rel_tol``; a
        population that cannot be rationalised inside the tolerance (i.e. a
        genuinely incommensurate pair) is rejected instead of being silently
        rounded onto a lattice that does not exist.
        """
        fitted: list[Fraction] = []
        for name, value in (("master_dt_s", master_dt_s), ("sub_dt_s", sub_dt_s)):
            exact = as_fraction(value, name)
            approx = rationalise(value, max_denominator=max_denominator)
            rel_error = abs(approx - exact) / exact
            if rel_error > rel_tol:
                raise RationalClockError(
                    f"{name}={value!r} cannot be rationalised to a denominator <= "
                    f"{max_denominator} within rel_tol={rel_tol:g} "
                    f"(best {approx} -> rel error {float(rel_error):.3e})"
                )
            fitted.append(approx)
        return cls(
            master_name=master_name,
            sub_name=sub_name,
            master_dt_s=fitted[0],
            sub_dt_s=fitted[1],
            exchange_periods=exchange_periods,
            disposition=disposition,
            ratio_rtol=rel_tol,
        )

    # -- serialisation --------------------------------------------------------------
    def to_dict(self, *, master_tick: int = 0) -> dict[str, Any]:
        """Contract artifact validated against :data:`COUPLING_CLOCK_RATIONAL_V1`."""
        if isinstance(master_tick, bool) or not isinstance(master_tick, int) or master_tick < 0:
            raise RationalClockError("master_tick must be a non-negative integer")
        if master_tick % self.master_ticks_per_exchange != 0 and master_tick != 0:
            raise RationalClockError(
                "artifact ticks must be exchange-aligned: master_tick must be a multiple of "
                f"{self.master_ticks_per_exchange}"
            )
        payload = {
            "schema_name": "coupling_clock_rational",
            "schema_version": self.schema_version,
            "disposition": self.disposition,
            "master": {
                "name": self.master_name,
                "dt_s": float(self.master_dt),
                "dt_exact": _frac_text(self.master_dt),
            },
            "sub": {
                "name": self.sub_name,
                "dt_s": float(self.sub_dt),
                "dt_exact": _frac_text(self.sub_dt),
            },
            "time_base": TIME_BASE,
            "ratio": float(self.ratio),
            "ratio_exact": _frac_text(self.ratio),
            "ratio_is_integer": self.ratio_is_integer,
            "atom_s": float(self.atom_s),
            "atom_exact": _frac_text(self.atom_s),
            "master_atoms_per_step": self.master_atoms,
            "sub_atoms_per_step": self.sub_atoms,
            "coprime_atoms": self.coprime,
            "exchange_periods": self.exchange_periods,
            "base_exchange_s": float(self.base_exchange_s),
            "base_exchange_exact": _frac_text(self.base_exchange_s),
            "exchange_interval_s": float(self.exchange_interval_s),
            "exchange_interval_exact": _frac_text(self.exchange_interval_s),
            "master_ticks_per_exchange": self.master_ticks_per_exchange,
            "sub_steps_per_exchange": self.sub_steps_per_exchange,
            "max_information_age_s": float(self.max_information_age_s),
            "ticks": {"master": master_tick, "sub_step": master_tick * self.master_atoms // self.sub_atoms},
        }
        validate(COUPLING_CLOCK_RATIONAL_V1, payload)
        return payload

    @classmethod
    def from_dict(cls, payload: Any) -> "RationalClockSpec":
        """Consumer gate: schema first, then exact cross-check of derived fields."""
        try:
            validate(COUPLING_CLOCK_RATIONAL_V1, payload)
        except SchemaError as exc:
            raise RationalClockError(f"rational coupling clock artifact rejected: {exc}") from exc
        spec = cls(
            master_name=payload["master"]["name"],
            sub_name=payload["sub"]["name"],
            master_dt_s=Fraction(payload["master"]["dt_exact"]),
            sub_dt_s=Fraction(payload["sub"]["dt_exact"]),
            exchange_periods=int(payload["exchange_periods"]),
            disposition=payload["disposition"],
            schema_version=payload["schema_version"],
        )
        _cross_check(spec, payload)
        return spec


def _frac_text(value: Fraction) -> str:
    return f"{value.numerator}/{value.denominator}"


def _number_matches(declared: float, exact: Fraction, ulps: float = 4.0) -> bool:
    target = float(exact)
    window = ulps * math.ulp(max(abs(target), 1e-300))
    return abs(declared - target) <= window


def _cross_check(spec: RationalClockSpec, payload: Any) -> None:
    """Recompute every derived field exactly; reject a tampered artifact."""
    exact_fields = {
        "ratio_exact": (payload["ratio_exact"], _frac_text(spec.ratio)),
        "atom_exact": (payload["atom_exact"], _frac_text(spec.atom_s)),
        "base_exchange_exact": (payload["base_exchange_exact"], _frac_text(spec.base_exchange_s)),
        "exchange_interval_exact": (
            payload["exchange_interval_exact"],
            _frac_text(spec.exchange_interval_s),
        ),
    }
    for name, (declared, recomputed) in exact_fields.items():
        if declared != recomputed:
            raise RationalClockError(
                f"derived field {name!r} declares {declared!r} but recomputes to {recomputed!r}"
            )
    numeric_fields = {
        "ratio": (payload["ratio"], spec.ratio),
        "atom_s": (payload["atom_s"], spec.atom_s),
        "base_exchange_s": (payload["base_exchange_s"], spec.base_exchange_s),
        "exchange_interval_s": (payload["exchange_interval_s"], spec.exchange_interval_s),
        "max_information_age_s": (payload["max_information_age_s"], spec.max_information_age_s),
    }
    for name, (declared, exact) in numeric_fields.items():
        if not _number_matches(float(declared), exact):
            raise RationalClockError(
                f"derived field {name!r} declares {declared!r} but recomputes to {float(exact)!r}"
            )
    integer_fields = {
        "master_atoms_per_step": (payload["master_atoms_per_step"], spec.master_atoms),
        "sub_atoms_per_step": (payload["sub_atoms_per_step"], spec.sub_atoms),
        "master_ticks_per_exchange": (
            payload["master_ticks_per_exchange"],
            spec.master_ticks_per_exchange,
        ),
        "sub_steps_per_exchange": (payload["sub_steps_per_exchange"], spec.sub_steps_per_exchange),
    }
    for name, (declared, recomputed) in integer_fields.items():
        if declared != recomputed:
            raise RationalClockError(
                f"derived field {name!r} declares {declared!r} but recomputes to {recomputed!r}"
            )
    if payload["ratio_is_integer"] != spec.ratio_is_integer:
        raise RationalClockError("ratio_is_integer disagrees with the exact ratio")
    if payload["coprime_atoms"] != spec.coprime:
        raise RationalClockError("coprime_atoms disagrees with the exact atom decomposition")


# ---------------------------------------------------------------------------
# closed artifact schema (module-local: keeps the v1 bilateral pin untouched)
# ---------------------------------------------------------------------------
_SOLVER_CLOCK = {
    "type": "object",
    "required": ["name", "dt_s", "dt_exact"],
    "properties": {
        "name": {"type": "string", "minLength": 1},
        "dt_s": {"type": "number", "exclusiveMinimum": 0},
        "dt_exact": {"type": "string", "minLength": 3},
    },
    "additionalProperties": False,
}

COUPLING_CLOCK_RATIONAL_V1: dict[str, Any] = {
    "type": "object",
    "required": [
        "schema_name",
        "schema_version",
        "disposition",
        "master",
        "sub",
        "time_base",
        "ratio",
        "ratio_exact",
        "ratio_is_integer",
        "atom_s",
        "atom_exact",
        "master_atoms_per_step",
        "sub_atoms_per_step",
        "coprime_atoms",
        "exchange_periods",
        "base_exchange_s",
        "base_exchange_exact",
        "exchange_interval_s",
        "exchange_interval_exact",
        "master_ticks_per_exchange",
        "sub_steps_per_exchange",
        "max_information_age_s",
        "ticks",
    ],
    "properties": {
        "schema_name": {"const": "coupling_clock_rational"},
        "schema_version": {"const": 1},
        "disposition": {"const": "M12"},
        "master": _SOLVER_CLOCK,
        "sub": _SOLVER_CLOCK,
        "time_base": {"const": TIME_BASE},
        "ratio": {"type": "number", "minimum": 1},
        "ratio_exact": {"type": "string", "minLength": 3},
        "ratio_is_integer": {"type": "boolean"},
        "atom_s": {"type": "number", "exclusiveMinimum": 0},
        "atom_exact": {"type": "string", "minLength": 3},
        "master_atoms_per_step": {"type": "integer", "minimum": 1},
        "sub_atoms_per_step": {"type": "integer", "minimum": 1},
        "coprime_atoms": {"type": "boolean"},
        "exchange_periods": {"type": "integer", "minimum": 1},
        "base_exchange_s": {"type": "number", "exclusiveMinimum": 0},
        "base_exchange_exact": {"type": "string", "minLength": 3},
        "exchange_interval_s": {"type": "number", "exclusiveMinimum": 0},
        "exchange_interval_exact": {"type": "string", "minLength": 3},
        "master_ticks_per_exchange": {"type": "integer", "minimum": 1},
        "sub_steps_per_exchange": {"type": "integer", "minimum": 1},
        "max_information_age_s": {"type": "number", "exclusiveMinimum": 0},
        "ticks": {
            "type": "object",
            "required": ["master", "sub_step"],
            "properties": {
                "master": {"type": "integer", "minimum": 0},
                "sub_step": {"type": "integer", "minimum": 0},
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}


def artifact_digest(spec: RationalClockSpec, *, master_tick: int = 0) -> dict[str, Any]:
    """Canonical digest of the contract artifact (producer/consumer lock value)."""
    payload = spec.to_dict(master_tick=master_tick)
    return {"artifact": "coupling_clock_rational.v1", "digest": payload_digest(payload)}


# ---------------------------------------------------------------------------
# stateful driver on the atom lattice
# ---------------------------------------------------------------------------
@dataclass
class RationalCouplingClock:
    """Drive a :class:`RationalClockSpec` on its exact atom lattice.

    The driver owns a single integer counter (``atom_index``); master ticks, sub
    steps and exchanges are *derived* by integer division, so no drift can
    accumulate and a fractional ratio needs no interpolation of the clock.
    """

    spec: RationalClockSpec
    _atom_index: int = field(default=0, init=False)
    _master_tick: int = field(default=0, init=False)
    _sub_step: int = field(default=0, init=False)
    _events: list[RationalExchangeEvent] = field(default_factory=list, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.spec, RationalClockSpec):
            raise RationalClockError("spec must be a RationalClockSpec")

    # -- state ---------------------------------------------------------------------
    @property
    def atom_index(self) -> int:
        return self._atom_index

    @property
    def master_tick(self) -> int:
        return self._master_tick

    @property
    def sub_step(self) -> int:
        return self._sub_step

    @property
    def events(self) -> tuple[RationalExchangeEvent, ...]:
        return tuple(self._events)

    @property
    def time_exact(self) -> Fraction:
        return self._atom_index * self.spec.atom_s

    @property
    def time_s(self) -> float:
        return float(self.time_exact)

    # -- advance -------------------------------------------------------------------
    def advance_master(self) -> tuple[RationalExchangeEvent, ...]:
        """Advance one master step; return the exchange events due at its end."""
        self._master_tick += 1
        self._atom_index += self.spec.master_atoms
        self._sub_step = self._atom_index // self.spec.sub_atoms
        if self._master_tick % self.spec.master_ticks_per_exchange != 0:
            return ()
        event = RationalExchangeEvent(
            index=self._master_tick // self.spec.master_ticks_per_exchange,
            atom_index=self._atom_index,
            master_tick=self._master_tick,
            sub_step=self._sub_step,
            time_exact=self.time_exact,
        )
        self._events.append(event)
        return (event,)

    def run(self, master_ticks: int) -> tuple[RationalExchangeEvent, ...]:
        if isinstance(master_ticks, bool) or not isinstance(master_ticks, int) or master_ticks < 0:
            raise RationalClockError("master_ticks must be a non-negative integer")
        emitted: list[RationalExchangeEvent] = []
        for _ in range(master_ticks):
            emitted.extend(self.advance_master())
        return tuple(emitted)

    def advance_sub_step(self) -> bool:
        """Advance exactly one sub step; return True if an exchange is due.

        Sub-driven advance is exposed because the sub clock may be the one that
        owns the outer loop (e.g. an explicit contact sub-solver with a global
        time-step controller).  Off-lattice master ticks are impossible: the atom
        index only ever lands on multiples of ``master_atoms``.
        """
        self._atom_index += self.spec.sub_atoms
        self._sub_step += 1
        if self._atom_index % self.spec.master_atoms == 0:
            self._master_tick = self._atom_index // self.spec.master_atoms
        if self._atom_index % self.spec.atoms_per_exchange != 0:
            return False
        event = RationalExchangeEvent(
            index=self._sub_step // self.spec.sub_steps_per_exchange,
            atom_index=self._atom_index,
            master_tick=self._master_tick,
            sub_step=self._sub_step,
            time_exact=self.time_exact,
        )
        self._events.append(event)
        return True

    def reset(self) -> None:
        self._atom_index = 0
        self._master_tick = 0
        self._sub_step = 0
        self._events.clear()

    def to_dict(self) -> dict[str, Any]:
        aligned_tick = self._master_tick
        return {
            "contract": self.spec.to_dict(master_tick=0),
            "state": {
                "atom_index": self._atom_index,
                "master_tick": self._master_tick,
                "sub_step": self._sub_step,
                "time_s": self.time_s,
            },
            "events": [event.to_dict() for event in self._events],
            "last_event_master_tick": aligned_tick,
        }


def expected_exchange_count(spec: RationalClockSpec, master_ticks: int) -> int:
    """Closed-form number of exchanges in a run of ``master_ticks`` master steps."""
    if isinstance(master_ticks, bool) or not isinstance(master_ticks, int) or master_ticks < 0:
        raise RationalClockError("master_ticks must be >= 0")
    return master_ticks // spec.master_ticks_per_exchange


# ---------------------------------------------------------------------------
# GCD cadence verification (machine check, not a claim)
# ---------------------------------------------------------------------------
def verify_gcd_cadence(spec: RationalClockSpec, *, master_ticks: int) -> dict[str, Any]:
    """Machine-verify that the exchange cadence is the LCM of the two clocks.

    Checks, all exact (integer arithmetic on the atom lattice):

    1. ``atom_s`` is the *greatest* common divisor: reducing by any prime factor
       of the atom would leave a number that no longer divides both steps.
    2. The atom coarsening is coprime: ``gcd(master_atoms, sub_atoms) == 1``.
    3. ``lcm(dt_m, dt_s) == dt_m*dt_s/gcd`` and equals ``base_exchange_s``.
    4. No smaller common period exists: every multiple of ``master_atoms`` below
       the LCM fails to be a multiple of ``sub_atoms`` (and vice versa).
    5. Exchange instants are exact multiples of ``exchange_interval_s`` and the
       sub-step position is an exact integer (no rounding, no drift).
    """
    checks: dict[str, Any] = {}
    a, b = spec.master_atoms, spec.sub_atoms
    checks["atom_s"] = float(spec.atom_s)
    checks["atom_exact"] = _frac_text(spec.atom_s)
    checks["master_atoms"] = a
    checks["sub_atoms"] = b
    checks["coprime_atoms"] = math.gcd(a, b) == 1
    checks["lcm_equals_product"] = spec.base_exchange_s == spec.master_dt * spec.sub_dt / spec.atom_s
    checks["fraction_lcm_matches"] = fraction_lcm(spec.master_dt, spec.sub_dt) == spec.base_exchange_s

    # (1) maximality of the atom: can the atom be divided by a prime p and still divide both?
    atom = spec.atom_s
    reducible: list[str] = []
    for prime in _small_primes(a * b):
        candidate = atom * prime  # a *larger* common divisor would factor the atom upward
        if _is_multiple(spec.master_dt, candidate) and _is_multiple(spec.sub_dt, candidate):
            reducible.append(str(prime))
    checks["atom_maximal"] = not reducible
    checks["atom_reducible_by"] = reducible

    # (3) independent integer lcm of the two atom counts equals the product (coprime)
    checks["lcm_atoms"] = math.lcm(a, b)
    checks["lcm_atoms_equals_product"] = math.lcm(a, b) == a * b

    # (4) exhaustive: no common period strictly between 0 and the LCM
    spurious: list[int] = []
    for k in range(1, b):  # multiples of the master step, k*b = LCM would be the first hit
        m = k * a
        if m % b == 0:
            spurious.append(m)
    for k in range(1, a):
        m = k * b
        if m % a == 0:
            spurious.append(m)
    checks["spurious_common_periods"] = sorted(set(spurious))
    checks["cadence_is_coarsest"] = not spurious

    # (5) exact exchange instants from the stateful driver
    clock = RationalCouplingClock(spec)
    events = clock.run(master_ticks)
    expected = expected_exchange_count(spec, master_ticks)
    interval = spec.exchange_interval_s
    instant_residuals = [event.time_exact - event.index * interval for event in events]
    substep_residuals = [
        event.time_exact - event.sub_step * spec.sub_dt for event in events
    ]
    checks["event_count"] = len(events)
    checks["event_count_matches_closed_form"] = len(events) == expected
    checks["instant_residual_zero"] = all(res == 0 for res in instant_residuals)
    checks["substep_residual_zero"] = all(res == 0 for res in substep_residuals)
    checks["substep_is_integer"] = all(
        event.atom_index % spec.sub_atoms == 0 for event in events
    )
    checks["max_abs_residual_s"] = float(
        max([abs(float(r)) for r in instant_residuals + substep_residuals] + [0.0])
    )

    checks["verified"] = all(
        checks[key]
        for key in (
            "coprime_atoms",
            "lcm_equals_product",
            "fraction_lcm_matches",
            "atom_maximal",
            "lcm_atoms_equals_product",
            "cadence_is_coarsest",
            "event_count_matches_closed_form",
            "instant_residual_zero",
            "substep_residual_zero",
            "substep_is_integer",
        )
    )
    checks["disposition"] = spec.disposition
    checks["ratio_exact"] = _frac_text(spec.ratio)
    checks["exchange_interval_exact"] = _frac_text(spec.exchange_interval_s)
    checks["master_ticks"] = master_ticks
    return checks


def _is_multiple(value: Fraction, unit: Fraction) -> bool:
    q = value / unit
    return q.denominator == 1


def _small_primes(limit: int, *, cap: int = 64) -> list[int]:
    """Small primes up to ``min(limit, 512)`` (used for atom maximality checks)."""
    limit = min(abs(int(limit)), 512)
    out: list[int] = []
    for n in range(2, limit + 1):
        if all(n % p for p in out if p * p <= n):
            out.append(n)
            if len(out) >= cap:
                break
    return out


# ---------------------------------------------------------------------------
# causality gate
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TimedSample:
    """A coupled quantity stamped with the exact time it was produced."""

    time_exact: Fraction
    value: float
    label: str = "sample"

    def __post_init__(self) -> None:
        if not isinstance(self.time_exact, Fraction) or self.time_exact < 0:
            raise RationalClockError("sample time must be a non-negative exact Fraction")

    @property
    def time_s(self) -> float:
        return float(self.time_exact)


@dataclass
class CausalBuffer:
    """Monotone sample history that refuses to answer questions about the future.

    ``latest_at_or_before(t)`` returns the freshest sample with ``time <= t``.
    If every sample is strictly later than ``t`` the request is *not* answered by
    extrapolation: :class:`CausalityViolation` is raised, which is the negative
    case M12 requires to be visible.
    """

    label: str = "buffer"
    _samples: list[TimedSample] = field(default_factory=list, init=False)

    def append(self, time_exact: Fraction | float, value: float) -> TimedSample:
        t = as_time(time_exact, "sample time")
        if self._samples and t < self._samples[-1].time_exact:
            raise RationalClockError(
                f"{self.label}: samples must be appended in time order "
                f"({_frac_text(t)} < {_frac_text(self._samples[-1].time_exact)})"
            )
        sample = TimedSample(time_exact=t, value=float(value), label=self.label)
        self._samples.append(sample)
        return sample

    def __len__(self) -> int:
        return len(self._samples)

    @property
    def samples(self) -> tuple[TimedSample, ...]:
        return tuple(self._samples)

    def latest_at_or_before(self, query: Fraction | float) -> TimedSample:
        q = as_time(query, "query time")
        for sample in reversed(self._samples):
            if sample.time_exact <= q:
                return sample
        raise CausalityViolation(
            f"{self.label}: no sample at or before t={_frac_text(q)}; "
            f"earliest available sample is at t={_frac_text(self._samples[0].time_exact) if self._samples else None} "
            "(future data cannot be used)"
        )

    def information_age(self, query: Fraction | float) -> Fraction:
        """Staleness of the sample that a causal consumer would use."""
        q = as_time(query, "query time")
        return q - self.latest_at_or_before(q).time_exact


def assert_causal_hold(
    exchange_time: Fraction, sample_time: Fraction, *, label: str = "exchange"
) -> Fraction:
    """Fail if a producer's data used at ``exchange_time`` was produced later.

    Returns the information age (``exchange_time - sample_time``).
    """
    if sample_time > exchange_time:
        raise CausalityViolation(
            f"{label}: sample produced at t={_frac_text(sample_time)} cannot be consumed at "
            f"t={_frac_text(exchange_time)} (overshoot "
            f"{_frac_text(sample_time - exchange_time)} s); refuses extrapolation"
        )
    return exchange_time - sample_time


def causality_audit(
    spec: RationalClockSpec,
    events: Sequence[RationalExchangeEvent],
    *,
    producer_times: Sequence[Fraction] | None = None,
) -> dict[str, Any]:
    """Audit an exchange sequence for monotonicity, alignment and information age.

    ``producer_times[k]`` is the latest time the *partner* solver had actually
    reached when exchange ``k`` consumed its data.  A producer time later than
    the exchange is a causality violation and is reported (not raised) so an
    artifact can carry the failure evidence.
    """
    intervals = [event.time_exact for event in events]
    monotone = all(b > a for a, b in zip(intervals, intervals[1:]))
    on_lattice = all(
        event.time_exact == event.index * spec.exchange_interval_s for event in events
    )
    ages: list[Fraction] = []
    violations: list[dict[str, Any]] = []
    if producer_times is not None:
        if len(producer_times) != len(events):
            raise RationalClockError("producer_times must match the number of events")
        for event, produced in zip(events, producer_times):
            if produced > event.time_exact:
                violations.append(
                    {
                        "event_index": event.index,
                        "exchange_time_exact": _frac_text(event.time_exact),
                        "producer_time_exact": _frac_text(produced),
                        "overshoot_exact": _frac_text(produced - event.time_exact),
                        "overshoot_s": float(produced - event.time_exact),
                    }
                )
            else:
                ages.append(event.time_exact - produced)
    max_age = max(ages) if ages else Fraction(0)
    return {
        "event_count": len(events),
        "monotone": monotone,
        "on_lattice": on_lattice,
        "is_causal": not violations,
        "violations": violations,
        "max_information_age_exact": _frac_text(max_age),
        "max_information_age_s": float(max_age),
        "declared_max_information_age_s": float(spec.max_information_age_s),
        "within_information_age_bound": max_age <= spec.max_information_age_s,
        "disposition": spec.disposition,
    }


def build_causal_buffer(
    label: str, samples: Iterable[tuple[Fraction | float, float]]
) -> CausalBuffer:
    """Convenience constructor used by tests and the multirate benchmark."""
    buf = CausalBuffer(label=label)
    for t, value in samples:
        buf.append(t, value)
    return buf