"""Exchange-operator algebra and interval-complete attribution ledger.

Motivation (campaign node ``p223w3-n06``)
-----------------------------------------
Two defects are reported by the coupling contract of the wake/weld/settling
chains:

1. **end-stamped handover** — the interface flux of the closed coarse interval
   is held over the coming interval, which injects an ``O(dt)`` bias into the
   receiver and, once the fine solver's own resolution floor dominates, makes
   the defect staircase-saturate instead of converging;
2. **interval attribution** — the exchange only books the sub-cycle samples of
   the interval that closes at the exchange instant, so the heat carried by the
   other ``r-1`` sub-intervals of the window is never booked: with a
   sub-cycling ratio ``r`` that drops ``(r-1)/r`` of the exchanged quantity
   (87.5 % at ``r = 8``, 93.75 % at ``r = 16``).

This module makes the algebra explicit.  Three operators implement one
interface, so a gate can compare them on the *same* interface history:

======================  ============================================  =======
operator                interval rate ``q_n``                        order
======================  ============================================  =======
``lag``                 newest closed sample (end stamp / ZOH hold)     1
``centered``            ``(q(t_n) + q(t_{n+1})) / 2`` (CN midpoint)      2
``predictor_corrector`` ``(q(t_n) + q_hat(t_{n+1})) / 2``, corrected     2
                        with the realised ``q(t_{n+1})`` once the
                        interval closes
======================  ============================================  =======

``centered`` needs the future endpoint, so it is the *idealised* member of the
algebra; ``predictor_corrector`` is its causal realisation and must reproduce
the ``centered`` final state once the deferred correction has been applied.

Attribution rule
----------------
Every produced sub-interval share belongs to exactly one exchange interval:

* the producer books a **release** into the interval that owns the sub-sample;
* the receiver books the **applied** integral of the same interval;
* a deferred correction is booked as a **correction** on that same interval.

The ledger therefore reports the three quantities a conservation audit needs::

    unattributed = (produced - released) / produced   -> must be 0
    deficit      = (released - applied)  / produced   -> must be 0
    double bookings / intervals closed twice          -> raise

A scheme that books only the tail of the sub-cycle window reports
``unattributed = (r-1)/r`` by construction, which is the quantitative form of
the 87-94 % heat loss.
"""
from __future__ import annotations

import abc
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from .ledger import CouplingLedgerError

__all__ = [
    "ExchangeScheme",
    "StampConvention",
    "StampRule",
    "SchemeSpec",
    "SCHEME_TABLE",
    "IntervalTransfer",
    "ExchangeOperator",
    "LagExchange",
    "CenteredExchange",
    "PredictorCorrectorExchange",
    "operator_for",
    "IntervalAttribution",
    "IntervalExchangeLedger",
    "AttributionSummary",
]


class ExchangeScheme(str, Enum):
    """Members of the exchange-operator algebra."""

    LAG = "lag"
    CENTERED = "centered"
    PREDICTOR_CORRECTOR = "predictor_corrector"


@dataclass(frozen=True)
class SchemeSpec:
    scheme: ExchangeScheme
    declared_order: int
    causal: bool
    description: str


#: Declared temporal order of every operator (order of the interval integral).
SCHEME_TABLE: dict[ExchangeScheme, SchemeSpec] = {
    ExchangeScheme.LAG: SchemeSpec(
        ExchangeScheme.LAG, 1, True,
        "end-stamped hold of the newest closed sample (the shipped practice)"),
    ExchangeScheme.CENTERED: SchemeSpec(
        ExchangeScheme.CENTERED, 2, False,
        "Crank-Nicolson style midpoint of both interval endpoints"),
    ExchangeScheme.PREDICTOR_CORRECTOR: SchemeSpec(
        ExchangeScheme.PREDICTOR_CORRECTOR, 2, True,
        "trapezoid on a linearly predicted endpoint, deferred correction on close"),
}


@dataclass(frozen=True)
class IntervalTransfer:
    """The exchange applied to one coarse interval ``[t_lo, t_hi]``."""

    index: int
    t_lo: float
    t_hi: float
    rate: float
    scheme: ExchangeScheme
    q_lo: float = 0.0
    q_hi_used: float = 0.0
    provisional: bool = False
    predicted: bool = False
    recentered: bool = False
    q_lo_raw: float = 0.0
    #: raw (un-de-stamped) sample of the previous interval, kept so the deferred
    #: correction can apply the same de-stamping order as the provisional step
    q_prev_raw: float | None = None

    @property
    def h(self) -> float:
        return self.t_hi - self.t_lo

    @property
    def integral(self) -> float:
        return self.rate * self.h

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t_lo": self.t_lo,
            "t_hi": self.t_hi,
            "rate": self.rate,
            "integral": self.integral,
            "scheme": self.scheme.value,
            "provisional": self.provisional,
            "predicted": self.predicted,
            "recentered": self.recentered,
        }


class StampConvention(str, Enum):
    """Where in the coarse interval the producer's sample really lives.

    ``ENDPOINT``
        the sample is the value at the exchange instant ``t_n``;
    ``WINDOW_MIDPOINT``
        the sample is the *average over the last fine window* published at
        ``t_n``, i.e. the value at ``t_n - offset`` (the end-stamping practice
        reported by the coupling contract).
    """

    ENDPOINT = "endpoint"
    WINDOW_MIDPOINT = "window_midpoint"


@dataclass(frozen=True)
class StampRule:
    """Declared stamping convention plus the optional de-stamping repair.

    With ``convention = WINDOW_MIDPOINT`` and ``recenter = False`` the operator
    consumes the published sample as if it were an endpoint value: that is the
    shipped practice, and it is what leaves a *constant* phase offset (hence a
    staircase-saturating defect).

    ``recenter = True`` applies the Taylor shift that maps a window-midpoint
    sample of displacement ``d`` back onto the interval endpoint.  Two orders are
    available:

    ``order = 1``
        one-sided slope from the two newest samples; the residual rate bias is
        ``-(d^2/2) q2`` (``q2`` = second derivative), a cadence-independent
        floor, so the flux-defect staircase stops at ``O(d^2)`` however fine the
        exchange cadence gets;
    ``order = 2``
        Newton forward difference over three samples (``q_lo, q_prev, q_prev2``);
        the residual bias is ``O(d h^2 q3)`` (``q3`` = third derivative), i.e. it
        is pushed below the trapezoid's own ``O(h^2)`` term, so the defect
        converges at second order in the exchange cadence.

    The order-2 rule needs a second history sample; without one it degrades to
    the order-1 rule and reports that through :meth:`effective_order` rather than
    silently claiming a floor it does not have.
    """

    convention: StampConvention = StampConvention.ENDPOINT
    offset: float = 0.0
    recenter: bool = False
    order: int = 1

    def __post_init__(self) -> None:
        if self.offset < 0.0:
            raise CouplingLedgerError("stamp offset must be non-negative")
        if self.convention is StampConvention.ENDPOINT and self.offset > 0.0:
            raise CouplingLedgerError(
                "an endpoint convention cannot carry a positive stamp offset")
        if self.recenter and self.convention is StampConvention.ENDPOINT:
            raise CouplingLedgerError(
                "recentring only applies to a displaced (window) stamp convention")
        if int(self.order) not in (1, 2):
            raise CouplingLedgerError(
                f"de-stamping order must be 1 or 2, got {self.order!r}")

    def effective_order(self, q_prev: float | None,
                        q_prev2: float | None) -> int:
        """De-stamping order actually available from the supplied history."""
        if not self.recenter:
            return 0
        if int(self.order) >= 2 and q_prev is not None and q_prev2 is not None:
            return 2
        return 1

    def endpoints(self, q_lo: float, q_hi: float | None, q_prev: float | None,
                  h: float, q_prev2: float | None = None
                  ) -> tuple[float, float | None]:
        """Map published samples to endpoint values of ``[t_lo, t_hi]``.

        ``q_lo``/``q_hi`` are stamped at ``t_lo``/``t_hi`` but actually live at
        ``t - offset``; ``q_prev``/``q_prev2`` are the same stream one and two
        coarse intervals further back.
        """
        if not math.isfinite(q_lo):
            raise CouplingLedgerError(f"non-finite sample {q_lo!r}")
        if self.convention is StampConvention.ENDPOINT or not self.recenter:
            return float(q_lo), (None if q_hi is None else float(q_hi))
        d = self.offset
        if h <= 0.0:
            raise CouplingLedgerError("interval length must be positive")
        u = d / h
        order = self.effective_order(q_prev, q_prev2)

        def shift(q_new: float, q_mid: float | None,
                  q_mid2: float | None) -> float:
            base = float(q_new) if q_mid is None else float(q_mid)
            d1 = float(q_new) - base
            if order >= 2:
                second = float(q_mid2) if q_mid2 is not None else float(q_prev2)
                d2 = float(q_new) - 2.0 * base + second
                return float(q_new) + u * d1 + 0.5 * u * (u + 1.0) * d2
            return float(q_new) + u * d1

        est_lo = shift(q_lo, q_prev, q_prev2)
        est_hi = None if q_hi is None else shift(q_hi, q_lo, q_prev)
        return est_lo, est_hi

    def as_dict(self) -> dict:
        return {"convention": self.convention.value, "offset": self.offset,
                "recenter": self.recenter, "order": int(self.order)}


#: The convention-free default: the producer publishes endpoint values.
ENDPOINT_STAMP = StampRule()


class ExchangeOperator(abc.ABC):
    """Unified interface of the exchange algebra.

    An operator is a pure function of the interface history; it never touches
    solver state.  That is what lets the same object be exercised by the
    manufactured solution (order) and by the three physical chains
    (conservation), without duplicating either.
    """

    scheme: ExchangeScheme

    def __init__(self, stamp: StampRule | None = None) -> None:
        self.spec = SCHEME_TABLE[self.scheme]
        self.stamp = stamp or ENDPOINT_STAMP

    @property
    def declared_order(self) -> int:
        return self.spec.declared_order

    @property
    def causal(self) -> bool:
        return self.spec.causal

    @abc.abstractmethod
    def provisional_transfer(self, index: int, t_lo: float, t_hi: float,
                             *, q_lo: float, q_hi: float | None,
                             q_prev: float | None,
                             q_prev2: float | None = None) -> IntervalTransfer:
        """Exchange that the receiver may apply *before* ``t_hi`` is known.

        ``q_hi`` is the realised value at ``t_hi`` when the caller already knows
        it (idealised operators); a causal operator works from ``q_lo`` /
        ``q_prev`` only.
        """

    def corrected_transfer(self, provisional: IntervalTransfer,
                           q_hi: float,
                           q_prev2: float | None = None) -> IntervalTransfer:
        """Deferred correction once ``q(t_hi)`` becomes available.

        The default is the identity: an operator whose provisional exchange is
        already exact for its known samples needs no correction.
        """
        if not math.isfinite(q_hi):
            raise CouplingLedgerError(f"non-finite correction sample {q_hi!r}")
        return provisional


class LagExchange(ExchangeOperator):
    """End-stamp: hold the newest closed sample over the coming interval.

    This is the operator the contract reported as the source of the ``O(dt)``
    bias: the exchange instant is the interval end, yet the value held is the
    one that was already known at the interval start, so the applied rate lags
    the physical rate by up to one interval.
    """

    scheme = ExchangeScheme.LAG

    def provisional_transfer(self, index: int, t_lo: float, t_hi: float,
                             *, q_lo: float, q_hi: float | None,
                             q_prev: float | None,
                             q_prev2: float | None = None) -> IntervalTransfer:
        if not math.isfinite(q_lo):
            raise CouplingLedgerError(f"non-finite lag sample {q_lo!r}")
        value, _ = self.stamp.endpoints(q_lo, None, q_prev, t_hi - t_lo, q_prev2)
        return IntervalTransfer(index=index, t_lo=t_lo, t_hi=t_hi, rate=float(value),
                                scheme=self.scheme, q_lo=float(value),
                                q_hi_used=float(value),
                                recentered=self.stamp.recenter, q_lo_raw=float(q_lo),
                                q_prev_raw=q_prev)


class CenteredExchange(ExchangeOperator):
    """Crank-Nicolson midpoint: trapezoid of the two interval endpoints.

    Second order, but not causal: it needs ``q(t_hi)`` to advance the interval.
    It stays in the algebra as the reference member that a causal operator has
    to reproduce after correction.
    """

    scheme = ExchangeScheme.CENTERED

    def provisional_transfer(self, index: int, t_lo: float, t_hi: float,
                             *, q_lo: float, q_hi: float | None,
                             q_prev: float | None,
                             q_prev2: float | None = None) -> IntervalTransfer:
        if q_hi is None:
            raise CouplingLedgerError(
                "the centered operator needs the interval endpoint q(t_hi); "
                "a causal solver must use predictor_corrector")
        q_lo_eff, q_hi_eff = self.stamp.endpoints(q_lo, q_hi, q_prev, t_hi - t_lo,
                                                  q_prev2)
        return IntervalTransfer(index=index, t_lo=t_lo, t_hi=t_hi,
                                rate=0.5 * (float(q_lo_eff) + float(q_hi_eff)),
                                scheme=self.scheme, q_lo=float(q_lo_eff),
                                q_hi_used=float(q_hi_eff),
                                recentered=self.stamp.recenter, q_lo_raw=float(q_lo),
                                q_prev_raw=q_prev)


class PredictorCorrectorExchange(ExchangeOperator):
    """Causal second-order exchange: predicted endpoint, deferred correction.

    The provisional rate is the trapezoid over ``(q_lo, q_hat_hi)`` with the
    endpoint predicted by linear extrapolation of the closed history
    (``q_hat_hi = 2 q_lo - q_prev``; the first interval degrades to the lag
    value, which the ladders report separately as a start-up interval).  When
    the realised ``q(t_hi)`` arrives the interval is re-booked with the true
    trapezoid and the receiver takes the difference as a correction impulse.
    """

    scheme = ExchangeScheme.PREDICTOR_CORRECTOR

    def provisional_transfer(self, index: int, t_lo: float, t_hi: float,
                             *, q_lo: float, q_hi: float | None,
                             q_prev: float | None,
                             q_prev2: float | None = None) -> IntervalTransfer:
        q_lo_eff, _ = self.stamp.endpoints(q_lo, None, q_prev, t_hi - t_lo, q_prev2)
        q_lo_eff = float(q_lo_eff)
        if q_prev is None:
            q_hat = q_lo_eff
            predicted = False
        else:
            q_prev_eff, _ = self.stamp.endpoints(q_prev, None, q_prev2, t_hi - t_lo,
                                                 q_prev2)
            q_hat = 2.0 * q_lo_eff - float(q_prev_eff)
            predicted = True
        return IntervalTransfer(index=index, t_lo=t_lo, t_hi=t_hi,
                                rate=0.5 * (q_lo_eff + q_hat),
                                scheme=self.scheme, q_lo=q_lo_eff, q_hi_used=q_hat,
                                provisional=True, predicted=predicted,
                                recentered=self.stamp.recenter, q_lo_raw=float(q_lo),
                                q_prev_raw=q_prev)

    def corrected_transfer(self, provisional: IntervalTransfer,
                           q_hi: float,
                           q_prev2: float | None = None) -> IntervalTransfer:
        super().corrected_transfer(provisional, q_hi)
        q_hi_eff = (self.stamp.endpoints(provisional.q_lo_raw, q_hi,
                                         provisional.q_prev_raw, provisional.h,
                                         q_prev2)[1]
                    if self.stamp.recenter and self.stamp.offset > 0.0 else float(q_hi))
        rate = 0.5 * (provisional.q_lo + float(q_hi_eff))
        return IntervalTransfer(index=provisional.index, t_lo=provisional.t_lo,
                                t_hi=provisional.t_hi, rate=rate,
                                scheme=self.scheme, q_lo=provisional.q_lo,
                                q_hi_used=float(q_hi_eff),
                                recentered=provisional.recentered,
                                q_lo_raw=provisional.q_lo_raw,
                                q_prev_raw=provisional.q_prev_raw)


def operator_for(scheme: ExchangeScheme | str,
                 stamp: StampRule | None = None) -> ExchangeOperator:
    """Factory keyed by the scheme enum (or its string value) and the stamp rule."""
    scheme = ExchangeScheme(scheme)
    return {
        ExchangeScheme.LAG: LagExchange,
        ExchangeScheme.CENTERED: CenteredExchange,
        ExchangeScheme.PREDICTOR_CORRECTOR: PredictorCorrectorExchange,
    }[scheme](stamp)


# ---------------------------------------------------------------------------
# interval-complete attribution ledger
# ---------------------------------------------------------------------------
@dataclass
class IntervalAttribution:
    """Booking state of one exchange interval."""

    index: int
    t_lo: float
    t_hi: float
    produced: float = 0.0
    released: float = 0.0
    applied: float = 0.0
    correction: float = 0.0
    sub_samples: int = 0
    released_sub_samples: int = 0
    closed: bool = field(default=False, repr=False)

    @property
    def h(self) -> float:
        return self.t_hi - self.t_lo

    @property
    def net_applied(self) -> float:
        return self.applied + self.correction

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "t_lo": self.t_lo,
            "t_hi": self.t_hi,
            "produced": self.produced,
            "released": self.released,
            "applied": self.applied,
            "correction": self.correction,
            "sub_samples": self.sub_samples,
            "released_sub_samples": self.released_sub_samples,
            "closed": self.closed,
        }


@dataclass(frozen=True)
class AttributionSummary:
    """Audit of one interface over the whole run."""

    interface: str
    intervals: int
    expected_intervals: int
    produced_total: float
    released_total: float
    applied_total: float
    correction_total: float
    closed_intervals: int
    open_intervals: tuple[int, ...]
    unattributed_share: float
    deficit_share: float
    max_interval_gap: float

    @property
    def complete(self) -> bool:
        return (self.intervals == self.expected_intervals
                and not self.open_intervals
                and self.unattributed_share <= 1e-14
                and self.deficit_share <= 1e-12
                and self.max_interval_gap <= 1e-12)

    def as_dict(self) -> dict:
        return {
            "interface": self.interface,
            "intervals": self.intervals,
            "expected_intervals": self.expected_intervals,
            "produced_total": self.produced_total,
            "released_total": self.released_total,
            "applied_total": self.applied_total,
            "correction_total": self.correction_total,
            "closed_intervals": self.closed_intervals,
            "open_intervals": list(self.open_intervals),
            "unattributed_share": self.unattributed_share,
            "deficit_share": self.deficit_share,
            "max_interval_gap": self.max_interval_gap,
            "complete": self.complete,
        }


class IntervalExchangeLedger:
    """Interval-complete attribution of one coupling interface.

    Parameters
    ----------
    interface:
        Name of the interface (``"weld_flux"``, ``"viv_force"``, ...).
    n_intervals:
        Number of coarse exchange intervals the run will close.
    sides:
        ``(producer, receiver)`` account names, for the audit record.
    """

    def __init__(self, interface: str, n_intervals: int,
                 sides: Sequence[str] = ("source", "target")) -> None:
        if n_intervals <= 0:
            raise CouplingLedgerError("n_intervals must be positive")
        if len(sides) != 2:
            raise CouplingLedgerError("an interface has exactly two sides")
        self.interface = interface
        self.sides = tuple(sides)
        self.n_intervals = int(n_intervals)
        self._records: dict[int, IntervalAttribution] = {}

    # -- interval access --------------------------------------------------------
    def interval(self, index: int) -> IntervalAttribution:
        if not 0 <= index < self.n_intervals:
            raise CouplingLedgerError(
                f"interval {index} outside [0, {self.n_intervals})")
        record = self._records.get(index)
        if record is None:
            raise CouplingLedgerError(f"interval {index} was never opened")
        return record

    def open_interval(self, index: int, t_lo: float, t_hi: float) -> IntervalAttribution:
        record = self.interval(index)
        if not math.isclose(record.t_lo, t_lo, rel_tol=0.0, abs_tol=1e-12) or \
                not math.isclose(record.t_hi, t_hi, rel_tol=0.0, abs_tol=1e-12):
            raise CouplingLedgerError(
                f"interval {index} already declared as [{record.t_lo}, {record.t_hi}]")
        return record

    def declare_interval(self, index: int, t_lo: float, t_hi: float) -> IntervalAttribution:
        if index in self._records:
            raise CouplingLedgerError(f"interval {index} declared twice")
        if not t_hi > t_lo:
            raise CouplingLedgerError("interval must be increasing")
        record = IntervalAttribution(index=index, t_lo=float(t_lo), t_hi=float(t_hi))
        self._records[index] = record
        return record

    # -- bookings ---------------------------------------------------------------
    def book_produced(self, index: int, integral: float) -> None:
        """Producer records the sub-interval share this interval owns."""
        record = self.interval(index)
        if record.closed:
            raise CouplingLedgerError(f"interval {index} is closed; produced booking refused")
        if not math.isfinite(integral):
            raise CouplingLedgerError(f"non-finite produced share {integral!r}")
        record.produced += float(integral)
        record.sub_samples += 1

    def book_release(self, index: int, integral: float) -> None:
        """Producer records the share actually handed to this interval."""
        record = self.interval(index)
        if record.closed:
            raise CouplingLedgerError(f"interval {index} is closed; release refused")
        if not math.isfinite(integral):
            raise CouplingLedgerError(f"non-finite release {integral!r}")
        record.released += float(integral)
        record.released_sub_samples += 1

    def book_applied(self, index: int, integral: float) -> None:
        """Receiver records the integral it applied over this interval."""
        record = self.interval(index)
        if record.applied != 0.0 or record.correction != 0.0:
            raise CouplingLedgerError(
                f"interval {index} already carries an applied booking "
                f"(double booking refused)")
        if not math.isfinite(integral):
            raise CouplingLedgerError(f"non-finite applied integral {integral!r}")
        record.applied = float(integral)

    def book_correction(self, index: int, integral: float) -> None:
        """Receiver records the deferred correction of this interval."""
        record = self.interval(index)
        if record.closed:
            raise CouplingLedgerError(f"interval {index} is closed; correction refused")
        if not math.isfinite(integral):
            raise CouplingLedgerError(f"non-finite correction {integral!r}")
        record.correction += float(integral)

    def close_interval(self, index: int) -> IntervalAttribution:
        record = self.interval(index)
        if record.closed:
            raise CouplingLedgerError(f"interval {index} closed twice")
        record.closed = True
        return record

    # -- audit ------------------------------------------------------------------
    def summary(self) -> AttributionSummary:
        records = [self._records[i] for i in sorted(self._records)]
        produced = sum(r.produced for r in records)
        released = sum(r.released for r in records)
        applied = sum(r.applied for r in records)
        correction = sum(r.correction for r in records)
        scale = max(sum(abs(r.released) for r in records),
                    sum(abs(r.produced) for r in records), 1e-300)
        open_intervals = tuple(r.index for r in records if not r.closed)
        gaps = [abs(r.produced - r.released) for r in records]
        return AttributionSummary(
            interface=self.interface,
            intervals=len(records),
            expected_intervals=self.n_intervals,
            produced_total=produced,
            released_total=released,
            applied_total=applied,
            correction_total=correction,
            closed_intervals=sum(1 for r in records if r.closed),
            open_intervals=open_intervals,
            unattributed_share=abs(produced - released) / scale,
            deficit_share=abs(released - applied - correction) / scale,
            max_interval_gap=(max(gaps) / scale) if gaps else 0.0,
        )

    def assert_complete(self, *, unattributed_tol: float = 1e-14,
                        deficit_tol: float = 1e-12) -> AttributionSummary:
        summary = self.summary()
        if summary.open_intervals:
            raise CouplingLedgerError(
                f"{self.interface}: {len(summary.open_intervals)} intervals never closed")
        if summary.unattributed_share > unattributed_tol:
            raise CouplingLedgerError(
                f"{self.interface}: {summary.unattributed_share:.3e} of the produced "
                "quantity is not attributed to any interval")
        if summary.deficit_share > deficit_tol:
            raise CouplingLedgerError(
                f"{self.interface}: {summary.deficit_share:.3e} of the released "
                "quantity was not applied by the receiver")
        return summary