"""Coupling manufactured solution (MMS) quadrature ladder.

The chains in :mod:`tensorphysics.coupling_energy.time_exchange` measure the
exchange *through a receiver*.  This module measures the exchange operator alone,
on an analytic flux with a closed-form interval integral, so a measured exponent
cannot be attributed to anything but the operator algebra.

Two conventions are exercised:

* ``endpoint`` — the producer publishes ``q(t_n)``, the operator integrates;
* ``window`` — the producer publishes the fine-window average stamped at the
  interval end (displaced by ``offset``), which is the practice the coupling
  contract reports.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .exchange_operators import (
    ExchangeScheme,
    StampConvention,
    StampRule,
    operator_for,
)
from .ledger import CouplingLedgerError
from .time_exchange import CouplingMMS, fit_order_with_r2

__all__ = [
    "MMSLevel",
    "MMSLadder",
    "mms_quadrature_level",
    "mms_order_ladder",
]


@dataclass(frozen=True)
class MMSLevel:
    h: float
    intervals: int
    defect_l1: float
    max_interval_defect: float

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class MMSLadder:
    mms: str
    scheme: str
    convention: str
    offset: float
    recenter: bool
    corrected: bool
    levels: list[MMSLevel]

    @property
    def steps(self) -> list[float]:
        return [lv.h for lv in self.levels]

    def defects(self) -> list[float]:
        return [lv.defect_l1 for lv in self.levels]

    def fit(self) -> tuple[float, float]:
        return fit_order_with_r2(self.steps, self.defects())

    def as_dict(self) -> dict:
        order, r2 = self.fit()
        return {
            "mms": self.mms,
            "scheme": self.scheme,
            "convention": self.convention,
            "offset": self.offset,
            "recenter": self.recenter,
            "corrected": self.corrected,
            "levels": [lv.as_dict() for lv in self.levels],
            "observed_order": order,
            "fit_r2": r2,
        }


def mms_quadrature_level(mms: CouplingMMS, scheme: ExchangeScheme | str, h: float, *,
                         offset: float = 0.0, recenter: bool = False,
                         corrected: bool = True, t_end: float | None = None) -> MMSLevel:
    """One quadrature level: interval integrals of ``q`` on a grid of step ``h``."""
    stamp = (StampRule(StampConvention.WINDOW_MIDPOINT, offset, recenter)
             if offset > 0.0 else StampRule())
    operator = operator_for(scheme, stamp)
    t_end = float(t_end if t_end is not None else mms.period)
    n = int(round(t_end / h))
    if n < 2:
        raise CouplingLedgerError("the ladder level is too short")
    q_prev: float | None = None
    defect_sum = 0.0
    exact_sum = 0.0
    worst = 0.0
    for i in range(n):
        t_lo, t_hi = i * h, (i + 1) * h
        if offset > 0.0:
            q_lo = (mms.interval_integral(t_lo - 2.0 * offset, t_lo) / (2.0 * offset))
            q_hi = (mms.interval_integral(t_hi - 2.0 * offset, t_hi) / (2.0 * offset))
        else:
            q_lo, q_hi = mms.flux(t_lo), mms.flux(t_hi)
        transfer = operator.provisional_transfer(i, t_lo, t_hi, q_lo=q_lo,
                                                 q_hi=q_hi, q_prev=q_prev)
        if corrected:
            transfer = operator.corrected_transfer(transfer, q_hi)
        exact = mms.interval_integral(t_lo, t_hi)
        defect_sum += abs(transfer.integral - exact)
        exact_sum += abs(exact)
        worst = max(worst, abs(transfer.integral - exact) / max(abs(exact), 1e-300))
        q_prev = q_lo
    return MMSLevel(h=h, intervals=n,
                    defect_l1=defect_sum / max(exact_sum, 1e-300),
                    max_interval_defect=worst)


def mms_order_ladder(mms: CouplingMMS | None, scheme: ExchangeScheme | str, *,
                     h_list: Sequence[float] = (0.25, 0.125, 0.0625, 0.03125, 0.015625),
                     offset: float = 0.0, offset_ratio: float = 0.0,
                     recenter: bool = False,
                     corrected: bool = True, t_end: float | None = None) -> MMSLadder:
    """Full quadrature ladder of one operator on one analytic flux."""
    mms = mms or CouplingMMS.harmonic()
    if offset > 0.0 and offset_ratio > 0.0:
        raise CouplingLedgerError("give either a fixed offset or an offset ratio")
    levels = []
    for h in h_list:
        offset_h = offset if offset_ratio <= 0.0 else offset_ratio * h
        levels.append(mms_quadrature_level(mms, scheme, h, offset=offset_h,
                                           recenter=recenter, corrected=corrected,
                                           t_end=t_end))
    return MMSLadder(
        mms=mms.name,
        scheme=ExchangeScheme(scheme).value,
        convention="window" if (offset > 0.0 or offset_ratio > 0.0) else "endpoint",
        offset=offset if offset_ratio <= 0.0 else offset_ratio,
        recenter=bool(recenter),
        corrected=bool(corrected),
        levels=levels,
    )