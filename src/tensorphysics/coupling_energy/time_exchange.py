"""Coupling manufactured solutions (MMS) and the three exchange chains.

The instrument this module builds answers one question: **can the temporal order
of a coupling exchange operator be observed at all, and if not, which of the two
reported defects masks it?**

Three layers:

``CouplingMMS``
    Analytic interface problem: a producer field ``u_f(t)``, a receiver field
    ``u_s(t)`` and an *analytic* interface flux ``q(t)`` with an exact interval
    integral.  Nothing on the exchange side is solved numerically, so a measured
    exponent is the operator's order and nothing else.

``ExchangeChain``
    The same exchange driven through a real receiver: a linear system integrated
    **exactly** for a piecewise-constant input (matrix exponential / closed
    form), with ``solve_ivp`` at ``rtol = 1e-12`` as the oracle.  Three chains
    follow the physics of the campaign node: ``viv`` (SDOF oscillator forced by
    the exchanged fluid force), ``weld`` (lumped weld-toe node heated by the
    exchanged flux) and ``settling`` (particle relaxing to the exchanged fluid
    velocity).

``run_ladder``
    Refines the *exchange cadence* while the fine solver step ``dt_fine`` is held
    fixed (the only knob a real partitioned code has), and reports per level the
    exchange defect, the receiver state error against the oracle, and the
    interval-attribution audit.

Two modelled defects are switchable so their effect can be separated:

``stamp_offset``
    The producer publishes the *window average* of the last fine window and the
    receiver stamps it at the interval end (``stamp_offset = dt_fine/2``), or the
    sample is a true endpoint value (``stamp_offset = 0``).  The first is the
    reported end-stamping practice: every sample is a midpoint value carrying a
    constant half-fine-step phase offset, so the defect it induces does not
    vanish when the exchange cadence is refined.

``tail_booking``
    The producer releases only the last sub-cycle share of the window instead of
    every sub-interval share.  That is the reported attribution defect; the
    resulting loss is ``(r-1)/r`` of the exchanged quantity in the smooth limit.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import expm

from .exchange_operators import (
    ExchangeScheme,
    IntervalExchangeLedger,
    StampConvention,
    StampRule,
    operator_for,
)
from .ledger import CouplingLedgerError

__all__ = [
    "CouplingMMS",
    "ExchangeChain",
    "ChainResult",
    "LadderResult",
    "run_ladder",
    "run_cadence_ladder",
    "fit_order_with_r2",
    "CHAIN_SPECS",
    "build_chain",
]


# ---------------------------------------------------------------------------
# order estimation with quality figure
# ---------------------------------------------------------------------------
def fit_order_with_r2(steps: Sequence[float], errors: Sequence[float]) -> tuple[float, float]:
    """Log-log least-squares slope and its R^2 (errors clamped to be positive)."""
    if len(steps) != len(errors):
        raise CouplingLedgerError("steps and errors must have equal length")
    if len(steps) < 2:
        raise CouplingLedgerError("at least two levels are required")
    xs = np.log(np.asarray(steps, dtype=float))
    ys = np.log(np.asarray([max(float(e), 1e-300) for e in errors], dtype=float))
    nx, ny = float(xs.mean()), float(ys.mean())
    sxx = float(((xs - nx) ** 2).sum())
    if sxx <= 0.0:
        raise CouplingLedgerError("all step sizes are identical")
    slope = float(((xs - nx) * (ys - ny)).sum()) / sxx
    pred = ny + slope * (xs - nx)
    ss_res = float(((ys - pred) ** 2).sum())
    ss_tot = float(((ys - ny) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 1.0
    return slope, r2


# ---------------------------------------------------------------------------
# analytic interface problem
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CouplingMMS:
    """Analytic two-field interface problem with an exact flux integral.

    ``fluid_field``/``solid_field`` are the producer/receiver interface fields and
    ``flux`` is the exchanged rate, with ``antiderivative`` giving its exact
    integral.  The families below are smooth or single-sided ramps, so a ladder
    measures the operator and not a boundary transient.
    """

    name: str
    period: float
    flux: Callable[[float], float]
    antiderivative: Callable[[float], float]
    fluid_field: Callable[[float], float]
    solid_field: Callable[[float], float]
    description: str = ""

    def interval_integral(self, t_lo: float, t_hi: float) -> float:
        return self.antiderivative(t_hi) - self.antiderivative(t_lo)

    def total_integral(self, t_end: float) -> float:
        return self.antiderivative(t_end) - self.antiderivative(0.0)

    @staticmethod
    def harmonic(q0: float = 1.0, period: float = 1.0) -> "CouplingMMS":
        """``q = q0 (1 + 0.6 sin(w t) + 0.25 sin(3 w t))``, ``w = 2 pi / period``."""
        w = 2.0 * math.pi / period

        def flux(t: float) -> float:
            return q0 * (1.0 + 0.6 * math.sin(w * t) + 0.25 * math.sin(3.0 * w * t))

        def anti(t: float) -> float:
            return q0 * (t - 0.6 * math.cos(w * t) / w
                         - 0.25 * math.cos(3.0 * w * t) / (3.0 * w))

        return CouplingMMS(
            name="harmonic", period=period, flux=flux, antiderivative=anti,
            fluid_field=lambda t: 1.0 + 0.6 * math.sin(w * t),
            solid_field=lambda t: 0.5 * (1.0 - math.cos(w * t)),
            description="smooth two-tone flux: order ladder of the exchange operator",
        )

    @staticmethod
    def ramp_pulse(q0: float = 1.0, period: float = 1.0) -> "CouplingMMS":
        """One-sided ramp pulse ``q = q0 (1 - cos(w t)) / 2`` (weld-like)."""
        w = 2.0 * math.pi / period

        def flux(t: float) -> float:
            return q0 * 0.5 * (1.0 - math.cos(w * t))

        def anti(t: float) -> float:
            return q0 * 0.5 * (t - math.sin(w * t) / w)

        return CouplingMMS(
            name="ramp_pulse", period=period, flux=flux, antiderivative=anti,
            fluid_field=lambda t: 0.5 * (1.0 - math.cos(w * t)),
            solid_field=lambda t: math.sin(w * t) / w,
            description="start-ramp pulse: exposes the end-stamp boundary sliver",
        )


# ---------------------------------------------------------------------------
# the three physical chains
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ExchangeChain:
    """A two-way chain driven by an exchanged interface quantity.

    The receiver is linear and forced piecewise-constant, so its interval update
    is *exact* (matrix exponential / closed form): every error the ladder measures
    belongs to the exchange, not to the receiver's own time scheme.
    """

    name: str
    unit: str
    mms: CouplingMMS
    t_end: float
    dt_fine: float
    y0: np.ndarray
    state_index: int
    #: exact update of the receiver state with a constant applied rate
    advance: Callable[[np.ndarray, float, float], np.ndarray]
    #: right-hand side of the same ODE driven by the analytic rate
    rhs: Callable[[float, np.ndarray], np.ndarray]
    fluid_field: str = "u_f"
    solid_field: str = "u_s"
    description: str = ""

    def sample(self, t: float, stamp_offset: float) -> float:
        """Producer sample at ``t`` under a declared stamping convention.

        ``stamp_offset = 0`` publishes the endpoint value; ``stamp_offset = d``
        publishes the window average of ``[t - 2d, t]`` stamped at ``t``, i.e. a
        midpoint value displaced by ``d`` — the end-stamping practice.
        """
        if stamp_offset <= 0.0:
            return self.mms.flux(t)
        return self.mms.interval_integral(t - 2.0 * stamp_offset, t) / (2.0 * stamp_offset)

    def oracle(self, times: np.ndarray) -> np.ndarray:
        """Oracle receiver trajectory (``solve_ivp``, ``rtol = 1e-12``)."""
        sol = solve_ivp(lambda t, y: np.asarray(self.rhs(t, y), dtype=float),
                        (0.0, float(times[-1]) + 1e-12), np.asarray(self.y0, dtype=float),
                        rtol=1e-12, atol=1e-12, method="DOP853", dense_output=True)
        if not sol.success:
            raise CouplingLedgerError(f"oracle integration failed for {self.name}: {sol.message}")
        return np.asarray(sol.sol(np.asarray(times, dtype=float)))


def _build_viv(period: float = 0.625, t_end: float = 2.5,
               dt_fine: float = 5.0e-4) -> ExchangeChain:
    """VIV chain: SDOF oscillator forced by the exchanged fluid force."""
    m, c, k, f0 = 12.0, 3.0, 1.2e3, 18.0
    w = 2.0 * math.pi / period
    mms = CouplingMMS(
        name="viv_force", period=period,
        flux=lambda t: f0 * math.sin(w * t),
        antiderivative=lambda t: f0 / w * (1.0 - math.cos(w * t)),
        fluid_field=lambda t: math.sin(w * t),
        solid_field=lambda t: 0.0,
        description="VIV lift force handed to a SDOF oscillator",
    )
    A = np.array([[0.0, 1.0, 0.0],
                  [-k / m, -c / m, 1.0 / m],
                  [0.0, 0.0, 0.0]])

    def advance(y: np.ndarray, rate: float, h: float) -> np.ndarray:
        z = np.array([float(y[0]), float(y[1]), float(rate)])
        return np.asarray(expm(A * h) @ z)[:2]

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.array([float(y[1]),
                         (f0 * math.sin(w * t) - c * float(y[1]) - k * float(y[0])) / m])

    return ExchangeChain(
        name="viv", unit="N", mms=mms, t_end=t_end, dt_fine=dt_fine,
        y0=np.array([0.0, 0.0]), state_index=1, advance=advance, rhs=rhs,
        fluid_field="F_L(t) [N]", solid_field="v_s(t) [m/s]",
        description="VIV lock-in force exchange onto a 1-DOF structure",
    )


def _build_weld(period: float = 1.0, t_end: float = 1.5,
                dt_fine: float = 2.0e-3) -> ExchangeChain:
    """Weld chain: lumped weld-toe node heated by the exchanged flux."""
    q0, cap, kth, t_inf = 6.0e3, 1.2e4, 18.0, 293.15
    w = 2.0 * math.pi / period
    mms = CouplingMMS(
        name="weld_flux", period=period,
        flux=lambda t: q0 * 0.5 * (1.0 - math.cos(w * t)),
        antiderivative=lambda t: q0 * 0.5 * (t - math.sin(w * t) / w),
        fluid_field=lambda t: 0.5 * (1.0 - math.cos(w * t)),
        solid_field=lambda t: 0.0,
        description="weld heat input handed to a lumped conduction node",
    )

    def advance(y: np.ndarray, rate: float, h: float) -> np.ndarray:
        t_eq = t_inf + rate / kth
        return np.array([t_eq + (float(y[0]) - t_eq) * math.exp(-kth * h / cap)])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        q = q0 * 0.5 * (1.0 - math.cos(w * t))
        return np.array([(q - kth * (float(y[0]) - t_inf)) / cap])

    return ExchangeChain(
        name="weld", unit="W", mms=mms, t_end=t_end, dt_fine=dt_fine,
        y0=np.array([t_inf]), state_index=0, advance=advance, rhs=rhs,
        fluid_field="q_weld(t) [W]", solid_field="T_s(t) [K]",
        description="weld-toe heat exchange onto a lumped conduction node",
    )


def _build_settling(period: float = 0.25, t_end: float = 1.0,
                    dt_fine: float = 5.0e-4) -> ExchangeChain:
    """Settling chain: particle relaxing to the exchanged fluid velocity."""
    d, rho_s, rho_f, mu, g = 4.0e-4, 2650.0, 1025.0, 1.6e-3, 9.81
    tau = rho_s * d * d / (18.0 * mu)
    v_t = (rho_s - rho_f) * g * d * d / (18.0 * mu)
    u0 = 0.02
    w = 2.0 * math.pi / period
    mms = CouplingMMS(
        name="settling_velocity", period=period,
        flux=lambda t: u0 * 0.5 * (1.0 - math.cos(w * t)),
        antiderivative=lambda t: u0 * 0.5 * (t - math.sin(w * t) / w),
        fluid_field=lambda t: 0.5 * (1.0 - math.cos(w * t)),
        solid_field=lambda t: 0.0,
        description="fluid velocity handed to a Stokes particle",
    )

    def advance(y: np.ndarray, rate: float, h: float) -> np.ndarray:
        target = rate + v_t
        return np.array([target + (float(y[0]) - target) * math.exp(-h / tau)])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        u = u0 * 0.5 * (1.0 - math.cos(w * t))
        return np.array([(u + v_t - float(y[0])) / tau])

    return ExchangeChain(
        name="settling", unit="m/s", mms=mms, t_end=t_end, dt_fine=dt_fine,
        y0=np.array([0.0]), state_index=0, advance=advance, rhs=rhs,
        fluid_field="u_f(t) [m/s]", solid_field="v_p(t) [m/s]",
        description="Stokes settling particle driven by the exchanged fluid velocity",
    )


CHAIN_SPECS: dict[str, Callable[..., ExchangeChain]] = {
    "viv": _build_viv,
    "weld": _build_weld,
    "settling": _build_settling,
}


def build_chain(name: str, **kwargs) -> ExchangeChain:
    """Build one of the three chains by name."""
    try:
        factory = CHAIN_SPECS[name]
    except KeyError as exc:
        raise CouplingLedgerError(
            f"unknown chain {name!r}; known: {sorted(CHAIN_SPECS)}") from exc
    return factory(**kwargs)


# ---------------------------------------------------------------------------
# ladder driver
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainResult:
    """Outcome of one (chain, scheme, cadence, defect) run."""

    chain: str
    scheme: str
    ratio: int
    h: float
    intervals: int
    defect_l1: float
    max_interval_defect: float
    state_error: float
    produced_total: float
    released_total: float
    applied_total: float
    unattributed_share: float
    deficit_share: float
    closed_intervals: int
    oracle_scale: float
    time_trace: tuple[float, ...] = ()
    state_trace: tuple[float, ...] = ()
    oracle_trace: tuple[float, ...] = ()

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()
                if k not in ("time_trace", "state_trace", "oracle_trace")}


@dataclass
class LadderResult:
    """A cadence ladder of one (chain, scheme, defect) configuration."""

    chain: str
    scheme: str
    stamp_offset: float
    tail_booking: bool
    recenter: bool = False
    subcycle: bool = False
    recenter_order: int = 1
    levels: list[ChainResult] = field(default_factory=list)

    @property
    def steps(self) -> list[float]:
        return [lv.h for lv in self.levels]

    def defects(self) -> list[float]:
        return [lv.defect_l1 for lv in self.levels]

    def state_errors(self) -> list[float]:
        return [lv.state_error for lv in self.levels]

    def fit(self) -> tuple[float, float]:
        return fit_order_with_r2(self.steps, self.defects())

    def fit_state(self) -> tuple[float, float]:
        return fit_order_with_r2(self.steps, self.state_errors())

    def saturation(self) -> float:
        """Plateau level of the defect: mean of the three finest cadences."""
        levels = sorted(self.defects())
        n = max(1, min(3, len(levels)))
        return sum(levels[:n]) / n

    def as_dict(self) -> dict:
        order, r2 = self.fit()
        sorder, sr2 = self.fit_state()
        return {
            "chain": self.chain,
            "scheme": self.scheme,
            "stamp_offset": self.stamp_offset,
            "tail_booking": self.tail_booking,
            "recenter": self.recenter,
            "recenter_order": int(self.recenter_order),
            "subcycle": self.subcycle,
            "levels": [lv.as_dict() for lv in self.levels],
            "observed_order": order,
            "fit_r2": r2,
            "state_order": sorder,
            "state_r2": sr2,
            "saturation": self.saturation(),
            "unattributed_last": self.levels[-1].unattributed_share if self.levels else None,
            "deficit_last": self.levels[-1].deficit_share if self.levels else None,
        }


def run_ladder(chain: ExchangeChain, scheme: ExchangeScheme | str, ratio: int, *,
               stamp_offset: float = 0.0, tail_booking: bool = False,
               recenter: bool = False, subcycle: bool = False,
               recenter_order: int = 1) -> ChainResult:
    """One exchange-cadence level: coarse step ``h = ratio * dt_fine``.

    The fine step is held fixed while ``ratio`` varies, which is the only knob a
    partitioned code really has: the exchange cadence, not the solver step.
    """
    if stamp_offset > 0.0:
        stamp = StampRule(StampConvention.WINDOW_MIDPOINT, float(stamp_offset),
                          bool(recenter), order=int(recenter_order))
    else:
        stamp = StampRule()
    operator = operator_for(scheme, stamp)
    ratio = int(ratio)
    if ratio < 1:
        raise CouplingLedgerError("ratio must be >= 1")
    h = ratio * chain.dt_fine
    n_intervals = int(round(chain.t_end / h))
    if n_intervals < 2:
        raise CouplingLedgerError("the run is too short for a ladder level")
    ledger = IntervalExchangeLedger(f"{chain.name}_exchange", n_intervals,
                                    sides=("producer", "receiver"))
    times = np.array([(n + 1) * h for n in range(n_intervals)])
    oracle = chain.oracle(times)[chain.state_index]

    if subcycle and tail_booking:
        raise CouplingLedgerError(
            "the sub-window arm publishes every window average by definition; "
            "tail booking removes them again")
    state = np.asarray(chain.y0, dtype=float)
    q_prev: float | None = None
    q_prev2: float | None = None
    m_prev: float | None = None  # kept for traceability of the published stream
    time_trace: list[float] = []
    state_trace: list[float] = []
    oracle_trace: list[float] = []
    defect_sum = 0.0
    exact_sum = 0.0
    interval_diffs: list[float] = []
    for n in range(n_intervals):
        t_lo, t_hi = n * h, (n + 1) * h
        ledger.declare_interval(n, t_lo, t_hi)

        # -- producer: exact sub-interval shares and their release --------------
        exact = chain.mms.interval_integral(t_lo, t_hi)
        shares = [chain.mms.interval_integral(t_lo + k * chain.dt_fine,
                                              t_lo + (k + 1) * chain.dt_fine)
                  for k in range(ratio)]
        for share in shares:
            ledger.book_produced(n, share)
        if tail_booking:
            ledger.book_release(n, shares[-1])
        else:
            for share in shares:
                ledger.book_release(n, share)

        if subcycle:
            # interval-complete exchange: every fine window average is published
            # and booked into the interval that owns it, and the receiver applies
            # them one by one.  Each average *is* the exact integral of its own
            # window, so the handover carries no quadrature error at all.  This
            # is the exact-attribution control arm of the algebra.
            sub_rates = [chain.sample(t_lo + (k + 1) * chain.dt_fine,
                                      0.5 * chain.dt_fine) for k in range(ratio)]
            applied = 0.0
            for rate in sub_rates:
                applied += rate * chain.dt_fine
                state = chain.advance(state, rate, chain.dt_fine)
            ledger.book_applied(n, applied)
            ledger.close_interval(n)
            defect_sum += abs(applied - exact)
            exact_sum += abs(exact)
            interval_diffs.append(abs(applied - exact))
            q_prev2 = q_prev
            q_prev = chain.sample(t_lo, 0.0)
            m_prev = q_prev
            time_trace.append(t_hi)
            state_trace.append(float(state[chain.state_index]))
            oracle_trace.append(float(oracle[n]))
            continue

        # -- exchange operator --------------------------------------------------
        # published samples: endpoint values, or the end-stamped window average
        # (midpoint value displaced by ``stamp_offset``).  The de-stamping rule
        # lives in the operator algebra, not here.
        q_lo = chain.sample(t_lo, stamp_offset)
        q_hi = chain.sample(t_hi, stamp_offset)
        transfer = operator.provisional_transfer(n, t_lo, t_hi, q_lo=q_lo,
                                                 q_hi=q_hi, q_prev=q_prev,
                                                 q_prev2=q_prev2)
        ledger.book_applied(n, transfer.integral)
        provisional = chain.advance(state, transfer.rate, h)

        # -- deferred correction (predictor-corrector only) ---------------------
        corrected = operator.corrected_transfer(transfer, q_hi, q_prev2=q_prev2)
        if abs(corrected.rate - transfer.rate) > 0.0:
            ledger.book_correction(n, corrected.integral - transfer.integral)
            state = chain.advance(state, corrected.rate, h)
        else:
            state = provisional
        applied = corrected.integral

        ledger.close_interval(n)
        interval_diffs.append(abs(applied - exact))
        defect_sum += abs(applied - exact)
        exact_sum += abs(exact)
        q_prev2 = q_prev
        q_prev = q_lo
        m_prev = q_lo
        time_trace.append(t_hi)
        state_trace.append(float(state[chain.state_index]))
        oracle_trace.append(float(oracle[n]))

    summary = ledger.summary()
    # the worst single interval, normalised by the run-mean |exchange| so that an
    # interval whose exact integral passes through zero is not scored as infinite
    interval_scale = (exact_sum / max(len(interval_diffs), 1)) if interval_diffs else 0.0
    max_interval_defect = (max(interval_diffs) / interval_scale) if interval_scale > 0.0 else 0.0
    state_error_scale = float(np.max(np.abs(oracle - oracle[0])))
    if state_error_scale <= 0.0:
        state_error_scale = 1.0
    state_error = abs(float(state[chain.state_index]) - float(oracle[-1])) / state_error_scale
    return ChainResult(
        chain=chain.name,
        scheme=operator.scheme.value,
        ratio=ratio,
        h=h,
        intervals=n_intervals,
        defect_l1=defect_sum / max(exact_sum, 1e-300),
        max_interval_defect=max_interval_defect,
        state_error=state_error,
        produced_total=summary.produced_total,
        released_total=summary.released_total,
        applied_total=summary.applied_total + summary.correction_total,
        unattributed_share=summary.unattributed_share,
        deficit_share=summary.deficit_share,
        closed_intervals=summary.closed_intervals,
        oracle_scale=state_error_scale,
        time_trace=tuple(time_trace),
        state_trace=tuple(state_trace),
        oracle_trace=tuple(oracle_trace),
    )


def run_cadence_ladder(chain: ExchangeChain | str, scheme: ExchangeScheme | str, *,
                       ratios: Sequence[int] = (1, 2, 4, 8, 16, 32),
                       stamp_offset_ratio: float = 0.0,
                       tail_booking: bool = False, recenter: bool = False,
                       subcycle: bool = False, recenter_order: int = 1,
                       **chain_kwargs) -> LadderResult:
    """Run a full cadence ladder for one chain and one operator.

    ``stamp_offset_ratio`` is expressed in units of ``dt_fine``: ``0`` publishes
    endpoint samples, ``0.5`` publishes the end-stamped window average
    (``stamp_offset = 0.5 dt_fine``), which is the reported practice.
    ``recenter = True`` additionally applies the de-stamping rule of the algebra
    (a first-order Taylor shift with the sampled slope), which is the repair the
    campaign node asks for on top of the CN midpoint operator.
    """
    if isinstance(chain, str):
        chain = build_chain(chain, **chain_kwargs)
    offset = stamp_offset_ratio * chain.dt_fine
    ladder = LadderResult(chain=chain.name, scheme=ExchangeScheme(scheme).value,
                          stamp_offset=offset, tail_booking=bool(tail_booking),
                          recenter=bool(recenter), subcycle=bool(subcycle),
                          recenter_order=int(recenter_order))
    for ratio in ratios:
        ladder.levels.append(run_ladder(chain, scheme, ratio,
                                        stamp_offset=offset,
                                        tail_booking=tail_booking,
                                        recenter=recenter,
                                        subcycle=subcycle,
                                        recenter_order=int(recenter_order)))
    return ladder