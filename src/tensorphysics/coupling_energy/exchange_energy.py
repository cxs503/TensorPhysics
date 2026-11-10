"""Interval-complete *energy* ledger for the coupling time-exchange algebra.

Why this module exists (campaign node ``p223w3-n05``)
----------------------------------------------------
:mod:`tensorphysics.coupling_energy.exchange_operators` closes the exchange
algebra on the **flux** (``int q dt``) and
:mod:`tensorphysics.coupling_energy.exchange_gate` accepts it on the flux, the
receiver state and the attribution bookkeeping.  Neither answers the question the
coupled energy budget actually asks:

    how much *energy* does the exchange operator put into the receiver, and is
    every unit of it booked into exactly one exchange interval?

A flux defect and an energy defect are not the same functional.  The exchange
enters the receiver through a power conjugate ``g`` (velocity for a force, the
identity for a heat flux, ``(v - v_t)/tau`` for a Stokes velocity handover), so
the interface energy of one interval is ``P = q_bar * int g dt``: a lagging
operator mis-weights the flux by whatever ``g`` is doing inside the interval.
A flux defect that is harmless where ``g`` is small is *not* harmless in energy.

Definitions used throughout
---------------------------
Every chain of :mod:`tensorphysics.coupling_energy.time_exchange` is linear and
forced piecewise-constant, and its closed-form update is an exact semigroup
(``advance(advance(y, q, a), q, b) == advance(y, q, a + b)``).  That yields three
measurable energy-valued quantities per run:

``stored`` ``E(y)``
    the receiver's stored-energy functional, chosen so that the *only* two terms
    in its balance are the interface power and a non-negative dissipation::

        dE/dt = q(t) * g(y) - dis(y),     dis >= 0

    ==========  ==============================  =====================  ==========================
    chain       ``E``                           ``g``                  ``dis``
    ==========  ==============================  =====================  ==========================
    ``viv``     ``m v^2/2 + k x^2/2``           ``v``                  ``c v^2``
    ``weld``    ``C (T - T_inf)``               ``1``                  ``k_th (T - T_inf)``
    ``settling````(v - v_t)^2 / 2``             ``(v - v_t)/tau``      ``(v - v_t)^2 / tau``
    ==========  ==============================  =====================  ==========================

``interface energy`` ``P_i = q_bar_i * int_{t_lo}^{t_hi} g dt``
    what the receiver actually receives on interval ``i``.

``interface energy defect`` ``eps_E``
    ``|W_applied - W_physical| / |W_physical|`` with ``W_physical = int q_true g dt``
    on the *same* trajectory.  Both integrals use the same sub-grid and the same
    quadrature weights, so ``eps_E`` isolates the operator: it is the spurious
    (or missing) interface energy of the scheme — first order for a lagging hold,
    second order for a time-centred one.

``energy balance residual``
    ``|E(T) - E(0) - W_applied + D| / scale`` through
    :class:`tensorphysics.coupling_energy.ledger.EnergyLedger`.  This is the
    instrument's own cross-check: ``E`` and ``(q g - dis)`` are two readings of
    the same trajectory, so the balance must close to quadrature accuracy
    whatever the operator does.

Interval-complete attribution, in energy
---------------------------------------
:class:`tensorphysics.coupling_energy.exchange_operators.IntervalExchangeLedger`
is reused unchanged with one booking per fine sub-window: the producer books the
energy every sub-window it owns, the receiver books the energy it applied.  Two
independent loss channels then separate, which is exactly what the coupling
contract needs to distinguish:

``unattributed`` (booking rule)
    ``(r-1)/r`` for the shipped "book only the tail of the window" practice —
    the 87-94 % heat loss of the contract, here expressed in the conserved
    quantity rather than in the flux;
``deficit`` (operator)
    the energy the released quantity does not deliver — ``O(dt)`` for a lagging
    hold, ``O(dt^2)`` for a time-centred one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np

from .exchange_operators import (
    ExchangeScheme,
    IntervalExchangeLedger,
    StampConvention,
    StampRule,
    operator_for,
)
from .ledger import CouplingLedgerError, EnergyLedger
from .time_exchange import (
    ExchangeChain,
    build_chain,
    fit_order_with_r2,
    run_ladder,
)

__all__ = [
    "ChainEnergy",
    "ENERGY_SPECS",
    "energy_account_for",
    "EnergyLevel",
    "EnergyLadder",
    "run_energy_level",
    "run_energy_ladder",
    "run_energy_study",
    "gate_energy",
    "ENERGY_ARMS",
    "DT_FRACTIONS",
    "TAIL_RATIOS",
]


# ---------------------------------------------------------------------------
# per-chain energy functionals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChainEnergy:
    """Stored-energy functional, power conjugate and dissipation of one chain.

    The three callables must satisfy ``dE/dt = q g - dis`` for the ODE of the
    chain.  The gate does not trust this table: it re-derives the identity from
    the chain's own right-hand side, so a drift between this table and the
    ``_build_*`` factories of :mod:`~tensorphysics.coupling_energy.time_exchange`
    is caught by a test instead of silently corrupting the audit.
    """

    name: str
    stored: Callable[[np.ndarray], float]
    conjugate: Callable[[np.ndarray], float]
    dissipation: Callable[[np.ndarray], float]
    energy_unit: str
    conjugate_description: str

    def stored_of(self, state: np.ndarray) -> float:
        return float(self.stored(np.asarray(state, dtype=float)))

    def conjugate_of(self, state: np.ndarray) -> float:
        return float(self.conjugate(np.asarray(state, dtype=float)))

    def dissipation_of(self, state: np.ndarray) -> float:
        return float(self.dissipation(np.asarray(state, dtype=float)))


def _viv_energy() -> ChainEnergy:
    m, c, k = 12.0, 3.0, 1.2e3
    return ChainEnergy(
        name="viv",
        stored=lambda y: 0.5 * m * float(y[1]) ** 2 + 0.5 * k * float(y[0]) ** 2,
        conjugate=lambda y: float(y[1]),
        dissipation=lambda y: c * float(y[1]) ** 2,
        energy_unit="J",
        conjugate_description="structural velocity v_s [m/s]",
    )


def _weld_energy() -> ChainEnergy:
    cap, kth, t_inf = 1.2e4, 18.0, 293.15
    return ChainEnergy(
        name="weld",
        stored=lambda y: cap * (float(y[0]) - t_inf),
        conjugate=lambda y: 1.0,
        dissipation=lambda y: kth * (float(y[0]) - t_inf),
        energy_unit="J",
        conjugate_description="unit thermal weight (a heat flux integrates to energy)",
    )


def _settling_energy() -> ChainEnergy:
    d, rho_s, rho_f, mu, g = 4.0e-4, 2650.0, 1025.0, 1.6e-3, 9.81
    tau = rho_s * d * d / (18.0 * mu)
    v_t = (rho_s - rho_f) * g * d * d / (18.0 * mu)
    return ChainEnergy(
        name="settling",
        stored=lambda y: 0.5 * (float(y[0]) - v_t) ** 2,
        conjugate=lambda y: (float(y[0]) - v_t) / tau,
        dissipation=lambda y: (float(y[0]) - v_t) ** 2 / tau,
        energy_unit="J/kg",
        conjugate_description="velocity slip (v_p - v_t)/tau",
    )


#: Energy functionals of the three chains.
ENERGY_SPECS: dict[str, Callable[[], ChainEnergy]] = {
    "viv": _viv_energy,
    "weld": _weld_energy,
    "settling": _settling_energy,
}


def energy_account_for(chain: ExchangeChain | str) -> ChainEnergy:
    """Energy functional of a chain (by object or by name)."""
    name = chain if isinstance(chain, str) else chain.name
    try:
        return ENERGY_SPECS[name]()
    except KeyError as exc:
        raise CouplingLedgerError(
            f"no energy functional registered for chain {name!r}; "
            f"known: {sorted(ENERGY_SPECS)}") from exc


# ---------------------------------------------------------------------------
# quadrature helpers
# ---------------------------------------------------------------------------
def _simpson(values: Sequence[float], h: float) -> float:
    """Composite Simpson rule on an odd number of equally spaced samples."""
    vals = [float(v) for v in values]
    n = len(vals)
    if n < 3 or n % 2 == 0:
        raise CouplingLedgerError("Simpson needs an odd number >= 3 of samples")
    total = vals[0] + vals[-1]
    for i in range(1, n - 1):
        total += (4.0 if i % 2 else 2.0) * vals[i]
    return total * h / 3.0


def _affine_stepper(chain: ExchangeChain, ds: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Affine update ``y -> M y + b q + c`` of one sub-step of length ``ds``.

    ``advance`` is affine in ``(y, q)`` for every chain (the exchange enters
    linearly and the non-coupling terms are constant), so the sub-step operator is
    recovered from ``n + 2`` probes of the chain's own ``advance`` and then applied
    by matrix-vector products.  This keeps the sub-grid cost off the
    matrix-exponential path without re-implementing any chain's update here.
    """
    key = (chain.name, float(chain.dt_fine), round(float(ds), 18))
    cached = _STEPPER_CACHE.get(key)
    if cached is not None:
        return cached
    n = int(np.asarray(chain.y0).size)
    zero = np.zeros(n)
    c = np.asarray(chain.advance(zero, 0.0, ds), dtype=float)
    columns = [np.asarray(chain.advance(np.eye(n)[i], 0.0, ds), dtype=float) - c
               for i in range(n)]
    b = np.asarray(chain.advance(zero, 1.0, ds), dtype=float) - c
    m = np.column_stack(columns)
    _STEPPER_CACHE[key] = (m, b, c)
    return m, b, c


_STEPPER_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _sub_trajectory(chain: ExchangeChain, state: np.ndarray, rate: float,
                    h: float, substeps: int) -> tuple[list[float], list[float], list[float]]:
    """Exact sub-trajectory of one constant-rate interval.

    ``advance`` is an exact semigroup (its closed form solves the linear ODE), so
    sampling it at ``s = j h / substeps`` is the exact trajectory of the receiver
    under the applied rate: the quadrature measures the exchange operator and
    never the receiver's own time integration.  The samples are produced by
    composing the sub-step operator of the chain, which is algebraically the same
    trajectory and keeps the sub-grid affordable.
    """
    if substeps < 2 or substeps % 2:
        raise CouplingLedgerError("substeps must be an even number >= 2")
    acct = energy_account_for(chain)
    ds = h / substeps
    m, b, c = _affine_stepper(chain, ds)
    g_vals: list[float] = []
    d_vals: list[float] = []
    t_vals: list[float] = []
    y = np.asarray(state, dtype=float)
    for j in range(substeps + 1):
        g_vals.append(acct.conjugate_of(y))
        d_vals.append(acct.dissipation_of(y))
        t_vals.append(j * ds)
        if j < substeps:
            y = m @ y + b * rate + c
    return g_vals, d_vals, t_vals


# ---------------------------------------------------------------------------
# one level of the energy ladder
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EnergyLevel:
    """Energy audit of one (chain, operator, dt, cadence) configuration."""

    chain: str
    scheme: str
    ratio: int
    dt: float
    h: float
    intervals: int
    interface_energy: float
    physical_energy: float
    energy_defect: float
    net_energy_error: float
    energy_balance_residual: float
    dissipation: float
    stored_initial: float
    stored_end: float
    reference_energy: float
    reference_energy_error: float
    unattributed_share: float
    deficit_share: float
    attribution_identity_residual: float
    max_interval_energy_gap: float
    closed_intervals: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class EnergyLadder:
    """A dt ladder of one (chain, operator, cadence) configuration."""

    chain: str
    scheme: str
    ratio: int
    stamp_offset: float
    tail_booking: bool = False
    recenter: bool = False
    subcycle: bool = False
    levels: list[EnergyLevel] = field(default_factory=list)

    @property
    def steps(self) -> list[float]:
        return [lv.dt for lv in self.levels]

    def defects(self) -> list[float]:
        return [max(lv.energy_defect, 1e-300) for lv in self.levels]

    def reference_errors(self) -> list[float]:
        return [max(lv.reference_energy_error, 1e-300) for lv in self.levels]

    def fit(self) -> tuple[float, float]:
        return fit_order_with_r2(self.steps, self.defects())

    def as_dict(self) -> dict:
        order, r2 = self.fit()
        sorder, sr2 = fit_order_with_r2(self.steps, self.reference_errors())
        return {
            "chain": self.chain,
            "scheme": self.scheme,
            "ratio": self.ratio,
            "stamp_offset": self.stamp_offset,
            "tail_booking": self.tail_booking,
            "recenter": self.recenter,
            "subcycle": self.subcycle,
            "levels": [lv.as_dict() for lv in self.levels],
            "energy_defect_order": order,
            "energy_defect_r2": r2,
            "reference_energy_order": sorder,
            "reference_energy_r2": sr2,
            "finest_energy_defect": self.defects()[0],
            "finest_reference_energy_error": self.reference_errors()[0],
            "balance_residual": max(lv.energy_balance_residual for lv in self.levels),
        }


def run_energy_level(chain: ExchangeChain | str, scheme: ExchangeScheme | str, *,
                     dt: float | None = None, ratio: int = 1,
                     stamp_offset_ratio: float = 0.0, recenter: bool = False,
                     subcycle: bool = False, tail_booking: bool = False,
                     substeps: int = 8) -> EnergyLevel:
    """One energy audit level: coarse step ``h = ratio * dt``.

    ``dt`` defaults to the chain's own fine step, so ``ratio`` is the only knob
    that changes when the exchange cadence is swept (as in the flux ladder);
    ``stamp_offset_ratio`` is in units of ``dt``, exactly as in
    :func:`tensorphysics.coupling_energy.time_exchange.run_cadence_ladder`.

    Two interface-energy figures are reported because they answer two different
    questions:

    ``energy_defect`` (L1)
        ``sum |P_applied - P_physical| / sum |P_physical|`` over the fine
        sub-windows — the local mis-weighting of the exchange, which is what an
        operator order is measured on.  The absolute values matter: the *signed*
        sum of the leading error of a symmetric rule telescopes to a boundary
        term and would hide the order (on the weld chain, whose run ends on an
        integer number of half periods, it cancels to machine precision).
    ``net_energy_error`` (signed)
        ``|sum (P_applied - P_physical)| / sum |P_physical|`` — the net spurious
        energy actually delivered to the receiver, boundary terms included.
    """
    if isinstance(chain, str):
        chain = build_chain(chain)
    acct = energy_account_for(chain)
    dt = float(chain.dt_fine if dt is None else dt)
    ratio = int(ratio)
    if ratio < 1:
        raise CouplingLedgerError("ratio must be >= 1")
    h = ratio * dt
    n_intervals = int(round(chain.t_end / h))
    if n_intervals < 2:
        raise CouplingLedgerError("the energy level is too short")
    t_final = n_intervals * h

    offset = stamp_offset_ratio * dt
    stamp = (StampRule(StampConvention.WINDOW_MIDPOINT, float(offset), bool(recenter))
             if offset > 0.0 else StampRule())
    operator = operator_for(scheme, stamp)

    energy_ledger = IntervalExchangeLedger(f"{chain.name}_energy", n_intervals,
                                           sides=("producer", "receiver"))
    balance = EnergyLedger(sides=("receiver",))

    state = np.asarray(chain.y0, dtype=float)
    stored_initial = acct.stored_of(state)
    balance.set("receiver", stored_initial)
    balance.sample(0.0)

    interface_total = 0.0
    physical_total = 0.0
    defect_l1 = 0.0
    dissipation_total = 0.0
    energy_scale = max(abs(stored_initial), 1e-300)
    ds = dt / substeps
    q_prev: float | None = None
    for n in range(n_intervals):
        t_lo = n * h
        t_hi = t_lo + h
        energy_ledger.declare_interval(n, t_lo, t_hi)

        # ---- the exchange the receiver applies on this coarse interval ------
        if subcycle:
            rates = [chain.sample(t_lo + (k + 1) * dt, 0.5 * dt) for k in range(ratio)]
        else:
            q_lo = chain.sample(t_lo, offset)
            q_hi = chain.sample(t_hi, offset)
            transfer = operator.provisional_transfer(n, t_lo, t_hi, q_lo=q_lo,
                                                     q_hi=q_hi, q_prev=q_prev)
            corrected = operator.corrected_transfer(transfer, q_hi)
            rates = [corrected.rate] * ratio
            q_prev = q_lo

        # ---- walk the fine sub-windows the interval owns --------------------
        applied_energy = 0.0
        dissipation_interval = 0.0
        physical_shares: list[float] = []
        for k, rate in enumerate(rates):
            a = t_lo + k * dt
            g_vals, d_vals, t_vals = _sub_trajectory(chain, state, rate, dt, substeps)
            q_vals = [chain.mms.flux(a + s) for s in t_vals]
            share = _simpson([gv * qv for gv, qv in zip(g_vals, q_vals)], ds)
            applied_share = rate * _simpson(g_vals, ds)
            physical_shares.append(share)
            applied_energy += applied_share
            defect_l1 += abs(applied_share - share)
            dissipation_interval += _simpson(d_vals, ds)
            state = chain.advance(state, rate, dt)

        for share in physical_shares:
            energy_ledger.book_produced(n, share)
        if tail_booking:
            energy_ledger.book_release(n, physical_shares[-1])
        else:
            for share in physical_shares:
                energy_ledger.book_release(n, share)
        energy_ledger.book_applied(n, applied_energy)
        energy_ledger.close_interval(n)

        interface_total += applied_energy
        physical_total += sum(physical_shares)
        dissipation_total += dissipation_interval

        stored = acct.stored_of(state)
        energy_scale = max(energy_scale, abs(stored), abs(interface_total),
                           abs(dissipation_total))
        balance.set("receiver", stored)
        # A *signed* sink is split instead of clamped: a negative sink means the
        # far field is heating the receiver (a weld node sitting exactly on the
        # reservoir temperature, plus its roundoff), which the ledger books as an
        # external input rather than silently flooring to zero.  The identity
        # ``E - external + dissipated == E(0)`` is unchanged by the split.
        balance.book_external(applied_energy - min(dissipation_interval, 0.0))
        balance.book_dissipation(max(dissipation_interval, 0.0))
        balance.sample(t_hi)

    summary = energy_ledger.summary()
    stored_end = acct.stored_of(state)
    scale = max(abs(physical_total), 1e-300)
    residual = max(abs(sample.total - stored_initial) for sample in balance.history)
    reference_states = np.asarray(chain.oracle(np.array([t_final])), dtype=float)
    reference_state = np.atleast_1d(reference_states[:, -1])
    reference_energy = acct.stored_of(reference_state)
    reference_scale = max(abs(reference_energy), abs(stored_end), scale, 1e-300)
    return EnergyLevel(
        chain=chain.name,
        scheme=operator.scheme.value,
        ratio=ratio,
        dt=dt,
        h=h,
        intervals=n_intervals,
        interface_energy=interface_total,
        physical_energy=physical_total,
        energy_defect=defect_l1 / scale,
        net_energy_error=abs(interface_total - physical_total) / scale,
        energy_balance_residual=residual / energy_scale,
        dissipation=dissipation_total,
        stored_initial=stored_initial,
        stored_end=stored_end,
        reference_energy=reference_energy,
        reference_energy_error=abs(stored_end - reference_energy) / reference_scale,
        unattributed_share=summary.unattributed_share,
        deficit_share=summary.deficit_share,
        attribution_identity_residual=(
            abs(summary.released_total - summary.produced_total)
            / max(abs(summary.produced_total), 1e-300)),
        max_interval_energy_gap=summary.max_interval_gap,
        closed_intervals=summary.closed_intervals,
    )


def run_energy_ladder(chain: ExchangeChain | str, scheme: ExchangeScheme | str, *,
                      dt_fractions: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
                      ratio: int = 1, stamp_offset_ratio: float = 0.0,
                      recenter: bool = False, subcycle: bool = False,
                      tail_booking: bool = False,
                      substeps: int = 8) -> EnergyLadder:
    """Full dt ladder of one (chain, operator, cadence) configuration."""
    if isinstance(chain, str):
        chain = build_chain(chain)
    ladder = EnergyLadder(chain=chain.name, scheme=ExchangeScheme(scheme).value,
                          ratio=int(ratio),
                          stamp_offset=float(stamp_offset_ratio) * chain.dt_fine,
                          tail_booking=bool(tail_booking),
                          recenter=bool(recenter), subcycle=bool(subcycle))
    for frac in dt_fractions:
        ladder.levels.append(run_energy_level(
            chain, scheme, dt=chain.dt_fine * float(frac), ratio=ratio,
            stamp_offset_ratio=stamp_offset_ratio, recenter=recenter,
            subcycle=subcycle, tail_booking=tail_booking, substeps=substeps))
    ladder.levels.sort(key=lambda lv: lv.dt)
    return ladder


# ---------------------------------------------------------------------------
# study + gate
# ---------------------------------------------------------------------------
#: Exchange arms compared in energy space.
ENERGY_ARMS: dict[str, dict] = {
    "lag_endpoint": {"scheme": "lag"},
    "centered_endpoint": {"scheme": "centered"},
    "predictor_corrector_endpoint": {"scheme": "predictor_corrector"},
    "centered_endstamp": {"scheme": "centered", "stamp_offset_ratio": 0.5},
    "centered_destamp": {"scheme": "centered", "stamp_offset_ratio": 0.5,
                         "recenter": True},
    "subcycle_complete": {"scheme": "centered", "subcycle": True, "ratio": 8},
}

#: dt ladder, as multiples of the chain's own fine step (finest first in records).
DT_FRACTIONS: tuple[float, ...] = (4.0, 2.0, 1.0, 0.5)

#: Cadences at which the booking rule (tail vs complete) is exercised.
TAIL_RATIOS: tuple[int, ...] = (2, 4, 8, 16)

#: Arms that carry no temporal delay by construction.
NO_DELAY_ARMS: tuple[str, ...] = ("centered_endpoint", "predictor_corrector_endpoint",
                                  "subcycle_complete")


def run_energy_study(*, chains: Sequence[str] = ("viv", "weld", "settling"),
                     dt_fractions: Sequence[float] = DT_FRACTIONS,
                     tail_ratios: Sequence[int] = TAIL_RATIOS,
                     substeps: int = 8) -> dict:
    """The whole energy study: dt ladders per chain plus the booking-rule sweep."""
    per_chain: dict[str, dict] = {}
    for name in chains:
        chain = build_chain(name)
        acct = energy_account_for(chain)
        arms: dict[str, dict] = {}
        for arm, kwargs in ENERGY_ARMS.items():
            kwargs = dict(kwargs)
            scheme = kwargs.pop("scheme")
            arms[arm] = run_energy_ladder(chain, scheme, dt_fractions=dt_fractions,
                                          substeps=substeps, **kwargs).as_dict()
        per_chain[name] = {
            "meta": {"unit": chain.unit, "t_end": chain.t_end,
                     "dt_fine": chain.dt_fine, "period": chain.mms.period,
                     "fluid_field": chain.fluid_field,
                     "solid_field": chain.solid_field,
                     "energy_unit": acct.energy_unit,
                     "conjugate": acct.conjugate_description,
                     "description": chain.description},
            "arms": arms,
        }

    # booking rule in energy space: tail-only versus complete sub-window booking
    weld = build_chain("weld")
    booking: dict[str, list[dict]] = {"tail": [], "complete": []}
    for ratio in tail_ratios:
        for key, tail in (("tail", True), ("complete", False)):
            level = run_energy_level(weld, "centered", ratio=ratio,
                                     tail_booking=tail, substeps=substeps)
            booking[key].append({
                "ratio": ratio,
                "unattributed_share": level.unattributed_share,
                "deficit_share": level.deficit_share,
                "predicted_loss": (ratio - 1.0) / ratio,
                "released_fraction": 1.0 - level.unattributed_share,
                "energy_defect": level.energy_defect,
                "attribution_identity_residual": level.attribution_identity_residual,
                "closed_intervals": level.closed_intervals,
                "intervals": level.intervals,
            })

    # quadrature sensitivity of the instrument itself: the balance residual must
    # fall with the sub-grid refinement (it is quadrature-limited), while the
    # energy defect must not move (it belongs to the operator, not to the grid).
    quadrature: dict[str, list[dict]] = {}
    for name in chains:
        chain = build_chain(name)
        rows = []
        for k in (4, 8, 16):
            level = run_energy_level(chain, "centered", dt=4.0 * dt_fine_scale(chain),
                                     substeps=k)
            rows.append({"substeps": k,
                         "balance_residual": level.energy_balance_residual,
                         "energy_defect": level.energy_defect})
        quadrature[name] = rows

    # the same configurations measured on the *flux* functional, so the report can
    # show that energy and flux are different readings of the same exchange
    cross: dict[str, dict] = {}
    for name, payload in per_chain.items():
        chain = build_chain(name)
        flux = {}
        for scheme in ("lag", "centered"):
            result = run_ladder(chain, scheme, 1)
            flux[scheme] = result.defect_l1
        energy = {scheme: payload["arms"][f"{scheme if scheme == 'lag' else 'centered'}"
                                          f"_{'endpoint'}"]
                  ["levels"][0]["energy_defect"] for scheme in ("lag", "centered")}
        cross[name] = {"flux_defect": flux, "energy_defect": energy,
                       "flux_over_energy": {k: flux[k] / max(energy[k], 1e-300)
                                            for k in flux}}

    return {
        "node": "p223w3-n05-physics-time-exchange",
        "dt_fractions": list(dt_fractions),
        "arms": sorted(ENERGY_ARMS),
        "chains": per_chain,
        "booking": booking,
        "quadrature": quadrature,
        "cross_check": cross,
    }


def dt_fine_scale(chain: ExchangeChain) -> float:
    """The chain's own fine step (the reference for every dt fraction)."""
    return float(chain.dt_fine)


def _fit(values: Sequence[float], steps: Sequence[float]) -> tuple[float, float]:
    return fit_order_with_r2(list(steps), [max(float(v), 1e-300) for v in values])


def _weld_energy_identity() -> dict:
    """Cross-check the energy table against each chain's own right-hand side.

    ``dE/dt`` from the chain's ``rhs`` must equal ``q g - dis`` from the energy
    table; the residual is reported so a drift between the two definitions can
    never pass silently.
    """
    out: dict[str, float] = {}
    for name in ("viv", "weld", "settling"):
        chain = build_chain(name)
        acct = energy_account_for(chain)
        eps = 1e-7
        y = np.asarray(chain.y0, dtype=float) + np.array([1e-3] * len(chain.y0))
        t0 = 0.37
        dy = chain.rhs(t0, y)
        dE = ((acct.stored_of(y + eps * np.asarray(dy, dtype=float))
               - acct.stored_of(y - eps * np.asarray(dy, dtype=float))) / (2.0 * eps))
        predicted = (chain.mms.flux(t0) * acct.conjugate_of(y)
                     - acct.dissipation_of(y))
        scale = max(abs(predicted), abs(dE), 1e-300)
        out[name] = abs(dE - predicted) / scale
    return out


def gate_energy(study: dict) -> dict:
    """Three-valued acceptance gate of the energy ledger."""
    criteria: list[dict] = []

    def add(key: str, description: str, required: str, measured, passed: bool) -> None:
        criteria.append({"key": key, "description": description,
                         "required": required, "measured": measured,
                         "verdict": "pass" if passed else "fail"})

    identity = _weld_energy_identity()
    add("energy_identity_matches_rhs",
        "the energy table must reproduce dE/dt = q g - dis of every chain rhs",
        "relative residual <= 1e-6 on all three chains",
        identity, all(v <= 1e-6 for v in identity.values()))

    # -- operator order in energy space ---------------------------------------
    orders: dict[str, dict] = {}
    order_ok = True
    for arm, (lo, hi) in (("lag_endpoint", (0.8, 1.2)),
                          ("centered_endpoint", (1.8, 2.2)),
                          ("predictor_corrector_endpoint", (1.8, 2.2))):
        per_chain = {}
        for name, payload in study["chains"].items():
            ladder = payload["arms"][arm]
            order, r2 = _fit(ladder["levels"] and
                             [lv["energy_defect"] for lv in ladder["levels"]],
                             [lv["dt"] for lv in ladder["levels"]])
            per_chain[name] = {"order": order, "r2": r2,
                               "finest": ladder["levels"][0]["energy_defect"],
                               "stored_order": ladder["energy_defect_order"]}
            order_ok &= lo <= order <= hi and r2 >= 0.99
        orders[arm] = per_chain
        add(f"energy_order_{arm}",
            f"the {arm} operator must show the declared order in interface energy",
            f"{lo} <= order <= {hi} and R^2 >= 0.99 on all three chains",
            per_chain, order_ok)

    # -- criterion (d): the no-delay implementation keeps the energy residual < 1e-5
    delay_numbers: dict[str, dict] = {}
    delay_ok = True
    for name, payload in study["chains"].items():
        per_arm = {}
        for arm in NO_DELAY_ARMS:
            ladder = payload["arms"][arm]
            finest = ladder["levels"][0]
            per_arm[arm] = {"energy_defect": finest["energy_defect"],
                            "reference_energy_error": finest["reference_energy_error"],
                            "dt": finest["dt"]}
            delay_ok &= (finest["energy_defect"] <= 1e-5
                         and finest["reference_energy_error"] <= 1e-5)
        delay_numbers[name] = per_arm
    add("no_delay_energy_residual",
        "a time-centred (no-delay) exchange must leave < 1e-5 relative energy "
        "defect and < 1e-5 stored-energy error at the finest step",
        "energy_defect <= 1e-5 and reference_energy_error <= 1e-5 on all three chains",
        delay_numbers, delay_ok)

    # -- the delay must be visible: lag leaks energy at first order ------------
    lag_numbers: dict[str, dict] = {}
    lag_ok = True
    for name, payload in study["chains"].items():
        lag = payload["arms"]["lag_endpoint"]["levels"]
        cn = payload["arms"]["centered_endpoint"]["levels"]
        ratios = [lv["energy_defect"] / max(cn_lv["energy_defect"], 1e-300)
                  for lv, cn_lv in zip(lag, cn)]
        lag_numbers[name] = {"lag_over_centered": ratios,
                             "worst_lag_defect": max(lv["energy_defect"] for lv in lag)}
        lag_ok &= min(ratios) >= 10.0
    add("lag_energy_leak_visible",
        "the lagging hold must leak interface energy at first order: >= 10x the "
        "centred operator at every step",
        "lag/centered energy defect >= 10x at every dt, all chains",
        lag_numbers, lag_ok)

    # -- de-stamping keeps the second order in energy --------------------------
    destamp_numbers: dict[str, dict] = {}
    destamp_ok = True
    for name, payload in study["chains"].items():
        raw = payload["arms"]["centered_endstamp"]
        fixed = payload["arms"]["centered_destamp"]
        gain = (raw["levels"][0]["energy_defect"]
                / max(fixed["levels"][0]["energy_defect"], 1e-300))
        order, r2 = _fit([lv["energy_defect"] for lv in fixed["levels"]],
                         [lv["dt"] for lv in fixed["levels"]])
        destamp_numbers[name] = {"gain_at_finest_dt": gain, "order": order, "r2": r2,
                                 "finest": fixed["levels"][0]["energy_defect"]}
        destamp_ok &= gain >= 10.0 and 1.8 <= order <= 2.2
    add("destamp_restores_energy_order",
        "de-stamping the window average must cut the energy defect >= 10x and "
        "restore the second order",
        "gain >= 10x and 1.8 <= order <= 2.2 on all three chains",
        destamp_numbers, destamp_ok)

    # -- predictor-corrector must reproduce the idealised centred operator ----
    pc_numbers: dict[str, float] = {}
    pc_ok = True
    for name, payload in study["chains"].items():
        cn = payload["arms"]["centered_endpoint"]["levels"][0]
        pc = payload["arms"]["predictor_corrector_endpoint"]["levels"][0]
        rel = abs(pc["interface_energy"] - cn["interface_energy"]) / max(
            abs(cn["interface_energy"]), 1e-300)
        pc_numbers[name] = rel
        pc_ok &= rel <= 1e-12
    add("predictor_corrector_equals_centered_energy",
        "the causal operator must reproduce the idealised centred interface "
        "energy after its deferred correction",
        "relative difference <= 1e-12 on all three chains",
        pc_numbers, pc_ok)

    # -- the instrument's own cross-check: energy balance closure -------------
    closure: dict[str, float] = {}
    for name, payload in study["chains"].items():
        closure[name] = max(ladder["balance_residual"]
                            for ladder in payload["arms"].values())
    add("energy_balance_closes",
        "E(T) - E(0) - W_applied + D must close for every arm: the stored energy "
        "and the exchange power are two readings of the same trajectory",
        "max relative balance residual <= 1e-8 over all arms and chains",
        closure, all(v <= 1e-8 for v in closure.values()))

    # -- the closure is quadrature-limited, not a leak -------------------------
    quad_numbers: dict[str, dict] = {}
    quad_ok = True
    for name, rows in study["quadrature"].items():
        balances = [row["balance_residual"] for row in rows]
        defects = [row["energy_defect"] for row in rows]
        r1 = balances[0] / max(balances[1], 1e-300)
        r2 = balances[1] / max(balances[2], 1e-300)
        spread = (max(defects) - min(defects)) / max(max(defects), 1e-300)
        # A chain whose closure is already at the roundoff floor cannot show a
        # quadrature reduction; that is reported as "roundoff_floor", not as a
        # pass by omission.
        if r1 >= 8.0 and r2 >= 8.0:
            regime = "quadrature_limited"
        elif max(balances) <= 1e-11:
            regime = "roundoff_floor"
        else:
            regime = "unexplained"
        quad_numbers[name] = {"balances": balances, "reduction_4_to_8": r1,
                              "reduction_8_to_16": r2, "defect_spread": spread,
                              "regime": regime}
        quad_ok &= regime != "unexplained" and spread <= 1e-4
    add("balance_is_quadrature_limited",
        "refining the sub-grid must shrink the balance residual (>= 8x per "
        "doubling) or already sit on the roundoff floor, while leaving the "
        "energy defect unchanged: the closure error belongs to the quadrature "
        "or to roundoff, never to the operator",
        "regime in {quadrature_limited, roundoff_floor} and defect spread <= 1e-4",
        quad_numbers, quad_ok)

    # -- energy and flux are different functionals, same verdict ---------------
    cross_numbers: dict[str, dict] = {}
    cross_ok = True
    for name, row in study["cross_check"].items():
        flux_gain = row["flux_defect"]["lag"] / max(row["flux_defect"]["centered"], 1e-300)
        energy_gain = (row["energy_defect"]["lag"]
                       / max(row["energy_defect"]["centered"], 1e-300))
        cross_numbers[name] = {"flux_gain": flux_gain, "energy_gain": energy_gain,
                               "flux_over_energy": row["flux_over_energy"]}
        cross_ok &= flux_gain >= 3.0 and energy_gain >= 50.0
    add("energy_and_flux_agree_on_verdict",
        "the two functionals are numerically distinct (flux/energy ratio 2-4x) "
        "yet both separate the lagging from the time-centred operator",
        "flux gain >= 3x and energy gain >= 50x on all three chains",
        cross_numbers, cross_ok)

    # -- interval completeness in energy --------------------------------------
    complete = study["booking"]["complete"]
    attribution_ok = all(row["unattributed_share"] <= 1e-12
                         and row["attribution_identity_residual"] <= 1e-12
                         for row in complete)
    add("energy_attribution_complete",
        "complete sub-window booking must leave zero unattributed energy and "
        "close every interval exactly once",
        "unattributed <= 1e-12 and identity residual <= 1e-12 for r in 2,4,8,16",
        [{k: row[k] for k in ("ratio", "unattributed_share",
                              "attribution_identity_residual", "closed_intervals",
                              "intervals")} for row in complete],
        attribution_ok)

    tail_rows = {row["ratio"]: row for row in study["booking"]["tail"]}
    band_ok = (0.85 <= tail_rows.get(8, {}).get("unattributed_share", 0.0) <= 0.90
               and 0.92 <= tail_rows.get(16, {}).get("unattributed_share", 0.0) <= 0.95)
    add("energy_attribution_tail_loss_band",
        "tail-only booking loses (r-1)/r of the exchanged energy: the 87-94 % "
        "band of the coupling contract, in the conserved quantity",
        "0.85 <= loss(r=8) <= 0.90 and 0.92 <= loss(r=16) <= 0.95",
        {str(r): {"unattributed": row["unattributed_share"],
                  "predicted": row["predicted_loss"],
                  "deficit": row["deficit_share"]}
         for r, row in tail_rows.items()}, band_ok)

    # -- the energy defect must also converge on the weld (the reported chain) -
    weld = study["chains"]["weld"]["arms"]["centered_endpoint"]
    weld_order, weld_r2 = _fit([lv["energy_defect"] for lv in weld["levels"]],
                               [lv["dt"] for lv in weld["levels"]])
    weld_finest = weld["levels"][0]["energy_defect"]
    add("weld_energy_staircase_second_order",
        "halving dt on the weld chain must drive the energy defect down as "
        "O(dt^2), well below the reported 1.2e-3 / 3e-4 staircase levels",
        "order in [1.8, 2.2], r2 >= 0.99 and finest defect <= 3e-4",
        {"order": weld_order, "r2": weld_r2, "finest_defect": weld_finest,
         "reported_flux_plateau": 1.2e-3, "weld_target": 3e-4},
        1.8 <= weld_order <= 2.2 and weld_r2 >= 0.99 and weld_finest <= 3e-4)

    passed = sum(1 for c in criteria if c["verdict"] == "pass")
    return {"criteria": criteria, "passed": passed, "total": len(criteria),
            "verdict": "pass" if passed == len(criteria) else "fail"}


def format_table(verdict: dict) -> str:
    rows = ["| criterion | required | measured (abridged) | verdict |",
            "| --- | --- | --- | --- |"]
    for c in verdict["criteria"]:
        measured = str(c["measured"])
        if len(measured) > 150:
            measured = measured[:147] + "..."
        rows.append(f"| {c['key']} | {c['required']} | {measured} | {c['verdict']} |")
    rows.append("")
    rows.append(f"**Overall verdict: {verdict['verdict'].upper()}** "
                f"({verdict['passed']}/{verdict['total']} criteria).")
    return "\n".join(rows)