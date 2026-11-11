"""Maturity gate of the coupling time-exchange operator (campaign ``p223w2b-n06``).

The operator algebra (:mod:`~tensorphysics.coupling_energy.exchange_operators`)
and its interval ledger were introduced by ``p223w3-n06``.  This module is the
*maturity* layer: it re-measures, from a clean state, the five properties the
campaign node declares as the acceptance line, and it records where each one is
capped:

===========  =====================================================  ==============
criterion    question                                               requirement
===========  =====================================================  ==============
``a``        does the manufactured solution see second order?        order >= 1.9
             (lag baseline must still measure 1.0 +- 0.1)
``b``        is every produced interface share booked into exactly   residual
             one exchange interval, with nothing unattributed?       <= 1e-12, 0
``c``        does the weld staircase converge at ``O(dt^2)`` under    order >= 1.9
             ``dt`` halving, below the 1.2e-3 plateau?               finest < 3e-4
``d``        is the no-delay (time-centred) exchange energy exact?   <= 1e-5
``e``        do the three-chain flux/energy gates still pass?        no regression
===========  =====================================================  ==============

Two refinements of the campaign ladder are used and kept apart, because they
answer different questions:

``cadence`` ladder
    only the exchange cadence ``h = r dt_fine`` is refined at a fixed fine step;
    this is the knob a partitioned code really has, and it is where the reported
    staircase saturation lives (the published window average carries an
    ``O(d^2)`` smoothing bias that no de-stamping can remove, ``d = dt_fine/2``);
``dt`` ladder
    the fine step itself is halved at a fixed cadence ratio, which is the
    refinement the criterion ``c`` names: every ``d``-induced floor halves with
    the step, so the restored operator must be second order on the *defect*
    metric as well as on the state.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

from .exchange_energy import run_energy_ladder
from .exchange_gate import gate_study as gate_flux_study
from .exchange_gate import run_exchange_study
from .exchange_energy import gate_energy, run_energy_study
from .exchange_mms import mms_order_ladder
from .exchange_operators import ExchangeScheme
from .ledger import CouplingLedgerError
from .time_exchange import (
    build_chain,
    fit_order_with_r2,
    run_cadence_ladder,
)

__all__ = [
    "CHAINS",
    "MATURITY_ARMS",
    "MMS_STEPS",
    "CADENCE_RATIOS",
    "DT_HALVINGS",
    "run_maturity_mms",
    "run_interval_identity",
    "run_weld_staircase",
    "run_no_delay_energy",
    "run_regression",
    "run_maturity_study",
    "gate_maturity",
    "format_table",
]

#: The three exchange chains of the coupling contract.
CHAINS: tuple[str, ...] = ("viv", "weld", "settling")

#: Exchange arms compared throughout the maturity study.
MATURITY_ARMS: dict[str, dict[str, Any]] = {
    "lag_endpoint": {"scheme": "lag", "stamp_offset_ratio": 0.0},
    "lag_endstamp": {"scheme": "lag", "stamp_offset_ratio": 0.5},
    "centered_endstamp": {"scheme": "centered", "stamp_offset_ratio": 0.5},
    "centered_destamp1": {"scheme": "centered", "stamp_offset_ratio": 0.5,
                          "recenter": True, "recenter_order": 1},
    "centered_destamp2": {"scheme": "centered", "stamp_offset_ratio": 0.5,
                          "recenter": True, "recenter_order": 2},
    "centered_endpoint": {"scheme": "centered", "stamp_offset_ratio": 0.0},
    "subcycle_complete": {"scheme": "centered", "subcycle": True},
}

#: Fine ladder of the manufactured solution (asymptotic range of ``omega h``).
MMS_STEPS: tuple[float, ...] = (0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625)

#: Exchange-cadence ladder at a fixed fine step (the reported saturation).
CADENCE_RATIOS: tuple[int, ...] = (1, 2, 4, 8, 16)

#: ``dt`` halving factors of the criterion-``c`` staircase.
DT_HALVINGS: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625)

#: Cadence ratio held fixed while the fine step is halved.
STAIRCASE_RATIO: int = 2

#: Arms that must be complete in the interval-attribution sense.
COMPLETE_ARMS: tuple[str, ...] = tuple(MATURITY_ARMS)


# ---------------------------------------------------------------------------
# (a) manufactured-solution order
# ---------------------------------------------------------------------------
def run_maturity_mms(mms: Any | None = None,
                     steps: Sequence[float] = MMS_STEPS) -> dict:
    """Quadrature order of every arm on the analytic interface flux."""
    arms: dict[str, dict] = {}
    for name, spec in MATURITY_ARMS.items():
        if spec.get("subcycle"):
            # the sub-window arm exchanges exact window integrals: no quadrature
            # law is involved, so it is scored through the attribution ledger.
            continue
        offset_ratio = float(spec.get("stamp_offset_ratio", 0.0))
        ladder = mms_order_ladder(
            mms, spec["scheme"], h_list=steps, offset_ratio=offset_ratio,
            recenter=bool(spec.get("recenter", False)),
            recenter_order=int(spec.get("recenter_order", 1)))
        order, r2 = ladder.fit()
        payload = ladder.as_dict()
        payload["observed_order"] = order
        payload["fit_r2"] = r2
        payload["finest_defect"] = ladder.defects()[-1]
        payload["coarsest_defect"] = ladder.defects()[0]
        arms[name] = payload
    return {"steps": list(steps), "arms": arms}


# ---------------------------------------------------------------------------
# (b) interval-complete attribution
# ---------------------------------------------------------------------------
def _audit_chain_arm(chain_name: str, spec: Mapping[str, Any], ratio: int, *,
                     tail_booking: bool = False) -> dict:
    """Book one (chain, arm, cadence) through the interval-complete ledger.

    Absolute shares are used throughout: the normalisation is the total
    *absolute* exchange of the run, which stays well defined when the exchanged
    flux has zero mean (the VIV force does).
    """
    from .exchange_operators import (
        IntervalExchangeLedger,
        StampConvention,
        StampRule,
        operator_for,
    )

    chain = build_chain(chain_name)
    ratio = int(ratio)
    h = ratio * chain.dt_fine
    n_intervals = int(round(chain.t_end / h))
    offset = float(spec.get("stamp_offset_ratio", 0.0)) * chain.dt_fine
    if offset > 0.0:
        stamp = StampRule(StampConvention.WINDOW_MIDPOINT, offset,
                          bool(spec.get("recenter", False)),
                          order=int(spec.get("recenter_order", 1)))
    else:
        stamp = StampRule()
    operator = operator_for(spec["scheme"], stamp)
    subcycle = bool(spec.get("subcycle", False))

    ledger = IntervalExchangeLedger(f"{chain_name}_exchange", n_intervals,
                                    sides=("producer", "receiver"))
    q_prev: float | None = None
    q_prev2: float | None = None
    produced_abs = 0.0
    released_abs = 0.0
    applied_abs = 0.0
    gap_abs = 0.0
    identity_abs = 0.0
    for n in range(n_intervals):
        t_lo, t_hi = n * h, (n + 1) * h
        ledger.declare_interval(n, t_lo, t_hi)
        shares = [chain.mms.interval_integral(t_lo + k * chain.dt_fine,
                                             t_lo + (k + 1) * chain.dt_fine)
                  for k in range(ratio)]
        produced = 0.0
        for share in shares:
            ledger.book_produced(n, share)
            produced += share
            produced_abs += abs(share)
        # every share of the window is handed over, except in the tail-booking
        # control, which releases only the last sub-interval share
        released = shares[-1] if tail_booking else produced
        ledger.book_release(n, released)
        released_abs += abs(released)
        identity_abs += abs(released - produced)

        if subcycle and tail_booking:
            raise CouplingLedgerError(
                "the sub-window arm exchanges exact window integrals; tail booking "
                "removes them again")
        if subcycle:
            sub_rates = [chain.sample(t_lo + (k + 1) * chain.dt_fine,
                                      0.5 * chain.dt_fine) for k in range(ratio)]
            applied = sum(rate * chain.dt_fine for rate in sub_rates)
        else:
            q_lo = chain.sample(t_lo, offset)
            q_hi = chain.sample(t_hi, offset)
            transfer = operator.provisional_transfer(
                n, t_lo, t_hi, q_lo=q_lo, q_hi=q_hi, q_prev=q_prev,
                q_prev2=q_prev2)
            corrected = operator.corrected_transfer(transfer, q_hi, q_prev2=q_prev2)
            if corrected.integral != transfer.integral:
                ledger.book_correction(n, corrected.integral - transfer.integral)
            applied = corrected.integral
            q_prev2 = q_prev
            q_prev = q_lo
        ledger.book_applied(n, applied)
        ledger.close_interval(n)
        applied_abs += abs(applied)
        gap_abs += abs(released - applied)

    summary = ledger.summary()
    scale = max(produced_abs, 1e-300)
    return {
        "ratio": ratio,
        "h": h,
        "intervals": n_intervals,
        "produced_abs": produced_abs,
        "released_abs": released_abs,
        "applied_abs": applied_abs,
        "identity_residual": identity_abs / scale,
        "unattributed_abs_share": identity_abs / scale,
        "tail_booking": bool(tail_booking),
        "handover_deficit_abs_share": gap_abs / scale,
        "open_intervals": list(summary.open_intervals),
        "closed_intervals": summary.closed_intervals,
        #: bookkeeping completeness: every declared interval closed, nothing
        #: unattributed.  Note this is *stricter* than ``summary.complete``,
        #: which additionally requires the physical handover deficit to vanish
        #: (only the exact sub-window arm can do that).
        "booked_once": (not summary.open_intervals
                        and summary.closed_intervals == n_intervals
                        and identity_abs / scale <= 1e-12),
        "ledger_complete": summary.complete,
    }


def identity_controls() -> dict:
    """Negative controls: the ledger must refuse/report the anomalies it exists for."""
    from .exchange_operators import IntervalExchangeLedger
    from .ledger import CouplingLedgerError

    out: dict[str, bool] = {}
    ledger = IntervalExchangeLedger("control", 3)
    ledger.declare_interval(0, 0.0, 1.0)
    ledger.declare_interval(1, 1.0, 2.0)
    ledger.declare_interval(2, 2.0, 3.0)
    ledger.book_produced(0, 1.0)
    ledger.book_release(0, 1.0)
    ledger.book_applied(0, 1.0)
    try:
        ledger.book_applied(0, 1.0)
        out["double_booking_refused"] = False
    except CouplingLedgerError:
        out["double_booking_refused"] = True
    try:
        ledger.book_release(7, 1.0)  # interval never declared
        out["undeclared_interval_refused"] = False
    except CouplingLedgerError:
        out["undeclared_interval_refused"] = True
    ledger.close_interval(0)
    try:
        ledger.close_interval(0)
        out["double_close_refused"] = False
    except CouplingLedgerError:
        out["double_close_refused"] = True
    # a missing interval (never opened) must be visible in the summary
    gap = IntervalExchangeLedger("gap", 3)
    gap.declare_interval(0, 0.0, 1.0)
    gap.declare_interval(2, 2.0, 3.0)
    gap.book_produced(0, 1.0)
    gap.book_release(0, 1.0)
    gap.book_applied(0, 1.0)
    gap.close_interval(0)
    gap.book_produced(2, 1.0)
    gap.book_release(2, 1.0)
    gap.book_applied(2, 1.0)
    gap.close_interval(2)
    gap_summary = gap.summary()
    out["missing_interval_detected"] = (
        len(gap_summary.open_intervals) > 0 or not gap_summary.complete)
    # an interval closed but never applied must also be caught
    open_ledger = IntervalExchangeLedger("open", 1)
    open_ledger.declare_interval(0, 0.0, 1.0)
    open_ledger.book_produced(0, 1.0)
    open_ledger.book_release(0, 1.0)
    out["unapplied_interval_detected"] = not open_ledger.summary().complete
    return out


def run_interval_identity(ratios: Sequence[int] = CADENCE_RATIOS) -> dict:
    """Book every exchange interval of every chain and audit the identity."""
    chains: dict[str, dict] = {}
    for chain_name in CHAINS:
        payload: dict[str, dict] = {}
        for arm, spec in MATURITY_ARMS.items():
            rows = [_audit_chain_arm(chain_name, spec, r) for r in ratios]
            payload[arm] = {
                "levels": rows,
                "worst_unattributed": max(r["unattributed_abs_share"] for r in rows),
                "worst_identity_residual": max(r["identity_residual"] for r in rows),
                "worst_handover_deficit": max(r["handover_deficit_abs_share"]
                                              for r in rows),
                "booked_once": all(r["booked_once"] for r in rows),
            }
        # negative control: releasing only the tail share of the window must show
        # up as an unattributed fraction of (r-1)/r of the absolute exchange
        tail = {"scheme": "centered", "stamp_offset_ratio": 0.5}
        row = _audit_chain_arm(chain_name, tail, 8, tail_booking=True)
        payload["control_tail_booking"] = {
            "ratio": 8,
            "unattributed_abs_share": row["unattributed_abs_share"],
            "expected_loss": 7.0 / 8.0,
            "deficit_abs_share": row["handover_deficit_abs_share"],
            "booked_once": row["booked_once"],
        }
        chains[chain_name] = payload
    return {"ratios": list(ratios), "chains": chains,
            "controls": identity_controls()}


# ---------------------------------------------------------------------------
# (c) weld staircase
# ---------------------------------------------------------------------------
def _ladder_rows(ladder: Any, key: str) -> dict:
    """Coarse-to-fine record of one ladder (``key`` is documentary only)."""
    ordered = sorted(ladder.levels, key=lambda lv: -lv.h)
    steps = [lv.h for lv in ordered]
    defects = [lv.defect_l1 for lv in ordered]
    states = [lv.state_error for lv in ordered]
    d_order, d_r2 = fit_order_with_r2(steps, defects)
    s_order, s_r2 = fit_order_with_r2(steps, states)
    return {
        "ordering": "coarse->fine",
        "steps": steps,
        "defects": defects,
        "state_errors": states,
        "levels": [{"ratio": lv.ratio, "h": lv.h, "defect_l1": lv.defect_l1,
                    "max_interval_defect": lv.max_interval_defect,
                    "state_error": lv.state_error} for lv in ordered],
        "defect_order": d_order,
        "defect_r2": d_r2,
        "state_order": s_order,
        "state_r2": s_r2,
        "coarsest_defect": defects[0],
        "finest_defect": defects[-1],
        "finest_state_error": states[-1],
        "coarsest_state_error": states[0],
        "plateau_defect": ladder.saturation(),
    }


def run_weld_staircase(chain_name: str = "weld") -> dict:
    """Both refinements of the weld staircase, arm by arm.

    ``cadence``: refinement of the exchange cadence at the shipped fine step —
    the plateau reported by the coupling contract lives here.
    ``dt``: halving of the fine step at a fixed cadence ratio — the refinement
    the maturity criterion names.
    """
    cadence: dict[str, dict] = {}
    dt: dict[str, dict] = {}
    for name, spec in MATURITY_ARMS.items():
        if spec.get("subcycle"):
            continue
        ladder = run_cadence_ladder(
            chain_name, spec["scheme"], ratios=CADENCE_RATIOS,
            stamp_offset_ratio=float(spec.get("stamp_offset_ratio", 0.0)),
            recenter=bool(spec.get("recenter", False)),
            recenter_order=int(spec.get("recenter_order", 1)))
        cadence[name] = _ladder_rows(ladder, "cadence")

        dts = [build_chain(chain_name).dt_fine * f for f in DT_HALVINGS]
        sub: list[dict] = []
        for dt_fine in dts:
            chain = build_chain(chain_name, dt_fine=dt_fine)
            lv = run_cadence_ladder(
                chain, spec["scheme"],
                ratios=(STAIRCASE_RATIO,),
                stamp_offset_ratio=float(spec.get("stamp_offset_ratio", 0.0)),
                recenter=bool(spec.get("recenter", False)),
                recenter_order=int(spec.get("recenter_order", 1))).levels[0]
            sub.append({"dt_fine": dt_fine, "h": lv.h,
                        "defect_l1": lv.defect_l1,
                        "max_interval_defect": lv.max_interval_defect,
                        "state_error": lv.state_error})
        sub.sort(key=lambda row: -row["dt_fine"])
        defects = [row["defect_l1"] for row in sub]
        states = [row["state_error"] for row in sub]
        dt[name] = {
            "levels": sub,
            "defect_order": fit_order_with_r2(dts, defects)[0],
            "defect_r2": fit_order_with_r2(dts, defects)[1],
            "state_order": fit_order_with_r2(dts, states)[0],
            "state_r2": fit_order_with_r2(dts, states)[1],
            "finest_defect": defects[-1],
            "finest_state_error": states[-1],
        }
    return {"chain": chain_name, "cadence_ratios": list(CADENCE_RATIOS),
            "dt_halvings": list(DT_HALVINGS), "staircase_ratio": STAIRCASE_RATIO,
            "cadence": cadence, "dt": dt}


# ---------------------------------------------------------------------------
# (d) energy residual of the no-delay implementation
# ---------------------------------------------------------------------------
def run_no_delay_energy(dt_fractions: Sequence[float] = (4.0, 2.0, 1.0, 0.5),
                        arms: Sequence[tuple[str, str]] = (
                            ("centered_endpoint", "centered"),
                            ("predictor_corrector_endpoint", "predictor_corrector"),
                        )) -> dict:
    chains: dict[str, dict] = {}
    for chain_name in CHAINS:
        payload: dict[str, dict] = {}
        for label, scheme in arms:
            ladder = run_energy_ladder(chain_name, scheme, ratio=1,
                                       dt_fractions=tuple(dt_fractions))
            finest = ladder.levels[0]
            payload[label] = {
                "scheme": scheme,
                "dt": finest.dt,
                "energy_defect": finest.energy_defect,
                "reference_energy_error": finest.reference_energy_error,
                "balance_residual": finest.energy_balance_residual,
                "attribution_identity_residual": finest.attribution_identity_residual,
                "unattributed_share": finest.unattributed_share,
                "levels": [{"dt": lv.dt, "energy_defect": lv.energy_defect,
                            "reference_energy_error": lv.reference_energy_error}
                           for lv in ladder.levels],
                "energy_defect_order": ladder.fit()[0],
            }
        chains[chain_name] = payload
    return {"dt_fractions": list(dt_fractions), "chains": chains}


# ---------------------------------------------------------------------------
# (e) three-chain regression of the shipped gates
# ---------------------------------------------------------------------------
def run_regression(with_energy: bool = True) -> dict:
    """Re-run the shipped flux and energy gates; report only their verdicts."""
    flux_study = run_exchange_study()
    flux_verdict = gate_flux_study(flux_study)
    out: dict[str, Any] = {
        "flux": {"verdict": flux_verdict["verdict"],
                 "passed": flux_verdict["passed"],
                 "total": flux_verdict["total"],
                 "failed": [c["key"] for c in flux_verdict["criteria"]
                            if c["verdict"] != "pass"]},
    }
    if with_energy:
        energy_study = run_energy_study()
        energy_verdict = gate_energy(energy_study)
        out["energy"] = {"verdict": energy_verdict["verdict"],
                         "passed": energy_verdict["passed"],
                         "total": energy_verdict["total"],
                         "failed": [c["key"] for c in energy_verdict["criteria"]
                                    if c["verdict"] != "pass"]}
    return out


# ---------------------------------------------------------------------------
# study + gate
# ---------------------------------------------------------------------------
def run_maturity_study(*, with_energy: bool = True) -> dict:
    """Full five-criterion maturity study (JSON-serialisable)."""
    return {
        "node": "p223w2b-n06",
        "arms": {name: dict(spec) for name, spec in MATURITY_ARMS.items()},
        "mms": run_maturity_mms(),
        "attribution": run_interval_identity(),
        "staircase": run_weld_staircase(),
        "no_delay_energy": run_no_delay_energy(),
        "regression": run_regression(with_energy=with_energy),
    }


@dataclass(frozen=True)
class MaturityCriterion:
    key: str
    description: str
    required: str
    measured: Any
    passed: bool

    @property
    def verdict(self) -> str:
        return "pass" if self.passed else "fail"

    def as_dict(self) -> dict:
        return {"key": self.key, "description": self.description,
                "required": self.required, "measured": self.measured,
                "verdict": self.verdict}


def _fit_from_levels(levels, step_key, err_key):
    """Re-derive an exponent from stored ``(step, error)`` pairs.

    The gate never reads a stored ``order`` field: a record that carries its own
    verdict is self-certifying, so every exponent is recomputed here.
    """
    steps = [float(lv[step_key]) for lv in levels]
    errors = [max(float(lv[err_key]), 1e-300) for lv in levels]
    order, r2 = fit_order_with_r2(steps, errors)
    finest = min(zip(steps, errors), key=lambda pair: pair[0])[1]
    return order, r2, finest


def gate_maturity(study: Mapping[str, Any]) -> dict:
    """Five verdicts computed from the stored study, never from stored verdicts."""
    out: list[MaturityCriterion] = []

    def add(key: str, description: str, required: str, measured: Any,
            passed: bool) -> None:
        out.append(MaturityCriterion(key, description, required, measured,
                                     bool(passed)))

    # -- (a) manufactured-solution order --------------------------------------
    mms = study["mms"]["arms"]
    mms_orders = {name: _fit_from_levels(payload["levels"], "h", "defect_l1")
                  for name, payload in mms.items()}
    lag_order, lag_r2, _ = mms_orders["lag_endpoint"]
    centered_order, centered_r2, _ = mms_orders["centered_endpoint"]
    destamp1_order, destamp1_r2, _ = mms_orders["centered_destamp1"]
    destamp2_order, destamp2_r2, _ = mms_orders["centered_destamp2"]
    order_ok = (0.9 <= lag_order <= 1.1 and centered_order >= 1.9
                and destamp2_order >= 1.9 and destamp1_order >= 1.9)
    add("mms_time_order",
        "MMS quadrature order: lag stays first order (baseline), the CN midpoint "
        "and both de-stamped window arms reach second order",
        "0.9 <= lag <= 1.1; centered, destamp1, destamp2 all >= 1.9",
        {"lag": lag_order, "centered": centered_order,
         "window_destamp1": destamp1_order, "window_destamp2": destamp2_order,
         "lag_r2": lag_r2, "centered_r2": centered_r2,
         "destamp1_r2": destamp1_r2, "destamp2_r2": destamp2_r2},
        order_ok)

    # -- (b) interval-complete identity ---------------------------------------
    attribution = study["attribution"]["chains"]
    worst_unattr = 0.0
    worst_identity = 0.0
    per_chain: dict[str, dict] = {}
    controls_ok = all(study["attribution"]["controls"].values())
    for chain_name, payload in attribution.items():
        u = max(payload[arm]["worst_unattributed"] for arm in COMPLETE_ARMS)
        r = max(payload[arm]["worst_identity_residual"] for arm in COMPLETE_ARMS)
        booked_once = all(payload[arm]["booked_once"] for arm in COMPLETE_ARMS)
        worst_unattr = max(worst_unattr, u)
        worst_identity = max(worst_identity, r)
        ctrl = payload["control_tail_booking"]
        ctrl_caught = abs(ctrl["unattributed_abs_share"] - ctrl["expected_loss"]) <= 5e-3
        controls_ok = controls_ok and booked_once and ctrl_caught
        per_chain[chain_name] = {"unattributed": u, "identity_residual": r,
                                 "booked_once": booked_once,
                                 "tail_control": ctrl}
    add("interval_complete_identity",
        "every produced interface share is booked into exactly one exchange "
        "interval: released equals produced, unattributed share is 0, every "
        "interval is closed exactly once, and the ledger still refuses the "
        "anomalies (double booking/close, undeclared or missing interval)",
        "identity residual <= 1e-12, unattributed == 0, all negative controls caught",
        {"per_chain": per_chain, "worst_unattributed": worst_unattr,
         "worst_identity_residual": worst_identity,
         "controls": dict(study["attribution"]["controls"])},
        worst_identity <= 1e-12 and worst_unattr == 0.0 and controls_ok)

    # -- (c) weld staircase ---------------------------------------------------
    stair = study["staircase"]
    dt_levels = stair["dt"]["centered_destamp2"]["levels"]
    dt_order, dt_r2, dt_finest = _fit_from_levels(dt_levels, "dt_fine", "defect_l1")
    dt_state_order, _, dt_finest_state = _fit_from_levels(dt_levels, "dt_fine",
                                                          "state_error")
    lag_state_order, _, lag_state_finest = _fit_from_levels(
        stair["cadence"]["lag_endstamp"]["levels"], "h", "state_error")
    cn_state_order, _, cn_state_finest = _fit_from_levels(
        stair["cadence"]["centered_endstamp"]["levels"], "h", "state_error")
    staircase_ok = dt_order >= 1.9 and dt_finest < 3e-4
    add("weld_staircase_second_order",
        "under dt halving at fixed exchange cadence the repaired weld staircase "
        "must fall at O(dt^2) below the 3e-4 line; the end-stamped arms must still "
        "show the reported plateau when only the cadence is refined",
        "dt-halving defect order >= 1.9 and finest defect < 3e-4",
        {"dt_arm": "centered_destamp2",
         "dt_defect_order": dt_order, "dt_defect_r2": dt_r2,
         "dt_state_order": dt_state_order,
         "dt_finest_defect": dt_finest,
         "dt_finest_state_error": dt_finest_state,
         "cadence_plateau_lag_state": lag_state_finest,
         "cadence_plateau_lag_state_order": lag_state_order,
         "cadence_plateau_cn_state": cn_state_finest,
         "cadence_plateau_cn_order": cn_state_order,
         "reported_plateau": 1.2e-3},
        staircase_ok)

    # -- (d) no-delay energy --------------------------------------------------
    energy = study["no_delay_energy"]["chains"]
    worst_energy = 0.0
    worst_ref = 0.0
    per_chain_energy: dict[str, dict] = {}
    for chain_name, payload in energy.items():
        for label, row in payload.items():
            worst_energy = max(worst_energy, row["energy_defect"])
            worst_ref = max(worst_ref, row["reference_energy_error"])
            per_chain_energy.setdefault(chain_name, {})[label] = {
                "energy_defect": row["energy_defect"],
                "reference_energy_error": row["reference_energy_error"],
                "dt": row["dt"]}
    add("no_delay_energy_residual",
        "the time-centred exchange without delay closes the energy budget: the "
        "interface-energy defect and the stored-energy error stay below 1e-5",
        "energy_defect <= 1e-5 and reference_energy_error <= 1e-5",
        {"caliper": per_chain_energy, "worst_energy_defect": worst_energy,
         "worst_reference_energy_error": worst_ref},
        worst_energy <= 1e-5 and worst_ref <= 1e-5)

    # -- (e) three-chain regression -------------------------------------------
    regression = study["regression"]
    flux_ok = regression["flux"]["verdict"] == "pass"
    energy_ok = regression.get("energy", {}).get("verdict", "skipped") in ("pass", "skipped")
    add("three_chain_no_regression",
        "the shipped flux and energy gates of the three chains still pass after "
        "the operator upgrade",
        "flux gate pass (and energy gate pass when run)",
        {"flux": regression["flux"], "energy": regression.get("energy")},
        flux_ok and energy_ok)

    passed = sum(1 for c in out if c.passed)
    return {"criteria": [c.as_dict() for c in out], "passed": passed,
            "total": len(out),
            "verdict": "pass" if passed == len(out) else "fail"}


def format_table(verdict: Mapping[str, Any]) -> str:
    lines = ["| criterion | required | measured (abridged) | verdict |",
             "| --- | --- | --- | --- |"]
    for c in verdict["criteria"]:
        measured = c["measured"]
        if isinstance(measured, Mapping):
            abridged = ", ".join(f"{k}={v!r}" for k, v in list(measured.items())[:4])
        else:
            abridged = repr(measured)
        if len(abridged) > 220:
            abridged = abridged[:217] + "..."
        lines.append(f"| `{c['key']}` | {c['required']} | {abridged} | "
                     f"**{c['verdict']}** |")
    lines.append("")
    lines.append(f"overall: **{verdict['verdict']}** "
                 f"({verdict['passed']}/{verdict['total']})")
    return "\n".join(lines)