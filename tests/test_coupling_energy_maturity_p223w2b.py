"""Acceptance tests of the coupling-operator maturity node ``p223w2b-n06``.

Five properties are guarded here, one test per criterion plus the unit tests of
the algebra upgrade that makes criterion (c) reachable:

* (a) the manufactured solution sees second order and the lag baseline stays
  first order;
* (b) the interval-complete ledger attributes every produced share exactly once,
  with unattributed share 0 and the anomaly controls still refused;
* (c) the weld staircase falls at ``O(dt^2)`` under ``dt`` halving;
* (d) the no-delay exchange keeps the interface-energy defect below ``1e-5``;
* (e) the stored study of the node, when present, re-gates to ``pass``.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from tensorphysics.coupling_energy.exchange_energy import run_energy_ladder
from tensorphysics.coupling_energy.exchange_mms import mms_order_ladder
from tensorphysics.coupling_energy.exchange_operators import (
    StampConvention,
    StampRule,
    operator_for,
)
from tensorphysics.coupling_energy.ledger import CouplingLedgerError
from tensorphysics.coupling_energy.maturity import (
    CADENCE_RATIOS,
    DT_HALVINGS,
    MMS_STEPS,
    STAIRCASE_RATIO,
    _audit_chain_arm,
    gate_maturity,
    identity_controls,
    run_interval_identity,
    run_maturity_mms,
    run_weld_staircase,
)
from tensorphysics.coupling_energy.time_exchange import (
    build_chain,
    fit_order_with_r2,
    run_cadence_ladder,
)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "p223w2b_n06" / "maturity.json"


# ---------------------------------------------------------------------------
# the algebra upgrade
# ---------------------------------------------------------------------------
def _quadratic(t: float) -> float:
    return 3.0 + 2.0 * t - 1.5 * t * t


def test_destamp_order2_is_exact_on_a_quadratic_stream():
    d, h, t_lo = 0.1, 0.5, 1.0
    q_lo = _quadratic(t_lo - d)
    q_prev = _quadratic(t_lo - h - d)
    q_prev2 = _quadratic(t_lo - 2.0 * h - d)
    rule = StampRule(StampConvention.WINDOW_MIDPOINT, d, recenter=True, order=2)
    est_lo, est_hi = rule.endpoints(q_lo, _quadratic(t_lo + h - d), q_prev, h, q_prev2)
    assert abs(est_lo - _quadratic(t_lo)) <= 1e-13
    assert est_hi is not None
    assert abs(est_hi - _quadratic(t_lo + h)) <= 1e-13
    # the first-order rule is *not* exact on the same stream: it keeps the
    # -(d^2/2) q'' bias that makes the cadence staircase saturate
    first = StampRule(StampConvention.WINDOW_MIDPOINT, d, recenter=True, order=1)
    est_lo_1, _ = first.endpoints(q_lo, None, q_prev, h, q_prev2)
    assert abs(est_lo_1 - _quadratic(t_lo)) > 1e-3


def test_destamp_degrades_openly_without_a_second_history_sample():
    rule = StampRule(StampConvention.WINDOW_MIDPOINT, 0.1, recenter=True, order=2)
    assert rule.effective_order(1.0, 0.5) == 2
    assert rule.effective_order(1.0, None) == 1
    one = StampRule(StampConvention.WINDOW_MIDPOINT, 0.1, recenter=True, order=1)
    assert rule.effective_order(None, None) == one.effective_order(None, None)
    est_deg, _ = rule.endpoints(1.0, None, 0.5, 0.5, None)
    est_one, _ = one.endpoints(1.0, None, 0.5, 0.5, None)
    assert est_deg == pytest.approx(est_one, rel=0.0, abs=0.0)


def test_destamp_rejects_an_unknown_order():
    with pytest.raises(CouplingLedgerError):
        StampRule(StampConvention.WINDOW_MIDPOINT, 0.1, recenter=True, order=3)


def test_predictor_corrector_threads_the_second_history_sample():
    """The deferred correction must de-stamp with the same order as the step."""
    chain = build_chain("weld")
    centered = run_cadence_ladder(chain, "centered", ratios=(4,),
                                 stamp_offset_ratio=0.5, recenter=True,
                                 recenter_order=2).levels[0]
    causal = run_cadence_ladder(chain, "predictor_corrector", ratios=(4,),
                                stamp_offset_ratio=0.5, recenter=True,
                                recenter_order=2).levels[0]
    assert abs(causal.defect_l1 - centered.defect_l1) <= 1e-3 * max(
        centered.defect_l1, 1e-30)
    # and the order-2 correction must actually differ from the order-1 one
    causal1 = run_cadence_ladder(chain, "predictor_corrector", ratios=(4,),
                                 stamp_offset_ratio=0.5, recenter=True,
                                 recenter_order=1).levels[0]
    assert abs(causal1.defect_l1 - centered.defect_l1) > 0.0


# ---------------------------------------------------------------------------
# (a) manufactured-solution order
# ---------------------------------------------------------------------------
def test_mms_lag_first_order_centered_second_order():
    lag = mms_order_ladder(None, "lag", h_list=MMS_STEPS)
    centered = mms_order_ladder(None, "centered", h_list=MMS_STEPS)
    lag_order, lag_r2 = lag.fit()
    cen_order, cen_r2 = centered.fit()
    assert 1.0 - 0.1 <= lag_order <= 1.0 + 0.1
    assert lag_r2 >= 0.99
    assert cen_order >= 1.9
    assert cen_r2 >= 0.99


def test_mms_window_destamp_reaches_second_order():
    raw = mms_order_ladder(None, "centered", h_list=MMS_STEPS,
                           offset_ratio=0.5, recenter=False)
    dst2 = mms_order_ladder(None, "centered", h_list=MMS_STEPS,
                            offset_ratio=0.5, recenter=True, recenter_order=2)
    raw_order, _ = raw.fit()
    dst_order, dst_r2 = dst2.fit()
    assert raw_order < 1.1, "an end-stamped window average must stay first order"
    assert dst_order >= 1.9
    assert dst_r2 >= 0.999


def test_maturity_mms_criterion_pass():
    study = {"mms": run_maturity_mms()}
    arms = study["mms"]["arms"]
    assert arms["lag_endpoint"]["observed_order"] == pytest.approx(0.997, abs=0.15)
    assert arms["centered_endpoint"]["observed_order"] >= 1.9
    assert arms["centered_destamp2"]["observed_order"] >= 1.9


# ---------------------------------------------------------------------------
# (b) interval-complete identity
# ---------------------------------------------------------------------------
def test_interval_identity_is_zero_for_every_chain_and_arm():
    for chain in ("weld", "viv", "settling"):
        for spec in ({"scheme": "lag", "stamp_offset_ratio": 0.0},
                     {"scheme": "centered", "stamp_offset_ratio": 0.5,
                      "recenter": True, "recenter_order": 2},
                     {"scheme": "centered", "subcycle": True}):
            row = _audit_chain_arm(chain, spec, 8)
            assert row["identity_residual"] == 0.0
            assert row["unattributed_abs_share"] == 0.0
            assert row["booked_once"] is True
            assert row["closed_intervals"] == row["intervals"]


def test_tail_booking_control_still_shows_r_minus_one_over_r():
    row = _audit_chain_arm("settling", {"scheme": "centered",
                                        "stamp_offset_ratio": 0.5}, 8,
                           tail_booking=True)
    assert row["unattributed_abs_share"] == pytest.approx(7.0 / 8.0, abs=1e-9)
    assert row["booked_once"] is False


def test_identity_controls_are_all_refused_or_reported():
    controls = identity_controls()
    assert controls
    assert all(controls.values()), controls


def test_interval_identity_study_closes_at_round_off():
    study = run_interval_identity(ratios=(2, 8))
    for chain, payload in study["chains"].items():
        for arm in ("lag_endpoint", "centered_destamp2", "subcycle_complete"):
            assert payload[arm]["worst_identity_residual"] <= 1e-12, (chain, arm)
            assert payload[arm]["worst_unattributed"] == 0.0, (chain, arm)
            assert payload[arm]["booked_once"], (chain, arm)
        ctrl = payload["control_tail_booking"]
        assert ctrl["unattributed_abs_share"] == pytest.approx(7.0 / 8.0, abs=5e-3)


# ---------------------------------------------------------------------------
# (c) weld staircase
# ---------------------------------------------------------------------------
def test_weld_staircase_is_second_order_under_dt_halving():
    defects, states, dts = [], [], []
    for factor in DT_HALVINGS:
        chain = build_chain("weld", dt_fine=2.0e-3 * factor)
        level = run_cadence_ladder(chain, "centered", ratios=(STAIRCASE_RATIO,),
                                   stamp_offset_ratio=0.5, recenter=True,
                                   recenter_order=2).levels[0]
        dts.append(chain.dt_fine)
        defects.append(level.defect_l1)
        states.append(level.state_error)
    order, r2 = fit_order_with_r2(dts, defects)
    assert order >= 1.9, f"defect order {order:.3f} is not second order"
    assert r2 >= 0.99
    assert defects[-1] < 3e-4, f"finest defect {defects[-1]:.3e} above the line"
    state_order, _ = fit_order_with_r2(dts, states)
    assert state_order >= 1.9


def test_endstamped_cadence_ladder_still_plateaus():
    """The defect the node repairs must still be reproducible on ``main``'s arms."""
    ladder = run_cadence_ladder("weld", "centered", ratios=CADENCE_RATIOS,
                                stamp_offset_ratio=0.5, recenter=False)
    order, _ = ladder.fit_state()
    assert abs(order) <= 0.3, f"the end-stamped cadence ladder did not plateau ({order})"
    states = ladder.state_errors()
    assert max(states) / min(states) <= 1.3, "the state error is not flat in cadence"
    plateau = min(states)
    assert 0.6e-3 <= plateau <= 2.4e-3, f"plateau {plateau:.3e} vs reported 1.2e-3"


def test_weld_staircase_study_reports_both_refinements():
    study = run_weld_staircase()
    assert study["dt"]["centered_destamp2"]["defect_order"] >= 1.9
    assert study["dt"]["centered_destamp2"]["finest_defect"] < 3e-4
    assert study["cadence"]["lag_endstamp"]["defect_order"] <= 0.9
    # the order-2 rule exists to halve the *floor* of the cadence ladder, so the
    # comparison is made at the finest cadence (the floor), not on the mean
    finest2 = study["cadence"]["centered_destamp2"]["finest_defect"]
    finest1 = study["cadence"]["centered_destamp1"]["finest_defect"]
    assert finest2 < finest1, (finest2, finest1)
    assert finest2 < 3e-4


# ---------------------------------------------------------------------------
# (d) no-delay energy
# ---------------------------------------------------------------------------
def test_no_delay_energy_residual_below_threshold():
    ladder = run_energy_ladder("weld", "centered", ratio=1,
                               dt_fractions=(2.0, 1.0))
    finest = ladder.levels[0]
    assert finest.energy_defect <= 1e-5
    assert finest.reference_energy_error <= 1e-5
    assert finest.unattributed_share <= 1e-12


# ---------------------------------------------------------------------------
# (e) the stored study of the node
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACTS.exists(),
                    reason="p223w2b-n06 artifacts absent in this checkout")
def test_stored_study_regates_to_pass():
    payload = json.loads(ARTIFACTS.read_text())
    study = {key: payload[key] for key in ("mms", "attribution", "staircase",
                                           "no_delay_energy", "regression")}
    verdict = gate_maturity(study)
    failed = [c["key"] for c in verdict["criteria"] if c["verdict"] != "pass"]
    assert verdict["verdict"] == "pass", failed
    assert verdict["passed"] == verdict["total"] == 5


@pytest.mark.skipif(not ARTIFACTS.exists(),
                    reason="p223w2b-n06 artifacts absent in this checkout")
def test_stored_study_is_not_self_certifying():
    """A tampered defect ladder must be re-gated as a failure, not trusted."""
    payload = json.loads(ARTIFACTS.read_text())
    study = {key: payload[key] for key in ("mms", "attribution", "staircase",
                                           "no_delay_energy", "regression")}
    arm = study["staircase"]["dt"]["centered_destamp2"]
    for level in arm["levels"]:
        level["defect_l1"] = 5e-3  # a flat ladder cannot be second order
    verdict = gate_maturity(study)
    keys = {c["key"]: c["verdict"] for c in verdict["criteria"]}
    assert keys["weld_staircase_second_order"] == "fail"
    assert verdict["verdict"] == "fail"