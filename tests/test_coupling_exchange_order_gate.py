"""Tests of the coupling MMS order ladder, the three chains and the acceptance gate."""
from __future__ import annotations

import math

import pytest

from tensorphysics.coupling_energy.exchange_gate import (
    gate_study,
    run_exchange_study,
)
from tensorphysics.coupling_energy.exchange_mms import mms_order_ladder
from tensorphysics.coupling_energy.time_exchange import (
    CHAIN_SPECS,
    CouplingMMS,
    build_chain,
    run_cadence_ladder,
    run_ladder,
)

H_LIST = (1.0 / 16, 1.0 / 32, 1.0 / 64, 1.0 / 128)


# ---------------------------------------------------------------------------
# manufactured solutions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("scheme,lo,hi", [
    ("lag", 0.85, 1.15),
    ("centered", 1.85, 2.15),
])
def test_mms_order_of_the_operator(scheme, lo, hi):
    ladder = mms_order_ladder(CouplingMMS.harmonic(), scheme, h_list=H_LIST)
    order, r2 = ladder.fit()
    assert lo <= order <= hi, (scheme, order)
    assert r2 >= 0.999, (scheme, r2)


def test_predictor_corrector_provisional_is_second_order_and_correction_matches_centered():
    harmonic = CouplingMMS.harmonic()
    prov = mms_order_ladder(harmonic, "predictor_corrector", h_list=H_LIST, corrected=False)
    corr = mms_order_ladder(harmonic, "predictor_corrector", h_list=H_LIST, corrected=True)
    centred = mms_order_ladder(harmonic, "centered", h_list=H_LIST)
    assert prov.fit()[0] >= 1.75
    assert corr.fit()[0] >= 1.85
    for a, b in zip(corr.defects(), centred.defects()):
        assert a == pytest.approx(b, rel=1e-12)


def test_window_convention_is_first_order_and_destamp_restores_second_order():
    harmonic = CouplingMMS.harmonic()
    raw = mms_order_ladder(harmonic, "centered", h_list=H_LIST, offset_ratio=0.25)
    repaired = mms_order_ladder(harmonic, "centered", h_list=H_LIST,
                                offset_ratio=0.25, recenter=True)
    assert 0.85 <= raw.fit()[0] <= 1.15
    assert repaired.fit()[0] >= 1.85


# ---------------------------------------------------------------------------
# chains
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", sorted(CHAIN_SPECS))
def test_chain_endpoint_convention_shows_declared_orders(name):
    chain = build_chain(name)
    lag = run_cadence_ladder(chain, "lag", ratios=(1, 2, 4, 8), stamp_offset_ratio=0.0)
    cen = run_cadence_ladder(chain, "centered", ratios=(1, 2, 4, 8), stamp_offset_ratio=0.0)
    assert lag.fit()[0] <= 1.05
    assert 1.85 <= cen.fit()[0] <= 2.15


@pytest.mark.parametrize("name", sorted(CHAIN_SPECS))
def test_endstamp_practice_plateaus_and_the_repair_removes_it(name):
    chain = build_chain(name)
    broken = run_cadence_ladder(chain, "centered", ratios=(1, 2, 4, 8),
                                stamp_offset_ratio=0.5)
    repaired = run_cadence_ladder(chain, "centered", ratios=(1, 2, 4, 8),
                                  stamp_offset_ratio=0.5, recenter=True)
    defects = broken.defects()
    assert max(defects) / min(defects) <= 1.3          # staircase, no convergence
    assert broken.fit()[0] <= 0.30
    assert broken.saturation() / repaired.levels[0].defect_l1 >= 50.0


@pytest.mark.parametrize("name", sorted(CHAIN_SPECS))
def test_subcycle_complete_arm_is_exact(name):
    chain = build_chain(name)
    ladder = run_cadence_ladder(chain, "centered", ratios=(1, 2, 4),
                                subcycle=True)
    for level in ladder.levels:
        assert level.unattributed_share <= 1e-15
        assert level.deficit_share <= 1e-12
        assert level.defect_l1 <= 1e-10


def test_tail_booking_loss_reproduces_the_reported_band():
    chain = build_chain("weld")
    losses = {}
    for ratio in (8, 16):
        level = run_ladder(chain, "centered", ratio, tail_booking=True)
        losses[ratio] = level.unattributed_share
    assert 0.85 <= losses[8] <= 0.90
    assert 0.92 <= losses[16] <= 0.95


def test_ladder_rejects_degenerate_configurations():
    chain = build_chain("weld")
    with pytest.raises(Exception):
        run_ladder(chain, "centered", 0)
    with pytest.raises(Exception):
        run_ladder(chain, "centered", 2, subcycle=True, tail_booking=True)
    with pytest.raises(Exception):
        build_chain("no_such_chain")


# ---------------------------------------------------------------------------
# gate end-to-end (short ladders, same code path as the shipped study)
# ---------------------------------------------------------------------------
def test_gate_passes_a_short_study():
    study = run_exchange_study(ratios=(1, 2, 4))
    verdict = gate_study(study)
    failing = [c["key"] for c in verdict["criteria"] if c["verdict"] != "pass"]
    assert verdict["verdict"] == "pass", failing
    assert verdict["passed"] == verdict["total"]


def test_gate_recomputes_the_order_from_the_ladder():
    """A gate that reads a stored exponent certifies itself; this one re-derives it."""
    study = run_exchange_study(ratios=(1, 2, 4))
    levels = study["mms"]["centered"]["levels"]
    # flatten the two coarsest levels onto the finest defect: the ladder no longer
    # converges, so a gate that recomputes must reject it even though the stored
    # ``observed_order`` field still claims 2.0
    levels[0]["defect_l1"] = levels[-1]["defect_l1"]
    levels[1]["defect_l1"] = levels[-1]["defect_l1"]
    verdict = gate_study(study)
    assert verdict["verdict"] == "fail"
    failing = [c["key"] for c in verdict["criteria"] if c["verdict"] == "fail"]
    assert "mms_order_centered" in failing
    assert study["mms"]["centered"]["observed_order"] > 1.85  # stale field, not used


def test_gate_fails_when_the_ledger_control_is_disabled():
    study = run_exchange_study(ratios=(1, 2, 4))
    study["attribution"]["controls"]["double_booking_refused"] = False
    verdict = gate_study(study)
    keys = [c["key"] for c in verdict["criteria"] if c["verdict"] == "fail"]
    assert "attribution_ledger_refuses_anomalies" in keys