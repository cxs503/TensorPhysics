"""Tests for campaign node P40W4-N09: three-field coupling energy contract.

The expensive coupled run is executed once in a module-scoped fixture with a
short time window; the production numbers are taken from the committed record
``docs/p40w4_n09/verification.json`` instead of being re-run.
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np
import pytest

from tensorphysics.coupling_energy.three_field import (
    CONTRACTS,
    CRITERIA,
    CRITERIA_DIGEST,
    ContractError,
    EnergyBook,
    EnergyBookError,
    History,
    LoopConfig,
    RateSchedule,
    SubcyclingError,
    assert_reciprocity,
    audit_record,
    contract,
    coupling_graph,
    criteria_digest,
    evaluate,
    integrate_history,
    lagrange_value,
    reconstruction_operator_study,
    run_negative_control,
    run_three_field_loop,
    validate_contracts,
)

RECORD_PATH = (pathlib.Path(__file__).resolve().parents[1] / "docs" / "p40w4_n09"
               / "verification.json")

FAST = dict(t_end=0.008, ny=8)


@pytest.fixture(scope="module")
def coupled():
    return run_three_field_loop(LoopConfig(**FAST))


@pytest.fixture(scope="module")
def record():
    if not RECORD_PATH.exists():  # pragma: no cover - record not generated yet
        pytest.skip("committed verification record is missing")
    return json.loads(RECORD_PATH.read_text())


# ---------------------------------------------------------------------------
# interface contracts
# ---------------------------------------------------------------------------
def test_contracts_form_a_closed_cycle():
    validate_contracts(CONTRACTS)
    graph = coupling_graph(CONTRACTS)
    assert set(graph) == {"fluid", "solid", "particles"}
    for src, targets in graph.items():
        assert targets, f"{src} has no outgoing crossing"
        for tgt in targets:
            assert src in graph[tgt] or tgt in graph, "cycle must close"


def test_contract_validation_rejects_a_chain():
    chain = tuple(c for c in CONTRACTS if c.key != "dem_lbm_drag")
    # dropping the drag crossing leaves solid->particles as a dead end
    with pytest.raises(ContractError):
        validate_contracts(chain)


def test_each_contract_declares_both_bookings_and_a_unit():
    for c in CONTRACTS:
        assert c.booked_by_source and c.booked_by_target
        assert c.unit in {"W", "N"}
        assert c.conservation
        assert c.as_dict()["key"] == c.key


def test_reciprocity_is_accepted_for_the_contact_crossing():
    # both entries are given in the contract's sign convention (action and the
    # reaction *after* the sign flip), so a reciprocal pair is the same number
    assert_reciprocity("fem_dem_contact_force", -12.5, -12.5)


def test_reciprocity_rejects_a_doctored_mismatch():
    with pytest.raises(ContractError):
        assert_reciprocity("fem_dem_contact_force", -12.5, -11.0)


def test_reconstructed_crossing_cannot_be_reciprocity_checked():
    with pytest.raises(ContractError):
        assert_reciprocity("lbm_fem_heat_flux", 1.0, 1.0)
    assert contract("lbm_fem_heat_flux").exact is False


# ---------------------------------------------------------------------------
# energy book
# ---------------------------------------------------------------------------
def test_energy_book_total_is_the_sum_of_four_terms():
    book = EnergyBook()
    book.set("kinetic", "particles", 2.0)
    book.set("potential", "solid", 0.5)
    book.set("internal", "fluid", 10.0)
    book.set("dissipated", "solid", 0.25)
    assert book.total() == pytest.approx(12.75)
    snap = book.sample(0.0)
    assert snap.component_total("internal") == pytest.approx(10.0)


def test_energy_book_detects_an_unbooked_source():
    book = EnergyBook()
    book.set("kinetic", "particles", 1.0)
    book.set("internal", "fluid", 1.0)
    book.sample(0.0)
    book.set("internal", "fluid", 1.01)   # 1 % created, nothing booked
    book.sample(1.0)
    assert book.residual() == pytest.approx(0.005, rel=1e-12)
    assert book.absolute_imbalance() == pytest.approx(0.01, rel=1e-12)


def test_energy_book_requires_increasing_time_and_a_scale():
    book = EnergyBook()
    book.sample(0.0)
    with pytest.raises(EnergyBookError):
        book.sample(0.0)
    with pytest.raises(EnergyBookError):
        book.initial_scale()          # zero initial energy -> undefined residual


# ---------------------------------------------------------------------------
# multi-rate schedule and reconstruction
# ---------------------------------------------------------------------------
def test_rate_schedule_reports_three_distinct_steps():
    sched = RateSchedule(dt_fluid=1.0e-3, dt_solid=5.0e-5, dt_particles=1.6e-4)
    d = sched.as_dict()
    assert d["dt_macro"] == pytest.approx(1.0e-3)
    assert d["ratio_fluid_over_solid"] == pytest.approx(20.0)
    assert sched.ratio("fluid", "particles") == pytest.approx(6.25)
    with pytest.raises(SubcyclingError):
        sched.ratio("particles", "fluid")   # asked for the wrong direction
    with pytest.raises(SubcyclingError):
        RateSchedule(dt_fluid=-1.0, dt_solid=1.0, dt_particles=1.0)


def test_lagrange_is_exact_on_a_cubic():
    f = lambda t: 1.0 - 2.0 * t + 0.5 * t ** 2 + 3.0 * t ** 3
    ts = [0.0, 0.1, 0.2, 0.3]
    ys = [f(t) for t in ts]
    for t in (0.05, 0.17, 0.3, 0.42):
        assert lagrange_value(ts, ys, t) == pytest.approx(f(t), rel=1e-12,
                                                          abs=1e-12)


def test_history_is_causal_and_bootstrap_aware():
    hist = History("t", degree=2)
    hist.push(0.0, 1.0)
    hist.push(1.0, 2.0)
    assert hist.value_at(0.5) == pytest.approx(1.0)     # only t=0 is available
    assert hist.value_at(1.5) == pytest.approx(2.5)     # linear extrapolation
    # a sample in the future must not be used
    hist.push(2.0, 100.0)
    assert hist.value_at(1.0) == pytest.approx(2.0)
    with pytest.raises(SubcyclingError):
        hist.push(1.5, 0.0)


def test_integrate_history_is_exact_for_a_constant():
    hist = History("p", degree=1)
    hist.push(0.0, 3.0)
    assert integrate_history(hist, 0.0, 2.0, n_quad=5) == pytest.approx(6.0)


def test_high_degree_extrapolation_overshoots_a_contact_step():
    """Why the contact load is exchanged with a zero-order hold.

    A degree-2 polynomial through a *discontinuous* contact history extrapolated
    one interval past the impact returns three times the applied load; feeding
    that back into the plate inside the same interval is what makes ``p > 0``
    contact-load exchange unstable.
    """
    hist = History("load", degree=2)
    for t, f in ((0.0, 0.0), (1.0e-4, 0.0), (2.0e-4, 20.0)):
        hist.push(t, f)
    assert hist.value_at(3.0e-4) == pytest.approx(60.0, rel=1e-9)
    hold = History("load", degree=0)
    hold.push(0.0, 0.0)
    hold.push(2.0e-4, 20.0)
    assert hold.value_at(3.0e-4) == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# the coupled loop
# ---------------------------------------------------------------------------
def test_coupled_loop_closes_the_ledger(coupled):
    diag = coupled.diagnostics
    assert diag["residual"] < 1.0e-3, diag["residual"]
    assert math.isfinite(diag["absolute_imbalance"])
    assert diag["energy_initial"]["internal"] > 0.0


def test_all_three_crossings_are_engaged(coupled):
    diag = coupled.diagnostics
    assert diag["contact_max_overlap"] > 0.0          # contact really happened
    assert diag["flux_fluid_peak"] > 0.0              # heat really crossed
    assert diag["drag_source_energy"] > 0.0           # drag really dissipated
    assert diag["dissipated"] > 0.0                   # dashpot really booked


def test_contact_crossing_is_reciprocal_in_the_run(coupled):
    assert coupled.diagnostics["momentum_reciprocity_error"] < 1e-12


def test_three_rates_are_distinct_in_the_run(coupled):
    sched = coupled.diagnostics["schedule"]
    steps = {sched["dt_fluid"], sched["dt_solid"], sched["dt_particles"]}
    assert len(steps) == 3
    assert coupled.diagnostics["n_panel_steps"] >= 2
    assert coupled.diagnostics["n_dem_steps"] >= 2


def test_drag_crossing_balances_within_a_few_percent(coupled):
    assert abs(coupled.diagnostics["drag_transfer_mismatch"]) < 5.0e-2


def test_negative_control_round_off_fields():
    ctrl = run_negative_control(LoopConfig(**FAST), n_steps=60)
    # the two diffusive fields and the free-flying particles conserve at round-off
    assert ctrl["fluid_thermal"] < 1e-10
    assert ctrl["panel_thermal"] < 1e-10
    assert ctrl["particles_kinetic"] < 1e-12
    # the explicit plate is limited by its symplectic O(dt^2) oscillation, and the
    # record reports it: the control must still be far below the coupled residual
    assert ctrl["plate_elastic"] < 1.0e-2


def test_operator_order_gain_is_at_least_four():
    study = reconstruction_operator_study(h=2.5e-4, t_end=0.04)
    assert study["gains"]["p1_over_p2"] >= 4.0
    # observed order of every scheme saturates near two: one sample per interval
    # is an interval average, which floors the operator at second order
    for p, run in study["runs"].items():
        assert 1.5 <= run["observed_order"] <= 2.6


# ---------------------------------------------------------------------------
# acceptance layer
# ---------------------------------------------------------------------------
def test_evaluate_withholds_a_missing_measurement():
    report = evaluate({"production_degree": 1, "runs": {"1": {}}})
    assert report.verdict == "WITHHELD"
    assert report.counts()["WITHHELD"] == len(CRITERIA)


def test_criteria_digest_pins_the_thresholds():
    assert criteria_digest(CRITERIA) == CRITERIA_DIGEST
    edited = tuple(
        c if c.id != "total_energy_residual"
        else type(c)(**{**c.as_dict(), "threshold": 1.0})
        for c in CRITERIA
    )
    assert criteria_digest(edited) != CRITERIA_DIGEST


def test_committed_record_regates_and_is_honest(record):
    report = audit_record(record)
    assert report.verdict == record["verdict"]
    # the node's primary conservation gate passes at the production degree
    assert report.outcomes["total_energy_residual"].outcome.value == "PASS"
    # and the order gate is an honest FAIL, not a withheld or a silent pass
    assert report.outcomes["interpolation_order_gain"].outcome.value == "FAIL"
    assert record["environment"]["git_rev"]


def test_record_rejects_a_foreign_criteria_digest(record):
    doctored = dict(record)
    doctored["criteria_digest"] = "0" * 64
    with pytest.raises(ValueError):
        audit_record(doctored)