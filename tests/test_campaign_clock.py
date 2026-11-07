"""Tests for the M12 coupling-clock contract.

Disposition M12 requires the master/sub step ratio *and* the exchange cadence to
be explicit contract fields, and requires exchange instants to live on an exact
integer lattice so a long campaign cannot drift.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.campaign import (
    ClockContractError,
    CouplingClock,
    CouplingClockSpec,
    ExchangeEvent,
    clock_drift,
    expected_exchange_count,
    verify_alignment,
)


def decimal_spec() -> CouplingClockSpec:
    return CouplingClockSpec(
        master_name="rigid_body",
        sub_name="lbm_flow",
        master_dt_s=0.004,
        sub_dt_s=0.0004,
        exchange_every=5,
    )


def test_ratio_and_cadence_are_explicit_contract_fields():
    spec = decimal_spec()
    assert spec.ratio == 10
    assert spec.exchange_interval_s == pytest.approx(0.02, rel=0, abs=0)
    assert spec.sub_steps_per_exchange == 50
    assert spec.max_information_age_s == spec.exchange_interval_s
    payload = spec.to_dict()
    assert payload["disposition"] == "M12"
    assert payload["time_base"] == "integer_sub_step"
    assert payload["ratio"] == 10 and payload["exchange_every"] == 5
    assert payload["sub_steps_per_exchange"] == 50


def test_exchange_cadence_has_no_implicit_default():
    with pytest.raises(TypeError):
        CouplingClockSpec("a", "b", 0.004, 0.0004)  # type: ignore[call-arg]
    with pytest.raises(ClockContractError):
        CouplingClockSpec("a", "b", 0.004, 0.0004, exchange_every=0)
    with pytest.raises(ClockContractError):
        CouplingClockSpec("a", "b", 0.004, 0.0004, exchange_every=True)


def test_non_integer_ratio_is_rejected():
    with pytest.raises(ClockContractError, match="integer ratio"):
        CouplingClockSpec("a", "b", 0.004, 0.00045, exchange_every=1)


def test_sub_clock_must_be_finer_than_master():
    with pytest.raises(ClockContractError, match="finer"):
        CouplingClockSpec("a", "b", 0.0004, 0.004, exchange_every=1)


def test_master_and_sub_must_differ():
    with pytest.raises(ClockContractError):
        CouplingClockSpec("solver", "solver", 0.004, 0.0004, exchange_every=1)


def test_exchange_instants_lie_on_the_exact_integer_lattice():
    spec = decimal_spec()
    clock = CouplingClock(spec)
    events = clock.run(250)
    assert len(events) == expected_exchange_count(spec, 250) == 50
    assert clock.sub_step == 250 * spec.ratio
    for event in events:
        assert event.sub_step == event.master_tick * spec.ratio
        assert event.time_s == event.master_tick * spec.master_dt_s
    report = verify_alignment(spec, events)
    assert report["sub_step_residual"] == 0
    assert report["time_residual_s"] == 0.0
    assert report["cadence_residual_s"] <= report["tolerance_s"]
    assert report["aligned"] is True


def test_partial_windows_do_not_emit_exchanges():
    spec = decimal_spec()
    clock = CouplingClock(spec)
    assert clock.run(4) == ()
    assert len(clock.advance_master()) == 1
    assert clock.master_tick == 5 and clock.sub_step == 50


def test_exchange_count_closed_form_matches_simulation():
    spec = decimal_spec()
    for ticks in (0, 1, 4, 5, 49, 50, 51, 500):
        clock = CouplingClock(spec)
        assert len(clock.run(ticks)) == expected_exchange_count(spec, ticks)


def test_integer_lattice_has_zero_drift_while_naive_accumulation_does_not():
    spec = decimal_spec()
    drift = clock_drift(spec, 500)
    assert drift["canonical_drift_s"] == 0.0
    assert drift["sub_steps"] == 5000
    assert abs(drift["naive_drift_s"]) > 0.0
    assert drift["naive_relative_drift"] < 1e-12  # small, but 60x larger than eps


def test_clock_artifact_roundtrip_and_state_ticks():
    spec = decimal_spec()
    clock = CouplingClock(spec)
    clock.run(15)
    payload = spec.to_dict(master_tick=clock.master_tick)
    assert payload["ticks"] == {"master": 15, "sub_step": 150}
    restored = CouplingClockSpec.from_dict(payload)
    assert restored == spec


def test_clock_artifact_rejects_extra_fields_and_bad_versions():
    payload = decimal_spec().to_dict()
    payload["exchange_every"] = 7  # would silently change the cadence
    with pytest.raises(ClockContractError, match="internally inconsistent"):
        CouplingClockSpec.from_dict(payload)
    payload = decimal_spec().to_dict()
    payload["schema_version"] = 2
    with pytest.raises(ClockContractError):
        CouplingClockSpec.from_dict(payload)
    payload = decimal_spec().to_dict()
    payload["extra"] = {"denormal": "field"}
    with pytest.raises(ClockContractError, match="additional properties"):
        CouplingClockSpec.from_dict(payload)


def test_exchange_event_rejects_invalid_state():
    with pytest.raises(ClockContractError):
        ExchangeEvent(index=0, master_tick=5, sub_step=50, time_s=0.02)
    with pytest.raises(ClockContractError):
        ExchangeEvent(index=1, master_tick=5, sub_step=50, time_s=math.nan)


def test_information_age_bound_is_the_exchange_interval():
    spec = decimal_spec()
    assert spec.max_information_age_s == spec.exchange_every * spec.master_dt_s
    assert spec.max_information_age_s == pytest.approx(
        spec.sub_steps_per_exchange * spec.sub_dt_s, rel=1e-15
    )