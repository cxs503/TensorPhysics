"""Tests for the rational (GCD-aligned) coupling clock and its causality gate.

Node IN16-K8: ``SYSTEM_OVERVIEW`` §5 only stated the *requirement* that solvers
synchronise physical clocks explicitly; the v1 contract
(:mod:`tensorphysics.campaign.clock`) enforced it for integer step ratios only.
These tests pin the implementation-level semantics:

* arbitrary **rational** ratios on an exact atom lattice (no drift, no rounding);
* exchange cadence = LCM(dt_master, dt_sub) -- machine-verified as the *coarsest*
  common period, not merely a common one;
* a **causality** gate that refuses future data (negative case);
* the multi-rate two-integrator cross-check against a closed-form reference,
  including the declared in-interval error band and the order it implies.
"""
from __future__ import annotations

import math
from fractions import Fraction

import pytest

from tensorphysics.campaign.clock_rational import (
    COUPLING_CLOCK_RATIONAL_V1,
    CausalityViolation,
    RationalClockError,
    RationalClockSpec,
    RationalCouplingClock,
    assert_causal_hold,
    build_causal_buffer,
    causality_audit,
    expected_exchange_count,
    fraction_gcd,
    fraction_lcm,
    verify_gcd_cadence,
)
from tensorphysics.campaign.multirate import (
    BAND_LINEAR_COEF,
    BAND_ZOH_COEF,
    ORDER_WINDOW,
    ORDER_WINDOW_SECOND,
    TwoIntegratorModel,
    band_scan,
    interpolation_band,
    interpolation_scan,
    refinement_scan,
    signal_amplitude,
    simulate_multirate,
)


# ---------------------------------------------------------------------------
# clock contract: rational ratio
# ---------------------------------------------------------------------------
def test_fractional_ratio_is_exact_not_rounded() -> None:
    """3 ms structure / 2 ms contact is ratio 3/2, not 1 or 2."""
    spec = RationalClockSpec(
        master_name="structure",
        sub_name="contact",
        master_dt_s=0.003,
        sub_dt_s=0.002,
        exchange_periods=1,
    )
    assert spec.ratio == Fraction(3, 2)
    assert spec.ratio_is_integer is False
    assert spec.master_dt == Fraction(3, 1000)
    assert spec.sub_dt == Fraction(2, 1000)
    assert spec.atom_s == Fraction(1, 1000)
    assert (spec.master_atoms, spec.sub_atoms) == (3, 2)
    assert spec.coprime is True
    # an exchange needs 2 master steps == 3 sub steps
    assert spec.master_ticks_per_exchange == 2
    assert spec.sub_steps_per_exchange == 3
    assert spec.base_exchange_s == Fraction(6, 1000)
    assert spec.exchange_interval_s == Fraction(6, 1000)


def test_gcd_atom_and_lcm_cadence_are_machine_verified() -> None:
    """The cadence must be the coarsest common period, checked exhaustively."""
    for dts in ((0.003, 0.002), (0.004, 0.003), (0.004, 0.0004), (0.02, 0.0075)):
        spec = RationalClockSpec("master", "sub", *dts, exchange_periods=1)
        report = verify_gcd_cadence(spec, master_ticks=spec.master_ticks_per_exchange * 12)
        assert report["verified"] is True, (dts, report)
        assert report["coprime_atoms"] is True
        assert report["atom_maximal"] is True
        assert report["lcm_atoms_equals_product"] is True
        assert report["spurious_common_periods"] == []
        assert report["cadence_is_coarsest"] is True
        assert report["instant_residual_zero"] is True
        assert report["substep_residual_zero"] is True
        # fraction gcd/lcm agree with the integer decomposition
        assert fraction_gcd(spec.master_dt, spec.sub_dt) == spec.atom_s
        assert fraction_lcm(spec.master_dt, spec.sub_dt) == spec.base_exchange_s
        assert report["event_count_matches_closed_form"] is True
        assert report["event_count"] == 12


def test_exchange_instants_are_exact_and_drift_free() -> None:
    """Long runs stay on the exact lattice: zero residual, zero drift."""
    spec = RationalClockSpec("lbm_flow", "dem_ice", 0.004, 0.0012, exchange_periods=3)
    clock = RationalCouplingClock(spec)
    events = clock.run(spec.master_ticks_per_exchange * 100)
    assert len(events) == 100
    for event in events:
        assert event.time_exact == event.index * spec.exchange_interval_s
        assert event.sub_step * spec.sub_dt == event.time_exact
    # float times are a single rounding of the exact value
    assert all(abs(event.time_s - float(event.time_exact)) <= 4 * math.ulp(event.time_s) for event in events)
    drift = events[-1].time_exact - 100 * spec.exchange_interval_s
    assert drift == 0


def test_cadence_is_mandatory_and_invalid_inputs_are_rejected() -> None:
    with pytest.raises(TypeError):
        RationalClockSpec("a", "b", 0.004, 0.002)  # type: ignore[call-arg]
    with pytest.raises(RationalClockError):
        RationalClockSpec("a", "b", 0.004, 0.002, exchange_periods=0)
    with pytest.raises(RationalClockError):
        RationalClockSpec("a", "b", 0.004, 0.002, exchange_periods=True)
    with pytest.raises(RationalClockError, match="coarser"):
        RationalClockSpec("a", "b", 0.002, 0.004, exchange_periods=1)
    with pytest.raises(RationalClockError, match="must differ"):
        RationalClockSpec("a", "a", 0.004, 0.002, exchange_periods=1)
    with pytest.raises(RationalClockError, match="M12"):
        RationalClockSpec("a", "b", 0.004, 0.002, exchange_periods=1, disposition="M11")


def test_rationalisation_of_float_steps_declares_its_tolerance() -> None:
    spec = RationalClockSpec.from_float_dts(
        "structure", "contact", 0.003, 0.002, exchange_periods=1, max_denominator=1000
    )
    assert spec.ratio == Fraction(3, 2)
    # a genuinely incommensurate pair cannot be rationalised inside the tolerance
    with pytest.raises(RationalClockError, match="cannot be rationalised"):
        RationalClockSpec.from_float_dts(
            "a",
            "b",
            math.sqrt(2) / 1000.0,
            math.sqrt(3) / 1000.0,
            exchange_periods=1,
            max_denominator=8,
            rel_tol=1e-12,
        )


def test_artifact_roundtrip_and_tamper_detection() -> None:
    spec = RationalClockSpec("structure", "contact", 0.003, 0.002, exchange_periods=2)
    payload = spec.to_dict(master_tick=spec.master_ticks_per_exchange)
    assert payload["schema_name"] == "coupling_clock_rational"
    assert payload["time_base"] == "rational_atom"
    assert payload["ratio_exact"] == "3/2"
    assert payload["exchange_periods"] == 2
    back = RationalClockSpec.from_dict(payload)
    assert back.ratio == spec.ratio and back.exchange_interval_s == spec.exchange_interval_s

    for field, bad in (
        ("atom_exact", "1/999"),
        ("ratio_exact", "3/1"),
        ("master_ticks_per_exchange", 3),
        ("exchange_interval_exact", "1/100"),
    ):
        tampered = dict(payload)
        tampered[field] = bad
        with pytest.raises(RationalClockError):
            RationalClockSpec.from_dict(tampered)

    tampered = dict(payload)
    tampered["extra"] = 1
    with pytest.raises(RationalClockError):
        RationalClockSpec.from_dict(tampered)
    # schema membership itself is a closed subset of JSON Schema
    assert COUPLING_CLOCK_RATIONAL_V1["additionalProperties"] is False


# ---------------------------------------------------------------------------
# causality gate (the required negative case)
# ---------------------------------------------------------------------------
def test_buffer_refuses_future_data() -> None:
    buf = build_causal_buffer("ice_force", [(Fraction(2, 100), 1.0), (Fraction(4, 100), 2.0)])
    assert buf.latest_at_or_before(Fraction(3, 100)).value == 1.0
    assert buf.information_age(Fraction(3, 100)) == Fraction(1, 100)
    with pytest.raises(CausalityViolation, match="future data cannot be used"):
        buf.latest_at_or_before(Fraction(1, 100))


def test_exchange_consumer_cannot_use_a_future_sample() -> None:
    # a sub-solver overshot the exchange instant by one sub step
    with pytest.raises(CausalityViolation, match="refuses extrapolation"):
        assert_causal_hold(Fraction(12, 1000), Fraction(14, 1000), label="exchange 1")
    # exactly at the exchange instant is legal, with zero information age
    assert assert_causal_hold(Fraction(12, 1000), Fraction(12, 1000)) == 0


def test_causality_audit_reports_overshoot_frequency() -> None:
    spec = RationalClockSpec("structure", "contact", 0.004, 0.002, exchange_periods=1)
    events = RationalCouplingClock(spec).run(spec.master_ticks_per_exchange * 4)
    good = causality_audit(spec, events, producer_times=[event.time_exact for event in events])
    assert good["is_causal"] is True
    assert good["max_information_age_s"] == 0.0
    assert good["within_information_age_bound"] is True

    late = [event.time_exact + spec.sub_dt for event in events]
    bad = causality_audit(spec, events, producer_times=late)
    assert bad["is_causal"] is False
    assert len(bad["violations"]) == len(events)
    assert bad["violations"][0]["overshoot_exact"] == "1/500"


# ---------------------------------------------------------------------------
# multi-rate two-integrator cross-check against the analytic solution
# ---------------------------------------------------------------------------
def test_analytic_reference_is_a_real_solution_of_the_coupled_system() -> None:
    """The closed form must satisfy the ODE and the initial conditions."""
    model = TwoIntegratorModel()
    x0, v0 = (1.0e-3, 0.0), (0.0, 0.0)
    x_at_0 = model.exact(0.0, x0=x0, v0=v0)
    assert x_at_0[0] == pytest.approx(x0[0], rel=1e-12)
    assert x_at_0[1] == pytest.approx(x0[1], abs=1e-12)
    # residual of the equations of motion via second-order central differences
    h = 1e-5
    t = 0.31
    xp = model.exact(t + h, x0=x0, v0=v0)
    xm = model.exact(t - h, x0=x0, v0=v0)
    xc = model.exact(t, x0=x0, v0=v0)
    acc = [(xp[i] - 2 * xc[i] + xm[i]) / h ** 2 for i in range(2)]
    f1 = (-(model.k_master + model.k_coupling) * xc[0] + model.k_coupling * xc[1]) / model.m_master
    f2 = (-(model.k_sub + model.k_coupling) * xc[1] + model.k_coupling * xc[0]) / model.m_sub
    assert acc[0] == pytest.approx(f1, rel=1e-5)
    assert acc[1] == pytest.approx(f2, rel=1e-5)


def test_multirate_solution_tracks_the_analytic_reference() -> None:
    model = TwoIntegratorModel()
    spec = RationalClockSpec("hull_section", "ice_contact", 0.004, 0.003, exchange_periods=1)
    result = simulate_multirate(model, spec, intervals=256)
    # one decade below the declared band: comfortable margin in the fine regime
    assert result.rel_error <= BAND_ZOH_COEF * model.omega_max * result.exchange_interval_s
    assert result.rel_error < 2e-2
    assert result.master_steps == 256 * spec.master_ticks_per_exchange
    assert result.sub_steps == 256 * spec.sub_steps_per_exchange


def test_exchange_error_band_and_first_order_convergence() -> None:
    """Refining clocks and cadence together gives clean first-order behaviour."""
    report = refinement_scan(TwoIntegratorModel(), levels=4)
    assert report["order_ok"] is True, report["order"]
    assert report["pair_orders"] == pytest.approx([1.0, 1.0, 1.0], abs=0.05)
    assert report["all_within_band"] is True
    assert report["worst_band_ratio"] <= BAND_ZOH_COEF
    # errors halve with the exchange interval
    errors = [row["rel_error"] for row in report["rows"]]
    for coarse, fine in zip(errors, errors[1:]):
        assert coarse / fine == pytest.approx(2.0, rel=0.05)


def test_band_holds_across_a_decade_of_exchange_intervals() -> None:
    report = band_scan(TwoIntegratorModel(), periods=(1, 2, 4, 8, 16), base_intervals=256)
    assert report["all_within_band"] is True
    assert all(row["within_band"] for row in report["rows"])
    # the exchange interval spans a decade and stays on the clock's own LCM
    hs = [row["exchange_interval_s"] for row in report["rows"]]
    assert hs[-1] / hs[0] == pytest.approx(16.0, rel=1e-12)
    # energy drift grows with the (stale) exchange interval
    drifts = [row["rel_energy_drift"] for row in report["rows"]]
    assert all(b > a for a, b in zip(drifts, drifts[1:]))


def test_in_interval_interpolation_band_and_causality_cost() -> None:
    """Linear interpolation is second order but needs future data -> rejected."""
    report = interpolation_scan(TwoIntegratorModel())
    assert report["all_zoh_within_band"] is True
    assert report["all_linear_within_band"] is True
    assert report["linear_order_ok"] is True, report["order_linear"]
    assert report["linear_order_ok"] is True and ORDER_WINDOW_SECOND[0] <= report["order_linear"] <= ORDER_WINDOW_SECOND[1]
    assert report["zoh_order_ok"] is True, report["order_zoh"]
    assert ORDER_WINDOW[0] <= report["order_zoh"] <= ORDER_WINDOW[1]
    # every linear reconstruction inside an interval is flagged non-causal
    assert report["linear_rejected_as_non_causal"] is True
    worst_zoh = report["worst_zoh_ratio"]
    worst_lin = report["worst_linear_ratio"]
    assert worst_zoh <= BAND_ZOH_COEF
    assert worst_lin <= BAND_LINEAR_COEF
    # causality costs about an order in accuracy at the same exchange interval
    assert worst_lin < worst_zoh


def test_interpolation_band_flags_the_non_causal_query() -> None:
    model = TwoIntegratorModel()
    row = interpolation_band(model, exchange_interval_s=0.012, intervals=8)
    assert row["linear_is_causal"] is False
    assert "future" in row["causality_violation"] or "refuses" in row["causality_violation"]
    x0, v0 = (1.0e-3, 0.0), (0.0, 0.0)
    assert row["amplitude"] == pytest.approx(
        signal_amplitude(model, x0=x0, v0=v0, duration_s=8 * 0.012), rel=1e-12
    )


def test_fractional_ratio_run_matches_integer_ratio_run() -> None:
    """A 3/2 clock reproduces the integer-ratio clock of the same exchange cadence.

    Rational semantics must be a *superset* of the v1 integer contract: two
    different decompositions of the same physical cadence (dt 2 ms/1.5 ms vs
    4 ms/3 ms halved) give the same exchange instants.
    """
    coarse = RationalClockSpec("m", "s", 0.004, 0.003, exchange_periods=1)
    fine = RationalClockSpec("m", "s", 0.002, 0.0015, exchange_periods=1)
    assert coarse.exchange_interval_s == Fraction(3, 250)  # 12 ms
    assert fine.exchange_interval_s == Fraction(3, 500)  # 6 ms (halved clocks, same 4:3 ratio)
    assert coarse.ratio == fine.ratio == Fraction(4, 3)
    events_coarse = RationalCouplingClock(coarse).run(coarse.master_ticks_per_exchange * 10)
    events_fine = RationalCouplingClock(fine).run(fine.master_ticks_per_exchange * 20)
    fine_times = {e.time_exact for e in events_fine}
    assert len(events_coarse) == 10 and len(events_fine) == 20
    # identical physical horizon: every coarse exchange instant is also a fine one
    assert events_coarse[-1].time_exact == events_fine[-1].time_exact == Fraction(3, 25)
    assert all(e.time_exact in fine_times for e in events_coarse)
    # the declared information age bound is the exchange interval in both cases
    assert coarse.max_information_age_s == coarse.exchange_interval_s
    assert fine.max_information_age_s == fine.exchange_interval_s
    # closed-form exchange counts agree with the drivers
    assert expected_exchange_count(coarse, coarse.master_ticks_per_exchange * 10) == len(events_coarse)
    assert expected_exchange_count(fine, fine.master_ticks_per_exchange * 20) == len(events_fine)
    assert expected_exchange_count(coarse, coarse.master_ticks_per_exchange * 10) == 10