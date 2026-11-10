"""Tests for Cowper-Symonds rate-dependent plasticity and impact DAF (p351w1_n05).

Acceptance-criteria coverage
----------------------------
(a) uniaxial tension at 0.001 / 0.1 / 10 s^-1 vs the closed form (<= 2 %),
(c) rate-independent limit (exact at zero rate; the literal 1e-4 s^-1 variant is
    documented as not achievable for D = 40.4 s^-1, p = 5),
(d) quasi-static limit of the DAF (long pulse -> DAF = 1),
(e) this module (>= 15 tests),
(f) fixtures use the generic names ``metric_alpha`` / ``metric_beta`` (no
    domain-specific vocabulary in the fixtures).

Negative controls: zero rate gives zero amplification; elastic amplitudes give
an amplitude-independent DAF with zero plastic strain rate; a rate-insensitive
material (D -> infinity) reproduces the rate-independent response.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.plasticity.cowper_symonds import (
    MILD_STEEL_D,
    MILD_STEEL_P,
    CowperSymonds,
    RateDependentJ2,
    analytic_uniaxial_flow_stress,
    uniaxial_tension_constant_rate,
)
from tensorphysics.plasticity.impact_daf import (
    BeamHingeFEM,
    ImpactCase,
    beam_clamped_static_moments,
    beam_first_mode_period,
    beam_modal_elastic_daf,
    half_sine_pulse,
    pulse_impulse,
    sdof_half_sine_daf,
    sdof_half_sine_daf_rk4,
    sdof_half_sine_response,
)

# ---------------------------------------------------------------------------
# generic fixtures (criterion f: no domain vocabulary in fixtures)
# ---------------------------------------------------------------------------
metric_alpha = 0.008  # plate thickness [m]
metric_beta = 0.010  # pulse duration [s]
metric_gamma = 1.0  # span [m]
metric_delta = 235.0e6  # first-yield stress [Pa]

RATE_FREE_D = 1.0e45  # stand-in for D -> infinity


@pytest.fixture(scope="module")
def material() -> CowperSymonds:
    return CowperSymonds()


@pytest.fixture(scope="module")
def unit_case() -> ImpactCase:
    base = ImpactCase(pressure_amplitude=1.0, pulse_duration=metric_beta,
                      thickness=metric_alpha, length=metric_gamma, yield_stress=metric_delta)
    return base


# ---------------------------------------------------------------------------
# (a) Cowper-Symonds law
# ---------------------------------------------------------------------------
def test_amplification_matches_closed_form(material):
    for rate in (1.0e-3, 0.1, 10.0, 100.0):
        expected = 1.0 + (rate / MILD_STEEL_D) ** (1.0 / MILD_STEEL_P)
        assert material.amplification(rate) == pytest.approx(expected, rel=1e-14)


def test_parameter_defaults_are_mild_steel(material):
    assert (material.D, material.p) == (40.4, 5.0)


def test_zero_rate_gives_exactly_unit_amplification(material):
    assert material.amplification(0.0) == 1.0
    assert material.dynamic_yield_stress(235.0e6, 0.0) == 235.0e6


def test_amplification_is_monotone_in_rate(material):
    rates = np.logspace(-6, 4, 40)
    amps = material.amplification(rates)
    assert np.all(np.diff(amps) > 0.0)


def test_analytic_inverse_round_trip(material):
    target = np.array([1.05, 1.5, 2.2, 3.0])
    assert np.allclose(material.amplification(material.rate_for_amplification(target)), target)


def test_negative_rate_rejected(material):
    with pytest.raises(ValueError):
        material.amplification(-1.0e-3)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        CowperSymonds(D=0.0)
    with pytest.raises(ValueError):
        CowperSymonds(p=-1.0)


def test_rate_threshold_plateau():
    cs = CowperSymonds(rate_threshold=1.0)
    assert cs.amplification(1.0) == pytest.approx(cs.amplification(1.0e6), rel=1e-14)
    assert cs.amplification(0.0) == 1.0


# ---------------------------------------------------------------------------
# (a) uniaxial tension vs the closed form
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("rate", [1.0e-3, 0.1, 10.0])
def test_uniaxial_perfectly_plastic_vs_analytic(rate, material):
    response = uniaxial_tension_constant_rate(rate, strain_max=0.02, n_steps=800, hardening=0.0)
    analytic = response.steady_state_analytic(material)
    assert abs(response.final_stress - analytic) / analytic <= 0.02


@pytest.mark.parametrize("rate", [1.0e-3, 0.1, 10.0])
def test_uniaxial_hardening_vs_steady_state(rate, material):
    hardening = 2.0e9
    response = uniaxial_tension_constant_rate(
        rate, strain_max=0.02, n_steps=800, hardening=hardening
    )
    analytic = response.steady_state_analytic(material)
    assert abs(response.final_stress - analytic) / analytic <= 0.02


def test_uniaxial_plastic_rate_equals_total_rate_when_perfectly_plastic(material):
    response = uniaxial_tension_constant_rate(1.0, strain_max=0.02, n_steps=600, hardening=0.0)
    assert response.final_plastic_strain_rate == pytest.approx(1.0, rel=1e-6)


def test_hardening_reduces_plastic_strain_rate(material):
    hardening = 2.0e9
    response = uniaxial_tension_constant_rate(1.0, strain_max=0.02, n_steps=800, hardening=hardening)
    expected = 1.0 / (1.0 + hardening / 210.0e9)
    assert response.final_plastic_strain_rate == pytest.approx(expected, rel=1e-2)


def test_return_map_elastic_branch_is_exact(material):
    model = RateDependentJ2(cs=material)
    stress_trial = 100.0e6  # below yield
    stress, eps_p, rate = model.return_map(stress_trial, 0.0, 1.0e-4)
    assert float(stress[0]) == pytest.approx(stress_trial, rel=1e-14)
    assert float(eps_p[0]) == 0.0
    assert float(rate[0]) == 0.0


# ---------------------------------------------------------------------------
# (c) rate-independent limit
# ---------------------------------------------------------------------------
def test_rate_independent_limit_literal_rate_is_over_threshold(material):
    """Documented honest limitation: classical CS amplifies > 1 % at 1e-4 s^-1."""
    deviation = abs(float(material.amplification(1.0e-4)) - 1.0)
    assert deviation > 0.01
    rate_needed = material.rate_for_target_amplification(1.01)
    assert abs(float(material.amplification(rate_needed)) - 1.0) <= 0.01 + 1e-12


def test_rate_independent_limit_d_infinity():
    cs = CowperSymonds(D=RATE_FREE_D)
    for rate in (1.0e-3, 1.0, 1.0e3):
        assert abs(float(cs.amplification(rate)) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# pulse shape and SDOF shock spectrum
# ---------------------------------------------------------------------------
def test_half_sine_pulse_shape():
    t = np.linspace(0.0, 2.0 * metric_beta, 401)
    pulse = half_sine_pulse(t, metric_beta, 3.0e6)
    assert np.all(pulse[t > metric_beta] == 0.0)
    assert pulse.max() == pytest.approx(3.0e6, rel=1e-3)
    assert pulse_impulse(metric_beta, 3.0e6) == pytest.approx(2.0 * 3.0e6 * metric_beta / np.pi)


def test_half_sine_pulse_rejects_bad_duration():
    with pytest.raises(ValueError):
        half_sine_pulse([0.0], 0.0)


@pytest.mark.parametrize("ratio", [0.1, 0.4, 0.8, 1.0, 2.0])
def test_sdof_closed_form_matches_rk4(ratio):
    exact = sdof_half_sine_daf(ratio)
    numeric = sdof_half_sine_daf_rk4(ratio, n_steps=12000)
    assert abs(numeric - exact) / exact <= 0.02


def test_sdof_peak_matches_textbook_value():
    ratios = np.linspace(0.2, 2.0, 200)
    peak = max(sdof_half_sine_daf(r) for r in ratios)
    assert peak == pytest.approx(1.77, rel=0.15)


def test_sdof_quasi_static_limit():
    assert abs(sdof_half_sine_daf(50.0) - 1.0) <= 0.02


def test_sdof_response_starts_at_zero():
    assert sdof_half_sine_response(np.array([0.0]), 0.5)[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# finite element plate strip
# ---------------------------------------------------------------------------
def test_first_period_clamped_stiffer_than_simply_supported():
    t_clamped = beam_first_mode_period(metric_gamma, 210.0e9, metric_alpha, 7850.0, 1.0, 0.3,
                                       "clamped")
    t_ss = beam_first_mode_period(metric_gamma, 210.0e9, metric_alpha, 7850.0, 1.0, 0.3,
                                  "simply_supported")
    assert t_clamped < t_ss


def test_fe_static_moment_matches_closed_form(unit_case):
    case = ImpactCase(pressure_amplitude=0.5 * unit_case.yield_pressure,
                      pulse_duration=metric_beta, thickness=metric_alpha,
                      length=metric_gamma, yield_stress=metric_delta)
    fem = BeamHingeFEM(case)
    closed = beam_clamped_static_moments(metric_gamma, case.pressure_amplitude * case.width)
    moments = fem.static_solution()["hinge_moments"]
    assert abs(moments[0]) == pytest.approx(closed["end"], rel=0.01)
    assert abs(moments[1]) == pytest.approx(closed["mid"], rel=0.01)


def test_fe_elastic_daf_matches_modal_reference(unit_case):
    case = ImpactCase(pressure_amplitude=0.5 * unit_case.yield_pressure,
                      pulse_duration=metric_beta, thickness=metric_alpha,
                      length=metric_gamma, yield_stress=metric_delta, n_elements=8)
    fem = BeamHingeFEM(case)
    t1 = beam_first_mode_period(metric_gamma, 210.0e9, metric_alpha, 7850.0, 1.0, 0.3, "clamped")
    result = fem.daf(dt=t1 / 120.0)
    ref = beam_modal_elastic_daf(metric_gamma, metric_alpha, case.pressure_amplitude,
                                 metric_beta, n_modes=9)
    assert abs(result["daf_global"] - ref["daf_global"]) / ref["daf_global"] <= 0.05


def test_negative_control_no_plasticity_below_yield(unit_case):
    case = ImpactCase(pressure_amplitude=0.3 * unit_case.yield_pressure,
                      pulse_duration=metric_beta, thickness=metric_alpha, n_elements=8)
    result = BeamHingeFEM(case).daf()
    assert result["max_plastic_strain_rate"] == 0.0
    assert max(result["final_plastic_strain"].values()) == 0.0


def test_negative_control_elastic_daf_amplitude_invariant(unit_case):
    dafs = []
    for ratio in (0.1, 0.25):
        case = ImpactCase(pressure_amplitude=ratio * unit_case.yield_pressure,
                          pulse_duration=metric_beta, thickness=metric_alpha, n_elements=8)
        dafs.append(BeamHingeFEM(case).daf()["daf_global"])
    assert abs(dafs[0] - dafs[1]) / dafs[0] <= 0.01


def test_negative_control_rate_insensitive_material_matches_elastic(unit_case):
    amplitude = 0.8 * unit_case.yield_pressure
    common = dict(pulse_duration=0.030, thickness=metric_alpha, n_elements=8)
    rate_free = CowperSymonds(D=RATE_FREE_D)
    cs_result = BeamHingeFEM(ImpactCase(pressure_amplitude=amplitude, **common)).daf()
    ri_result = BeamHingeFEM(
        ImpactCase(pressure_amplitude=amplitude, cs=rate_free, **common)
    ).daf()
    assert cs_result["daf_global"] >= ri_result["daf_global"] - 1e-9
    assert cs_result["max_plastic_strain_rate"] > ri_result["max_plastic_strain_rate"]


def test_rate_effect_increases_dynamic_stress(unit_case):
    amplitude = 0.9 * unit_case.yield_pressure
    common = dict(pulse_duration=0.020, thickness=metric_alpha, n_elements=8)
    cs_result = BeamHingeFEM(ImpactCase(pressure_amplitude=amplitude, **common)).daf()
    ri_result = BeamHingeFEM(
        ImpactCase(pressure_amplitude=amplitude, cs=CowperSymonds(D=RATE_FREE_D), **common)
    ).daf()
    assert cs_result["daf_global"] > ri_result["daf_global"]


def test_quasi_static_limit_of_daf(unit_case):
    case = ImpactCase(pressure_amplitude=0.5 * unit_case.yield_pressure,
                      pulse_duration=0.5, thickness=metric_alpha, n_elements=8)
    t1 = beam_first_mode_period(metric_gamma, 210.0e9, metric_alpha, 7850.0, 1.0, 0.3, "clamped")
    result = BeamHingeFEM(case).daf(dt=t1 / 40.0)
    assert abs(result["daf_global"] - 1.0) <= 0.02


def test_invalid_case_configuration_rejected():
    with pytest.raises(ValueError):
        ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, n_elements=7)
    with pytest.raises(ValueError):
        ImpactCase(pressure_amplitude=1.0, pulse_duration=0.01, support="free")


def test_analytic_flow_stress_helper(material):
    value = analytic_uniaxial_flow_stress(
        eps_p=0.0, total_strain_rate=1.0, sigma_0=235.0e6, hardening=0.0,
        youngs_modulus=210.0e9, cs=material,
    )
    assert value == pytest.approx(235.0e6 * float(material.amplification(1.0)), rel=1e-12)