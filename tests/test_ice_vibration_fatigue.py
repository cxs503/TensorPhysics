"""Verification suite: ice-induced vibration lock-in + ice-pressure low-cycle fatigue.

Covers the acceptance criteria of node P180W2-N07:

* C1 -- lock-in band and amplification (``test_lock_in_band_amplification``,
  ``test_entrainment_frequency_equals_natural_frequency``)
* C2 -- amplification cross-check (describing function vs RK4 time marching,
  ``test_describing_function_matches_time_domain``); the Frederking (1984)
  experimental anchor is *not* reproducible offline and is graded WITHHELD in
  ``benchmarks/ice_vibration_fatigue`` -- no test here pretends otherwise.
* C3 -- epsilon-N curve consistency (``test_strain_life_curve_branches_sum``,
  ``test_life_inversion_roundtrip``)
* C4 -- Neuber notch correction (``test_neuber_identity_and_elastic_limit``,
  ``test_glinka_vs_neuber_small_scale_yield``)
* C5 -- LCF/HCF transition life (``test_transition_life_equal_branches``,
  ``test_ice_pressure_chain_hcf_vs_lcf_transition``)
* C6 -- three negative controls (``test_no_lock_in_outside_band``,
  ``test_zero_amplitude_low_strain_runs_out``,
  ``test_notch_factor_tends_to_one_for_large_radius``)
* C7 -- Miner accumulation (``test_miner_damage_scales_with_exposure``,
  ``test_allowable_hours_reaches_damage_one``)
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fatigue.ice_pressure_lowcycle import (
    RAYLEIGH_COV,
    cycles_from_exposure,
    ice_pressure_low_cycle_assessment,
    nominal_stress_range_MPa,
    pulse_amplitude_blocks,
    weibull_shape_from_cov,
)
from tensorphysics.fatigue.strain_life import (
    DamageBlock,
    StrainLifeMaterial,
    cycles_to_failure,
    cyclic_strain_amplitude,
    elastic_strain_amplitude,
    endurance_limit_strain_amplitude,
    glinka_notch_strain,
    k_prime_from_coefficients,
    neuber_notch_strain,
    notch_factor,
    notch_factor_neuber,
    notch_factor_peterson,
    notch_factor_from_geometry,
    plastic_strain_amplitude,
    reversals_for_strain_amplitude,
    strain_life_damage,
    stress_concentration_elliptical_hole,
    stress_amplitude_from_strain,
    total_strain_amplitude,
    transition_life,
    transition_reversals,
)
from tensorphysics.ice_vibration.lockin_coupling import (
    LockInCase,
    ice_force_velocity_slope,
    ice_negative_damping,
    limit_cycle_amplitude,
    lock_in_amplification_check,
    lock_in_state,
    self_excited_speed_window,
    time_domain_limit_cycle,
)


# ---------------------------------------------------------------------------
# Fixtures / cases
# ---------------------------------------------------------------------------


def steel() -> StrainLifeMaterial:
    """Representative structural-steel strain-life constants (dimensionally
    consistent with the cyclic curve through the Morrow compatibility rule)."""
    return StrainLifeMaterial(
        name="structural steel (representative)",
        E_MPa=206000.0,
        sigma_f_MPa=1000.0,
        b=-0.10,
        eps_f=0.5,
        c=-0.50,
        notch_constant_mm=0.5,
    )


def nominal_lockin_case() -> LockInCase:
    """Ice-loaded 1 Hz mode with a velocity-weakening ice force."""
    return LockInCase(
        name="nominal ice lock-in case",
        natural_frequency_hz=1.0,
        modal_mass_kg=1.0e6,
        damping_ratio=0.02,
        ice_force_max_n=1.0e5,
        ice_critical_velocity_m_s=0.05,
        flaking_length_m=0.05,
        velocity_exponent=2.0,
    )


# ---------------------------------------------------------------------------
# C3 / C5: strain-life curve
# ---------------------------------------------------------------------------


def test_cyclic_parameters_derived_from_strain_life_constants():
    mat = steel()
    assert mat.n_prime == pytest.approx(mat.b / mat.c, rel=1e-12)
    assert mat.K_prime_MPa == pytest.approx(
        k_prime_from_coefficients(mat.sigma_f_MPa, mat.eps_f, mat.n_prime), rel=1e-12
    )
    # Morrow compatibility: the elastic and plastic lines cross the cyclic
    # curve consistently, because K' was derived from them.
    assert mat.K_prime_MPa == pytest.approx(
        mat.sigma_f_MPa / mat.eps_f ** mat.n_prime, rel=1e-12
    )


def test_strain_life_curve_branches_sum():
    mat = steel()
    for rev in (1.0e2, 1.0e3, 1.0e4, 1.0e5, 1.0e6):
        total = total_strain_amplitude(mat, rev)
        parts = elastic_strain_amplitude(mat, rev) + plastic_strain_amplitude(mat, rev)
        assert total == pytest.approx(parts, rel=1e-15)
    # monotone decreasing in life
    amps = [total_strain_amplitude(mat, r) for r in np.logspace(2, 7, 11)]
    assert all(a > b for a, b in zip(amps[:-1], amps[1:]))


def test_life_inversion_roundtrip():
    mat = steel()
    for rev in (2.0e3, 1.0e4, 5.0e4, 1.0e5, 3.0e6):
        amp = total_strain_amplitude(mat, rev)
        recovered = reversals_for_strain_amplitude(mat, amp)
        assert recovered == pytest.approx(rev, rel=1e-9)
        assert cycles_to_failure(mat, amp) == pytest.approx(rev / 2.0, rel=1e-9)


def test_transition_life_equal_branches_and_analytic():
    mat = steel()
    rev_t = transition_reversals(mat)
    assert elastic_strain_amplitude(mat, rev_t) == pytest.approx(
        plastic_strain_amplitude(mat, rev_t), rel=1e-10
    )
    analytic = (mat.eps_f * mat.E_MPa / mat.sigma_f_MPa) ** (1.0 / (mat.b - mat.c))
    assert rev_t == pytest.approx(analytic, rel=1e-12)
    assert transition_life(mat) == pytest.approx(rev_t / 2.0, rel=1e-12)
    # regime flip exactly at N_T
    assert total_strain_amplitude(mat, rev_t * 0.5) > 2.0 * elastic_strain_amplitude(
        mat, rev_t * 0.5
    )
    assert total_strain_amplitude(mat, rev_t * 10.0) < 1.5 * elastic_strain_amplitude(
        mat, rev_t * 10.0
    )


def test_morrow_mean_stress_shortens_life():
    mat = steel()
    amp = 2.0e-3
    n0 = cycles_to_failure(mat, amp, 0.0)
    n1 = cycles_to_failure(mat, amp, 100.0)
    n2 = cycles_to_failure(mat, amp, 200.0)
    assert n0 > n1 > n2


def test_zero_amplitude_low_strain_runs_out():
    """C6 (negative control): strain amplitude -> 0 gives life -> infinity."""
    mat = steel()
    runout = endurance_limit_strain_amplitude(mat)
    small = strain_life_damage(mat, [DamageBlock(strain_amplitude=runout / 2.0, cycles=1.0e6)])
    assert small.damage_total == 0.0
    assert small.blocks[0]["N_f"] is None  # run-out: infinite life
    finite = strain_life_damage(mat, [DamageBlock(strain_amplitude=2.0e-3, cycles=1.0e6)])
    assert finite.damage_total > 0.0
    assert finite.blocks[0]["N_f"] is not None


# ---------------------------------------------------------------------------
# C4: Neuber notch correction
# ---------------------------------------------------------------------------


def test_neuber_identity_and_elastic_limit():
    mat = steel()
    for nominal in (5.0, 20.0, 60.0, 150.0, 300.0):
        res = neuber_notch_strain(mat, 3.0, nominal)
        target = (3.0 * nominal) ** 2 / mat.E_MPa
        assert res.notch_stress_range_MPa * res.notch_strain_range == pytest.approx(
            target, rel=1e-9
        )
        assert res.K_sigma * res.K_eps == pytest.approx(9.0, rel=1e-9)
        assert res.neuber_identity_error < 1e-9
        assert res.notch_strain_range >= nominal / mat.E_MPa * 3.0 * (1.0 - 1e-9)
    # small-load limit collapses to the elastic notch strain K_t dS / E
    res = neuber_notch_strain(mat, 3.0, 1.0)
    assert res.notch_strain_range == pytest.approx(3.0 * 1.0 / mat.E_MPa, rel=1e-4)
    assert res.plastic_fraction < 1e-3


def test_neuber_kt_one_matches_elastic_limit():
    """K_t = 1 with a vanishing nominal stress collapses to the elastic notch strain.

    Note the closure is *not* the cyclic curve for finite ``dS``: the Neuber
    hyperbola always enforces ``dsigma deps = (K_t dS)^2/E``, which for K_t = 1
    is only compatible with an elastic response.  The limit case is therefore
    checked at small load, where the plastic term vanishes.
    """
    mat = steel()
    for nominal in (0.2, 1.0):
        res = neuber_notch_strain(mat, 1.0, nominal)
        assert res.notch_strain_range == pytest.approx(nominal / mat.E_MPa, rel=1e-4)
        assert res.plastic_fraction < 1e-4
    with pytest.raises(ValueError):
        neuber_notch_strain(mat, 0.9, 100.0)


def test_glinka_vs_neuber_small_scale_yield():
    mat = steel()
    # elastic / small-scale yielding: both closures agree
    for nominal in (5.0, 20.0, 60.0):
        n_ = neuber_notch_strain(mat, 3.0, nominal)
        g = glinka_notch_strain(mat, 3.0, nominal)
        assert abs(n_.notch_strain_range - g.notch_strain_range) / g.notch_strain_range < 0.01
    # large-scale yielding: Neuber is the conservative (larger strain) closure
    n_ = neuber_notch_strain(mat, 3.0, 600.0)
    g = glinka_notch_strain(mat, 3.0, 600.0)
    assert n_.notch_strain_range > g.notch_strain_range
    assert n_.plastic_fraction > 0.5


def test_notch_factor_tends_to_one_for_large_radius():
    """C6 (negative control): notch radius -> infinity gives K_t -> 1 and K_f -> 1."""
    K_t = 3.0
    # fixed K_t: the notch sensitivity (not K_f) decreases as the root sharpens
    sensitivities = [notch_factor_neuber(K_t, r, 0.5) for r in (1.0, 2.0, 10.0, 100.0)]
    assert all(a < b for a, b in zip(sensitivities[:-1], sensitivities[1:]))
    assert sensitivities[0] == pytest.approx(1.0 + 2.0 / (1.0 + math.sqrt(0.5)), rel=1e-12)
    # geometry driven: blunt notch -> K_t -> 1 -> K_f -> 1, q -> 1
    depth = 5.0
    geom = [notch_factor_from_geometry(depth, r, 0.5) for r in (0.5, 2.0, 20.0, 5.0e3, 5.0e7)]
    assert all(a["K_t"] > b["K_t"] for a, b in zip(geom[:-1], geom[1:]))
    assert all(a["K_f"] > b["K_f"] for a, b in zip(geom[:-1], geom[1:]))
    assert geom[-1]["K_f"] == pytest.approx(1.0, abs=1e-3)
    assert geom[0]["q"] < geom[-1]["q"] < 1.0 + 1e-3
    assert stress_concentration_elliptical_hole(depth, 0.5) == pytest.approx(
        1.0 + 2.0 * math.sqrt(depth / 0.5), rel=1e-12
    )
    peterson = notch_factor(K_t, 2.0, 0.5, method="peterson")
    assert peterson == pytest.approx(1.0 + 2.0 / (1.0 + 0.5 / 2.0), rel=1e-12)
    assert notch_factor(K_t, 2.0, material=steel()) == pytest.approx(
        notch_factor_neuber(K_t, 2.0, 0.5), rel=1e-12
    )
    assert notch_factor_peterson(K_t, 100.0, 0.5) > notch_factor_peterson(K_t, 1.0, 0.5)
    with pytest.raises(ValueError):
        notch_factor(K_t, 2.0, method="unknown")


# ---------------------------------------------------------------------------
# C1 / C2 / C6: lock-in physics
# ---------------------------------------------------------------------------


def test_ice_negative_damping_small_amplitude_matches_slope():
    case = nominal_lockin_case()
    u = 0.05
    slope = ice_force_velocity_slope(
        u, case.ice_force_max_n, case.ice_critical_velocity_m_s, case.velocity_exponent
    )
    assert slope < 0.0
    damping = ice_negative_damping(case, u, 1e-9 * case.drift_speed_for_ratio(1.0))
    assert damping == pytest.approx(abs(slope), rel=1e-6)
    # saturation reduces the ice damping at finite amplitude
    assert ice_negative_damping(case, u, 0.02) < damping
    assert ice_negative_damping(case, u, 0.0) == pytest.approx(abs(slope), rel=1e-12)


def test_self_excited_window_bounds():
    case = nominal_lockin_case()
    window = self_excited_speed_window(case)
    assert window is not None
    lo, hi = window
    c_s = case.structural_damping_ns_m
    assert lo < hi
    assert ice_negative_damping(case, 0.5 * (lo + hi), 0.0) > c_s
    assert ice_negative_damping(case, lo * 0.5, 0.0) < c_s
    assert ice_negative_damping(case, hi * 2.0, 0.0) < c_s
    # no excitation at all for an overdamped structure
    soft = LockInCase(
        name="highly damped",
        natural_frequency_hz=1.0,
        modal_mass_kg=1.0e6,
        damping_ratio=0.5,
        ice_force_max_n=1.0e3,
        ice_critical_velocity_m_s=0.05,
        flaking_length_m=0.05,
    )
    assert self_excited_speed_window(soft) is None
    assert limit_cycle_amplitude(soft, 0.05) is None
    assert lock_in_state(soft, 0.05).locked_in is False


def test_lock_in_band_amplification():
    """C1: inside 0.8 <= f_ice/f_n <= 1.2 the response locks in and amplifies > 3x."""
    case = nominal_lockin_case()
    check = lock_in_amplification_check(case, amplification_threshold=3.0)
    assert check["all_ratios_locked"] is True
    assert check["min_amplification"] > 3.0
    assert check["pass"] is True
    for row in check["rows"]:
        assert 0.8 - 1e-12 <= row["frequency_ratio"] <= 1.2 + 1e-12
        assert row["locked_in"] is True
        assert row["amplification"] > 3.0
        assert row["ice_damping_ns_m"] > row["structural_damping_ns_m"]
    # the entrained response is far flatter in detuning than a linear resonance
    assert check["entrainment_insensitivity"] < 0.5


def test_no_lock_in_outside_band():
    """C6 (negative control): ice frequency far from f_n gives no lock-in."""
    case = nominal_lockin_case()
    for ratio in (0.2, 0.3, 0.5, 2.0, 3.0):
        state = lock_in_state(case, case.drift_speed_for_ratio(ratio))
        assert state.locked_in is False
        assert state.frequency_ratio == pytest.approx(ratio, rel=1e-12)
        assert state.amplification < 3.0
        assert "no entrainment" in state.reason


def test_describing_function_matches_time_domain():
    """C2 (cross-method): describing function vs RK4 time marching."""
    case = nominal_lockin_case()
    for ratio in (0.8, 1.2):
        u = case.drift_speed_for_ratio(ratio)
        a_df = limit_cycle_amplitude(case, u)
        td = time_domain_limit_cycle(case, u, n_cycles=250, steps_per_period=200)
        assert a_df == pytest.approx(td["amplitude_m"], rel=0.30)
        assert a_df == pytest.approx(td["amplitude_m"], rel=0.02)


def test_entrainment_frequency_equals_natural_frequency():
    """C1 (discriminating signature): the response frequency is f_n, not f_ice."""
    case = nominal_lockin_case()
    freqs = {}
    for ratio in (0.8, 1.0, 1.2):
        u = case.drift_speed_for_ratio(ratio)
        td = time_domain_limit_cycle(case, u, n_cycles=200, steps_per_period=200)
        freqs[ratio] = td["frequency_hz"]
        assert td["frequency_hz"] == pytest.approx(case.natural_frequency_hz, rel=0.02)
    # the three forcing frequencies differ by 40 %, the response by < 0.5 %
    assert max(freqs.values()) / min(freqs.values()) < 1.005


# ---------------------------------------------------------------------------
# C5 / C7: ice-pressure load spectrum -> damage
# ---------------------------------------------------------------------------


def test_nominal_stress_range_analytic():
    value = nominal_stress_range_MPa(3.0e6, 1.2, 0.006)
    assert value == pytest.approx(3.0e6 * 1.2 / 4.0 / 0.006 / 1.0e6, rel=1e-12)
    assert cycles_from_exposure(1.0, 10.0) == pytest.approx(36000.0)


def test_pulse_spectrum_properties():
    assert weibull_shape_from_cov(RAYLEIGH_COV) == pytest.approx(2.0, rel=1e-3)
    blocks = pulse_amplitude_blocks(600.0, RAYLEIGH_COV, 6)
    multipliers = [m for m, _ in blocks]
    assert all(a < b for a, b in zip(multipliers[:-1], multipliers[1:]))
    assert sum(multipliers) / len(multipliers) == pytest.approx(1.0, rel=0.02)
    assert sum(n for _, n in blocks) == pytest.approx(600.0, rel=1e-12)


def test_ice_pressure_chain_hcf_vs_lcf_transition():
    """C5: the same lock-in load with a weaker detail crosses N_T into LCF."""
    case = nominal_lockin_case()
    state = lock_in_state(case, case.drift_speed_for_ratio(1.0))
    mat = steel()
    mild = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, 10.0, 1.2, 0.006,
        K_t=3.0, notch_radius_mm=2.0,
    )
    severe = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, 10.0, 1.2, 0.003,
        K_t=3.0, notch_radius_mm=2.0,
    )
    assert mild.primary_regime == "HCF"
    assert mild.primary_N_f > mild.transition_life
    assert mild.damage.damage_hcf > mild.damage.damage_lcf
    assert severe.primary_regime == "LCF"
    assert severe.primary_N_f < severe.transition_life
    assert severe.damage.damage_lcf > severe.damage.damage_hcf
    assert severe.nominal_stress_range_MPa == pytest.approx(
        2.0 * mild.nominal_stress_range_MPa, rel=1e-12
    )
    assert severe.damage.damage_total > mild.damage.damage_total
    # notch strain amplitude must exceed the nominal elastic strain amplitude
    assert mild.notch["strain_amplitude"] > mild.nominal_stress_range_MPa / mat.E_MPa / 2.0


def test_miner_damage_scales_with_exposure():
    """C7: Miner's sum is linear in exposure for a constant amplitude block."""
    case = nominal_lockin_case()
    state = lock_in_state(case, case.drift_speed_for_ratio(1.0))
    mat = steel()
    short = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, 10.0, 1.2, 0.003,
        K_t=3.0, notch_radius_mm=2.0,
    )
    long = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, 20.0, 1.2, 0.003,
        K_t=3.0, notch_radius_mm=2.0,
    )
    assert long.damage.damage_total == pytest.approx(2.0 * short.damage.damage_total, rel=1e-12)
    assert long.cycles == pytest.approx(2.0 * short.cycles)


def test_allowable_hours_reaches_damage_one():
    """C7: the design exposure for D = 1.0 is recovered exactly."""
    case = nominal_lockin_case()
    state = lock_in_state(case, case.drift_speed_for_ratio(1.0))
    mat = steel()
    ref = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, 10.0, 1.2, 0.003,
        K_t=3.0, notch_radius_mm=2.0,
    )
    design = ice_pressure_low_cycle_assessment(
        mat, state.force_range_n, case.natural_frequency_hz, ref.allowable_hours,
        1.2, 0.003, K_t=3.0, notch_radius_mm=2.0,
    )
    assert design.damage.damage_total == pytest.approx(1.0, rel=1e-9)
    assert design.damage.damage_total <= 1.0 + 1e-9


def test_zero_load_gives_zero_damage():
    """C6 (negative control): vanishing load range -> no damage, infinite life."""
    mat = steel()
    tiny = ice_pressure_low_cycle_assessment(
        mat, 1.0, 1.0, 10.0, 1.2, 0.006, K_t=2.0, notch_radius_mm=2.0
    )
    assert tiny.damage.damage_total == 0.0
    assert math.isinf(tiny.allowable_hours)
    assert all(row["regime"] == "run-out" for row in tiny.damage.blocks)


def test_strain_life_damage_matches_analytic_miner():
    mat = steel()
    amp = 3.0e-3
    n_f = cycles_to_failure(mat, amp)
    res = strain_life_damage(mat, [DamageBlock(strain_amplitude=amp, cycles=n_f / 4.0)])
    assert res.damage_total == pytest.approx(0.25, rel=1e-9)
    by_regime = "LCF" if n_f < transition_life(mat) else "HCF"
    assert res.blocks[0]["regime"] == by_regime


def test_stress_range_inverse_roundtrip():
    mat = steel()
    for sigma_a in (50.0, 150.0, 300.0, 450.0):
        eps_a = cyclic_strain_amplitude(mat, sigma_a)
        assert stress_amplitude_from_strain(mat, eps_a) == pytest.approx(sigma_a, rel=1e-9)