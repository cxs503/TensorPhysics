"""Tests for the W7-N07 propeller blade ice-hydrodynamic fatigue package.

Coverage: S-N fitting, mean-stress correction, unsteady airfoil theory
(Theodorsen / Sears / Kuessner), beam impulse response against closed form,
finite-difference PDE solution, cycle counting, Miner damage, ice load model and
the full stress-history chain.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.propeller_fatigue.beam_impact import (
    AnalyticImpulse,
    Beam,
    finite_difference_beam,
    numerical_impulse,
)
from tensorphysics.propeller_fatigue.blade_stress import BladeGeometry, root_stress_components
from tensorphysics.propeller_fatigue.hydro_unsteady import (
    WakeField,
    blade_element_hydro,
    harmonic_amplitude,
    harmonic_amplitude_angle,
    kussner_gust_response,
    sears_gust_function,
    sears_root_moment,
    theodorsen_function,
)
from tensorphysics.propeller_fatigue.ice_spectrum import (
    ICE_CASES,
    blade_impact_train,
    ice_milling_torque,
    impact_spectrum,
)
from tensorphysics.propeller_fatigue.materials_sn import (
    MATERIALS,
    BasquinSN,
    SNReference,
    goodman_amplitude,
)
from tensorphysics.propeller_fatigue.rainflow_miner import (
    aggregate_realisations,
    initiation_life,
    miner_damage,
    rainflow_count,
    safety_factor,
    turning_points,
)

NAB = MATERIALS["NAB"]


@pytest.fixture(scope="module")
def geom():
    return BladeGeometry()


@pytest.fixture(scope="module")
def sn_nab():
    return BasquinSN.fit(SNReference("NAB"))


@pytest.fixture(scope="module")
def beam():
    """Equivalent flatwise blade span (simply supported anchor configuration)."""
    return Beam(length=1.75, E=118.0e9, I=1.5e-4, mass_per_length=990.0,
                bc="simply_supported", n_modes=16, outer_fibre_distance=0.076)


# --------------------------------------------------------------------------- #
#  S-N curve and mean-stress correction
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("material", ["NAB", "AL6082"])
def test_sn_fit_matches_reference_points(material):
    """Criterion (d): Basquin fit within 10 % of every reference point."""
    ref = SNReference(material)
    sn = BasquinSN.fit(ref)
    err = sn.fit_error(ref)
    assert err["max_abs_rel"] <= 0.10, err
    assert sn.b < 0.0


def test_sn_law_is_invertible(sn_nab):
    for N in (1.0e4, 1.0e6, 1.0e8):
        sigma = sn_nab.sigma_a(N)
        assert sn_nab.cycles(sigma) == pytest.approx(N, rel=1e-9)


def test_sn_exponent_is_physics_plausible(sn_nab):
    """Inverse Basquin slope between 6 and 14 decades^-1 for NAB."""
    assert 6.0 <= sn_nab.exponent_sigma_range() <= 14.0


def test_goodman_correction():
    assert goodman_amplitude(100.0e6, 0.0, 650.0e6) == pytest.approx(100.0e6)
    assert goodman_amplitude(100.0e6, 325.0e6, 650.0e6) == pytest.approx(200.0e6)
    # tensile mean stress always increases the equivalent amplitude
    assert goodman_amplitude(80.0e6, 200.0e6, 650.0e6) > 80.0e6


# --------------------------------------------------------------------------- #
#  Unsteady airfoil theory
# --------------------------------------------------------------------------- #
def test_theodorsen_limits():
    assert abs(theodorsen_function(0.0) - 1.0) < 1e-12
    assert abs(theodorsen_function(1e-4) - 1.0) < 0.01
    assert abs(abs(theodorsen_function(200.0)) - 0.5) < 0.01


def test_sears_limits_and_decay():
    assert abs(sears_gust_function(0.0) - 1.0) < 1e-12
    prev = 1.0
    for k in (0.1, 0.5, 1.0, 2.0, 5.0, 10.0):
        val = abs(sears_gust_function(k))
        assert val < prev
        prev = val
    # high-frequency asymptotic decay 1/sqrt(2 pi k)
    assert abs(sears_gust_function(20.0)) / (1 / np.sqrt(2 * np.pi * 20.0)) == pytest.approx(1.0, rel=0.05)


@pytest.mark.parametrize("k", [0.2, 0.3, 0.5])
def test_sears_time_domain_matches_frequency_domain(k):
    """Kuessner indicial integration reproduces the Sears amplitude."""
    V, c = 20.0, 1.0
    f = k * V / (np.pi * c)
    t = np.arange(0.0, 40.0 / f, 1.0 / (f * 300.0))
    CL = kussner_gust_response(t, np.cos(2 * np.pi * f * t), V, c)
    n_settle = int(0.6 * t.size)
    amp = abs(harmonic_amplitude(CL[n_settle:], t[n_settle:], f)) * V
    ref = abs(sears_gust_function(k))
    assert abs(amp - ref) / ref < 0.12


def test_blade_element_unsteady_reduces_alternating_load(geom):
    """The unsteady correction must reduce the quasi-steady gust response."""
    st = geom.stations()
    st.span_factor = 1.0
    qs = blade_element_hydro(st, 5.85, 130.0 / 60.0, WakeField(), unsteady=False)
    un = blade_element_hydro(st, 5.85, 130.0 / 60.0, WakeField(), unsteady=True)
    amp_qs = abs(harmonic_amplitude_angle(qs["moment"][0], qs["theta"], 1))
    amp_un = abs(harmonic_amplitude_angle(un["moment"][0], un["theta"], 1))
    assert amp_un < amp_qs


def test_hydro_pipeline_vs_sears_reference_two_conditions(geom):
    """Criterion (b): blade-element unsteady load vs analytic Sears <= 15 %, 2 OPs."""
    st = geom.stations()
    st.span_factor = 1.0
    wake = WakeField()
    for n_rps, J in ((130.0 / 60.0, 0.60), (110.0 / 60.0, 0.45)):
        V = J * n_rps * geom.D
        un = blade_element_hydro(st, V, n_rps, wake, unsteady=True)
        amp_bem = abs(harmonic_amplitude_angle(un["moment"][0], un["theta"], 1))
        amp_ref = sears_root_moment(st, V, n_rps, wake, harmonic=1)["moment_amplitude"]
        assert abs(amp_bem - amp_ref) / amp_ref <= 0.15


# --------------------------------------------------------------------------- #
#  Beam impulse response
# --------------------------------------------------------------------------- #
def test_beam_modes_and_frequency(beam):
    assert beam.fundamental_frequency == pytest.approx(68.6, rel=0.02)
    # modal mass of a simply supported sine mode is m L / 2
    assert beam.modal_mass[0] == pytest.approx(990.0 * 1.75 / 2.0)


def test_beam_static_stress_matches_section_formula(beam):
    """Modal static solution must equal M/W = (P L/4)/(I/c)."""
    P, L = 3.46e5, 1.75
    sigma_modal = beam.static_stress(0.0, P, L / 2, L / 2)
    sigma_hand = (P * L / 4.0) / (beam.I / beam.outer_fibre_distance)
    # the curvature series converges as 1/N modes, so a few % is expected here
    assert sigma_modal == pytest.approx(sigma_hand, rel=0.03)


def test_beam_analytic_matches_numerical_integration(beam):
    """Criterion (a), solver check: closed form vs RK4 state-space integration."""
    P, tau = 3.46e5, 0.015
    analytic = AnalyticImpulse(beam, P, tau, beam.length / 2, beam.length / 2).peak_stress()
    _, sig = numerical_impulse(beam, P, tau, duration=0.05)
    assert abs(float(sig.max()) - analytic) / analytic <= 0.02


def test_beam_analytic_matches_finite_difference(beam):
    """Criterion (a): independent PDE discretisation within 5 %."""
    P, tau = 3.46e5, 0.015
    analytic = AnalyticImpulse(beam, P, tau, beam.length / 2, beam.length / 2).peak_stress()
    _, sig = finite_difference_beam(beam.length, beam.E, beam.I, beam.mass_per_length,
                                    P, tau, beam.outer_fibre_distance, n_nodes=61,
                                    duration=0.05)
    assert abs(float(sig.max()) - analytic) / analytic <= 0.05


def test_beam_dynamic_load_factor_limits(beam):
    """DLF -> 1 for slow pulses, -> 0 for impulsive ones, peak near tau ~ T1."""
    P = 3.46e5
    mid = beam.length / 2
    slow = AnalyticImpulse(beam, P, 20.0 * beam.fundamental_period, mid, mid).dynamic_load_factor()
    fast = AnalyticImpulse(beam, P, 0.10 * beam.fundamental_period, mid, mid).dynamic_load_factor()
    peak = AnalyticImpulse(beam, P, 1.00 * beam.fundamental_period, mid, mid).dynamic_load_factor()
    assert abs(slow - 1.0) < 0.05
    assert fast < 0.6
    assert 1.5 < peak < 1.7


# --------------------------------------------------------------------------- #
#  Cycle counting and Miner damage
# --------------------------------------------------------------------------- #
def test_turning_points_keeps_extrema():
    x = np.array([0.0, 1.0, 2.0, 1.0, 0.0, 3.0, 2.0])
    tp = turning_points(x)
    assert tp[0] == 0.0 and tp[-1] == 2.0
    assert tp.max() == 3.0 and len(tp) == 5


def test_rainflow_constant_amplitude():
    """A stationary sine of amplitude A yields full cycles of range 2A."""
    A, n, per = 100.0, 6, 200
    x = A * np.cos(2 * np.pi * np.arange(0, n * per) / per)
    cyc = rainflow_count(x)
    full = cyc["count"] == 1.0
    assert np.allclose(cyc["range"][full], 2 * A, atol=1e-6)
    # the last (unclosed) period of an open record is counted as half cycles
    assert abs(int(full.sum()) - n) <= 1


def test_rainflow_damage_equivalent_to_constant_amplitude(sn_nab):
    """Rainflow damage of a sine record equals n/N(A) for the same cycles."""
    A, n, per = 120.0e6, 40, 100
    x = A * np.cos(2 * np.pi * np.arange(0, n * per) / per)
    cyc = rainflow_count(x)
    D = miner_damage(cyc, sn_nab, NAB.sigma_uts)["D"]
    D_ref = n / sn_nab.cycles(A)
    assert abs(D - D_ref) / D_ref < 0.15


def test_rainflow_nested_cycles():
    """Classic nested sequence: one large cycle plus one small one."""
    x = np.array([0.0, 3.0, 0.0, 1.0, 0.0])
    cyc = rainflow_count(x)
    ranges = np.sort(cyc["range"][cyc["count"] == 1.0])
    assert np.allclose(ranges, [1.0, 3.0])


def test_miner_damage_is_linear_in_counts(sn_nab):
    hist = np.concatenate([[0.0], np.tile([120.0e6, -120.0e6], 5), [0.0]])
    cyc = rainflow_count(hist)
    d1 = miner_damage(cyc, sn_nab, NAB.sigma_uts)["D"]
    d2 = miner_damage(cyc, sn_nab, NAB.sigma_uts, scale=2.0)["D"]
    assert d2 == pytest.approx(2.0 * d1, rel=1e-9)


def test_safety_factor_and_life_are_inverse_damage():
    assert safety_factor(0.25) == pytest.approx(4.0)
    assert initiation_life(0.25, 1000.0) == pytest.approx(4000.0)
    assert safety_factor(0.0) == float("inf")


def test_damage_grows_with_stress_amplitude(sn_nab):
    low = rainflow_count(np.concatenate([[0.0], np.tile([80.0e6, -80.0e6], 4), [0.0]]))
    high = rainflow_count(np.concatenate([[0.0], np.tile([140.0e6, -140.0e6], 4), [0.0]]))
    d_low = miner_damage(low, sn_nab, NAB.sigma_uts)["D"]
    d_high = miner_damage(high, sn_nab, NAB.sigma_uts)["D"]
    assert d_high > 100.0 * d_low


def test_aggregate_realisations_reports_sem():
    mean, sem = aggregate_realisations([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert sem == pytest.approx(0.577, rel=0.01)


# --------------------------------------------------------------------------- #
#  Ice load model
# --------------------------------------------------------------------------- #
def test_milling_torque_scales_with_crushing_strength(geom):
    env = ICE_CASES["design"]
    n = 130.0 / 60.0
    q1 = ice_milling_torque(geom, env, n)["Q_mill_Nm"]
    q2 = ice_milling_torque(geom, env, n, sigma_ice=2.0 * env.sigma_ice)["Q_mill_Nm"]
    assert q2 == pytest.approx(2.0 * q1, rel=1e-9)
    # depth of cut per blade pass = v / (n Z)
    out = ice_milling_torque(geom, env, n)
    assert out["depth_of_cut_m"] == pytest.approx(env.advance_speed / (n * geom.Z))


def test_impact_spectrum_percentiles():
    env = ICE_CASES["design"]
    spec = impact_spectrum(env, Q_mill=2.5e5, Q_design=4.4e5, n_samples=20000, seed=3)
    assert spec["p99"] == pytest.approx(spec["Q_max_envelope_Nm"], rel=0.10)
    assert spec["p50"] == pytest.approx(2.5e5, rel=0.10)
    assert spec["k_ice_envelope"] == pytest.approx(env.k_torque_max, rel=1e-9)


def test_impact_train_is_phase_locked(geom):
    """Every impact hits the blade that is inside the ice contact sector."""
    env = ICE_CASES["heavy"]
    spec = impact_spectrum(env, 2.5e5, 4.4e5, seed=5)
    train = blade_impact_train(geom, env, 130.0 / 60.0, 30.0, spec, dt=1e-3, seed=7)
    omega = 2 * np.pi * 130.0 / 60.0
    for t0, b in zip(train["t_on"], train["blade_index"]):
        theta_b = (omega * t0 + 2 * np.pi * b / geom.Z) % (2 * np.pi)
        # angle measured from the ice sector centre, wrapped to +-pi
        delta = (theta_b + np.pi) % (2 * np.pi) - np.pi
        assert abs(delta) <= 2 * np.pi / geom.Z + env.sector_half_width


# --------------------------------------------------------------------------- #
#  Geometry / stress components / end-to-end stress history
# --------------------------------------------------------------------------- #
def test_geometry_is_consistent(geom):
    props = geom.root_properties()
    assert props["W_flat_m3"] < props["W_edge_m3"]
    r = geom.stations().r
    assert np.all(np.diff(geom.chord(r)) < 0)
    ear = geom.expanded_area_ratio()
    assert 0.30 <= ear <= 0.90
    assert props["planform_area_m2"] > 0.5
    assert geom.R == pytest.approx(geom.D / 2)


def test_root_stress_components(geom):
    out = root_stress_components(geom, NAB, 130.0 / 60.0, M_axial=1.0e5, M_tangential=1.0e5)
    assert out["sigma_axial_bending"] > out["sigma_tangential_bending"]
    assert out["sigma_combined_outer_fibre"] > out["sigma_axial_bending"]
    assert out["sigma_centrifugal"] > 0.0


def test_stress_history_chain(geom, sn_nab):
    """Stress superposition -> rainflow -> Miner, on a short window."""
    from tensorphysics.propeller_fatigue.pipeline import build_stress_history, design_point

    dp = design_point(geom.D, 130.0 / 60.0, 0.60)
    out = build_stress_history(geom, NAB, ICE_CASES["design"], WakeField(), dp,
                              n_real=1, dlf=1.245, n_revolutions=20.0)
    assert out["sigma_total_max_Pa"] > out["sigma_mean_Pa"]
    assert out["n_cycles_window"] > 0
    assert out["D_window"] >= 0.0
    assert out["D_exact"] > 0.0
    assert out["SF_exact"] > 1.0
    assert np.isfinite(out["life_hours_exact"])


def test_damage_monotone_in_ice_severity(geom):
    from tensorphysics.propeller_fatigue.pipeline import build_stress_history, design_point

    dp = design_point(geom.D, 130.0 / 60.0, 0.60)
    d = []
    for key in ("light", "heavy"):
        out = build_stress_history(geom, NAB, ICE_CASES[key], WakeField(), dp,
                                  n_real=1, dlf=1.245, n_revolutions=20.0)
        d.append(out["D_exact"])
    assert d[1] > d[0]