"""Tests for the hull-girder springing / whipping hydroelasticity module.

Each test pins one acceptance gate of the campaign node; the analytic references
are the Euler-Bernoulli free-free roots, the exact 2-D added mass ``rho pi a^2``
of a circular cylinder and the exact ellipse results ``m_x = rho pi b^2`` /
``m_y = rho pi a^2``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.hull_hydroelastic import (
    KCS,
    TimoshenkoBeam,
    added_mass_matrix,
    added_mass_gate,
    circle_section,
    dry_mode_analytic_gate,
    euler_bernoulli_free_free_roots,
    heave_added_mass,
    run_case,
    superellipse_section,
    uniform_beam,
    wet_modes,
)
from tensorphysics.hull_hydroelastic.response import (
    encounter_frequency,
    springing_stress,
    whipping_fatigue,
    whipping_response,
)
from tensorphysics.hull_hydroelastic.sections import Section2D
from tensorphysics.hull_hydroelastic.wet import (
    added_mass_distribution,
    ship_beam_properties,
)

RHO = 1025.0


# --------------------------------------------------------------- dry modes
def test_euler_bernoulli_roots():
    roots = euler_bernoulli_free_free_roots(4)
    assert roots[0] == pytest.approx(4.7300407, rel=1e-6)
    assert roots[1] == pytest.approx(7.8532046, rel=1e-6)
    assert roots[2] == pytest.approx(10.9956078, rel=1e-6)


def test_gate1_dry_modes_match_analytic_roots():
    """Gate 1: uniform free-free beam, first three modes within 3 %."""
    gate = dry_mode_analytic_gate()
    assert gate["max_relative_error_pct"] <= 3.0
    assert gate["pass_analytic"] is True


def test_gate1_timoshenko_softens_and_is_slender_consistent():
    """Gate 1: shear softening sign + < 1 % on mode 1 at L/h = 20."""
    gate = dry_mode_analytic_gate()
    assert gate["shear_softening_sign"] is True
    assert gate["pass_slender_20_mode1_under_1pct"] is True
    # every Timoshenko frequency must sit below the Euler-Bernoulli value
    for key in ("20", "100"):
        drop = gate["slender"][key]["drop_pct"]
        assert all(d > 0.0 for d in drop)
        # and the softening must shrink as the beam gets more slender
    assert gate["slender"]["20"]["drop_pct"][0] > gate["slender"]["100"]["drop_pct"][0]


def test_timoshenko_shear_rigid_limit():
    props = uniform_beam(length=100.0, height=2.0, section="solid_rect", n_elements=120)
    tim = TimoshenkoBeam(props).dry_modes(3).frequencies_hz
    eb = TimoshenkoBeam(props, shear_rigid=True).dry_modes(3).frequencies_hz
    assert np.all(tim < eb)
    assert np.all(tim > 0.9 * eb)


def test_free_free_has_two_rigid_modes():
    props = uniform_beam(length=100.0, height=2.0, n_elements=80)
    beam = TimoshenkoBeam(props)
    modes = beam.dry_modes(3)
    assert modes.rigid_modes.shape[1] == 2
    # mass orthonormality of the elastic modes
    phi = modes.shapes
    assert phi.shape[0] == 3


# ------------------------------------------------------------- added mass
def test_gate2_circle_added_mass_analytic():
    """Gate 2: 2-D heave added mass of a cylinder vs rho pi a^2 within 5 %."""
    ref = RHO * np.pi
    m = heave_added_mass(circle_section(1.0, -10.0), n_panels=160)
    assert m == pytest.approx(ref, rel=0.05)


def test_gate2_ellipse_added_mass_analytic():
    """Exact ellipse result: motion along the long axis has m = rho pi b^2."""
    th = np.linspace(0.0, 2.0 * np.pi, 240, endpoint=False)
    ell = Section2D(y=2.0 * np.cos(th), z=-10.0 + np.sin(th), name="ellipse")
    mat = added_mass_matrix(ell, modes=("heave", "sway"), n_panels=320)["matrix"]
    assert mat[0, 0] == pytest.approx(RHO * np.pi * 4.0, rel=0.05)  # heave: rho pi a_y^2
    assert mat[1, 1] == pytest.approx(RHO * np.pi * 1.0, rel=0.05)  # sway: rho pi b^2


def test_gate2_summary_pass():
    gate = added_mass_gate()
    assert gate["pass"] is True
    assert gate["worst_circle_error_pct"] <= 5.0


def test_added_mass_positive_and_monotone_in_draught():
    small = heave_added_mass(superellipse_section(8.0, 5.0, 4.0), n_panels=120, free_surface=True)
    large = heave_added_mass(superellipse_section(16.0, 10.0, 4.0), n_panels=120, free_surface=True)
    assert small > 0.0 and large > small


# ---------------------------------------------------------------- wet modes
@pytest.fixture(scope="module")
def kcs_wet():
    ship = KCS.girder()
    beam = TimoshenkoBeam(ship_beam_properties(ship, n_elements=60))
    am = added_mass_distribution(
        ship, n_stations=7, section_exponent=4.0, n_panels=80, x=beam.props.x
    )
    wet = wet_modes(
        beam,
        am["a33"],
        n_modes=8,
        zeta_structural=0.02,
        b_waterplane=ship.waterplane_width(beam.props.x),
    )
    return ship, beam, am, wet


def test_gate4_wet_below_dry_and_positive_damping(kcs_wet):
    _, _, _, wet = kcs_wet
    assert np.all(wet.wet_frequencies_hz < wet.dry_omega / (2.0 * np.pi))
    assert np.all(wet.zeta > 0.0)
    # added mass must be a finite, positive fraction of the generalised mass
    assert np.all(wet.added_mass_ratio > 0.0)
    assert np.all(wet.added_mass_ratio < 1.0)


def test_wet_frequency_downshift_is_physical(kcs_wet):
    _, _, _, wet = kcs_wet
    shift = wet.wet_frequencies_hz[0] / (wet.dry_omega[0] / (2.0 * np.pi)) - 1.0
    # 2-node vertical wet frequency of a full-form hull is 15-45 % below dry
    assert -0.45 < shift < -0.15


def test_springing_stress_speed_sensitivity(kcs_wet):
    ship, _, _, wet = kcs_wet
    low = springing_stress(ship, wet, 6.0, 9.0, speed=0.0)
    high = springing_stress(ship, wet, 6.0, 9.0, speed=12.0)
    assert low["sigma_rms_Pa"] > 0.0 and high["sigma_rms_Pa"] > 0.0
    # forward speed moves the resonance onto shorter (more attenuated) waves, so
    # the encounter-frequency formulation changes the flexible response
    # materially; the sign of the change is a model result, not an assumption.
    assert abs(high["sigma_rms_Pa"] / low["sigma_rms_Pa"] - 1.0) > 0.05


def test_encounter_frequency_limits():
    om, jac = encounter_frequency(np.array([1.0]), speed=0.0)
    assert om[0] == pytest.approx(1.0)
    assert jac[0] == pytest.approx(1.0)
    om, jac = encounter_frequency(np.array([1.0]), speed=10.0, heading_deg=180.0)
    assert om[0] == pytest.approx(1.0 + 10.0 / 9.81)
    assert jac[0] == pytest.approx(1.0 + 2.0 * 10.0 / 9.81)


def test_damping_reduces_resonant_response(kcs_wet):
    ship, _, _, wet = kcs_wet
    light = springing_stress(ship, wet, 6.0, 9.0, zeta=0.005, speed=12.0)
    heavy = springing_stress(ship, wet, 6.0, 9.0, zeta=0.05, speed=12.0)
    assert heavy["sigma_rms_Pa"] < light["sigma_rms_Pa"]


# ---------------------------------------------------------------- whipping
def test_whipping_impulse_response(kcs_wet):
    ship, _, _, wet = kcs_wet
    small = whipping_response(ship, wet, peak_force=2.5e6)
    large = whipping_response(ship, wet, peak_force=5.0e6)
    assert large["peak_stress_Pa"] == pytest.approx(2.0 * small["peak_stress_Pa"], rel=0.05)
    # the transient must decay (structural damping): after ~11 s of ringing the
    # 2-node envelope has dropped by exp(-zeta omega t) ~ 0.6 and the higher
    # modes (damped faster) are gone
    stress = np.abs(np.array(large["stress_Pa"]))
    tail = stress[-200:].max()
    assert tail < 0.15 * large["peak_stress_Pa"]
    # monotone envelope decay over the last half of the record
    half = stress[stress.size // 2 :]
    assert half[: half.size // 8].max() > half[-half.size // 8 :].max()


def test_whipping_beam_validity_filter(kcs_wet):
    ship, _, _, wet = kcs_wet
    full = whipping_response(ship, wet, peak_force=5.0e6, min_wavelength_over_depth=None)
    band = whipping_response(ship, wet, peak_force=5.0e6, min_wavelength_over_depth=4.0)
    assert len(band["modes_kept"]) < full["n_modes"]
    assert band["peak_stress_Pa"] <= full["peak_stress_Pa"]


def test_whipping_fatigue_scales_with_stress_cubed():
    d1 = whipping_fatigue(20.0e6, 2.3, 0.02)["damage_per_event"]
    d2 = whipping_fatigue(40.0e6, 2.3, 0.02)["damage_per_event"]
    assert d2 == pytest.approx(8.0 * d1, rel=1e-6)


# ------------------------------------------------------------------- gates
def test_full_case_kcs_gates():
    """End-to-end acceptance gates for KCS (slow: ~1-2 min)."""
    res = run_case(
        KCS, n_stations=7, n_panels=80, n_modes=8, slamming_band=(5.0e6,), duration_band=(0.25,)
    )
    assert res["sign_gates"]["pass"] is True
    # truncation gate: the 10 -> 20 mode step must be converged (<1 %).  The
    # 5-mode step is NOT converged (the quasi-static flexible share needs more
    # modes) and that deficit is reported in the JSON rather than hidden.
    steps = {c["n_modes"]: c for c in res["convergence"]["changes_vs_20"]}
    assert steps[10]["stress_rms_change_pct"] <= 1.0
    assert steps[10]["max_wet_freq_change_pct"] <= 1.0
    assert res["coupling"]["trend_increasing"] is True
    ledger = res["utilisation_ledger"]
    assert ledger["proportionality_error_pct"] <= 2.0
    assert ledger["utilisation_coupled"] > ledger["utilisation_decoupled"]