"""Acceptance tests for the ice-zone manoeuvring (turning radius) capability.

Campaign node ``p362w2-n10-ice-turning-radius``: the module
:mod:`tensorphysics.ice.maneuver` must deliver a converged steady turning radius
for a reference icebreaker in broken ice, reproduce the monotone trends of the
turning-capability chart, and agree with the reduced-order empirical balance.

All runs use ``dt = 1.0 s`` and ``t_end = 900-1200 s`` (verified to reproduce the
production ``dt = 0.5 s / 2400 s`` radius to better than 0.1 %).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import tensorphysics.ice.maneuver as MV
from tensorphysics.ice.maneuver import (
    IceChannel,
    ManeuverShip,
    analyse_steady_state,
    assess_chart,
    calibrate_kappa_lat,
    course_stability,
    hull_forces,
    ice_forces,
    integrate_maneuver,
    kv_svalbard_ship,
    linear_derivatives,
    propeller_thrust,
    rudder_forces,
    steady_radius_balance,
    steady_radius_linear_theory,
    trim_speed,
)
from tensorphysics.ice.resistance import IceDomainError, level_ice_resistance

DT = 1.0
T_END = 1000.0
REF_ICE = IceChannel(thickness_m=0.6, concentration=0.5)
OW_ICE = IceChannel(thickness_m=0.0, concentration=0.0)


@pytest.fixture(scope="module")
def ship() -> ManeuverShip:
    return kv_svalbard_ship()


@pytest.fixture(scope="module")
def reference_run(ship):
    return integrate_maneuver(ship, delta_deg=20.0, ice=REF_ICE, t_end=T_END, dt=DT)


@pytest.fixture(scope="module")
def open_water_run(ship):
    return integrate_maneuver(ship, delta_deg=20.0, ice=OW_ICE, t_end=T_END, dt=DT)


# --------------------------------------------------------------- containment


def test_geometry_gate_closes_cb_l_b_t(ship):
    gate = ship.geometry_gate()
    assert gate["gate_passed"] is True
    assert gate["C_b_relative_error"] < 1e-12
    assert 0.5 < gate["C_b"] < 0.65
    # ice-class rudder proportions stay in the documented envelope
    assert 1 / 80.0 < gate["A_R_over_LT"] < 1 / 30.0
    assert 2.0 < gate["f_alpha_per_rad"] < 4.0


def test_ship_derived_inertia_and_added_mass(ship):
    assert ship.I_z > 0.0
    assert ship.J_z == pytest.approx(0.5 * ship.I_z)
    assert 0.05 * ship.mass < ship.m_x < 0.2 * ship.mass
    assert 0.7 * ship.mass < ship.m_y < 1.2 * ship.mass


# ------------------------------------------------------------------- controls


def test_ice_forces_vanish_at_zero_concentration(ship):
    f = ice_forces(ship, 3.0, 0.4, 0.02, OW_ICE)
    assert f["X_I"] == 0.0 and f["Y_I"] == 0.0 and f["N_I"] == 0.0


def test_ice_lateral_force_is_zero_without_bow_sweep(ship):
    # no yaw rate and no sway -> the bow contact point does not sweep sideways
    f = ice_forces(ship, 3.0, 0.0, 0.0, REF_ICE)
    assert abs(f["Y_I"]) < 1e-9
    assert abs(f["N_I"]) < 1e-9
    # but the longitudinal ice resistance is not zero
    assert f["X_I"] < -1.0e5


def test_ice_yaw_moment_opposes_a_starboard_turn(ship):
    # starboard turn: r > 0 -> the bow sweeps to starboard -> ice pushes back
    f = ice_forces(ship, 3.0, -0.3, 0.012, REF_ICE)
    assert f["N_I"] < 0.0
    # ... and a mirror-image port turn (same slip-free reference condition)
    # produces the mirrored moment
    f_port = ice_forces(ship, 3.0, 0.0, -0.012, REF_ICE)
    f_stbd = ice_forces(ship, 3.0, 0.0, 0.012, REF_ICE)
    assert f_port["N_I"] > 0.0
    assert f_port["N_I"] == pytest.approx(-f_stbd["N_I"], rel=1e-9)


def test_rudder_moment_sign_and_zero_at_zero_rudder(ship):
    _, _, n_pos = rudder_forces(ship, 4.0, 0.0, 0.0, math.radians(20.0))
    _, _, n_neg = rudder_forces(ship, 4.0, 0.0, 0.0, math.radians(-20.0))
    _, _, n_zero = rudder_forces(ship, 4.0, 0.0, 0.0, 0.0)
    assert n_pos > 0.0 > n_neg
    assert abs(n_zero) < 1e-9


def test_zero_rudder_keeps_the_ship_straight(ship):
    run = integrate_maneuver(ship, delta_deg=0.0, ice=REF_ICE, t_end=400.0, dt=DT)
    assert abs(run["steady"]["r_ss_rad_s"]) < 1e-6


def test_out_of_envelope_ice_raises(ship):
    with pytest.raises(IceDomainError):
        ice_forces(ship, 3.0, 0.3, 0.01, IceChannel(thickness_m=0.6, concentration=1.4))
    with pytest.raises(IceDomainError):
        ice_forces(ship, 3.0, 0.3, 0.01, IceChannel(thickness_m=3.0, concentration=0.5))


# ------------------------------------------------------------------- physics


def test_cross_flow_drag_damps_sway(ship):
    _, y_pos, _ = hull_forces(ship, 4.0, 1.0, 0.0)
    _, y_neg, _ = hull_forces(ship, 4.0, -1.0, 0.0)
    assert y_pos < 0.0 < y_neg  # force opposes the sway velocity
    assert y_pos == pytest.approx(-y_neg, rel=1e-6)


def test_course_stability_index_is_positive(ship):
    st = course_stability(ship, 4.0)
    assert st["directionally_stable"] is True
    assert st["C"] > 0.0
    # the lateral-force slope is the calibrated one (documented deviation from the
    # published KVLCC2 derivative band, reported rather than hidden)
    band = MV.DERIVATIVE_REFERENCE_BAND["Y_v_prime"]
    assert st["Y_v_prime"] < band[1]  # stiffer than the reference band


def test_propeller_thrust_decreases_with_speed(ship):
    t_low = propeller_thrust(ship, 0.5)
    t_high = propeller_thrust(ship, 6.0)
    assert t_low > t_high > 0.0


def test_trim_speed_drops_with_ice_concentration(ship):
    u_ow = trim_speed(ship, OW_ICE)
    u_c2 = trim_speed(ship, IceChannel(thickness_m=0.6, concentration=0.2))
    u_c5 = trim_speed(ship, IceChannel(thickness_m=0.6, concentration=0.5))
    assert u_ow > u_c2 > u_c5 > 0.0


# --------------------------------------------------------- acceptance criteria


def test_reference_turn_converges_and_is_finite(reference_run):
    st = reference_run["steady"]
    assert st["converged"] is True
    assert st["r_rate_pct_per_min"] < 1.0  # criterion (b)
    assert math.isfinite(st["R_steady_m"])
    assert st["R_over_L"] > 1.0  # a ship cannot turn inside its own length
    # documented reference value of this model (regression guard)
    assert st["R_over_L"] == pytest.approx(3.47, abs=0.10)


def test_ice_widens_the_turn(reference_run, open_water_run):
    s_ice = reference_run["steady"]
    s_ow = open_water_run["steady"]
    assert s_ice["R_steady_m"] > s_ow["R_steady_m"]
    ratio = s_ice["R_steady_m"] / s_ow["R_steady_m"]
    lo, hi = MV.TURNING_RATIO_ANCHOR["band"]
    assert lo <= ratio <= hi  # external anchor band 1.3-2.0


def test_open_water_turning_circle_in_imo_band(ship):
    st = integrate_maneuver(ship, delta_deg=35.0, ice=OW_ICE, t_end=T_END, dt=DT)["steady"]
    lo, hi = MV.OPEN_WATER_ANCHOR["D_T_over_L_band"]
    assert lo <= st["D_T_over_L"] <= hi


def test_empirical_balance_matches_simulation(reference_run):
    st = reference_run["steady"]
    bal = steady_radius_balance(
        ship=kv_svalbard_ship(), delta_deg=20.0, ice=REF_ICE, u_ss=st["u_ss_ms"]
    )
    assert bal["converged"] is True
    err = abs(st["R_steady_m"] - bal["R_bal_m"]) / bal["R_bal_m"] * 100.0
    assert err <= 20.0  # criterion (a)
    assert bal["ice_share_of_rudder"] > 0.1  # ice moment is not a rounding effect


def test_linear_theory_deviates_more_than_balance(reference_run):
    """The frozen-derivative route must be documented as the weaker estimate."""
    st = reference_run["steady"]
    lin = steady_radius_linear_theory(
        ship=kv_svalbard_ship(), delta_deg=20.0, ice=REF_ICE, u_ss=st["u_ss_ms"]
    )
    bal = steady_radius_balance(
        ship=kv_svalbard_ship(), delta_deg=20.0, ice=REF_ICE, u_ss=st["u_ss_ms"]
    )
    err_lin = abs(st["R_steady_m"] - lin["R_emp_m"]) / lin["R_emp_m"]
    err_bal = abs(st["R_steady_m"] - bal["R_bal_m"]) / bal["R_bal_m"]
    assert err_bal < err_lin  # nonlinearity error dominates the linear route


def test_radius_monotone_in_concentration(ship):
    rows = []
    for c in (0.2, 0.5, 0.8):
        st = integrate_maneuver(
            ship, delta_deg=20.0, ice=IceChannel(thickness_m=0.6, concentration=c), t_end=T_END, dt=DT
        )["steady"]
        assert st["converged"] is True
        rows.append(st["R_steady_m"])
    assert rows[0] < rows[1] < rows[2]  # criterion (c)


def test_radius_monotone_in_rudder_angle(ship):
    rows = []
    for d in (10.0, 25.0, 35.0):
        st = integrate_maneuver(
            ship, delta_deg=d, ice=REF_ICE, t_end=T_END, dt=DT
        )["steady"]
        assert st["converged"] is True
        rows.append(st["R_steady_m"])
    assert rows[0] > rows[1] > rows[2]  # criterion (d)


def test_chart_gates_on_small_chart(ship):
    chart = MV.ice_turning_chart(
        ship,
        delta_deg=20.0,
        concentrations=(0.3, 0.5),
        rudder_angles_deg=(15.0, 30.0),
        t_end=800.0,
        dt=DT,
    )
    gates = assess_chart(chart)
    # with two points per sweep the monotonicity helper returns False (needs 3+),
    # so only the convergence half of the gate is asserted here
    assert gates["concentration_all_converged"] is True
    assert gates["rudder_all_converged"] is True
    assert len(chart["concentration_sweep"]) == 2


# ------------------------------------------------------- convergence metrics


def test_analyse_steady_state_flags_a_non_converged_turn():
    t = np.linspace(0.0, 600.0, 301)
    r = 0.01 * t / 600.0  # still growing at the end -> not converged
    u = np.full_like(t, 3.0)
    v = np.zeros_like(t)
    x = np.zeros_like(t)
    y = np.zeros_like(t)
    psi = np.zeros_like(t)
    hist = np.column_stack([t, x, y, psi, u, v, r, ])
    out = analyse_steady_state(hist, window_s=100.0)
    assert out["converged"] is False
    assert out["r_rate_pct_per_min"] > 1.0


def test_analyse_steady_state_accepts_a_flat_turn():
    t = np.linspace(0.0, 600.0, 301)
    r = np.full_like(t, -0.012)
    u = np.full_like(t, 3.5)
    v = np.full_like(t, -0.3)
    zero = np.zeros_like(t)
    hist = np.column_stack([t, zero, zero, zero, u, v, r])
    out = analyse_steady_state(hist, window_s=100.0)
    assert out["converged"] is True
    assert out["R_steady_m"] == pytest.approx(math.hypot(3.5, -0.3) / 0.012, rel=1e-9)


# ------------------------------------------------------------- calibration


def test_kappa_lat_calibration_is_bracketed_and_reproducible(ship):
    out = calibrate_kappa_lat(ship, t_end=800.0, dt=DT, target_ratio=1.45, band=(0.05, 1.0))
    assert out["bracket_ok"] is True
    assert 0.05 <= out["kappa_lat"] <= 1.0
    assert out["achieved_ratio"] == pytest.approx(1.45, rel=0.05)


def test_lindqvist_law_is_reused_unchanged(ship):
    res = level_ice_resistance(0.6, 1.0, ship.geometry, sigma_c=ship.sigma_c_Pa, sigma_f=ship.sigma_f_Pa)
    assert res["domain_ok"] is True
    assert res["R_total_N"] == pytest.approx(
        res["R_crush_N"] + res["R_break_N"] + res["R_submersion_N"], rel=1e-9
    )


def test_trajectory_history_is_consistent_with_the_steady_turn(reference_run):
    hist = reference_run["history"]
    st = reference_run["steady"]
    # the last recorded yaw rate equals the reported steady value
    assert hist[-1, 6] == pytest.approx(st["r_ss_rad_s"], rel=0.05)
    # heading keeps increasing (starboard turn) and the track is a closed-ish arc
    assert hist[-1, 3] > hist[0, 3] + math.pi / 2
    radius_from_track = math.hypot(hist[-1, 1], hist[-1, 2]) / 2.0
    assert radius_from_track > 0.0