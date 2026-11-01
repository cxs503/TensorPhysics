"""Acceptance tests for the pod-ice interaction module (physics + API).

Every criterion of the mission is covered by at least one machine-checkable
test, plus the underlying model validations (BEMT, beam FEM, Hertz, FEM contact
refinement, rotor-lag phase).
"""
from __future__ import annotations

import json
from math import atan, degrees, pi, sqrt
from pathlib import Path

import numpy as np
import pytest

from scenarios.pod_ice_interaction.api import PodIceRequest, compute
from scenarios.pod_ice_interaction.contact import (
    QuarterContactFEM,
    fem_stress_refinement,
    hertz_point_contact,
    hertz_pressure,
    ice_pod_contact_case,
)
from scenarios.pod_ice_interaction.envelope import (
    IceLoadCase,
    PodSpec,
    motor_thrust_limit,
    thrust_domain,
    verify_domain,
)
from scenarios.pod_ice_interaction.interaction import (
    InteractionConfig,
    compare_coupling,
    ice_torque_shape,
    simulate_interaction,
    verify_phase,
)
from scenarios.pod_ice_interaction.propeller import BladeGeometry, OpenWaterCurve, open_water
from scenarios.pod_ice_interaction.structure import (
    StrutSection,
    beam_root_forces,
    max_thrust_for_strength,
    pod_root_response,
)

V_SHIP = 8.0


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def geometry() -> BladeGeometry:
    return BladeGeometry()


@pytest.fixture(scope="module")
def curve(geometry: BladeGeometry) -> OpenWaterCurve:
    return OpenWaterCurve(geometry, n_points=21)


@pytest.fixture(scope="module")
def pod() -> PodSpec:
    return PodSpec(power_rated=15.0e6, n_rated_rpm=125.0)


@pytest.fixture(scope="module")
def ice() -> IceLoadCase:
    return IceLoadCase(thickness=1.5)


# ------------------------------------------------------------------- propeller
def test_bemt_open_water_is_physical(geometry):
    ow = open_water(geometry, 0.65)
    assert 0.05 < ow.kt < 1.0
    assert 0.005 < ow.kq < 0.2
    assert 0.0 < ow.eta < 1.0


def test_bemt_thrust_decreases_with_advance_ratio(geometry):
    kts = [open_water(geometry, j).kt for j in (0.3, 0.5, 0.7, 0.9)]
    assert all(a > b for a, b in zip(kts, kts[1:]))


def test_bemt_efficiency_below_unity_over_design_range(curve):
    for j in (0.5, 0.6, 0.7, 0.8):
        assert 0.0 < curve.eta_at(j) < 1.0


def test_expanded_area_ratio_is_wide_blade(geometry):
    assert 0.30 < geometry.expanded_area_ratio < 0.80


def test_rated_power_is_recovered(curve, pod):
    n = curve.solve_rpm_for_power(pod.power_rated, V_SHIP)
    assert curve.delivered_power(n, V_SHIP) == pytest.approx(pod.power_rated, rel=1e-6)


def test_torque_and_thrust_gradients_positive(curve):
    n = 1.9
    assert curve.torque_gradient(n, V_SHIP) > 0.0
    assert curve.thrust_gradient(n, V_SHIP) > 0.0


# -------------------------------------------------------------- strut / hull
def test_beam_fem_root_moment_matches_cantilever():
    section = StrutSection()
    root = beam_root_forces(section, fx=1.2e6, fy=4.0e5, my=1.0e5)
    assert root["my"] == pytest.approx(1.2e6 * section.height + 1.0e5, rel=1e-6)
    assert root["mx"] == pytest.approx(4.0e5 * section.height, rel=1e-6)
    assert root["reconciliation_error"] < 1e-6


def test_beam_fem_tip_deflection_matches_cantilever():
    section = StrutSection()
    root = beam_root_forces(section, fx=1.0e5, fy=1.0e5)
    analytic = 1.0e5 * section.height ** 3 / (
        3.0 * section.young_modulus * section.second_moment_y
    )
    assert abs(root["tip_deflection_x"]) == pytest.approx(analytic, rel=0.02)


def test_section_stress_grows_with_thrust():
    section = StrutSection()
    low = pod_root_response(section, thrust=5.0e5, azimuth_deg=30.0, ice_force=1.2e6)
    high = pod_root_response(section, thrust=1.3e6, azimuth_deg=30.0, ice_force=1.2e6)
    assert high["sigma_vm"] > low["sigma_vm"]


def test_strength_limit_sits_on_allowable_stress():
    section = StrutSection()
    t_lim = max_thrust_for_strength(section, azimuth_deg=60.0, ice_force=1.2e6)
    resp = pod_root_response(section, thrust=t_lim, azimuth_deg=60.0, ice_force=1.2e6)
    assert resp["utilization"] == pytest.approx(1.0, abs=1e-3)


# ----------------------------------------------------------- thrust domain
@pytest.fixture(scope="module")
def domain(pod, ice, curve):
    return thrust_domain(
        pod, ice, v_ship=V_SHIP, azimuths_deg=(0.0, 45.0, 90.0, 135.0), curve=curve
    )


def test_domain_boundary_is_minimum_of_constraints(domain):
    for st in domain["stations"]:
        assert st["t_max"] == pytest.approx(
            min(st["t_motor"], st["t_bearing"], st["t_structure"]), rel=1e-12
        )


def test_domain_has_no_overstressed_region(pod, ice, domain):
    check = verify_domain(pod, ice, domain)
    assert check["overstressed_points"] == []
    assert check["max_utilization_inside"] <= 1.0 + 1e-9


def test_domain_boundary_matches_strength_bisection(pod, ice, domain):
    check = verify_domain(pod, ice, domain)
    assert check["boundary_error_max"] < 0.01
    assert check["maximality_violations"] == 0
    assert check["passed"] is True


def test_motor_limit_is_rated_power_torque(pod, ice, curve):
    motor = motor_thrust_limit(pod, curve, V_SHIP)
    assert motor["torque"] == pytest.approx(pod.motor_torque_limit, rel=1e-3)
    assert motor["n_rps"] < pod.n_rated_rps


# ---------------------------------------------------------------------- contact
def test_hertz_peak_pressure_formula():
    force, a = 1.0e6, 0.04
    h = hertz_point_contact(force, 1.0, 1.2, 8.0e9, 0.33, 2.06e11, 0.30)
    assert h["peak_pressure"] == pytest.approx(3.0 * force / (2.0 * pi * h["contact_radius"] ** 2), rel=1e-12)
    assert h["contact_radius"] == pytest.approx(
        (3.0 * force * h["r_effective"] / (4.0 * h["e_star"])) ** (1.0 / 3.0), rel=1e-12
    )
    assert h["contact_radius"] > 0.0 and a != h["contact_radius"]


def test_hertz_scaling_laws():
    h1 = hertz_point_contact(1.0e6, 1.0, 1.2, 8.0e9, 0.33, 2.06e11, 0.30)
    h8 = hertz_point_contact(8.0e6, 1.0, 1.2, 8.0e9, 0.33, 2.06e11, 0.30)
    assert h8["contact_radius"] / h1["contact_radius"] == pytest.approx(2.0, rel=1e-9)
    assert h8["peak_pressure"] / h1["peak_pressure"] == pytest.approx(2.0, rel=1e-9)


def test_hertz_pressure_distribution_integrates_to_force():
    h = hertz_point_contact(1.2e6, 1.0, 1.2, 8.0e9, 0.33, 2.06e11, 0.30)
    r = np.linspace(0.0, h["contact_radius"], 4001)
    p = hertz_pressure(r, h["peak_pressure"], h["contact_radius"])
    integral = np.trapezoid(2.0 * pi * r * p, r)
    assert integral == pytest.approx(h["force"], rel=1e-2)


def test_fem_quarter_model_force_scales_with_face_areas():
    model = QuarterContactFEM(0.03, young_modulus=8.6e9, h_ratio=0.30)
    assert model._face_areas.sum() == pytest.approx(model.xs[-1] ** 2, rel=1e-9)


def test_fem_contact_stress_matches_hertz_theory():
    """Acceptance criterion (b): contact stress maximum vs Hertz within 10 %."""
    case = ice_pod_contact_case(1.2e6, h_ratio=0.28)
    assert abs(case["von_mises_deviation_rel"]) < 0.10
    assert case["contact_pressure_ok"] is True
    assert 0.4 < case["fem"]["fem_max_depth_over_a"] < 0.6


def test_fem_center_displacement_matches_boussinesq():
    fem = fem_stress_refinement(1.2e6, 1.0, 1.2, 8.0e9, 0.33, 2.06e11, 0.30, h_ratio=0.28)
    assert abs(fem["center_displacement_error_rel"]) < 0.10


# --------------------------------------------------------------- interaction
def test_ice_torque_shape_duty_and_periodicity():
    theta = np.linspace(0.0, 2.0 * pi, 3001)
    shape = ice_torque_shape(theta, 4, 0.18)
    assert shape.max() == pytest.approx(1.0, abs=1e-9)
    assert np.mean(shape) == pytest.approx(4 * 0.18 * 0.5, rel=0.02)
    shifted = ice_torque_shape(theta + 2.0 * pi, 4, 0.18)
    assert np.allclose(shape, shifted, atol=1e-9)


def test_ice_torque_shape_rejects_overlapping_duty():
    with pytest.raises(ValueError):
        ice_torque_shape(np.array([0.0, 1.0]), 4, 0.4)


@pytest.fixture(scope="module")
def interaction_result(curve, pod, ice):
    cfg = InteractionConfig(ice_torque_mean=0.30e6, blade_duty=0.18, duration=2.4, dt=3.0e-4)
    one_way = simulate_interaction(curve, pod, ice, cfg, v_ship=V_SHIP, two_way=False, n_cmd_rpm=100.0)
    two_way = simulate_interaction(curve, pod, ice, cfg, v_ship=V_SHIP, two_way=True, n_cmd_rpm=100.0)
    return cfg, one_way, two_way


def test_interaction_histories_are_consistent(interaction_result):
    _, one_way, _ = interaction_result
    assert one_way["t"].size == one_way["thrust"].size == one_way["torque_ice"].size
    assert np.all(one_way["thrust"] > 0.0)
    assert np.all(one_way["n_rpm"] > 0.0)
    assert np.max(one_way["torque_ice"]) > np.mean(one_way["torque_ice"])


def test_phase_relation_matches_analytic_rotor_lag(interaction_result, curve, pod):
    """Acceptance criterion (c): thrust/torque vs ice spectrum phase."""
    cfg, one_way, _ = interaction_result
    phase = verify_phase(one_way, curve, pod, cfg)
    assert phase["phase_error_deg"] < 5.0
    assert phase["amp_error_rel"] < 0.10
    assert phase["passed"] is True


def test_analytic_phase_formula_consistency(interaction_result, curve, pod):
    cfg, one_way, _ = interaction_result
    phase = verify_phase(one_way, curve, pod, cfg)
    n_mean = float(np.mean(one_way["n_rps"]))
    w = 2.0 * pi * phase["blade_frequency"]
    k_p = curve.torque_gradient(n_mean, V_SHIP) / (2.0 * pi)
    expected = 180.0 - degrees(atan(w * cfg.rotor_inertia / (k_p + cfg.controller_gain)))
    assert phase["phase_diff_analytic_deg"] == pytest.approx(expected, rel=1e-6)


def test_two_way_coupling_is_small_but_non_zero(interaction_result):
    _, one_way, two_way = interaction_result
    coupling = compare_coupling(one_way, two_way)
    assert coupling["pod_velocity_rms"] > 0.0
    assert 0.0 < coupling["ice_torque_change_rel"] < 0.10


# ------------------------------------------------------------------------ API
def test_api_compute_fast_path():
    req = PodIceRequest(
        azimuths_deg=[0.0, 60.0],
        contact_h_ratio=0.30,
        interaction_duration=1.2,
        interaction_dt=5.0e-4,
        interaction_rpm=100.0,
    )
    out = compute(req)
    assert out["status"] == "ok"
    assert len(out["thrust_domain"]["stations"]) == 2
    assert out["structural_check"]["passed"] is True
    assert out["verdicts"]["contact_within_10pct_of_hertz"] is True
    assert out["verdicts"]["phase_relation_correct"] is True
    assert out["verdicts"]["all_checks_passed"] is True


def test_api_endpoint_is_reachable():
    from fastapi.testclient import TestClient

    from scenarios.pod_ice_interaction.api import app

    client = TestClient(app)
    resp = client.post(
        "/api/v1/calculations/pod_ice_interaction",
        json={
            "azimuths_deg": [0.0],
            "contact_h_ratio": 0.32,
            "include_interaction": False,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["thrust_domain"]["stations"][0]["t_max"] > 0.0


# ----------------------------------------------------------------- artefacts
def test_committed_case_artifacts_pass_all_criteria():
    path = Path("artifacts/pod_ice/summary.json")
    if not path.exists():
        pytest.skip("case artefacts not present in this worktree")
    summary = json.loads(path.read_text())
    assert summary["acceptance"]["a_domain_matches_strength"] is True
    assert summary["acceptance"]["b_contact_within_10pct"] is True
    assert summary["acceptance"]["c_phase_correct"] is True
    assert summary["acceptance"]["e_full_case_produced"] is True
    assert summary["contact"]["von_mises_deviation_rel"] > -0.10
    assert summary["interaction"]["phase_error_deg"] < 5.0
    assert summary["structural_check"]["overstressed_points"] == 0
    assert 1.0e3 < summary["thrust_domain"]["motor_limit_N"] / 1e3 < 2.0e3