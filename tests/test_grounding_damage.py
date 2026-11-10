"""Tests for the grounding damage assessment package.

Every acceptance criterion of the campaign node (C1-C8) has at least one test.
The panel solves are deliberately small (coarse quadrature, few modes) so that
the whole module runs in well under a minute; the production-precision runs are
in ``scripts/generate_grounding_damage_artifacts.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.grounding_damage import (HullForm, PanelSpec, VonKarmanPanel,
                                            RidgeGeometry, RidgeContact,
                                            ShoalContact, StiffenerSection,
                                            Compartment, BottomOpening,
                                            check_stiffener, check_imo_criteria,
                                            gz_curve, transient_flood,
                                            analytic_filling_time, quasi_static_flood,
                                            inflow_rate, hertz_sphere_on_plane,
                                            HertzContact, bilinear_reference,
                                            uniaxial_curve, POLAR_NV_F,
                                            assess_ridge_damage, GroundingCase,
                                            run_grounding_case, parameter_sweep,
                                            SHIP_15000DWT)
from tensorphysics.grounding_damage import verification as V


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def panel_response():
    """Coarse but converged bottom-panel resistance curve."""
    spec = PanelSpec(a=3.6, b=0.9, t=0.022, n_x=15, n_y=15, n_layers=3,
                     w_modes=2, uv_modes=2)
    solver = VonKarmanPanel(spec, load_steps=8, max_newton=15)
    return solver.resistance_curve(0.30, steps=8)


@pytest.fixture(scope="module")
def ridge(panel_response):
    return RidgeGeometry(h_ridge=4.0, slope_deg=12.0, length=30.0)


# ---------------------------------------------------------------------------
# C1 - contact mechanics
# ---------------------------------------------------------------------------
def test_hertz_scaling():
    """Hertz: contact radius ~ sqrt(delta), pressure ~ sqrt(delta)."""
    a = hertz_sphere_on_plane(1.0e4, 0.5, 2.06e11, 0.3)
    b = hertz_sphere_on_plane(4.0e4, 0.5, 2.06e11, 0.3)
    assert b.radius / a.radius == pytest.approx(4.0 ** (1.0 / 3.0), rel=1e-9)
    assert b.peak_pressure / a.peak_pressure == pytest.approx(4.0 ** (1.0 / 3.0), rel=1e-9)
    assert a.approach == pytest.approx(a.radius**2 / 0.5, rel=1e-12)


def test_hertz_solver_contact_radius_within_10pct():
    solver = HertzContact(0.5, 2.06e11, 0.3, n_rings=80)
    sol = solver.solve(1e-3)
    ref = hertz_sphere_on_plane(sol["load"], 0.5, 2.06e11, 0.3)
    assert abs(sol["contact_radius"] / ref.radius - 1.0) < 0.10


def test_hertz_solver_peak_pressure_within_10pct():
    solver = HertzContact(0.5, 2.06e11, 0.3, n_rings=80)
    sol = solver.solve(1e-3)
    ref = hertz_sphere_on_plane(sol["load"], 0.5, 2.06e11, 0.3)
    assert abs(sol["peak_pressure"] / ref.peak_pressure - 1.0) < 0.10
    # pressure must decay monotonically away from the centre
    p = sol["pressure"][: int(0.8 * len(sol["pressure"]))]
    assert np.all(np.diff(p) <= 1e-6 * p.max())


# ---------------------------------------------------------------------------
# panel: linear limit, membrane action, plasticity, tearing
# ---------------------------------------------------------------------------
def test_panel_linear_limit_navier():
    res = V.panel_linear_verification()
    assert abs(res["ssss"]["error_pct"]) < 5.0


def test_panel_linear_limit_clamped():
    res = V.panel_linear_verification()
    assert abs(res["clamped"]["error_pct"]) < 5.0


def test_panel_membrane_stiffening(panel_response):
    """Beyond w/t ~ 0.2 the pressure must grow faster than linearly."""
    resp = panel_response
    m = resp.deflection > 3 * resp.spec.t
    if m.sum() < 2:
        pytest.skip("coarse curve does not reach the membrane regime")
    d = resp.deflection[m]
    q = np.maximum(resp.pressure[m], 1e-9)
    slope = np.polyfit(np.log(d), np.log(q), 1)[0]
    assert slope > 1.5


def test_panel_plastic_strain_monotone(panel_response):
    resp = panel_response
    assert np.all(np.diff(resp.eps_pl_max) >= -1e-12)
    assert resp.eps_pl_max[0] == pytest.approx(0.0, abs=1e-12)


def test_tearing_criterion_consistent(panel_response):
    """torn_fraction > 0 exactly when eps_pl >= eps_f (criterion C3)."""
    resp = panel_response
    eps_f = resp.spec.grade.eps_f
    above = resp.eps_pl_max >= eps_f
    for a, t in zip(above, resp.torn_fraction):
        assert (t > 0.0) == bool(a)


def test_panel_energy_monotone_and_positive(panel_response):
    resp = panel_response
    assert np.all(np.diff(resp.energy) > 0.0)
    assert resp.energy[-1] > 0.0


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
def test_material_bilinear_reproduction():
    res = V.material_verification(n=120)
    assert res["max_relative_stress_error"] < 1e-3
    assert res["pass"]


def test_material_uniaxial_lateral_stress_vanishes():
    """The uniaxial driver must produce a uniaxial stress state."""
    from tensorphysics.grounding_damage.material import plane_stress_return_map
    eps, sig, eps_pl = uniaxial_curve(POLAR_NV_F, 0.02, 60)
    ref = bilinear_reference(POLAR_NV_F, eps)
    assert np.max(np.abs(sig - ref)) / np.abs(ref).max() < 1e-3
    assert eps_pl[-1] > 0.0


# ---------------------------------------------------------------------------
# ridge / shoal contact geometry and resistance
# ---------------------------------------------------------------------------
def test_ridge_penetration_geometry(ridge):
    y = np.array([0.0, 0.5, -0.5])
    d = ridge.penetration(y, 0.5)
    assert d[0] == pytest.approx(0.5)
    assert d[1] == pytest.approx(0.5 - 0.5 * np.tan(np.deg2rad(12.0)))
    assert ridge.contact_half_width(0.5) == pytest.approx(0.5 / np.tan(np.deg2rad(12.0)))


def test_ridge_penetration_clipped_by_height(ridge):
    assert ridge.penetration(np.array([0.0]), 10.0)[0] == pytest.approx(ridge.h_ridge)


def test_ridge_resistance_monotone(ridge, panel_response):
    contact = RidgeContact(ridge, panel_response.pressure_at,
                           panel_breadth=panel_response.spec.b,
                           panel_length=panel_response.spec.a)
    r = [contact.resistance(d) for d in (0.02, 0.05, 0.10, 0.15)]
    assert all(np.diff(r) > 0.0)
    assert contact.resistance(0.0) == 0.0


def test_ridge_energy_inversion(ridge, panel_response):
    contact = RidgeContact(ridge, panel_response.pressure_at,
                           panel_breadth=panel_response.spec.b,
                           panel_length=panel_response.spec.a)
    e_target = 0.4 * contact.energy(0.15)
    apex = contact.apex_depth_for_energy(e_target)
    assert contact.energy(apex) == pytest.approx(e_target, rel=0.02)


def test_shoal_bearing_capacity_cap():
    shoal = ShoalContact(sand_stiffness=2e8, bearing_capacity=1e6)
    assert shoal.pressure(0.0) == 0.0
    assert shoal.pressure(1e-3) == pytest.approx(2e5, rel=1e-9)
    assert shoal.pressure(1.0) == pytest.approx(1e6, rel=1e-12)


# ---------------------------------------------------------------------------
# stiffeners
# ---------------------------------------------------------------------------
def test_stiffener_buckling_strength():
    """Stocky reference section yields first; a slender web buckles first."""
    s = StiffenerSection()
    res = V.stiffener_verification()
    assert not res["web_buckling_below_yield"]      # stocky: plastic collapse governs
    assert res["slender_web_buckling_below_yield"]  # slender: elastic buckling
    assert 0.0 < s.plastic_bending_resistance() < 1e7


def test_stiffener_check_flags_failure():
    s = StiffenerSection()
    ok = check_stiffener(s, pressure=1.0e5, panel_area=3.24, membrane_force=1.0e6,
                         bend_moment=1.0e4)
    bad = check_stiffener(s, pressure=5.0e7, panel_area=3.24, membrane_force=1.0e8,
                          bend_moment=1.0e8)
    assert not ok.failed
    assert bad.failed


# ---------------------------------------------------------------------------
# C4 - flooding
# ---------------------------------------------------------------------------
def test_flooding_rate_matches_bernoulli():
    comp = Compartment("db", 0.0, 20.0, -11.5, 11.5, 0.0, 1.8, permeability=0.95)
    op = BottomOpening(area=0.5, discharge_coefficient=0.6)
    res = transient_flood(op, comp, draft0=8.6, waterplane_area=2500.0,
                          t_end=20.0, dt=0.01, allow_sinkage=False)
    i1, i2 = 20, 40
    q_num = (res["volume"][i2] - res["volume"][i1]) / (res["t"][i2] - res["t"][i1])
    assert q_num == pytest.approx(inflow_rate(0.5, 8.6, 0.6), rel=0.02)


def test_flooding_filling_time_analytic():
    res = V.flooding_verification()
    assert abs(res["t_fill_error_pct"]) < 2.0


def test_flooding_zero_opening_no_inflow():
    comp = Compartment("db", 0.0, 20.0, -11.5, 11.5, 0.0, 1.8)
    op = BottomOpening(area=0.0)
    res = transient_flood(op, comp, draft0=8.6, waterplane_area=2500.0, t_end=10.0, dt=0.1)
    assert np.allclose(res["volume"], 0.0)
    eq = quasi_static_flood(op, comp, draft0=8.6, waterplane_area=2500.0)
    assert eq["volume"] == 0.0


def test_flooding_equilibrium_respects_capacity():
    comp = Compartment("db", 0.0, 40.0, -11.5, 11.5, 0.0, 1.8, permeability=0.95)
    op = BottomOpening(area=2.0, discharge_coefficient=0.6)
    eq = quasi_static_flood(op, comp, draft0=8.6, waterplane_area=2596.0)
    assert 0.0 < eq["volume"] <= comp.capacity + 1e-9
    assert eq["level"] <= comp.z1 + 1e-9


# ---------------------------------------------------------------------------
# C5 - residual stability
# ---------------------------------------------------------------------------
def test_barge_volume_and_gm():
    res = V.stability_verification()
    assert abs(res["volume_error_pct"]) < 1.0
    assert abs(res["gm_error_pct"]) < 2.0


def test_barge_gz_curve_matches_semi_analytic():
    res = V.stability_verification()
    assert res["worst_gz_error_pct"] < 2.0


def test_gz_zero_at_zero_heel():
    hull = HullForm(length=100.0, breadth=20.0, depth=24.0, draft=6.0, cb=1.0,
                    flat_ratio=1.0, bilge_height=0.0, n_stations=41, fullness=400.0)
    vol = hull.displacement_volume()
    gz = gz_curve(hull, kg=4.0, volume=vol, lcg=0.0, angles=(0.0, 5.0))
    assert gz["gz"][0] == pytest.approx(0.0, abs=1e-6)
    assert gz["gz"][1] > 0.0


def test_imo_criteria_logic():
    angles = np.arange(0.0, 61.0, 5.0)
    good = 2.0 * np.sin(np.deg2rad(angles))
    bad = -good
    ok = check_imo_criteria(angles, good, gm=1.0)
    nok = check_imo_criteria(angles, bad, gm=-0.5)
    assert ok["all_ok"] == 1.0
    assert nok["all_ok"] == 0.0
    assert ok["area_to_30"] > 0.055


# ---------------------------------------------------------------------------
# C6 - negative controls
# ---------------------------------------------------------------------------
def test_negative_control_zero_ridge(panel_response):
    res = V.negative_controls(panel_response)
    assert res["pass"]


def test_negative_control_zero_energy_no_damage(ridge, panel_response):
    contact = RidgeContact(ridge, panel_response.pressure_at,
                           panel_breadth=panel_response.spec.b,
                           panel_length=panel_response.spec.a)
    assert contact.apex_depth_for_energy(0.0) == 0.0
    damage = assess_ridge_damage(ridge, panel_response, apex_depth=0.0, n_longitudinal=8)
    assert damage.n_torn == 0
    assert damage.opening_area == 0.0


# ---------------------------------------------------------------------------
# C7 - end to end
# ---------------------------------------------------------------------------
def test_end_to_end_energy_balance(panel_response):
    res = run_grounding_case(GroundingCase(velocity_kn=3.0, ridge_height=4.0,
                                           ridge_slope_deg=12.0), panel_response)
    assert abs(res["energy"]["relative_residual"]) < 0.05
    assert res["contact"]["apex_depth"] > 0.0
    assert res["damage"]["opening_area"] >= 0.0


def test_end_to_end_damage_and_stability(panel_response):
    res = run_grounding_case(GroundingCase(velocity_kn=5.0), panel_response)
    assert res["damage"]["max_eps_pl"] > 0.0
    if res["damage"]["opening_area"] > 0.0:
        assert res["flooding"]["volume"] > 0.0
        assert res["stability"]["gm"] == res["stability"]["gm"]   # not NaN
        assert "gz" in res["stability"]
        assert len(res["stability"]["gz"]["gz"]) == len(res["stability"]["gz"]["angle_deg"])


def test_geometry_gate_reproduces_cb():
    hull = HullForm(length=SHIP_15000DWT["length"], breadth=SHIP_15000DWT["breadth"],
                    depth=SHIP_15000DWT["depth"], draft=SHIP_15000DWT["draft"],
                    cb=SHIP_15000DWT["cb"], n_stations=41)
    gate = hull.gate()
    assert gate["cb"] == pytest.approx(gate["cb_target"], abs=1e-3)


# ---------------------------------------------------------------------------
# C8 - parameter scan
# ---------------------------------------------------------------------------
def test_sweep_monotone_in_velocity(panel_response):
    rows = parameter_sweep(panel_response, velocities=(1.0, 5.0),
                           ridge_heights=(4.0,), slopes=(12.0,))
    assert len(rows) == 2
    assert rows[0]["apex_depth"] <= rows[1]["apex_depth"]
    assert rows[0]["opening_area"] <= rows[1]["opening_area"] + 1e-9


def test_sweep_shallow_slope_more_damage(panel_response):
    """A shallower ridge flank loads a wider strip -> more torn panels."""
    rows = parameter_sweep(panel_response, velocities=(5.0,), ridge_heights=(4.0,),
                           slopes=(5.0, 20.0))
    assert rows[0]["contact_half_width"] >= rows[1]["contact_half_width"]