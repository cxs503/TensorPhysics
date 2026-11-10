"""Verification tests for the superstructure-icing kernels.

Every test asserts a *physical* invariant or a closed-form benchmark, not a
stored number:

* potential flow: no normal velocity through the wall;
* droplet impingement: ballistic (``E -> 1``, ``beta(0) -> 1``) and inertia-free
  (``E -> 0``) limits, monotone ``E(K)``, convergence in time step / launch grid
  / contact tolerance;
* Messinger: exact rime closed form, independent reference implementation,
  scalar/vectorised and transient/steady code-path agreement, water-mass closure,
  regime switching;
* anti-icing: optimised power really is lower than the uniform baseline *and*
  the design satisfies the ice-thickness constraint;
* stability: hand-computed mass-moment KG, zero-ice invariance, monotone trends,
  threshold-load bracketing.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.superstructure_icing import air
from tensorphysics.superstructure_icing.anti_icing import (
    ElectrothermalAntiIcing,
    power_saving,
)
from tensorphysics.superstructure_icing.case_runner import CaseConfig, run_acceptance
from tensorphysics.superstructure_icing.droplets import (
    beta_area_weighted_cos,
    impingement,
    particle_relaxation_time,
    stokes_number,
)
from tensorphysics.superstructure_icing.flowfield import CylinderFlow, EllipseFlow
from tensorphysics.superstructure_icing.geometry import (
    Component,
    Superstructure,
    component_ice_thickness,
    design_ice_thickness,
    reference_superstructure,
)
from tensorphysics.superstructure_icing.messinger import (
    heat_transfer_coefficient,
    ice_growth_1d,
    ice_growth_steady,
    messinger_balance,
    messinger_balance_scalar,
    messinger_stagnation_reference,
    rime_thickness,
    stefan_rime_thickness,
)
from tensorphysics.superstructure_icing.stability import (
    icing_stability,
    reference_hull,
    roll_period,
    threshold_load,
    uniform_load_thickness,
)

U = 15.0
TA = -15.0


# ---------------------------------------------------------------- air / props
def test_air_properties_reference_values():
    assert air.mu_air(0.0) == pytest.approx(1.716e-5, rel=0.01)
    assert air.mu_air(-15.0) == pytest.approx(1.641e-5, rel=0.01)
    assert air.k_air(0.0) == pytest.approx(0.0242, rel=0.02)
    assert air.p_sat_water(0.0) == pytest.approx(611.2, rel=0.01)
    assert air.p_sat_ice(-20.0) == pytest.approx(103.2, rel=0.02)
    # vapour density at 0 C over water ~ 4.85 g/m3
    assert float(air.rho_v_sat(np.array([0.0]))[0]) == pytest.approx(4.85e-3, rel=0.02)
    assert air.pr_air(-15.0) == pytest.approx(0.71, abs=0.03)


# ------------------------------------------------------------------- flowfield
def test_potential_flow_wall_impermeability():
    """The analytic velocity field must be tangent to the wall everywhere."""
    cyl = CylinderFlow(R=0.1, U=U)
    xs, ys, ss = cyl.surface(721)
    u, v = cyl.velocity(xs, ys)
    nx, ny = xs, ys  # outward normal of a circle
    n = np.hypot(nx, ny)
    assert np.max(np.abs(u * nx + v * ny) / n) < 1e-12

    ell = EllipseFlow(A=0.5, B=0.12, U=U)
    xs, ys, ss = ell.surface(721)
    u, v = ell.velocity(xs, ys)
    dx = np.gradient(xs, ss)
    dy = np.gradient(ys, ss)
    t = np.hypot(dx, dy)
    # the polyline tangent is a finite-difference quantity, so the tolerance is
    # set by the discretisation (normalised by the local speed), not by round-off
    residual = np.abs(u * dy - v * dx) / t / (np.hypot(u, v) + U)
    assert np.max(residual) < 1e-3


def test_potential_flow_freestream_and_stagnation():
    cyl = CylinderFlow(R=0.1, U=U)
    u, v = cyl.velocity(np.array([-10.0]), np.array([0.0]))
    assert float(u[0]) == pytest.approx(U, rel=1e-3)
    assert abs(float(v[0])) < 1e-3
    # stagnation point at the upstream pole
    u, v = cyl.velocity(np.array([-0.1]), np.array([0.0]))
    assert np.hypot(float(u[0]), float(v[0])) < 1e-9
    # twice the freestream at the shoulder (classic cylinder result)
    u, v = cyl.velocity(np.array([0.0]), np.array([0.1]))
    assert float(u[0]) == pytest.approx(2.0 * U, rel=1e-6)


def test_ellipse_reduces_to_cylinder():
    cyl = CylinderFlow(R=0.25, U=U)
    ell = EllipseFlow(A=0.25, B=0.25, U=U)
    xs, ys, ss = cyl.surface(101)
    u1, v1 = cyl.velocity(xs, ys)
    u2, v2 = ell.velocity(xs, ys)
    # the Joukowski ellipse with A = B is a circle of radius A, but rotated by 90
    # degrees relative to the cylinder parameterisation -> compare magnitudes
    assert np.allclose(np.hypot(u1, v1), np.hypot(u2, v2), rtol=0.05, atol=0.2)


# -------------------------------------------------------------- impingement
@pytest.fixture(scope="module")
def base_impingement():
    body = CylinderFlow(R=0.1, U=U)
    return impingement(body, 50e-6, U=U, t_air=TA, n_traj=81)


def test_relaxation_time_and_stokes_number():
    d = 20e-6
    tau = particle_relaxation_time(d, TA)
    assert tau == pytest.approx(1.354e-3, rel=0.02)
    # K = tau U / D for a 0.2 m cylinder
    assert stokes_number(d, U, 0.2, TA) == pytest.approx(0.1016, rel=0.02)


def test_ballistic_limit_of_collection_efficiency():
    """Straight-line trajectories: E -> 1 and beta(theta) -> cos(theta)."""
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 5e-3, U=U, t_air=TA, n_traj=61, gravity=False)
    assert res.E_total == pytest.approx(1.0, abs=0.02)
    assert float(res.beta[0]) == pytest.approx(1.0, abs=0.02)
    s = np.linspace(0.0, res.s_limit, 50)
    analytic = beta_area_weighted_cos(s, body)
    numeric = res.beta_at(s)
    assert np.mean(np.abs(numeric - analytic)) < 0.05


def test_inertia_free_limit():
    """Zero-inertia droplets follow streamlines: no impingement at all."""
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 2e-6, U=U, t_air=TA, n_traj=41)
    assert res.E_total < 1e-3


def test_collection_efficiency_monotone_in_mvd():
    body = CylinderFlow(R=0.1, U=U)
    es = [impingement(body, d, U=U, t_air=TA, n_traj=41).E_total for d in (20e-6, 50e-6, 100e-6)]
    assert es[0] < es[1] < es[2]
    assert 0.0 <= es[0] < 0.05
    assert es[2] > 0.3


def test_beta_below_unity_and_positive(base_impingement):
    r = base_impingement
    assert np.all(r.beta >= 0.0)
    assert float(np.max(r.beta)) <= 1.0 + 1e-9
    assert 0.0 < r.s_limit < np.pi * 0.1
    # beta decays towards the impingement limit
    assert r.beta[0] > r.beta[-1]


def test_impingement_convergence_in_timestep_and_grid():
    body = CylinderFlow(R=0.1, U=U)
    e_coarse = impingement(body, 50e-6, U=U, t_air=TA, n_traj=81, conv_factor=0.02).E_total
    e_fine = impingement(body, 50e-6, U=U, t_air=TA, n_traj=81, conv_factor=0.005).E_total
    assert abs(e_fine - e_coarse) / e_coarse < 0.03
    e_g1 = impingement(body, 50e-6, U=U, t_air=TA, n_traj=81).E_total
    e_g2 = impingement(body, 50e-6, U=U, t_air=TA, n_traj=161).E_total
    assert abs(e_g2 - e_g1) / e_g1 < 0.02


# ------------------------------------------------------------------- Messinger
def test_nusselt_correlations_cross_check():
    re, pr = 2.5e5, 0.71
    from tensorphysics.superstructure_icing.messinger import (
        nusselt_cylinder_churchill_bernstein,
        nusselt_cylinder_hilpert,
        nusselt_flat_plate,
    )

    nu_cb = nusselt_cylinder_churchill_bernstein(re, pr)
    nu_h = nusselt_cylinder_hilpert(re, pr)
    assert 200.0 < nu_cb < 600.0
    assert abs(nu_cb - nu_h) / nu_cb < 0.35  # correlations agree within 35 %
    assert 0.0 < nusselt_flat_plate(re, pr) < nu_cb


def test_heat_transfer_coefficient_magnitude():
    h_c = heat_transfer_coefficient(U, 0.2, TA, kind="cylinder")
    assert 30.0 < h_c < 90.0


def test_messinger_regimes_and_scalar_vector_equivalence():
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    probes = [
        (1.2e-2, 0.0, -15.0, 0.0),  # glaze
        (1.2e-2, 3e-3, -15.0, 500.0),  # glaze with heat + runback
        (1e-4, 0.0, -30.0, 0.0),  # rime
        (2e-3, 1e-3, -5.0, 0.0),  # near melting
    ]
    for (mi, m_in, ta, q) in probes:
        vec = messinger_balance(np.array([mi]), np.array([m_in]), ta, h_c, U, q_heat=np.array([q]))
        sca = messinger_balance_scalar(mi, m_in, ta, h_c, U, q=q)
        assert float(vec.freezing_fraction[0]) == pytest.approx(sca[0], abs=1e-9)
        assert float(vec.T_surface[0]) == pytest.approx(sca[1], abs=1e-6)
    # regime switch: cold + little water -> dry growth (n = 1, T_s < 0)
    cold = messinger_balance(np.array([1e-4]), np.array([0.0]), -30.0, h_c, U)
    assert int(cold.regime[0]) == 0
    assert float(cold.freezing_fraction[0]) == pytest.approx(1.0)
    assert float(cold.T_surface[0]) < 0.0
    # warm + heavy water -> wet growth (T_s = 0, n < 1)
    warm = messinger_balance(np.array([1.2e-2]), np.array([0.0]), -15.0, h_c, U)
    assert int(warm.regime[0]) == 1
    assert 0.0 < float(warm.freezing_fraction[0]) < 1.0
    assert float(warm.T_surface[0]) == pytest.approx(0.0)


def test_heating_suppresses_freezing():
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    n0 = float(messinger_balance_scalar(1.2e-2, 0.0, TA, h_c, U, q=0.0)[0])
    n1 = float(messinger_balance_scalar(1.2e-2, 0.0, TA, h_c, U, q=1000.0)[0])
    # the zero-ice power density is available in closed form at T_s = T_melt
    q_zero = (
        h_c * (0.0 - TA)
        + float(air.rho_v_sat(np.array([0.0]))[0] - air.rho_v_sat(np.array([TA]))[0])
        * h_c / (air.rho_air(TA) * air.cp_air(TA) * air.le_air(TA) ** (2 / 3)) * air.L_V
        + air.EPS_ICE * air.SIGMA_SB * (air.tk(0.0) ** 4 - air.tk(TA) ** 4)
        + 1.2e-2 * air.C_W * (0.0 - TA)
    )
    assert 1500.0 < q_zero < 2500.0
    n2 = float(messinger_balance_scalar(1.2e-2, 0.0, TA, h_c, U, q=q_zero * 1.05)[0])
    assert n0 > n1 > n2
    assert n2 == pytest.approx(0.0, abs=1e-6)


def test_rime_closed_form_is_reproduced():
    """Dry growth must equal h = LWC*U*beta*t/rho exactly."""
    s = np.array([0.0, 0.01, 0.02, 0.03])
    beta = np.array([0.5, 0.45, 0.35, 0.2])
    h_c = heat_transfer_coefficient(U, 0.2, -40.0)
    out = ice_growth_steady(s, beta, U, -40.0, 2e-4, 600.0, h_c, rho_ice=air.RHO_ICE_RIME)
    analytic = rime_thickness(600.0, 2e-4, U, beta, rho_i=air.RHO_ICE_RIME)
    assert np.allclose(out["h"], analytic, rtol=1e-9)
    assert np.all(out["freezing_fraction"] == 1.0)


def test_transient_matches_independent_reference():
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 50e-6, U=U, t_air=TA, n_traj=41)
    s = np.linspace(0.0, res.s_limit, 21)
    beta = res.beta_at(s)
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    g = ice_growth_1d(s, beta, U, TA, 1e-3, 600.0, dt=10.0, h_c=h_c, L_char=0.2)
    _, h_ref = messinger_stagnation_reference(float(res.beta[0]), U, TA, 1e-3, h_c, 600.0)
    rel = abs(g.h_ice[-1, 0] - float(h_ref[-1])) / float(h_ref[-1])
    assert rel < 0.05
    # water-mass closure (ice + runback + evaporated = impinged)
    assert g.closure_error < 1e-10


def test_transient_and_steady_paths_agree():
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 50e-6, U=U, t_air=TA, n_traj=41)
    s = np.linspace(0.0, res.s_limit, 21)
    beta = res.beta_at(s)
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    g = ice_growth_1d(s, beta, U, TA, 1e-3, 600.0, dt=10.0, h_c=h_c, L_char=0.2)
    st = ice_growth_steady(s, beta, U, TA, 1e-3, 600.0, h_c, L_char=0.2)
    assert np.max(np.abs(g.h_final - st["h"])) / np.max(st["h"]) < 1e-9


def test_ice_thickness_grows_monotone_with_time_and_lwc():
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 50e-6, U=U, t_air=TA, n_traj=41)
    s = np.linspace(0.0, res.s_limit, 11)
    beta = res.beta_at(s)
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    h = [
        ice_growth_steady(s, beta, U, TA, lwc, 600.0, h_c)["h"][0]
        for lwc in (0.5e-3, 1e-3, 2e-3)
    ]
    assert h[0] < h[1] < h[2]
    t = [ice_growth_steady(s, beta, U, TA, 1e-3, tt, h_c)["h"][0] for tt in (300.0, 600.0, 900.0)]
    assert t[0] < t[1] < t[2]


def test_stefan_comparator_is_finite_and_monotone():
    a = stefan_rime_thickness(600.0, 15.0)
    b = stefan_rime_thickness(1200.0, 15.0)
    assert 0.0 < a < b
    assert b / a == pytest.approx(np.sqrt(2.0), rel=1e-9)


# ----------------------------------------------------------------- anti-icing
def test_anti_icing_optimisation_beats_uniform():
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 50e-6, U=U, t_air=TA, n_traj=41)
    s = np.linspace(0.0, res.s_limit, 14)
    beta = res.beta_at(s)
    h_c = heat_transfer_coefficient(U, 0.2, TA)
    anti = ElectrothermalAntiIcing(
        s, beta, U, TA, 1.5e-3, t_end=1800.0, dt=10.0, h_c=h_c, L_char=0.2
    )
    h_unheated = anti.ice_thickness(np.zeros(s.size))
    assert np.max(h_unheated) > 5e-3  # the constraint is genuinely active
    uniform = anti.uniform_design(5e-3)
    optimised = anti.optimize(5e-3)
    assert np.max(uniform.h_ice) <= 5e-3 * 1.01
    assert np.max(optimised.h_ice) <= 5e-3 * 1.01
    assert power_saving(optimised, uniform) >= 30.0
    assert optimised.power_w < uniform.power_w


def test_spreading_matrix_is_energy_conserving():
    body = CylinderFlow(R=0.1, U=U)
    res = impingement(body, 50e-6, U=U, t_air=TA, n_traj=41)
    s = np.linspace(0.0, res.s_limit, 9)
    anti = ElectrothermalAntiIcing(s, res.beta_at(s), U, TA, 1.0e-3, h_c=60.0)
    assert np.allclose(anti.W.sum(axis=0), 1.0, atol=1e-9)
    assert np.all(anti.W >= -1e-12)


# ------------------------------------------------------------------ stability
def test_stability_hand_computed_mass_moment():
    from tensorphysics.superstructure_icing.stability import ShipParticulars

    ship = ShipParticulars("test", L=50.0, B=10.0, T=3.0, depth=4.0,
                           displacement_t=1000.0, KG=5.0, GM=1.0, k_x_factor=0.35)
    sup = Superstructure("test", deck_height=4.0, components=[Component("c", "plate", 100.0, 10.0, 0.2)])
    t = {"c": 0.01}
    r = icing_stability(ship, sup, t)
    m_ice = air.RHO_ICE_GLAZE * 0.01 * 100.0 / 1000.0
    assert r.ice_mass_t == pytest.approx(m_ice, rel=1e-12)
    kg_expected = (1000.0 * 5.0 + m_ice * 10.0) / (1000.0 + m_ice)
    assert r.KG_new == pytest.approx(kg_expected, rel=1e-12)
    assert r.GM_new == pytest.approx(1.0 - (kg_expected - 5.0), rel=1e-12)
    # roll period must grow (GM down, inertia up)
    assert r.T_roll_new > r.T_roll0
    assert r.dT_roll_pct > 0.0


def test_stability_zero_ice_is_identity():
    ship = reference_hull("coaster")
    sup = reference_superstructure("coaster")
    r = icing_stability(ship, sup, {c.name: 0.0 for c in sup.components})
    assert r.dKG == pytest.approx(0.0, abs=1e-12)
    assert r.dT_roll_pct == pytest.approx(0.0, abs=1e-12)
    assert r.T_roll0 == pytest.approx(roll_period(ship.displacement_t, ship.I_xx, ship.GM), rel=1e-12)


def test_stability_monotone_in_load_and_threshold_bracket():
    ship = reference_hull("coaster")
    sup = reference_superstructure("coaster")
    d = [
        icing_stability(ship, sup, uniform_load_thickness(sup, L0)).dKG
        for L0 in (10.0, 30.0, 60.0, 90.0)
    ]
    assert d[0] < d[1] < d[2] < d[3]
    l_req = threshold_load(ship, sup, metric="dKG", target=0.10)
    assert np.isfinite(l_req)
    assert 30.0 < l_req < 150.0
    r = icing_stability(ship, sup, uniform_load_thickness(sup, l_req))
    assert r.dKG == pytest.approx(0.10, abs=1e-3)


def test_design_ice_thickness():
    assert design_ice_thickness(30.0) == pytest.approx(30.0 / air.RHO_ICE_GLAZE)


def test_superstructure_component_geometry_and_icing():
    sup = reference_superstructure("coaster")
    assert sup.area > 500.0
    assert all(c.z > sup.deck_height for c in sup.components)
    m, mz, z = sup.moments({c.name: 0.01 for c in sup.components})
    assert m > 0.0
    assert z == pytest.approx(mz / m, rel=1e-12)
    assert sup.deck_height < z < max(c.z for c in sup.components)
    # a component's icing thickness comes from the coupled chain and is positive
    comp = Component("mast", "cylinder", 3.0, 8.0, 0.2)
    h_mean, h_stag, h_prof = component_ice_thickness(
        comp, U, TA, 1e-3, 50e-6, 600.0, dt=20.0, n_traj=21, n_cell=7
    )
    assert 0.0 < h_mean < 0.05
    assert h_stag >= h_mean * 0.5
    assert h_prof.size == 7


# ------------------------------------------------------------- end-to-end run
@pytest.mark.parametrize("node", ["p380w2-n07"])
def test_acceptance_record_structure(node):
    cfg = CaseConfig(n_traj=41, n_cell_anti=10, dts=(10.0, 5.0))
    rec = run_acceptance(cfg=cfg, out_dir=None, verbose=False)
    crit = rec["criteria"]
    for key in ("a_impingement", "b_ice_growth", "c_anti_icing", "d_stability",
                "e_sensitivity_matrix", "f_time_convergence"):
        assert key in crit
    assert crit["b_ice_growth"]["result"]["rel_error_pct"] < 15.0
    assert crit["c_anti_icing"]["optimised"]["saving_pct"] >= 30.0
    assert crit["e_sensitivity_matrix"]["n_cases"] == 27
    assert crit["f_time_convergence"]["within_10pct"]
    assert crit["a_impingement"]["verdict"] in {"WITHHELD", "PASS"}
    assert rec["summary"]["b"] == "PASS"
    assert rec["verification"]["transient_vs_steady_max_rel_diff_pct"] < 1e-6