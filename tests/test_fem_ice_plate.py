"""Tests for the FEM ice-plate cohesive-zone / J2-plasticity verification face.

Covers the cohesive traction-separation laws and the moment-rotation hinge,
the J2 return mapping and consistent tangent, the radial-return FE solution of
the pressurised thick cylinder, the axisymmetric plate on an elastic foundation
with a cohesive ring crack, the coupled bar energy audit, and the three
negative controls required by the task.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.fem_ice_plate.cohesive_zone import (
    BilinearTSL,
    CohesiveHinge,
    ExponentialTSL,
    dissipated_energy_numerical,
)
from tensorphysics.fem_ice_plate.coupled import (
    bulk_1d_return_map,
    solve_bar_with_interface,
)
from tensorphysics.fem_ice_plate.foundation_plate import (
    plate_flexural_rigidity,
    solve_axisym_plate_foundation,
)
from tensorphysics.fem_ice_plate.j2_plasticity import (
    J2Material,
    J2State,
    consistent_tangent_numerical,
    equivalent_stress,
    lamé_compliance,
    return_map,
    thick_cylinder_elastic,
    thick_cylinder_fe,
    thick_cylinder_plastic_reference,
    uniaxial_elastoplastic_modulus,
    uniaxial_reference,
    uniaxial_tension_curve,
)

# ---------------------------------------------------------------------------
# cohesive zone: traction-separation laws
# ---------------------------------------------------------------------------
def test_bilinear_fracture_energy_matches_numerical_integral():
    tsl = BilinearTSL(sigma_max=1.2e6, delta_f=5e-4)
    exact = tsl.G_f
    assert exact == pytest.approx(0.5 * 1.2e6 * 5e-4, rel=1e-12)
    numeric = dissipated_energy_numerical(lambda d: tsl.traction(d), tsl.delta_f)
    assert numeric == pytest.approx(exact, rel=1e-3)


def test_bilinear_damage_is_monotone_and_irreversible():
    tsl = BilinearTSL(sigma_max=1e6, delta_f=1e-4, delta_0=1e-5)
    d = [tsl.damage(x) for x in np.linspace(0, 1.2e-4, 50)]
    assert d[0] == 0.0 and d[-1] == 1.0
    assert all(np.diff(d) >= -1e-12), "damage must never decrease"
    # unloading from a damaged state follows the secant, not the virgin slope
    t_after = tsl.traction(0.5 * 1e-4, delta_history=0.8e-4)
    assert 0.0 < t_after < tsl.sigma_max


def test_bilinear_traction_vanishes_beyond_failure_separation():
    tsl = BilinearTSL(sigma_max=1e6, delta_f=1e-4)
    assert tsl.traction(2e-4, delta_history=2e-4) == 0.0
    assert tsl.dissipate(2e-4) == pytest.approx(tsl.G_f)


def test_exponential_tsl_peak_and_energy():
    tsl = ExponentialTSL(sigma_max=2e6, G_f=300.0)
    # peak traction is reached exactly at delta_0
    assert tsl.traction(tsl.delta_0) == pytest.approx(2e6, rel=1e-12)
    numeric = dissipated_energy_numerical(
        lambda d: tsl.traction(d), 20.0 * tsl.delta_0, n=400001
    )
    assert numeric == pytest.approx(300.0, rel=2e-3)


def test_cohesive_hinge_fracture_energy_round_trip():
    M_peak, G_f, h = 1.2e5, 20.0, 1.0
    hinge = CohesiveHinge.from_fracture_energy(M_peak, G_f, h)
    assert hinge.G_f == pytest.approx(G_f, rel=1e-12)
    # closed-form dissipation at complete failure equals int M dphi = G_f * h
    assert hinge.dissipate(hinge.phi_f) == pytest.approx(G_f * h, rel=1e-12)
    numeric = dissipated_energy_numerical(lambda p: hinge.moment(p), hinge.phi_f)
    assert numeric == pytest.approx(G_f * h, rel=1e-3)


# ---------------------------------------------------------------------------
# J2 plasticity: return mapping
# ---------------------------------------------------------------------------
def test_return_map_elastic_below_yield():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=250e6)
    eps = np.array([1e-3, -3e-4, -3e-4, 0.0, 0.0, 0.0])
    sigma, state = return_map(mat, eps)
    assert state.ep == 0.0
    assert np.allclose(sigma, mat.elastic_matrix() @ eps)


def test_return_map_yields_on_the_yield_surface():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=250e6, hardening="linear", H=2e9)
    eps = np.array([3e-3, -1e-3, -1e-3, 0.0, 0.0, 0.0])
    sigma, state = return_map(mat, eps)
    assert state.ep > 0.0
    assert equivalent_stress(sigma) == pytest.approx(
        mat.yield_stress(state.ep), rel=1e-10
    )


def test_consistent_tangent_matches_uniaxial_elastoplastic_modulus():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=250e6, hardening="linear", H=2e9)
    eps = np.array([3e-3, -9e-4, -9e-4, 0.0, 0.0, 0.0])
    D = consistent_tangent_numerical(mat, eps)
    # a uniaxial-stress perturbation on the plastic branch gives E_ep
    Hp = mat.hardening_modulus(0.0)
    E_ep = mat.E * Hp / (mat.E + Hp)
    assert D[0, 0] > 0.0
    assert uniaxial_elastoplastic_modulus(mat) == pytest.approx(E_ep)
    # the tangent must be symmetric (associated plasticity)
    assert np.allclose(D, D.T, rtol=1e-6, atol=1e3)


def test_uniaxial_curve_matches_closed_form():
    b = 25.0
    Q = (510e6 - 355e6) / (1.0 - math.exp(-b * 0.15))
    mat = J2Material(E=206e9, nu=0.3, sigma_y0=355e6, hardening="voce", Q=Q, b=b)
    num = uniaxial_tension_curve(mat, eps_max=0.06, n_steps=200)
    ref = uniaxial_reference(mat, num["eps"])
    mask = num["eps"] > 0
    rel = np.abs(num["sigma"][mask] - ref["sigma"][mask]) / ref["sigma"][mask]
    assert float(np.sqrt(np.mean(rel**2))) < 0.01


def test_uniaxial_tangent_after_yield():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=5e9)
    num = uniaxial_tension_curve(mat, eps_max=0.02, n_steps=400)
    slope = (num["sigma"][-1] - num["sigma"][-2]) / (num["eps"][-1] - num["eps"][-2])
    Hp = mat.hardening_modulus(0.0)
    assert slope == pytest.approx(mat.E * Hp / (mat.E + Hp), rel=1e-3)


# ---------------------------------------------------------------------------
# thick-walled cylinder: FE vs closed form
# ---------------------------------------------------------------------------
def test_cylinder_elastic_matches_lame():
    a, b, p, E, nu = 0.05, 0.10, 60e6, 200e9, 0.3
    mat = J2Material(E=E, nu=nu, sigma_y0=250e6)
    res = thick_cylinder_fe(mat, a, b, p, n_el=160, n_inc=6, tol=1e-9)
    ref = thick_cylinder_elastic(a, b, p, E, nu, r=res.r_gauss)
    for i, key in enumerate(("sigma_r", "sigma_theta", "sigma_z")):
        scale = np.max(np.abs(ref[key]))
        # linear radial elements: O(h) convergence of the Gauss-point stresses
        assert np.max(np.abs(res.sigma_gauss[:, i] - ref[key])) / scale < 0.02
    assert np.max(np.abs(res.ep_gauss)) == 0.0


def test_cylinder_elastoplastic_matches_autofrettage_solution():
    a, b, p, sy = 0.05, 0.10, 170e6, 250e6
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=sy, hardening="perfect")
    res = thick_cylinder_fe(mat, a, b, p, n_el=80, n_inc=24, tol=1e-8)
    ref = thick_cylinder_plastic_reference(a, b, p, sy, E=200e9)
    fld = ref["fields"](res.r_gauss)
    for i, key in enumerate(("sigma_r", "sigma_theta")):
        scale = np.max(np.abs(fld[key]))
        assert np.max(np.abs(res.sigma_gauss[:, i] - fld[key])) / scale < 0.05
    assert abs(res.r_plastic - ref["c"]) / ref["c"] < 0.05


def test_cylinder_plane_strain_constraint_holds():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=250e6, hardening="perfect")
    res = thick_cylinder_fe(mat, 0.05, 0.10, 160e6, n_el=40, n_inc=12, tol=1e-8)
    assert np.max(np.abs(res.eps_gauss[:, 2])) == 0.0


def test_cylinder_unloading_is_elastic():
    a, b, pmax, E, nu = 0.05, 0.10, 170e6, 200e9, 0.3
    mat = J2Material(E=E, nu=nu, sigma_y0=250e6, hardening="perfect")
    path = list(np.linspace(0, pmax, 20)) + list(np.linspace(pmax, 0, 21)[1:])
    res = thick_cylinder_fe(mat, a, b, pmax, n_el=50, p_path=path, tol=1e-8)
    i_peak = int(np.argmax(res.p_history))
    ph, uh = res.p_history[i_peak:], res.u_bore_history[i_peak:]
    slope = np.polyfit(ph, uh, 1)[0]
    assert slope == pytest.approx(lamé_compliance(a, b, E, nu), rel=1e-3)
    assert abs(res.u_bore_history[-1]) > 0.0


def test_cylinder_energy_closure():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=250e6, hardening="perfect")
    res = thick_cylinder_fe(mat, 0.05, 0.10, 160e6, n_el=50, n_inc=16, tol=1e-8)
    e = res.energies()
    closure = abs(e["W_ext"] - e["W_elastic"] - e["W_plastic"]) / e["W_ext"]
    assert closure < 0.02


# ---------------------------------------------------------------------------
# plate on elastic foundation with a cohesive ring crack
# ---------------------------------------------------------------------------
def _plate_config():
    E, h, nu, k = 30e9, 1.0, 0.3, 1e4
    D = plate_flexural_rigidity(E, h, nu)
    return D, nu, k, (D / k) ** 0.25


def test_intact_plate_matches_hetenyi_deflection():
    from tensorphysics.elastic_foundation.plates import winkler_plate_patch_load

    D, nu, k, l0 = _plate_config()
    a, P = 0.1 * l0, 1e5
    w0 = float(winkler_plate_patch_load(np.array([0.0]), P, a, D, k)[0])
    res = solve_axisym_plate_foundation(
        D, nu, k, patch_radius=a, n_el=60, grading=1.6, w_center_steps=[w0]
    )
    ana = winkler_plate_patch_load(res.r, res.load_factor[-1], a, D, k)
    mask = (res.r > 0.5 * l0) & (res.r < 5 * l0)
    assert np.max(np.abs(res.w[mask] - ana[mask])) / np.max(np.abs(ana[mask])) < 0.03


def test_intact_plate_matches_hetenyi_moment():
    from tensorphysics.elastic_foundation.plates import (
        winkler_plate_patch_load,
        winkler_plate_patch_moment_r,
    )

    D, nu, k, l0 = _plate_config()
    a = 0.1 * l0
    w0 = float(winkler_plate_patch_load(np.array([0.0]), 1e5, a, D, k)[0])
    res = solve_axisym_plate_foundation(
        D, nu, k, patch_radius=a, n_el=60, grading=1.6, w_center_steps=[w0]
    )
    sel = (res.r_elem > 0.4 * l0) & (res.r_elem < 3.5 * l0)
    ana = winkler_plate_patch_moment_r(
        res.r_elem[sel], res.load_factor[-1], a, D, k, nu
    )
    assert np.max(np.abs(res.M_r_elem[sel] - ana)) / np.max(np.abs(ana)) < 0.03


def test_cohesive_ring_crack_dissipates_the_prescribed_fracture_energy():
    E, h, nu, k = 5e9, 1.0, 0.3, 9810.0
    D = plate_flexural_rigidity(E, h, nu)
    l0 = (D / k) ** 0.25
    M_peak, G_f = 0.7e6 * h**2 / 6.0, 20.0
    hinge = CohesiveHinge.from_fracture_energy(M_peak, G_f, h, phi_0_frac=1e-5)
    res = solve_axisym_plate_foundation(
        D, nu, k, patch_radius=0.2 * l0, n_el=80, grading=1.6,
        crack_radius=0.3 * l0, hinge=hinge,
        w_center_steps=np.linspace(0, 0.35, 31)[1:], tol=1e-9,
    )
    area = 2.0 * math.pi * res.crack_radius * h
    assert res.fracture_energy / area == pytest.approx(G_f, rel=0.02)
    eb = res.energy_balance()
    assert eb["residual_rel"] < 0.02


def test_cracked_plate_is_softer_than_intact():
    E, h, nu, k = 5e9, 1.0, 0.3, 9810.0
    D = plate_flexural_rigidity(E, h, nu)
    l0 = (D / k) ** 0.25
    hinge = CohesiveHinge.from_fracture_energy(0.7e6 * h**2 / 6.0, 20.0, h)
    steps = np.linspace(0, 0.30, 21)[1:]
    kw = dict(D=D, nu=nu, k=k, patch_radius=0.2 * l0, n_el=60, grading=1.6,
              w_center_steps=steps, tol=1e-9)
    intact = solve_axisym_plate_foundation(**kw)
    cracked = solve_axisym_plate_foundation(crack_radius=0.3 * l0, hinge=hinge, **kw)
    assert cracked.load_factor[-1] < intact.load_factor[-1]


# ---------------------------------------------------------------------------
# coupled bar and negative controls
# ---------------------------------------------------------------------------
def test_bulk_1d_return_map_matches_closed_form():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=2e9)
    strain = 0.003
    sigma, ep, dw = bulk_1d_return_map(mat, eps=strain, ep=0.0)
    assert sigma == pytest.approx(mat.yield_stress(ep), rel=1e-10)
    assert strain == pytest.approx(sigma / mat.E + ep, rel=1e-10)
    assert dw > 0.0


def test_coupled_bar_energy_conservation():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=2e9)
    tsl = BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=2e-4)
    res = solve_bar_with_interface(mat, tsl, L=0.1, n_steps=300)
    eb = res.energy_balance()
    assert eb["W_plastic"] > 0.0 and eb["W_fracture"] > 0.0
    assert eb["residual_rel"] < 0.02


def test_negative_control_zero_fracture_energy():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=2e9)
    ref = solve_bar_with_interface(
        mat, BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=2e-4), L=0.1, n_steps=200
    )
    brittle = solve_bar_with_interface(
        mat, BilinearTSL(sigma_max=420e6, delta_f=1e-7, delta_0=1e-8), L=0.1, n_steps=200
    )
    assert brittle.W_fracture < 1e-3 * ref.W_fracture


def test_negative_control_infinite_yield_stress_is_elastic():
    elastic = J2Material(E=200e9, nu=0.3, sigma_y0=1e15)
    tsl = BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=2e-4)
    res = solve_bar_with_interface(elastic, tsl, L=0.1, n_steps=200)
    assert np.max(res.ep_bulk) == 0.0
    assert res.W_plastic == 0.0


def test_negative_control_rigid_interface():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=2e9)
    rigid = solve_bar_with_interface(
        mat, BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=1e-7), L=0.1, n_steps=300
    )
    compliant = solve_bar_with_interface(
        mat, BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=4e-4), L=0.1, n_steps=300
    )
    i_r = int(np.argmax(rigid.force >= 0.99 * rigid.peak_force))
    i_c = int(np.argmax(compliant.force >= 0.99 * compliant.peak_force))
    assert rigid.delta_interface[i_r] < compliant.delta_interface[i_c]


# ---------------------------------------------------------------------------
# verification harness
# ---------------------------------------------------------------------------
def test_verification_harness_all_checks_pass():
    from tensorphysics.fem_ice_plate.verification import (
        check_c1_hetényi,
        check_c2_uniaxial,
        check_c3_cylinder,
        check_c5_unloading,
        check_c6_negative_controls,
        check_c7_energy,
    )

    assert check_c1_hetényi(n_el=50)["pass"]
    assert check_c2_uniaxial()["pass"]
    assert check_c3_cylinder()["pass"]
    assert check_c5_unloading()["pass"]
    c6 = check_c6_negative_controls()
    assert c6["gf_to_zero"]["pass"] and c6["sigma_y_to_infinity"]["pass"]
    assert c6["rigid_interface"]["pass"]
    assert check_c7_energy()["pass"]


def test_results_json_is_written_and_complete(tmp_path):
    from tensorphysics.fem_ice_plate.verification import write_results

    res = {
        "summary": {"checks": {"C1": True}, "all_pass": True},
        "C1": {"pass": True, "deflection_max_dev": 1e-5},
    }
    path = write_results(res, tmp_path / "out.json")
    import json

    loaded = json.loads(path.read_text())
    assert loaded["summary"]["all_pass"] is True
    assert loaded["C1"]["deflection_max_dev"] == pytest.approx(1e-5)