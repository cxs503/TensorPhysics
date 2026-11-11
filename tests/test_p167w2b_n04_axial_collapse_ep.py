"""P167W2B-N04 verification tests for ``axial_collapse_ep``.

Covers the five acceptance items of the campaign node:

* von Mises + isotropic hardening closed form / incremental integrator / consistent tangent;
* Smith panel collapse with elastoplastic element curves vs the Paik & Thayamballi
  (2003) simplified method;
* IACS ice-belt pressure-area exponent inversion;
* simply supported plate elastic buckling vs Timoshenko;
* negative controls (wrong parameters must be rejected, trends must be physical).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.elastoplastic_collapse import axial_collapse_ep as ep

MAT = ep.IsotropicHardeningMaterial()
PANEL = ep.StiffenedPanelScantlings()


# ---------------------------------------------------------------------------
# material model
# ---------------------------------------------------------------------------
def test_material_rejects_invalid_parameters():
    with pytest.raises(ValueError):
        ep.IsotropicHardeningMaterial(E=-1.0)
    with pytest.raises(ValueError):
        ep.IsotropicHardeningMaterial(nu=0.5)
    with pytest.raises(ValueError):
        ep.IsotropicHardeningMaterial(sigma_y0=0.0)
    with pytest.raises(ValueError):
        ep.IsotropicHardeningMaterial(H=-1.0e9)


def test_uniaxial_elastic_branch_is_hooke():
    sigma, tangent, ebar = ep.uniaxial_closed_form(0.5 * MAT.eps_yield, MAT)
    assert sigma == pytest.approx(0.5 * MAT.sigma_y0, rel=1e-12)
    assert tangent == pytest.approx(MAT.E, rel=1e-12)
    assert ebar == 0.0


def test_uniaxial_yield_onset():
    sigma, tangent, _ = ep.uniaxial_closed_form(MAT.eps_yield, MAT)
    assert sigma == pytest.approx(MAT.sigma_y0, rel=1e-12)
    assert tangent == pytest.approx(MAT.E, rel=1e-12)


def test_uniaxial_hardening_slope_is_algorithmic_tangent():
    et = ep.IsotropicHardeningMaterial(E=200.0e9, nu=0.0, sigma_y0=300.0e6, H=20.0e9)
    eps = 0.01
    h = 1e-12
    s_plus, _, _ = ep.uniaxial_closed_form(eps + h, et)
    s_minus, _, _ = ep.uniaxial_closed_form(eps - h, et)
    fd = (s_plus - s_minus) / (2 * h)
    assert fd == pytest.approx(et.tangent_modulus, rel=1e-6)
    assert et.tangent_modulus == pytest.approx(et.E * et.H / (et.E + et.H), rel=1e-12)


def test_uniaxial_closed_form_matches_analytic_bilinear_within_1pct():
    eps = np.linspace(1e-6, 0.01, 50)
    closed = np.array([ep.uniaxial_closed_form(e, MAT)[0] for e in eps])
    ref = np.where(eps <= MAT.eps_yield, MAT.E * eps,
                   MAT.sigma_y0 + MAT.H * (eps - MAT.eps_yield) / (1.0 + MAT.H / MAT.E))
    assert np.max(np.abs(closed - ref)) / ref[-1] < 0.01


def test_uniaxial_incremental_integrator_matches_closed_form():
    eps = np.linspace(0.0, 0.01, 101)
    closed = np.array([ep.uniaxial_closed_form(e, MAT)[0] for e in eps])
    inc = ep.uniaxial_incremental(eps, MAT)["sigma"]
    assert np.max(np.abs(inc - closed)) / closed[-1] < 1e-12


def test_inverse_strain_solves_back_to_stress():
    for target in (0.5 * MAT.sigma_y0, MAT.sigma_y0, 1.4 * MAT.sigma_y0):
        e = ep.strain_for_uniaxial_stress(target, MAT)
        assert ep.uniaxial_closed_form(e, MAT)[0] == pytest.approx(target, rel=1e-12)


def test_j2_return_mapping_satisfies_yield_surface():
    strain = np.array([0.004, -0.0012, -0.0012, 0.0015, 0.0, 0.0])
    rm = ep.j2_return_mapping(strain, np.zeros(6), 0.0, MAT)
    assert rm["yielded"] is True
    s = rm["stress"]
    S = np.array([[s[0], s[3], s[5]], [s[3], s[1], s[4]], [s[5], s[4], s[2]]])
    S_dev = S - np.trace(S) / 3.0 * np.eye(3)
    vm = math.sqrt(1.5 * float(np.sum(S_dev * S_dev)))
    assert vm == pytest.approx(MAT.sigma_y0 + MAT.H * rm["ebar_p"], rel=1e-12)


def test_j2_elastic_branch_returns_hooke_tangent():
    strain = np.array([1e-5, 0.0, 0.0, 0.0, 0.0, 0.0])
    rm = ep.j2_return_mapping(strain, np.zeros(6), 0.0, MAT)
    assert rm["yielded"] is False
    C = ep.j2_consistent_tangent(strain, np.zeros(6), 0.0, MAT)
    # uniaxial elastic stiffness must reproduce E for a uniaxial strain path
    eps_tensor = np.array([1e-5, -MAT.nu * 1e-5, -MAT.nu * 1e-5, 0.0, 0.0, 0.0])
    sig = C @ eps_tensor
    assert sig[0] == pytest.approx(MAT.E * 1e-5, rel=1e-9)


def test_j2_consistent_tangent_matches_finite_difference():
    strain = np.array([0.004, -0.0012, -0.0012, 0.0015, 0.0, 0.0])
    ep0 = np.zeros(6)
    C = ep.j2_consistent_tangent(strain, ep0, 0.0, MAT)
    C_fd = np.zeros((6, 6))
    h = 1e-9
    for j in range(6):
        dp = np.zeros(6)
        dp[j] = h
        sp = np.array(ep.j2_return_mapping(strain + dp, ep0, 0.0, MAT)["stress"], dtype=float)
        sm = np.array(ep.j2_return_mapping(strain - dp, ep0, 0.0, MAT)["stress"], dtype=float)
        C_fd[:, j] = (sp - sm) / (2.0 * h)
    assert np.max(np.abs(C - C_fd)) / np.max(np.abs(C)) < 1e-6


def test_hardening_raises_the_yield_stress():
    soft = ep.IsotropicHardeningMaterial(H=0.0)
    hard = ep.IsotropicHardeningMaterial(H=2.0e9)
    assert ep.uniaxial_closed_form(0.01, hard)[0] > ep.uniaxial_closed_form(0.01, soft)[0]


# ---------------------------------------------------------------------------
# plate elastic buckling
# ---------------------------------------------------------------------------
def test_timoshenko_k_factor_minimum():
    # square plate -> k = 4 for m = 1
    tim = ep.elastic_plate_buckling_timoshenko(0.7, 0.7, 0.012, MAT)
    assert tim["k"] == pytest.approx(4.0, rel=1e-12)
    assert tim["m"] == 1.0
    # a/b = 2 -> k = 4 with m = 2
    tim2 = ep.elastic_plate_buckling_timoshenko(1.4, 0.7, 0.012, MAT)
    assert tim2["k"] == pytest.approx(4.0, rel=1e-12)
    assert tim2["m"] == 2.0


@pytest.mark.parametrize("aspect", [1.0, 2.0, 3.0])
def test_fd_buckling_matches_timoshenko_within_5pct(aspect):
    a = aspect * PANEL.b
    tim = ep.elastic_plate_buckling_timoshenko(a, PANEL.b, PANEL.t, MAT)
    fd = ep.fd_plate_buckling_stress(a, PANEL.b, PANEL.t, MAT, h=PANEL.b / 4.0, refinements=1)
    assert abs(fd["sigma_cr"] - tim["sigma_cr"]) / tim["sigma_cr"] < 0.05


def test_fd_buckling_converges_at_second_order():
    fd = ep.fd_plate_buckling_stress(2 * PANEL.b, PANEL.b, PANEL.t, MAT,
                                     h=PANEL.b / 4.0, refinements=2)
    assert fd["observed_order"] == pytest.approx(2.0, abs=0.15)


def test_buckling_scales_with_thickness_squared():
    t1, t2 = 0.010, 0.020
    s1 = ep.elastic_plate_buckling_timoshenko(1.4, 0.7, t1, MAT)["sigma_cr"]
    s2 = ep.elastic_plate_buckling_timoshenko(1.4, 0.7, t2, MAT)["sigma_cr"]
    assert s2 / s1 == pytest.approx((t2 / t1) ** 2, rel=1e-12)


# ---------------------------------------------------------------------------
# IACS pressure-area law
# ---------------------------------------------------------------------------
def test_iacs_pressure_clamps_below_reference_area():
    assert ep.iacs_pressure(0.1, 1.0e6, 0.5) == pytest.approx(1.0e6, rel=1e-12)
    assert ep.iacs_pressure(0.5, 1.0e6, 0.5) == pytest.approx(1.0e6, rel=1e-12)


def test_iacs_pressure_is_monotone_decreasing():
    areas = np.geomspace(0.5, 20.0, 30)
    p = [ep.iacs_pressure(a, 1.0e6, 0.5) for a in areas]
    assert all(p[i] > p[i + 1] for i in range(len(p) - 1))


def test_iacs_pressure_invalid_input():
    with pytest.raises(ValueError):
        ep.iacs_pressure(0.0, 1.0e6, 0.5)
    with pytest.raises(ValueError):
        ep.iacs_pressure(1.0, 1.0e6, 0.0)


def test_pressure_area_exponent_recovered_clean():
    ident = ep.pressure_area_identifiability(1.0e6, 0.5)
    assert ident["clean"]["exponent"] == pytest.approx(-0.3, abs=1e-6)


def test_pressure_area_exponent_in_band_under_noise():
    ident = ep.pressure_area_identifiability(1.0e6, 0.5, noise_cov=0.05)
    for fit in ident["noisy"]:
        assert -0.35 <= fit["exponent"] <= -0.25
        assert fit["r2"] > 0.90


def test_fit_rejects_bad_input():
    with pytest.raises(ValueError):
        ep.fit_pressure_area_exponent([1.0, 2.0], [1.0, 0.5])
    with pytest.raises(ValueError):
        ep.fit_pressure_area_exponent([1.0, 2.0, 3.0], [1.0, 0.0, 0.5])


# ---------------------------------------------------------------------------
# element curves and panel collapse
# ---------------------------------------------------------------------------
def test_faulkner_ultimate_limits():
    stocky = ep.faulkner_plate_ultimate(0.2, 0.02, MAT, residual_stress_ratio=0.0)
    slender = ep.faulkner_plate_ultimate(1.2, 0.01, MAT, residual_stress_ratio=0.0)
    assert stocky["beta"] < 1.0
    # the empirical curve is only fitted for beta >= 1; stocky plates clip at yield
    assert stocky["phi"] == pytest.approx(1.0, rel=1e-12)
    assert stocky["sigma_u"] == pytest.approx(MAT.sigma_y0, rel=1e-12)
    assert 0.0 < slender["sigma_u"] < stocky["sigma_u"]


def test_column_curves_are_continuous_at_slenderness_transition():
    sy = MAT.sigma_y0
    lo = ep.johnson_ostenfeld(sy, 0.5 * sy - 1e-6)
    hi = ep.johnson_ostenfeld(sy, 0.5 * sy + 1e-6)
    assert lo == pytest.approx(hi, rel=1e-5)
    # Perry-Robertson reduces to min(yield, Euler) for a perfect strut
    assert ep.perry_robertson(sy, 0.4 * sy, 0.0) == pytest.approx(0.4 * sy, rel=1e-9)
    assert ep.perry_robertson(sy, 3.0 * sy, 0.0) == pytest.approx(sy, rel=1e-9)


def test_section_properties_match_hand_calculation():
    sec = ep.stiffener_section_properties(0.200, 0.010, 0.100, 0.015)
    area = 0.200 * 0.010 + 0.100 * 0.015
    assert sec["A"] == pytest.approx(area, rel=1e-12)
    assert 0.0 < sec["y_bar"] < 0.215
    assert sec["I"] > 0.0 and sec["r"] == pytest.approx(math.sqrt(sec["r2"]), rel=1e-12)


def test_panel_collapse_has_limit_point_and_stays_below_yield():
    res = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=400)
    assert 0.0 < res["eps_u"] < 0.012
    assert 0.0 < res["sigma_u"] <= MAT.sigma_y0
    assert res["post_peak_ratio"] < 1.0
    assert res["elastic_slope"] == pytest.approx(MAT.E, rel=1e-6)


def test_panel_ultimate_matches_published_simplified_method_within_10pct():
    res = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=800)
    ref = ep.paik_thayamballi_simplified_ultimate(PANEL, MAT)
    assert abs(res["sigma_u"] - ref["sigma_u"]) / ref["sigma_u"] < 0.10


def test_panel_ultimate_is_monotone_in_plate_slenderness():
    sigmas = []
    for b in (0.40, 0.55, 0.70, 0.85, 1.00):
        res = ep.smith_panel_axial_collapse(ep.StiffenedPanelScantlings(b=b), MAT, n_steps=400)
        sigmas.append(res["sigma_u"])
    assert all(sigmas[i] > sigmas[i + 1] for i in range(len(sigmas) - 1))


def test_panel_ultimate_is_converged_in_strain_steps():
    a = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=200)["sigma_u"]
    b = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=1600)["sigma_u"]
    assert abs(a - b) / b < 0.01


def test_hardening_raises_the_post_buckling_plateau():
    """Hardening enters through the current yield stress of the plating plateau.

    At the collapse strain the plastic strain is small, so the *ultimate* is
    almost insensitive; the effect must nevertheless be visible on the plateau
    (large shortening), which is what the test checks.
    """
    soft = ep.smith_panel_axial_collapse(PANEL, ep.IsotropicHardeningMaterial(H=0.0),
                                        n_steps=800, eps_max=0.012)
    hard = ep.smith_panel_axial_collapse(PANEL, ep.IsotropicHardeningMaterial(H=50.0e9),
                                        n_steps=800, eps_max=0.012)
    idx = int(np.argmin(np.abs(hard["eps"] - 0.008)))
    assert hard["sigma_plate"][idx] > soft["sigma_plate"][idx]
    assert hard["cap_plate"] >= soft["cap_plate"]


def test_residual_stress_lowers_panel_ultimate():
    low = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=400, residual_stress_ratio=0.0)
    high = ep.smith_panel_axial_collapse(PANEL, MAT, n_steps=400, residual_stress_ratio=0.30)
    assert high["sigma_u"] < low["sigma_u"]


def test_longer_frame_spacing_lowers_ultimate():
    short = ep.smith_panel_axial_collapse(ep.StiffenedPanelScantlings(a=2.4), MAT, n_steps=400)
    long = ep.smith_panel_axial_collapse(ep.StiffenedPanelScantlings(a=9.6), MAT, n_steps=400)
    assert long["sigma_u"] <= short["sigma_u"] + 1.0


def test_scantlings_validate_input():
    with pytest.raises(ValueError):
        ep.StiffenedPanelScantlings(t=0.0)
    with pytest.raises(ValueError):
        ep.euler_column_stress(MAT, 1e-4, 0.0)


def test_artifacts_report_all_criteria_pass():
    """Regression guard: the published artifact must stay green."""
    import json
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "artifacts" / "p167w2b_n04" / "results.json"
    if not path.exists():
        pytest.skip("results.json not generated yet")
    data = json.loads(path.read_text())
    failed = [k for k, v in data["criteria"].items() if not v["pass"]]
    assert not failed, f"failed criteria: {failed}"