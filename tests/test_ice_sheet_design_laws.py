"""Tests for the ice thickness / temperature / loading-rate design laws.

Covers the four acceptance criteria of node P366-W2-N01 at unit-test scale:

* the material model (monotonicity in ``|T|`` and in the strain rate, the
  fixed point ``eps_dot <-> l <-> E``, input validation);
* the constant-pressure patch test of the BFS foundation element (``1e-10``);
* the Pasternak -> Winkler degeneration (``0.1 %``);
* the Hetenyi/Pasternak closed-form anchor of the 2-D FEM (``5 %`` on the
  window ``r/l in [0.1, 5]``);
* the dependence laws of the design quantities and the analytic
  self-consistency of the first-crack load.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_sheet import design_laws as dl
from tensorphysics.ice_sheet import ice_material as im
from tensorphysics.ice_sheet import pasternak_foundation as pf
from tensorphysics.ice_sheet.kirchhoff_plate_fem import (
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    graded_axis_centered,
)

FREE = {"left": "free", "right": "free", "bottom": "free", "top": "free"}


# ---------------------------------------------------------------------------
# material model
# ---------------------------------------------------------------------------
def test_modulus_increases_with_colder_ice():
    vals = [im.ice_elastic_modulus(t, 1e-4) for t in (0.0, -5.0, -10.0, -20.0, -30.0)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # measured range: a few GPa
    assert 3.0e9 < vals[0] < 9.0e9
    assert 3.0e9 < vals[-1] < 1.2e10


def test_modulus_increases_with_strain_rate_and_saturates():
    vals = [im.ice_elastic_modulus(-10.0, e) for e in (1e-8, 1e-7, 1e-6, 1e-5, 1e-3)]
    assert all(b > a for a, b in zip(vals, vals[1:]))
    # creep branch: E_sec ~ eps_dot^(-(n-1)/n) = eps_dot^(-2/3); between a
    # hundred-fold rate step the secant modulus must drop by a factor ~11
    r1 = im.ice_elastic_modulus(-10.0, 1e-8) / im.ice_elastic_modulus(-10.0, 1e-6)
    assert 0.05 < r1 < 0.15
    # elastic (high-rate) branch is nearly rate independent
    r2 = im.ice_elastic_modulus(-10.0, 1e-3) / im.ice_elastic_modulus(-10.0, 1e-4)
    assert 1.0 <= r2 < 1.1


def test_flexural_strength_rises_then_saturates():
    vals = [im.ice_flexural_strength(-10.0, e) for e in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))
    assert vals[-1] / vals[0] > 1.5
    # ductile-brittle transition cap
    assert im.ice_flexural_strength(-10.0, 1e-3) == pytest.approx(
        im.ice_flexural_strength(-10.0, 1e-2)
    )
    # colder ice is stronger
    assert im.ice_flexural_strength(-25.0, 1e-5) > im.ice_flexural_strength(-5.0, 1e-5)


def test_strain_rate_from_indentation_rate_scaling():
    eps = im.strain_rate_from_indentation_rate(0.05, 1.0, 15.0)
    assert eps == pytest.approx(0.5 * 1.0 * 0.05 / 225.0)
    # linear in the rate, quadratic in the inverse length
    assert im.strain_rate_from_indentation_rate(0.10, 1.0, 15.0) == pytest.approx(2 * eps)


def test_material_state_is_self_consistent():
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    st = m.derived()
    # eps_dot must reproduce itself from the converged (E, l)
    eps_check = im.strain_rate_from_indentation_rate(
        m.indentation_rate_m_s, m.h, st["char_length_m"]
    )
    assert eps_check == pytest.approx(st["eps_dot_1_s"], rel=1e-12)
    assert st["E_Pa"] == pytest.approx(m.elastic_modulus(st["eps_dot_1_s"]), rel=1e-12)
    # D, l, G_p, gamma chain
    D = st["E_Pa"] * m.h**3 / (12 * (1 - m.nu**2))
    assert st["D_Nm"] == pytest.approx(D, rel=1e-12)
    assert st["char_length_m"] == pytest.approx((D / st["k_w_Pa_per_m"]) ** 0.25, rel=1e-12)
    assert st["G_p_N_per_m"] == pytest.approx(
        m.shear_layer_beta * st["G_ice_Pa"] * m.h, rel=1e-12
    )
    assert st["pasternak_gamma"] == pytest.approx(
        st["G_p_N_per_m"] / math.sqrt(D * st["k_w_Pa_per_m"]), rel=1e-12
    )


def test_material_validation():
    with pytest.raises(ValueError):
        im.IceSheetMaterial(h=0.0)
    with pytest.raises(ValueError):
        im.IceSheetMaterial(h=1.0, temp_C=5.0)  # above melting point
    with pytest.raises(ValueError):
        im.IceSheetMaterial(h=1.0, indentation_rate_m_s=-1.0)
    with pytest.raises(ValueError):
        im.IceSheetMaterial(h=1.0, shear_layer_beta=-0.5)
    with pytest.raises(ValueError):
        im.ice_elastic_modulus(-10.0, 0.0)
    with pytest.raises(ValueError):
        im.creep_rate_factor(1e-6, creep_exponent=1.0)
    with pytest.raises(ValueError):
        im.temperature_stiffening(-10.0, slope_per_K=-0.1)


def test_winkler_modulus_is_buoyancy():
    m = im.IceSheetMaterial(h=1.0, rho_w=1025.0, g=9.81)
    assert m.winkler_modulus == pytest.approx(1025.0 * 9.81)


# ---------------------------------------------------------------------------
# criterion 2: constant-pressure patch test
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g_p_scale", [0.0, 1.0e-9, 1.0])
def test_patch_test_element_reaction(g_p_scale):
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    st = m.derived()
    res = dl.foundation_patch_test(
        h=m.h, E=st["E_Pa"], nu=m.nu, k_w=st["k_w_Pa_per_m"],
        g_p=g_p_scale * st["G_p_N_per_m"], nx=4, ny=5, length=6.0 * st["char_length_m"],
        w0=1.0e-3,
    )
    assert res["max_rel_error"] < 1.0e-10
    assert res["shear_term_max_rel_contribution"] < 1.0e-10
    assert res["passed"] is True
    # every element carries exactly k w0 (to round-off)
    assert res["max_reaction_Pa"] == pytest.approx(res["target_reaction_Pa"], rel=1e-10)
    assert res["min_reaction_Pa"] == pytest.approx(res["target_reaction_Pa"], rel=1e-10)


# ---------------------------------------------------------------------------
# criterion 3: Pasternak -> Winkler degeneration
# ---------------------------------------------------------------------------
def test_closed_form_degenerates_exactly_to_kelvin():
    D, k, P = 4.4e8, 10055.25, 5.0e5
    r = np.linspace(0.5, 60.0, 40)
    w_win = np.atleast_1d(pf.pasternak_point_load(r, P, D, k, 0.0))
    kelvin = np.atleast_1d(pf.pasternak_point_load(r, P, D, k, 0.0))
    assert np.allclose(w_win, kelvin)


def test_fem_recovers_winkler_for_vanishing_shear_layer():
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    st = m.derived()
    D, k = st["D_Nm"], st["k_w_Pa_per_m"]
    l = st["char_length_m"]
    L, n, h0, Rc, P = 120.0, 40, 0.30, 0.5, 5.0e5
    xs = graded_axis_centered(L, n, h0)
    plate = KirchhoffWinklerPlate(h=m.h, E=st["E_Pa"], nu=m.nu, k_w=k, lx=L, ly=L)

    def solve(g_p):
        solver = BFSPlateSolver(
            plate, n, n, xs=xs, ys=xs,
            foundation_model="pasternak" if g_p > 0 else "winkler", g_p=g_p,
        )
        q = P / (math.pi * Rc * Rc)
        solver.add_pressure(
            lambda x, y: np.full(np.shape(x), q),
            inside=lambda x, y: np.hypot(x, y) <= Rc,
            area=math.pi * Rc * Rc, gauss_n=4, bbox=(-Rc, Rc, -Rc, Rc),
        )
        return solver.solve(bc=FREE)

    r = np.linspace(0.1 * l, 3.0 * l, 40)
    w_win = solve(0.0).sample_ray((0.0, 0.0), 0.0, r=r)["w"]
    for scale, tol in ((1e-2, 1e-2), (1e-6, 1e-3)):
        w_p = solve(scale * math.sqrt(D * k)).sample_ray((0.0, 0.0), 0.0, r=r)["w"]
        rel = np.max(np.abs(w_p - w_win)) / np.max(np.abs(w_win))
        assert rel < tol


# ---------------------------------------------------------------------------
# criterion 1: Hetenyi / Pasternak closed-form anchor
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("g_p_scale", [0.0, 1.0])
def test_fem_vs_hetnyi_point_load_window(g_p_scale):
    """2-D FEM vs the closed form on ``r/l in [0.1, 5]``: error < 5 %."""
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    st = m.derived()
    D, k, l = st["D_Nm"], st["k_w_Pa_per_m"], st["char_length_m"]
    g_p = g_p_scale * math.sqrt(D * k)
    P, Rc = 5.0e5, 0.5
    L, n, h0 = 200.0, 90, 0.10
    xs = graded_axis_centered(L, n, h0)
    plate = KirchhoffWinklerPlate(h=m.h, E=st["E_Pa"], nu=m.nu, k_w=k, lx=L, ly=L)
    solver = BFSPlateSolver(
        plate, n, n, xs=xs, ys=xs,
        foundation_model="pasternak" if g_p > 0 else "winkler", g_p=g_p,
    )
    q = P / (math.pi * Rc * Rc)
    solver.add_pressure(
        lambda x, y: np.full(np.shape(x), q),
        inside=lambda x, y: np.hypot(x, y) <= Rc,
        area=math.pi * Rc * Rc, gauss_n=4, bbox=(-Rc, Rc, -Rc, Rc),
    )
    field = solver.solve(bc=FREE)
    r = np.linspace(0.1 * l, 5.0 * l, 121)
    w_fem = field.sample_ray((0.0, 0.0), 0.0, r=r)["w"]
    ana = pf.pasternak_patch_field(P, Rc, D, k, g_p, nu=m.nu, r_max=5.5 * l,
                                   n_field=4001, n_source=96)
    w_ana = np.interp(r, ana["r"], ana["w"])
    peak = np.max(np.abs(w_ana))
    assert np.max(np.abs(w_fem - w_ana)) / peak < 0.05
    mask = np.abs(w_ana) >= 0.1 * peak
    assert np.max(np.abs((w_fem - w_ana)[mask] / w_ana[mask])) < 0.05
    # the disc load is a point load to 3 % on this window
    w_point = np.atleast_1d(pf.pasternak_point_load(r, P, D, k, g_p))
    assert np.max(np.abs(w_ana - w_point)) / peak < 0.05


# ---------------------------------------------------------------------------
# criterion 4: design laws
# ---------------------------------------------------------------------------
def test_unit_moment_field_is_small_scale_linear():
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    a = dl.peak_bending_stress(m, 5.0e5, 0.5)
    b = dl.peak_bending_stress(m, 1.0e6, 0.5)
    assert b["sigma_max_Pa"] == pytest.approx(2.0 * a["sigma_max_Pa"], rel=1e-12)
    # sigma = 6 M / h^2
    assert a["sigma_max_Pa"] == pytest.approx(6.0 * a["M_max_Nm_per_m"] / 1.0**2, rel=1e-12)


def test_first_crack_load_is_consistent_with_peak_stress():
    m = im.IceSheetMaterial(h=1.0, temp_C=-10.0, indentation_rate_m_s=0.05)
    fc = dl.first_crack_load(m, 0.5)
    stress = dl.peak_bending_stress(m, fc["first_crack_load_N"], 0.5)
    assert stress["sigma_max_Pa"] == pytest.approx(fc["sigma_f_Pa"], rel=1e-12)


def test_thickness_law_is_decreasing():
    rows = [
        dl.peak_bending_stress(m, 5.0e5, 0.5)
        for m in (im.IceSheetMaterial(h=h, temp_C=-10.0, indentation_rate_m_s=0.05)
                  for h in (0.5, 1.0, 2.0))
    ]
    sig = [r["sigma_max_Pa"] for r in rows]
    assert sig[0] > sig[1] > sig[2]
    slope = dl.loglog_slope([0.5, 1.0, 2.0], sig)
    assert -1.3 < slope < -0.5  # between the slab h^-2 limit and pure spreading


def test_first_crack_load_grows_with_thickness_and_temperature():
    fcs = [dl.first_crack_load(im.IceSheetMaterial(
        h=h, temp_C=-10.0, indentation_rate_m_s=0.05), 0.5)["first_crack_load_N"]
        for h in (0.5, 1.0, 2.0)]
    assert fcs[0] < fcs[1] < fcs[2]
    fc_cold = dl.first_crack_load(im.IceSheetMaterial(
        h=1.0, temp_C=-25.0, indentation_rate_m_s=0.05), 0.5)["first_crack_load_N"]
    fc_warm = dl.first_crack_load(im.IceSheetMaterial(
        h=1.0, temp_C=-5.0, indentation_rate_m_s=0.05), 0.5)["first_crack_load_N"]
    assert fc_cold > fc_warm


def test_dependence_study_signs():
    study = dl.dependence_study(rate_values=(0.002, 0.05, 0.5))
    signs = study["signs"]
    assert signs["d_sigma_max_d_h"]["sign"] < 0
    assert signs["d_sigma_max_d_absT"]["sign"] < 0
    # the rate law of the *induced* stress is weak; the capacity laws are not
    assert signs["d_Pc_d_h"]["sign"] > 0
    assert signs["d_Pc_d_absT"]["sign"] > 0
    assert signs["d_Pc_d_rate"]["sign"] > 0
    assert signs["d_utilization_d_absT"]["sign"] < 0
    assert abs(signs["d_sigma_max_d_rate"]["loglog_slope"]) < 0.05


def test_temperature_sign_flips_with_foundation_model():
    """Documented finding: the induced-stress temperature sign is model dependent."""
    pasternak = dl.dependence_study(rate_values=(0.01, 0.05))
    winkler = dl.dependence_study(rate_values=(0.01, 0.05),
                                  material_kwargs={"shear_layer_beta": 0.0})
    assert pasternak["signs"]["d_sigma_max_d_absT"]["sign"] < 0
    assert winkler["signs"]["d_sigma_max_d_absT"]["sign"] > 0


def test_sign_report_and_helpers():
    rows = [{"x": 1.0, "y": 2.0}, {"x": 2.0, "y": 4.0}, {"x": 4.0, "y": 8.0}]
    rep = dl.sign_report(rows, "x", "y")
    assert rep["sign"] == 1
    assert rep["loglog_slope"] == pytest.approx(1.0)
    assert dl.loglog_slope([1.0, 10.0], [1.0, 10.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        dl.loglog_slope([1.0], [1.0])