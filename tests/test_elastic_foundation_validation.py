"""Validation tests for the elastic-foundation (Winkler / Pasternak) surface.

Node p270w1-n09 "elastic foundation validation maturity": beams and plates on
elastic foundations, analytical anchors + independent numerical checks.

Run with::

    PYTHONPATH=src python -m pytest tests/test_elastic_foundation_validation.py -q
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.elastic_foundation import (
    beam_char_length,
    beam_foundation_roots,
    cosine_only_beam_point_load,
    finite_plate_center_deflection,
    finite_plate_solution,
    hetenyi_beam_point_load,
    pasternak_beam_point_load,
    point_load_shear_jump,
    solve_axisym_plate_fem,
    solve_beam_foundation_fem,
    winkler_plate_biharmonic_residual,
    winkler_plate_char_length,
    winkler_plate_flexural_rigidity,
    winkler_plate_moment_r,
    winkler_plate_patch_load,
    winkler_plate_point_load,
    winkler_plate_point_load_hankel,
    winkler_plate_load_recovery,
)

# reference data set: steel-ish beam on a soft foundation
EI = 1.0e7          # N m^2
K_BEAM = 1.0e6      # N/m^2
P_BEAM = 1.0e5      # N

# reference data set: floating ice sheet (fresh-water ice, sea-water buoyancy)
E_ICE = 5.0e9
NU_ICE = 0.33
H_ICE = 1.0
RHO_G = 10.0e3      # rho_w g [N/m^3] -> Winkler modulus for a floating sheet
P_ICE = 1.0e5


# ---------------------------------------------------------------------------
# beams
# ---------------------------------------------------------------------------
def test_beam_pasternak_reduces_to_hetenyi():
    """G_p = 0 must reproduce Hetényi's infinite Winkler beam exactly."""
    lb = beam_char_length(EI, K_BEAM)
    x = np.linspace(-6 * lb, 6 * lb, 121)
    w_het = hetenyi_beam_point_load(x, P_BEAM, EI, K_BEAM)
    w_pas = pasternak_beam_point_load(x, P_BEAM, EI, K_BEAM, 0.0)
    assert np.max(np.abs(w_pas - w_het)) <= 1e-12 * np.max(np.abs(w_het))


def test_beam_hetenyi_centre_deflection():
    """w(0) = P/(2 k l_b) with l_b = (4 EI/k)^(1/4)."""
    lb = beam_char_length(EI, K_BEAM)
    w0 = float(hetenyi_beam_point_load(0.0, P_BEAM, EI, K_BEAM)[0])
    assert w0 == pytest.approx(P_BEAM / (2.0 * K_BEAM * lb), rel=1e-12)


def test_beam_derivative_consistency():
    """The analytic derivative chain w'..w'''' must be self-consistent."""
    lb = beam_char_length(EI, K_BEAM)
    x = np.array([0.37, 1.0, 2.3]) * lb
    h = 1e-4 * lb
    for d in (1, 2, 3, 4):
        num = [(pasternak_beam_point_load(x + h, P_BEAM, EI, K_BEAM, 0.0, d - 1)
                - pasternak_beam_point_load(x - h, P_BEAM, EI, K_BEAM, 0.0, d - 1))
               / (2 * h)][0]
        ana = pasternak_beam_point_load(x, P_BEAM, EI, K_BEAM, 0.0, d)
        assert np.max(np.abs(num - ana)) <= 2e-4 * np.max(np.abs(ana))


def test_beam_pasternak_ode_residual():
    """EI w'''' - G_p w'' + k w = 0 away from the load (both regimes)."""
    lb = beam_char_length(EI, K_BEAM)
    x = np.array([0.4, 1.3, 2.7]) * lb
    for G in (0.0, 0.3e6, 5.0e6):
        w = pasternak_beam_point_load(x, P_BEAM, EI, K_BEAM, G, 0)
        w2 = pasternak_beam_point_load(x, P_BEAM, EI, K_BEAM, G, 2)
        w4 = pasternak_beam_point_load(x, P_BEAM, EI, K_BEAM, G, 4)
        resid = EI * w4 - G * w2 + K_BEAM * w
        assert np.max(np.abs(resid)) <= 1e-9 * K_BEAM * np.max(np.abs(w))


def test_beam_source_strength_correct_formula():
    """EI [w'''] jump across the load must equal P (correct solution)."""
    for G in (0.0, 1.0e6):
        ratio = point_load_shear_jump(
            lambda z: pasternak_beam_point_load(z, P_BEAM, EI, K_BEAM, G, 3),
            P_BEAM, EI)
        assert ratio == pytest.approx(1.0, rel=1e-6)


def test_cosine_only_formula_falsified():
    """The cosine-only formula ``P/(2 k l) e^{-|x|/l} cos(x/l)`` carries only
    half of the point-load source strength -> it is NOT the point-load solution.

    This is a recorded falsification: the task brief's beam formula reproduces
    the centre deflection but only ``P/(2EI)`` of shear jump.
    """
    ratio = point_load_shear_jump(
        lambda z: cosine_only_beam_point_load(z, P_BEAM, EI, K_BEAM, 3),
        P_BEAM, EI)
    assert ratio == pytest.approx(0.5, rel=1e-6)
    # its centre deflection nevertheless equals the correct value
    lb = beam_char_length(EI, K_BEAM)
    assert float(cosine_only_beam_point_load(0.0, P_BEAM, EI, K_BEAM)[0]) == \
        pytest.approx(P_BEAM / (2.0 * K_BEAM * lb), rel=1e-12)
    # and it already deviates strongly from the true shape at x = l
    x = 1.0 * lb
    w_true = float(hetenyi_beam_point_load(x, P_BEAM, EI, K_BEAM)[0])
    w_cos = float(cosine_only_beam_point_load(x, P_BEAM, EI, K_BEAM)[0])
    assert abs(w_cos / w_true - 1.0) > 0.2


def test_beam_pasternak_two_real_roots_regime():
    """gamma = G/sqrt(EI k) > 2 -> two real, positive wavenumbers."""
    for G in (2.1 * np.sqrt(EI * K_BEAM), 10.0 * np.sqrt(EI * K_BEAM)):
        a, b = beam_foundation_roots(EI, K_BEAM, G)
        assert abs(a.imag) < 1e-9 * abs(a)
        assert abs(b.imag) < 1e-9 * abs(b)
        assert a.real > 0 and b.real > 0
    a, b = beam_foundation_roots(EI, K_BEAM, 0.5 * np.sqrt(EI * K_BEAM))
    assert abs(a.real - b.real) < 1e-12
    assert abs(a.imag + b.imag) < 1e-12


def test_beam_fem_matches_analytic_winkler():
    lb = beam_char_length(EI, K_BEAM)
    res = solve_beam_foundation_fem(EI, K_BEAM, P_BEAM, half_length=8 * lb,
                                    n_elem=800)
    ana = pasternak_beam_point_load(res.x, P_BEAM, EI, K_BEAM, 0.0)
    mask = np.abs(res.x) <= 4 * lb
    rel = np.max(np.abs(res.w[mask] - ana[mask])) / np.max(np.abs(ana))
    assert rel < 5e-4
    w0 = float(res.evaluate(0.0)[0])
    assert w0 == pytest.approx(float(ana[len(ana) // 2]), rel=1e-5)


def test_beam_fem_matches_analytic_pasternak():
    lb = beam_char_length(EI, K_BEAM)
    G = 1.5 * np.sqrt(EI * K_BEAM)      # oscillatory Pasternak regime
    res = solve_beam_foundation_fem(EI, K_BEAM, P_BEAM, half_length=8 * lb,
                                    n_elem=800, G_p=G)
    ana = pasternak_beam_point_load(res.x, P_BEAM, EI, K_BEAM, G)
    mask = np.abs(res.x) <= 4 * lb
    rel = np.max(np.abs(res.w[mask] - ana[mask])) / np.max(np.abs(ana))
    assert rel < 5e-4


def test_beam_fem_truncation_error_decays():
    """Free-end truncation error must decay with the half length."""
    lb = beam_char_length(EI, K_BEAM)
    errs = []
    for half in (4.0, 6.0, 8.0):
        res = solve_beam_foundation_fem(EI, K_BEAM, P_BEAM,
                                        half_length=half * lb, n_elem=400)
        ana = pasternak_beam_point_load(res.x, P_BEAM, EI, K_BEAM, 0.0)
        errs.append(np.max(np.abs(res.w - ana)) / np.max(np.abs(ana)))
    assert errs[0] > errs[1] > errs[2]


# ---------------------------------------------------------------------------
# plates
# ---------------------------------------------------------------------------
def test_kelvin_ker_kei_identity():
    """Re/Im of K_0(r e^{i pi/4}/l) are the Kelvin functions ker/kei."""
    from scipy.special import kei, ker
    from tensorphysics.elastic_foundation.plates import kelvin_k0_derivatives
    rho = np.array([0.05, 0.5, 2.0, 7.0])
    F, _d1, _d2 = kelvin_k0_derivatives(rho * 1.0, 1.0)
    assert np.allclose(F.real, ker(rho), rtol=1e-9, atol=0)
    assert np.allclose(F.imag, kei(rho), rtol=1e-9, atol=1e-15)


def test_plate_centre_deflection_hetenyi_value():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    w0 = float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0])
    assert w0 == pytest.approx(P_ICE / (8.0 * np.sqrt(RHO_G * D)), rel=1e-12)
    assert w0 == pytest.approx(P_ICE / (8.0 * RHO_G * l ** 2), rel=1e-12)
    # kei(0) = -pi/4 identity behind the formula
    assert w0 == pytest.approx(-P_ICE * (-np.pi / 4.0) / (2 * np.pi * RHO_G * l ** 2),
                               rel=1e-12)


def test_plate_kelvin_matches_hankel_integral():
    """The Kelvin closed form and the defining Hankel integral must agree."""
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    for rho in (0.0, 0.3, 1.0, 2.0, 5.0):
        kelvin = float(winkler_plate_point_load(rho * l, P_ICE, D, RHO_G)[0])
        hankel = winkler_plate_point_load_hankel(rho * l, P_ICE, D, RHO_G)
        assert kelvin == pytest.approx(hankel, rel=1e-5)


def test_plate_load_recovery():
    """Equilibrium: the foundation must carry the whole point load."""
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    assert winkler_plate_load_recovery(P_ICE, D, RHO_G) == pytest.approx(1.0,
                                                                        rel=1e-6)


def test_plate_biharmonic_residual_vanishes():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    r = np.array([0.3, 0.7, 1.5, 3.0, 6.0]) * l
    resid = winkler_plate_biharmonic_residual(r, P_ICE, D, RHO_G)
    assert np.max(np.abs(resid)) < 1e-10


def test_plate_patch_tends_to_point_load():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    for rho in (1.0, 3.0):
        patch = float(winkler_plate_patch_load(rho * l, P_ICE, 0.01 * l, D, RHO_G)[0])
        point = float(winkler_plate_point_load(rho * l, P_ICE, D, RHO_G)[0])
        assert patch == pytest.approx(point, rel=1e-4)


def test_axisym_plate_fem_matches_patch_analytic():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    a = 0.05 * l
    res = solve_axisym_plate_fem(D, NU_ICE, RHO_G, 40 * l, P=P_ICE,
                                 load_radius=a, n_center=10, n_outer=300,
                                 grading=2.0, bc="free")
    rq = np.linspace(0.0, 8 * l, 81)
    w_fe = res.evaluate(rq)
    w_an = winkler_plate_patch_load(rq, P_ICE, a, D, RHO_G)
    scale = np.max(np.abs(w_an))
    assert abs(w_fe[0] / w_an[0] - 1.0) < 1e-4
    assert np.sqrt(np.mean((w_fe - w_an) ** 2)) / scale < 1e-4


def test_axisym_plate_fem_point_load_centre():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    res = solve_axisym_plate_fem(D, NU_ICE, RHO_G, 40 * l, P=P_ICE,
                                 load_radius=0.0, n_outer=400, grading=2.0,
                                 bc="free")
    w0 = float(res.w[0])
    w0_an = float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0])
    assert abs(w0 / w0_an - 1.0) < 1e-5


def test_axisym_plate_fem_moment_matches_analytic():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    res = solve_axisym_plate_fem(D, NU_ICE, RHO_G, 40 * l, P=P_ICE,
                                 load_radius=0.05 * l, n_center=10,
                                 n_outer=300, grading=2.0, bc="free")
    rq = np.array([1.0, 2.0, 3.0]) * l
    M_fe = res.moment_r(rq)
    M_an = winkler_plate_moment_r(rq, P_ICE, D, RHO_G, NU_ICE)
    assert np.max(np.abs(M_fe - M_an)) <= 0.01 * np.max(np.abs(M_an))


def test_finite_plate_solution_tends_to_infinite():
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    w_inf = float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0])
    for bc in ("clamped", "simply_supported"):
        w_fin = finite_plate_center_deflection(40 * l, P_ICE, D, RHO_G,
                                               NU_ICE, bc)
        assert abs(w_fin / w_inf - 1.0) < 1e-10


def test_boundary_effect_decays_exponentially():
    """Rigid-edge boundary effect at the centre decays with R/l."""
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    w_inf = float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0])
    errs = []
    for span in (5.0, 10.0, 20.0):
        w = finite_plate_center_deflection(span * l, P_ICE, D, RHO_G,
                                           NU_ICE, "clamped")
        errs.append(abs(w / w_inf - 1.0))
    assert errs[0] > errs[1] > errs[2]
    # more than one order of magnitude per 5 l
    assert errs[1] < 1e-2 * errs[0]
    assert errs[2] < 1e-2 * errs[1]


def test_axisym_plate_fem_boundary_conditions():
    """Clamped / simply supported edges reproduce the exact finite solution."""
    D = winkler_plate_flexural_rigidity(E_ICE, H_ICE, NU_ICE)
    l = winkler_plate_char_length(D, RHO_G)
    a = 0.05 * l
    for bc in ("clamped", "simply_supported"):
        res = solve_axisym_plate_fem(D, NU_ICE, RHO_G, 12 * l, P=P_ICE,
                                     load_radius=a, n_center=10, n_outer=400,
                                     grading=2.0, bc=bc)
        w_ex = finite_plate_solution(12 * l, P_ICE, D, RHO_G, NU_ICE, bc)
        # compare with the finite-domain exact solution using the patch-load
        # centre value as reference amplitude (patch/point ratio is analytic)
        ratio = float(winkler_plate_patch_load(0.0, P_ICE, a, D, RHO_G)[0]) / \
            float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0])
        assert abs(float(res.w[0]) / (w_ex.w_center * ratio) - 1.0) < 5e-3


# ---------------------------------------------------------------------------
# ice-sheet application: scaling law and first-crack load
# ---------------------------------------------------------------------------
def test_scaling_law_char_length_three_quarters():
    """l = (D/k)^{1/4} ~ h^{3/4} for a floating ice sheet."""
    hs = np.array([0.5, 1.0, 1.5])
    ls = []
    for h in hs:
        D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
        ls.append(winkler_plate_char_length(D, RHO_G))
    slope = np.polyfit(np.log(hs), np.log(ls), 1)[0]
    assert slope == pytest.approx(0.75, abs=1e-10)
    # reference value for h = 1 m, E = 5 GPa, nu = 0.33, k = 10 kN/m^3
    assert ls[1] == pytest.approx(14.705, rel=1e-4)


def test_scaling_law_centre_deflection():
    """w(0) = P/(8 sqrt(kD)) ~ h^{-3/2}."""
    hs = np.array([0.5, 1.0, 1.5])
    ws = []
    for h in hs:
        D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
        ws.append(float(winkler_plate_point_load(0.0, P_ICE, D, RHO_G)[0]))
    slope = np.polyfit(np.log(hs), np.log(ws), 1)[0]
    assert slope == pytest.approx(-1.5, abs=1e-10)


def test_ice_beam_char_length_scaling():
    """Beam characteristic length l_b = (4EI/k)^{1/4} ~ h^{3/4} as well."""
    b = 1.0
    hs = np.array([0.5, 1.0, 1.5])
    lbs = [beam_char_length(E_ICE * b * h ** 3 / 12.0, RHO_G * b) for h in hs]
    slope = np.polyfit(np.log(hs), np.log(lbs), 1)[0]
    assert slope == pytest.approx(0.75, abs=1e-10)


def test_first_crack_load_fem_vs_analytic():
    """First-crack load from the FEM moment field matches the analytic one."""
    sigma_f = 7.0e5
    a_contact = 0.5                     # contact patch radius [m]
    for h in (0.5, 1.0, 1.5):
        D = winkler_plate_flexural_rigidity(E_ICE, h, NU_ICE)
        l = winkler_plate_char_length(D, RHO_G)
        M_f = sigma_f * h ** 2 / 6.0
        # analytic point-load moment at the patch edge (r = a) as the driver
        M_an_unit = float(winkler_plate_moment_r(
            np.array([a_contact]), 1.0, D, RHO_G, NU_ICE)[0])
        P_cr_an = M_f / abs(M_an_unit)
        res = solve_axisym_plate_fem(D, NU_ICE, RHO_G, 40 * l, P=P_cr_an,
                                     load_radius=a_contact, n_center=12,
                                     n_outer=320, grading=2.0, bc="free")
        M_fe_max = float(np.max(np.abs(res.moment_r(np.array([a_contact])))))
        assert abs(M_fe_max / M_f - 1.0) < 0.05


def test_input_guards():
    with pytest.raises(ValueError):
        beam_foundation_roots(0.0, K_BEAM)
    with pytest.raises(ValueError):
        beam_foundation_roots(EI, -1.0)
    with pytest.raises(ValueError):
        solve_axisym_plate_fem(1.0, 0.3, 1.0, 1.0, bc="pinned")
    with pytest.raises(ValueError):
        solve_beam_foundation_fem(EI, K_BEAM, P_BEAM, 1.0, n_elem=3)