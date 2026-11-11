"""p387w2-n07: three-tier ice-plate foundation verification surface.

Winkler (Hetényi) -> Pasternak/Vlasov -> elastic half-space (Boussinesq), with
point-load / patch-load anchors, ring-crack spacing scaling law, first-crack
load and negative controls.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.elastic_foundation.crack_spacing import (
    bearing_capacity_ratio,
    crack_spacing,
    first_crack_load,
    flexural_strength_moment,
    local_log_slope,
    moment_extrema_radii,
    pasternak_ring_crack_period,
    spacing_scaling_exponent,
)
from tensorphysics.elastic_foundation.halfspace import (
    boussinesq_patch_deflection,
    boussinesq_point_deflection,
    boussinesq_punch_deflection,
    halfspace_foundation_modulus,
    halfspace_plate_char_length,
    halfspace_plate_center_deflection,
    halfspace_plate_moment_r,
    halfspace_plate_patch_load,
)
from tensorphysics.elastic_foundation.pasternak_plate import (
    match_pasternak_to_halfspace,
    pasternak_plate_bilaplacian_residual,
    pasternak_plate_center_deflection,
    pasternak_plate_char_length,
    pasternak_plate_moment_r,
    pasternak_plate_patch_load,
    pasternak_plate_patch_moment_r,
    pasternak_plate_point_load,
    pasternak_plate_point_load_hankel,
    pasternak_plate_roots,
    vlasov_foundation_params,
    vlasov_layer_depth,
)
from tensorphysics.elastic_foundation.plates import (
    solve_axisym_plate_fem,
    winkler_plate_point_load,
)

# level ice sheet (the case used by the node report)
E, NU, H_ICE = 5.0e9, 0.30, 1.0
D = E * H_ICE ** 3 / (12.0 * (1.0 - NU ** 2))
K_WATER = 1025.0 * 9.81
SOIL = dict(E_s=100.0e6, nu_s=0.35)
C_HS = halfspace_foundation_modulus(**SOIL)
P = 1.0e5
A_PATCH = 0.5


# ---------------------------------------------------------------------------
# spectral roots / Winkler limit
# ---------------------------------------------------------------------------
def test_pasternak_roots_reduce_to_kelvin_wavenumbers():
    l = pasternak_plate_char_length(D, K_WATER)
    _m1, _m2, mu1, mu2 = pasternak_plate_roots(D, K_WATER, 0.0)
    assert abs(mu1 - np.exp(-1j * np.pi / 4) / l) < 1e-12 * abs(mu1)
    assert abs(mu2 - np.exp(1j * np.pi / 4) / l) < 1e-12 * abs(mu2)


def test_pasternak_roots_reject_degenerate_shear_modulus():
    G_crit = 2.0 * np.sqrt(D * K_WATER)
    with pytest.raises(ValueError):
        pasternak_plate_roots(D, K_WATER, G_crit)


def test_pasternak_roots_reject_bad_inputs():
    with pytest.raises(ValueError):
        pasternak_plate_roots(0.0, K_WATER, 1.0)
    with pytest.raises(ValueError):
        pasternak_plate_roots(D, 0.0, 1.0)
    with pytest.raises(ValueError):
        pasternak_plate_roots(D, K_WATER, -1.0)


def test_winkler_limit_matches_hetenyi_kelvin_solution():
    l = pasternak_plate_char_length(D, K_WATER)
    r = np.linspace(0.0, 5.0 * l, 101)
    het = winkler_plate_point_load(r, P, D, K_WATER)
    past = pasternak_plate_point_load(r, P, D, K_WATER, 0.0)
    assert np.max(np.abs(past / het - 1.0)) < 1e-12


def test_centre_deflection_matches_hetenyi_closed_value():
    w0 = pasternak_plate_center_deflection(P, D, K_WATER, 0.0)
    assert abs(w0 / (P / (8.0 * np.sqrt(K_WATER * D))) - 1.0) < 1e-12


def test_closed_form_satisfies_pasternak_plate_equation():
    G_p = 5.0e7
    res = pasternak_plate_bilaplacian_residual(np.array([0.5, 1.0, 3.0]), P, D,
                                               K_WATER, G_p)
    assert np.max(np.abs(res)) < 1e-8


def test_closed_form_matches_independent_hankel_integral():
    G_p = 5.0e7
    for r in (0.0, 1.0, 5.0, 20.0):
        cf = pasternak_plate_point_load(r, P, D, K_WATER, G_p)[0]
        hk = pasternak_plate_point_load_hankel(r, P, D, K_WATER, G_p)
        assert abs(cf / hk - 1.0) < 1e-4


def test_point_load_derivative_and_sign_conventions():
    G_p = 5.0e7
    assert pasternak_plate_point_load(10.0, P, D, K_WATER, G_p)[0] > 0.0
    assert abs(float(pasternak_plate_point_load(0.0, P, D, K_WATER, G_p,
                                                deriv=1)[0])) < 1e-30
    with pytest.raises(ValueError):
        pasternak_plate_point_load(1.0, P, D, K_WATER, G_p, deriv=5)
    with pytest.raises(ValueError):
        pasternak_plate_point_load(-1.0, P, D, K_WATER, G_p)
    # w'' diverges logarithmically at the load point -> reported as nan
    assert np.isnan(pasternak_plate_point_load(0.0, P, D, K_WATER, G_p,
                                               deriv=2)[0])


def test_shear_layer_changes_the_response_monotonically():
    r = np.array([0.0, 2.0, 10.0])
    w0 = pasternak_plate_point_load(r, P, D, K_WATER, 0.0)
    w1 = pasternak_plate_point_load(r, P, D, K_WATER, 1.0e8)
    w2 = pasternak_plate_point_load(r, P, D, K_WATER, 1.0e9)
    assert np.all(w2 < w1) and np.all(w1 < w0)


# ---------------------------------------------------------------------------
# Vlasov / half-space parameter identification
# ---------------------------------------------------------------------------
def test_vlasov_parameter_round_trip():
    k, G_p = vlasov_foundation_params(SOIL["E_s"], SOIL["nu_s"], 8.0)
    assert k == pytest.approx(SOIL["E_s"] / ((1 - SOIL["nu_s"] ** 2) * 8.0))
    assert G_p == pytest.approx(SOIL["E_s"] * 8.0 / (6.0 * (1 + SOIL["nu_s"])))
    assert vlasov_layer_depth(SOIL["E_s"], SOIL["nu_s"], k) == pytest.approx(8.0)
    with pytest.raises(ValueError):
        vlasov_foundation_params(SOIL["E_s"], 0.5, 8.0)


def test_kernel_match_equals_value_and_slope_of_halfspace_kernel():
    k, G_p, alpha_ref = match_pasternak_to_halfspace(D, SOIL["E_s"], SOIL["nu_s"])
    assert G_p == pytest.approx(C_HS / (2.0 * alpha_ref))
    assert k == pytest.approx(C_HS * alpha_ref / 2.0)
    # value and slope both match the exact half-space kernel C alpha
    value = k + G_p * alpha_ref ** 2
    slope = 2.0 * G_p * alpha_ref
    assert value == pytest.approx(C_HS * alpha_ref)
    assert slope == pytest.approx(C_HS)


def test_boussinesq_closed_forms():
    p = P / (np.pi * A_PATCH ** 2)
    w0 = boussinesq_patch_deflection(0.0, p, A_PATCH, SOIL["E_s"], SOIL["nu_s"])[0]
    w_edge = boussinesq_patch_deflection(A_PATCH, p, A_PATCH, SOIL["E_s"],
                                         SOIL["nu_s"])[0]
    assert w0 == pytest.approx(2.0 * p * A_PATCH * (1 - SOIL["nu_s"] ** 2) / SOIL["E_s"])
    assert w_edge == pytest.approx(4.0 * p * A_PATCH * (1 - SOIL["nu_s"] ** 2)
                                  / (np.pi * SOIL["E_s"]))
    # far field tends to the Boussinesq point-load value
    far = boussinesq_patch_deflection(200.0 * A_PATCH, p, A_PATCH, SOIL["E_s"],
                                      SOIL["nu_s"])[0]
    assert far == pytest.approx(boussinesq_point_deflection(P, 200.0 * A_PATCH,
                                                            SOIL["E_s"], SOIL["nu_s"]),
                                rel=1e-4)
    with pytest.raises(ValueError):
        boussinesq_patch_deflection(1.0, p, 0.0, SOIL["E_s"], SOIL["nu_s"])


def test_rigid_punch_closed_form():
    w = boussinesq_punch_deflection(P, A_PATCH, SOIL["E_s"], SOIL["nu_s"])
    assert w == pytest.approx(P * (1 - SOIL["nu_s"] ** 2)
                             / (2.0 * SOIL["E_s"] * A_PATCH))


def test_halfspace_constants():
    assert C_HS == pytest.approx(SOIL["E_s"] / (2.0 * (1 - SOIL["nu_s"] ** 2)))
    assert halfspace_plate_char_length(D, C_HS) == pytest.approx((D / C_HS) ** (1 / 3))
    with pytest.raises(ValueError):
        halfspace_foundation_modulus(SOIL["E_s"], 0.5)
    with pytest.raises(ValueError):
        halfspace_plate_char_length(0.0, C_HS)


# ---------------------------------------------------------------------------
# spectral plate-on-half-space solver
# ---------------------------------------------------------------------------
def test_plate_free_limit_recovers_boussinesq():
    p = P / (np.pi * A_PATCH ** 2)
    D_small = 1.0e-3 * C_HS * A_PATCH ** 3
    w_solver = halfspace_plate_center_deflection(P, A_PATCH, D_small,
                                                 SOIL["E_s"], SOIL["nu_s"])
    w_ref = boussinesq_patch_deflection(0.0, p, A_PATCH, SOIL["E_s"],
                                        SOIL["nu_s"])[0]
    assert abs(w_solver / w_ref - 1.0) < 0.01


def test_far_field_of_plate_on_halfspace_is_boussinesq():
    r_far = 50.0 * A_PATCH
    w = halfspace_plate_patch_load(r_far, P, A_PATCH, D, SOIL["E_s"],
                                   SOIL["nu_s"])[0]
    ref = boussinesq_point_deflection(P, r_far, SOIL["E_s"], SOIL["nu_s"])
    assert abs(w / ref - 1.0) < 0.01


def test_stiff_plate_limit_law_exponent():
    Ds = np.array([1.0e9, 1.0e10, 1.0e11, 1.0e12])
    w0 = np.array([halfspace_plate_center_deflection(P, A_PATCH, Dv, SOIL["E_s"],
                                                     SOIL["nu_s"]) for Dv in Ds])
    alpha, _ = spacing_scaling_exponent(Ds, w0)
    assert alpha == pytest.approx(-1.0 / 3.0, rel=0.05)


def test_halfspace_deflection_and_moment_match_arbitrary_precision_quadrature():
    """Independent check with mpmath: no panel heuristics, 20 significant digits.

    (Finite-difference moments are *not* usable here: the O(1e-4) quadrature
    noise of the deflection is amplified by 1/h^2, see the node report.)
    """
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 20
    Dm, am, Cm, Pm, num = float(D), float(A_PATCH), float(C_HS), float(P), float(NU)

    def w_exact(r):
        f = lambda al: (mp.j1(al * am) * mp.j0(al * r)
                        / (mp.pi * am * al * (Dm * al ** 3 + Cm)))
        return Pm * mp.quad(f, [0, 0.05, 0.2, 1, 3, 10, 30, 100, 300, 1000])

    def m_exact(r):
        f = lambda al: (Dm * (mp.j1(al * am) / (mp.pi * am * al * (Dm * al ** 3 + Cm)))
                        * al ** 2
                        * (mp.j0(al * r) - (1 - num) * mp.j1(al * r) / (al * r)))
        return Pm * mp.quad(f, [0, 0.05, 0.2, 1, 3, 10, 30, 100, 300, 1000])

    for r in (0.5 * A_PATCH, 1.5 * A_PATCH):
        w = halfspace_plate_patch_load(r, P, A_PATCH, D, SOIL["E_s"], SOIL["nu_s"])[0]
        m = halfspace_plate_moment_r(r, P, A_PATCH, D, SOIL["E_s"], SOIL["nu_s"], NU)[0]
        assert abs(w / float(w_exact(r)) - 1.0) < 1e-3
        assert abs(m / float(m_exact(r)) - 1.0) < 1e-2


def test_patch_moment_is_smooth_at_the_axis():
    m = pasternak_plate_patch_moment_r(np.array([0.0, 0.25, 0.5, 1.0]), P,
                                       A_PATCH, D, K_WATER, NU, 0.0)
    assert np.all(np.isfinite(m))
    assert m[0] > 0.0


# ---------------------------------------------------------------------------
# axisymmetric FEM with the Pasternak term
# ---------------------------------------------------------------------------
def test_fem_winkler_backward_compatible():
    l = pasternak_plate_char_length(D, K_WATER)
    res = solve_axisym_plate_fem(D, NU, K_WATER, 30.0 * l, P=P, n_outer=800)
    assert res.G_p == 0.0
    assert res.w[0] == pytest.approx(pasternak_plate_center_deflection(P, D, K_WATER, 0.0),
                                     rel=1e-4)


def test_fem_pasternak_matches_closed_form():
    k, G_p = vlasov_foundation_params(SOIL["E_s"], SOIL["nu_s"], 8.0)
    scale = 1.0 / min(abs(mu) for mu in pasternak_plate_roots(D, k, G_p)[2:])
    res = solve_axisym_plate_fem(D, NU, k, 60.0 * scale, P=P, n_outer=900,
                                 grading=2.2, G_p=G_p)
    ref = pasternak_plate_center_deflection(P, D, k, G_p)
    assert abs(res.w[0] / ref - 1.0) < 0.01
    r = np.array([0.5, 1.0, 2.0]) * pasternak_plate_char_length(D, k)
    assert np.max(np.abs(res.evaluate(r)
                         / pasternak_plate_point_load(r, P, D, k, G_p) - 1.0)) < 0.02


# ---------------------------------------------------------------------------
# crack spacing / bearing capacity
# ---------------------------------------------------------------------------
def test_ring_crack_spacing_scaling_exponent_closed_form():
    ks = np.array([1.0e3, 1.0e4, 1.0e5])
    sp = []
    for k in ks:
        l = pasternak_plate_char_length(D, k)
        sp.append(crack_spacing(
            lambda r, _k=k: pasternak_plate_moment_r(r, 1.0, D, _k, NU, 0.0),
            11.0 * l, n=4001, n_keep=4))
    alpha, _ = spacing_scaling_exponent(ks, sp)
    assert alpha == pytest.approx(-0.25, rel=0.02)
    cs = np.array(sp) / np.array([pasternak_plate_char_length(D, k) for k in ks])
    assert np.ptp(cs) / np.mean(cs) < 1e-3      # s/l is a universal constant


def test_ring_crack_spacing_scaling_exponent_from_fem():
    ks = np.array([1.0e4, 1.0e5])
    sp = []
    for k in ks:
        l = pasternak_plate_char_length(D, k)
        res = solve_axisym_plate_fem(D, NU, k, 15.0 * l, P=1.0, n_outer=800,
                                     grading=2.0)
        sp.append(crack_spacing(lambda r: res.moment_r(r), 11.0 * l,
                                n=3001, n_keep=3))
    alpha, _ = spacing_scaling_exponent(ks, sp)
    assert alpha == pytest.approx(-0.25, rel=0.02)


def test_pasternak_crack_period_is_analytic_and_rejects_real_roots():
    k, G_p = vlasov_foundation_params(SOIL["E_s"], SOIL["nu_s"], 8.0)
    period = pasternak_ring_crack_period(D, k, G_p)
    mu = pasternak_plate_roots(D, k, G_p)[2]
    assert period == pytest.approx(2.0 * np.pi / abs(mu.imag))
    with pytest.raises(ValueError):          # two real roots: monotonic field
        pasternak_ring_crack_period(D, K_WATER, 1.923e9)


def test_first_crack_load_is_foundation_independent_for_a_point_load():
    M_f = flexural_strength_moment(7.0e5, H_ICE)
    vals = []
    for k in (1.0e3, 1.0e4, 1.0e5):
        l = pasternak_plate_char_length(D, k)
        vals.append(first_crack_load(
            lambda r, _k=k: pasternak_plate_moment_r(r, 1.0, D, _k, NU, 0.0),
            M_f, 8.0 * l, n=4001))
    assert np.ptp(np.array(vals)) / np.mean(vals) < 1e-6
    assert bearing_capacity_ratio(vals[0], vals[1]) == pytest.approx(1.0, rel=1e-6)
    with pytest.raises(ValueError):
        flexural_strength_moment(-1.0, 1.0)


def test_local_log_slope_reproduces_a_power_law():
    assert local_log_slope(lambda x: 3.0 * x ** -0.25, 7.0) == pytest.approx(-0.25,
                                                                            rel=1e-8)
    with pytest.raises(ValueError):
        local_log_slope(lambda x: -x, 1.0)


# ---------------------------------------------------------------------------
# negative controls
# ---------------------------------------------------------------------------
def test_negative_control_winkler_placeholder_is_rejected():
    """Dropping the shear layer must move w by far more than the 5 % gate."""
    k, G_p = K_WATER, 1.923e9
    w_past = pasternak_plate_center_deflection(P, D, k, G_p)
    w_wink = pasternak_plate_center_deflection(P, D, k, 0.0)
    assert abs(w_wink / w_past - 1.0) > 0.05


def test_negative_control_wrong_shear_layer_sign_is_detected():
    """A negative shear modulus would soften instead of stiffen the bed; the
    model refuses it outright, so a sign error cannot pass silently."""
    k, G_p = K_WATER, 1.923e9
    with pytest.raises(ValueError):
        pasternak_plate_point_load(np.array([0.0, 1.0]), P, D, k, -G_p)


def test_negative_control_continuum_has_no_finite_point_load():
    """Plate-free continuum diverges like 1/a; a finite plate saturates."""
    a = np.array([0.5, 0.25])
    w_free = np.array([halfspace_plate_center_deflection(
        P, aa, 1.0e-3 * C_HS * aa ** 3, SOIL["E_s"], SOIL["nu_s"]) for aa in a])
    exp, _ = spacing_scaling_exponent(a, w_free)
    assert exp == pytest.approx(-1.0, rel=0.01)          # diverges, no w_max
    w_plate = np.array([halfspace_plate_center_deflection(P, aa, D, SOIL["E_s"],
                                                          SOIL["nu_s"]) for aa in a])
    assert abs(w_plate[0] / w_plate[1] - 1.0) < 0.05     # plate regularises it


def test_negative_control_linearity_and_zero_load():
    w1 = pasternak_plate_center_deflection(P, D, K_WATER, 0.0)
    w2 = pasternak_plate_center_deflection(2.0 * P, D, K_WATER, 0.0)
    assert w2 == pytest.approx(2.0 * w1, rel=1e-12)
    assert pasternak_plate_center_deflection(0.0, D, K_WATER, 0.0) == pytest.approx(0.0)


def test_patch_load_tends_to_point_load():
    l = pasternak_plate_char_length(D, K_WATER)
    w_patch = pasternak_plate_patch_load(10.0 * l, P, 1e-3 * l, D, K_WATER, 0.0)[0]
    w_point = pasternak_plate_point_load(10.0 * l, P, D, K_WATER, 0.0)[0]
    assert abs(w_patch / w_point - 1.0) < 1e-3


def test_moment_extrema_metadata():
    l = pasternak_plate_char_length(D, K_WATER)
    rr = moment_extrema_radii(lambda r: pasternak_plate_moment_r(r, 1.0, D, K_WATER, NU, 0.0),
                              11.0 * l, n=4001)
    assert rr.size >= 2
    assert np.all(np.diff(rr) > 0.0)
    empty = moment_extrema_radii(lambda r: np.zeros_like(r), 1.0, n=11)
    assert empty.size == 0                  # no extremum -> caller must react
    with pytest.raises(ValueError):
        crack_spacing(lambda r: np.zeros_like(r), 1.0, n=11)