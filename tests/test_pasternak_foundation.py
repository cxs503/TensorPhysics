"""Tests for the Pasternak two-parameter elastic foundation (P54-W3-N02).

Covers
------
* closed-form layer: dispersion roots, characteristic lengths, Winkler
  degeneration (exact), Bessel kernel accuracy vs an ``mpmath`` reference,
  moment derivatives, contact-patch field, first-crack loads, calibration;
* FEM layer: shear-layer element matrices, Winkler degeneration of the assembled
  operator, monotone deflection reduction, agreement with the closed form for a
  point load and for a finite contact patch, free-plate negative control,
  linearity, and input validation.

The FEM cases use deliberately coarse graded meshes so the suite stays fast
(< ~30 s); the production-grade mesh study lives in
``scripts/p54w3_pasternak_foundation_verify.py``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_sheet import pasternak_foundation as pf
from tensorphysics.ice_sheet.kirchhoff_plate_fem import (
    BFSPlateSolver,
    KirchhoffWinklerPlate,
    graded_axis_centered,
    kelvin_point_load,
    kelvin_point_load_moments,
)

H, E, NU = 1.0, 5.0e9, 0.3
D = E * H**3 / (12.0 * (1.0 - NU**2))
K = 1025.0 * 9.81
L_C = (D / K) ** 0.25
FREE = {"left": "free", "right": "free", "bottom": "free", "top": "free"}
P = 1.0e6


def g_of(gamma: float) -> float:
    """Shear-layer modulus for a prescribed dimensionless coupling gamma."""
    return gamma * math.sqrt(D * K)


# ---------------------------------------------------------------------------
# closed-form layer
# ---------------------------------------------------------------------------
def test_dispersion_roots_identities():
    for gamma in (0.0, 0.5, 1.0, 3.0):
        roots = pf.pasternak_roots(D, K, g_of(gamma))
        m1, m2 = roots["m1"], roots["m2"]
        g_p = roots["g_p"]
        # D m^2 - G_p m + k = 0  ==>  sum / product in complex arithmetic
        assert abs((m1 + m2) - g_p / D) < 1e-6 * abs(g_p / D) + 1e-12
        assert abs((m1 * m2) - K / D) < 1e-9 * (K / D)
        a, b = roots["a"], roots["b"]
        assert abs(a * a - m1) < 1e-8 * abs(m1)
        assert abs(b * b - m2) < 1e-8 * abs(m2)
        assert a.real > 0.0 and b.real > 0.0


def test_regimes_follow_the_discriminant():
    assert pf.pasternak_roots(D, K, 0.0)["regime"] == "winkler"
    assert pf.pasternak_roots(D, K, g_of(1.0))["regime"] == "complex"
    assert pf.pasternak_roots(D, K, g_of(2.0 + 1e-6))["regime"] == "real"


def test_length_scales():
    fnd = pf.PasternakFoundation(k_w=K, g_p=g_of(2.0))
    assert fnd.char_length(D) == pytest.approx(L_C, rel=1e-12)
    assert fnd.shear_length() == pytest.approx(math.sqrt(g_of(2.0) / K), rel=1e-12)
    assert fnd.coupling(D) == pytest.approx(2.0, rel=1e-12)
    assert pf.PasternakFoundation().is_winkler


def test_winkler_degeneration_is_exact():
    r = np.linspace(0.0, 80.0, 81)
    a = np.atleast_1d(pf.pasternak_point_load(r, P, D, K, 0.0))
    b = np.atleast_1d(kelvin_point_load(r, P, D, K))
    assert np.array_equal(a, b)  # delegation, not an approximation


def test_moment_degeneration_is_exact():
    r = np.linspace(1e-3, 8.0 * L_C, 41)
    a = pf.pasternak_point_load_moments(r, P, D, K, 0.0, NU)
    b = kelvin_point_load_moments(r, P, D, K, NU)
    for key in ("w", "w_r", "w_rr", "M_r", "M_t"):
        assert np.array_equal(np.asarray(a[key]), np.asarray(b[key]))


def test_center_deflection_limit_winkler():
    """w(0) = P l^2 / (8 D) for the Hetényi point load (kei(0) = -pi/4)."""
    w0 = pf.pasternak_point_load(0.0, P, D, K, 0.0)
    assert w0 == pytest.approx(P * L_C**2 / (8.0 * D), rel=1e-12)


def test_center_deflection_finite_and_decreasing_for_shear_layer():
    vals = [pf.pasternak_point_load(0.0, P, D, K, g_of(gm)) for gm in (0.0, 0.5, 1.0, 2.0, 4.0)]
    assert all(np.isfinite(vals))
    assert all(vals[i + 1] < vals[i] for i in range(len(vals) - 1))


@pytest.mark.parametrize(
    "mag,angle",
    [(0.2, 0.4), (1.5, 0.785), (4.0, 1.2), (8.4, 0.05), (8.6, 0.3), (15.0, -0.7), (30.0, 1.1)],
)
def test_bessel_kernel_matches_mpmath(mag, angle):
    """The vectorised K_0/K_1 kernel vs a 30-digit mpmath reference."""
    z = np.array([mag * np.exp(1j * angle)])
    f0, f1 = pf._bessel_k0_k1(z)
    r0 = pf.bessel_k_reference(z, 0)
    r1 = pf.bessel_k_reference(z, 1)
    assert abs(f0[0] - r0[0]) / abs(r0[0]) < 1e-7
    assert abs(f1[0] - r1[0]) / abs(r1[0]) < 1e-7


@pytest.mark.parametrize("gamma", [0.5, 1.0, 5.0])
def test_patch_field_tends_to_point_load(gamma):
    """A vanishing contact radius must reproduce the point-load Green function."""
    g_p = g_of(gamma)
    field = pf.pasternak_patch_field(
        P, 1.0e-4, D, K, g_p, NU, r_max=3.0 * L_C, n_field=801, n_source=48
    )
    ref = np.atleast_1d(pf.pasternak_point_load(field["r"], P, D, K, g_p))
    assert np.max(np.abs(field["w"] - ref)) / np.max(np.abs(ref)) < 1e-6


def test_moments_are_derivatives_of_the_deflection():
    """M_r = -D (w,rr + nu w,r / r) checked against finite differences of w."""
    g_p = g_of(1.0)
    r = np.linspace(0.3 * L_C, 6.0 * L_C, 601)
    mom = pf.pasternak_point_load_moments(r, P, D, K, g_p, NU)
    w = mom["w"]
    dr = r[1] - r[0]
    w_r = np.gradient(w, dr)
    w_rr = np.gradient(w_r, dr)
    m_fd = -D * (w_rr + NU * w_r / r)
    core = slice(20, -20)   # np.gradient is one-sided at the ends
    scale = np.max(np.abs(mom["M_r"][core]))
    assert np.max(np.abs(m_fd[core] - mom["M_r"][core])) / scale < 1e-3


def test_point_and_patch_critical_loads_agree():
    """Two independent formulations (point Green function / ring-integrated disc)."""
    m_f = 0.7e6 * H**2 / 6.0
    for gamma in (0.0, 1.0, 5.0):
        point = pf.pasternak_critical_point_load(m_f, D, K, g_of(gamma), NU)
        patch = pf.pasternak_critical_patch_load(m_f, D, K, g_of(gamma), 0.5, NU)
        assert point["interior_minimum"] is True          # finite ring-crack peak
        assert abs(point["P_crit_N"] / patch["P_crit_N"] - 1.0) < 1e-2


def test_critical_patch_load_increases_with_shear_layer():
    m_f = 0.7e6 * H**2 / 6.0
    loads = [
        pf.pasternak_critical_patch_load(m_f, D, K, g_of(gm), 0.5, NU)["P_crit_N"]
        for gm in (0.0, 0.5, 1.0, 2.0, 4.0)
    ]
    assert all(loads[i + 1] > loads[i] for i in range(len(loads) - 1))
    # Winkler reference: the first-crack load of a floating sheet under a patch
    assert loads[0] == pytest.approx(6.14e6, rel=0.02)


def test_critical_patch_load_is_radius_insensitive():
    m_f = 0.7e6 * H**2 / 6.0
    vals = [
        pf.pasternak_critical_patch_load(m_f, D, K, g_of(1.0), rc, NU)["P_crit_N"]
        for rc in (0.25, 0.5, 1.0, 2.0)
    ]
    assert np.max(np.abs(np.diff(vals))) / vals[0] < 0.02


def test_ice_calibration_matches_the_formulas():
    cal = pf.ice_foundation_calibration(H, E=E, nu=NU, beta=1.0)
    g_ice = E / (2.0 * (1.0 + NU))
    assert cal["k_w_Pa_per_m"] == pytest.approx(1025.0 * 9.81, rel=1e-12)
    assert cal["G_ice_Pa"] == pytest.approx(g_ice, rel=1e-12)
    assert cal["g_p_N_per_m"] == pytest.approx(g_ice * H, rel=1e-12)
    assert cal["D_Nm"] == pytest.approx(D, rel=1e-12)
    assert cal["gamma"] == pytest.approx(g_ice * H / math.sqrt(D * K), rel=1e-12)
    assert cal["regime"] == "real"  # gamma ~ 900 >> 2


def test_calibration_table_is_consistent_and_monotone():
    rows = pf.calibration_table(H, E=E, nu=NU)
    assert len(rows) >= 5
    base = rows[0]
    assert base["beta"] == 0.0
    assert base["w_center_ratio_vs_winkler"] == pytest.approx(1.0, rel=1e-12)
    assert base["P_crit_ratio_vs_winkler"] == pytest.approx(1.0, rel=1e-12)
    gammas = [r["gamma"] for r in rows]
    assert all(gammas[i + 1] > gammas[i] for i in range(len(gammas) - 1))
    for row in rows:
        assert row["w_center_ratio_vs_winkler"] <= 1.0
        assert row["P_crit_deviation_vs_winkler_pct"] >= 0.0


def test_validation_errors():
    with pytest.raises(ValueError):
        pf.PasternakFoundation(k_w=0.0)
    with pytest.raises(ValueError):
        pf.PasternakFoundation(g_p=-1.0)
    with pytest.raises(ValueError):
        pf.pasternak_roots(D, K, -1.0)
    with pytest.raises(ValueError):
        pf.pasternak_point_load(-1.0, P, D, K, g_of(1.0))
    with pytest.raises(ValueError):
        pf.pasternak_point_load_moments(0.0, P, D, K, g_of(1.0), NU)
    with pytest.raises(ValueError):
        pf.pasternak_critical_patch_load(0.0, D, K, g_of(1.0), 0.5, NU)
    with pytest.raises(ValueError):
        pf.pasternak_patch_field(P, 0.0, D, K, g_of(1.0), NU)
    with pytest.raises(ValueError):
        pf.ice_foundation_calibration(0.0)


# ---------------------------------------------------------------------------
# FEM layer
# ---------------------------------------------------------------------------
def _plate(L: float) -> KirchhoffWinklerPlate:
    return KirchhoffWinklerPlate(h=H, E=E, nu=NU, k_w=K, lx=L, ly=L)


def test_shear_element_matrix_properties():
    L, n, h0 = 60.0, 12, 2.5
    xs = graded_axis_centered(L, n, h0)
    wink = BFSPlateSolver(_plate(L), n, n, xs=xs, ys=xs)
    past = BFSPlateSolver(_plate(L), n, n, xs=xs, ys=xs, foundation_model="pasternak", g_p=g_of(1.0))
    assert np.all(past.element_shear_matrix() != 0.0)
    assert np.all(wink.element_shear_matrix() == 0.0)
    Kg = past.element_shear_matrix()[0]
    assert np.allclose(Kg, Kg.T)                       # symmetric
    ev = np.linalg.eigvalsh(0.5 * (Kg + Kg.T))
    assert ev.min() > -1e-8 * max(1.0, ev.max())       # positive semi-definite
    # rigid-body translation (w = const, slopes = 0) carries no shear energy
    rigid = np.zeros(16)
    rigid[[0, 4, 8, 12]] = 1.0
    assert abs(rigid @ Kg @ rigid) < 1e-9 * ev.max()


def test_fem_winkler_degeneration_is_bit_identical():
    L, n = 80.0, 20
    sw = BFSPlateSolver(_plate(L), n, n)
    sp = BFSPlateSolver(_plate(L), n, n, foundation_model="pasternak", g_p=0.0)
    Kw, Kp = sw._stiffness(), sp._stiffness()
    assert (Kw != Kp).nnz == 0
    fw = sw.add_point_load(P, 0.0, 0.0).solve(bc=FREE)
    fp = sp.add_point_load(P, 0.0, 0.0).solve(bc=FREE)
    assert np.array_equal(fw.w, fp.w)
    assert np.array_equal(fw.Mx, fp.Mx)
    assert fp.foundation_model == "pasternak" and fp.g_p == 0.0


def test_solver_validation_errors():
    plate = _plate(20.0)
    # pasternak with g_p = 0 is the (documented) Winkler degeneration, not an error
    assert BFSPlateSolver(plate, 4, 4, foundation_model="pasternak").g_p == 0.0
    with pytest.raises(ValueError):
        BFSPlateSolver(plate, 4, 4, g_p=g_of(1.0))           # needs the pasternak model
    with pytest.raises(ValueError):
        BFSPlateSolver(plate, 4, 4, foundation_model="pasternak", g_p=-1.0)
    with pytest.raises(ValueError):
        BFSPlateSolver(plate, 4, 4, foundation_model="soil")


def test_fem_deflection_decreases_with_shear_layer():
    vals = []
    for g_p in (0.0, g_of(1.0), g_of(4.0)):
        sol = BFSPlateSolver(_plate(160.0), 60, 60,
                             xs=graded_axis_centered(160.0, 60, 0.5),
                             ys=graded_axis_centered(160.0, 60, 0.5),
                             foundation_model="pasternak", g_p=g_p)
        fld = sol.add_point_load(P, 0.0, 0.0).solve(bc=FREE)
        i0 = int(np.argmin(np.abs(fld.x)))
        vals.append(float(fld.w[i0, i0]))
    assert vals[0] > vals[1] > vals[2] > 0.0


def test_fem_point_load_matches_closed_form():
    """(b) criterion at test scale: gamma = 1, r in [0, 5 l_c]."""
    g_p = g_of(1.0)
    L, n, h0 = 240.0, 60, 0.5
    sol = BFSPlateSolver(_plate(L), n, n,
                         xs=graded_axis_centered(L, n, h0),
                         ys=graded_axis_centered(L, n, h0),
                         foundation_model="pasternak", g_p=g_p)
    fld = sol.add_point_load(P, 0.0, 0.0).solve(bc=FREE)
    r = np.linspace(0.0, 5.0 * L_C, 101)
    ray = fld.sample_ray((0.0, 0.0), 0.0, r=r)
    ana = np.atleast_1d(pf.pasternak_point_load(r, P, D, K, g_p))
    assert abs(ray["w"][0] - ana[0]) / ana[0] < 0.03
    assert np.sqrt(np.mean(((ray["w"] - ana) / ana) ** 2)) < 0.05


def test_fem_patch_moment_matches_closed_form():
    """(d) support: the peak hogging moment under a disc load."""
    Rc = 0.5
    g_p = g_of(1.0)
    L, n, h0 = 240.0, 80, 0.25
    sol = BFSPlateSolver(_plate(L), n, n,
                         xs=graded_axis_centered(L, n, h0),
                         ys=graded_axis_centered(L, n, h0),
                         foundation_model="pasternak", g_p=g_p)
    q = P / (math.pi * Rc * Rc)
    sol.add_pressure(lambda x, y: np.full(np.shape(x), q),
                     inside=lambda x, y: np.hypot(x, y) <= Rc,
                     area=math.pi * Rc * Rc, gauss_n=4, bbox=(-Rc, Rc, -Rc, Rc))
    fld = sol.solve(bc=FREE)
    ray = fld.sample_ray((0.0, 0.0), 0.0, r=np.linspace(0.5 * Rc, 5.0 * L_C, 160))
    ana = pf.pasternak_patch_field(P, Rc, D, K, g_p, NU, r_max=8.0 * L_C,
                                   n_field=4001, n_source=96)
    peak_fem = float(np.min(ray["M_r"]))
    peak_ana = float(ana["M_r"].min())
    assert abs(peak_fem - peak_ana) / abs(peak_ana) < 0.03


def test_free_plate_negative_control():
    """(e) k = G_p = 0 reproduces the simply supported Navier plate."""
    a, q, n = 20.0, 1.0e4, 24
    sol = BFSPlateSolver(_plate(a), n, n, foundation_model="none")
    sol.add_pressure(lambda x, y: np.full(np.shape(x), q), area=a * a, gauss_n=4)
    bc = {"left": "simply", "right": "simply", "bottom": "simply", "top": "simply"}
    fld = sol.solve(bc=bc)
    total = 0.0
    for m in range(1, 200, 2):
        for k in range(1, 200, 2):
            total += (math.sin(m * math.pi / 2) * math.sin(k * math.pi / 2)
                      / (m * k * ((m / a) ** 2 + (k / a) ** 2) ** 2))
    w_ref = 16.0 * q / (math.pi**6 * D) * total   # total carries the a^4 scaling
    assert abs(float(np.max(fld.w)) - w_ref) / w_ref < 5e-3
    assert np.max(np.abs(sol.element_matrices()[1])) == 0.0
    assert np.max(np.abs(sol.element_shear_matrix())) == 0.0


def test_load_linearity_and_zero_load():
    sols = {}
    for load in (0.0, 1.0e3, 1.0e6):
        sol = BFSPlateSolver(_plate(80.0), 20, 20, foundation_model="pasternak", g_p=g_of(1.0))
        fld = sol.add_point_load(load, 0.0, 0.0).solve(bc=FREE)
        i0 = int(np.argmin(np.abs(fld.x)))
        sols[load] = float(fld.w[i0, i0])
    assert sols[0.0] == 0.0
    assert sols[1.0e6] / sols[1.0e3] == pytest.approx(1.0e3, rel=1e-9)

@pytest.mark.parametrize("gamma", [0.0, 1.0, 5.0])
def test_disc_load_quadrature_matches_ring_averaged_field(gamma):
    """Two independent disc-load formulations: 2-D polar quadrature vs Graf ring average."""
    g_p = g_of(gamma)
    r = np.array([0.0, 0.5, 1.0, 2.0, 4.0]) * L_C
    field = pf.pasternak_patch_field(P, 0.5, D, K, g_p, NU, r_max=8.0 * L_C,
                                     n_field=801, n_source=64)
    w_ref = np.interp(r, field["r"], field["w"])
    w_quad = pf.pasternak_patch_load(r, P, 0.5, D, K, g_p, n_radial=24, n_theta=48)
    assert np.max(np.abs(w_quad - w_ref) / np.abs(w_ref)) < 1e-3
