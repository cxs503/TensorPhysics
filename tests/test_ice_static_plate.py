"""Hetenyi / Kelvin-function verification of the steady plate deflection."""
import numpy as np
import pytest

from tensorphysics.ice.dispersion import FlexuralDispersion
from tensorphysics.ice.static_plate import (
    biharmonic_symbol,
    hankel_gauss_deflection,
    hankel_gauss_stress_r,
    hetenyi_center_deflection,
    hetenyi_point_load,
    solve_biharmonic_fft,
)


def _static_problem(L=768.0, dx=2.0, P=4.36e6, sigma=4.0, h=1.5):
    disp = FlexuralDispersion.from_geometry(9.0e9, h, 0.3)
    nx = int(round(L / dx))
    x = np.arange(nx) * dx - 0.5 * L
    X, Y = np.meshgrid(x, x, indexing="xy")
    load = P / (2.0 * np.pi * sigma**2) * np.exp(-(X**2 + Y**2) / (2.0 * sigma**2))
    w = solve_biharmonic_fft(load, dx, disp.rigidity)
    iy, ix = np.unravel_index(np.argmin(np.hypot(X, Y)), X.shape)
    return disp, w, X, Y, x, iy, ix


def test_hetenyi_centre_matches_classical_formula():
    D = FlexuralDispersion.from_geometry(9.0e9, 1.0, 0.3).rigidity
    P = 1.0e6
    assert hetenyi_point_load(P, D, 0.0) == pytest.approx(hetenyi_center_deflection(P, D), rel=1e-12)
    # kei(0) = -pi/4  ->  w(0) = P lc^2 / (8 D)
    assert hetenyi_point_load(P, D, 0.0) > 0.0  # downward positive
    assert np.all(np.diff(hetenyi_point_load(P, D, np.linspace(0, 5, 20))) < 0.0)


def test_biharmonic_symbol_limits():
    S = biharmonic_symbol(32, 32, 0.5, kind="finite_difference")
    Ssp = biharmonic_symbol(32, 32, 0.5, kind="spectral")
    # low wavenumber: the discrete 13-point stencil reproduces k^4
    kx = 2 * np.pi * np.fft.fftfreq(32, d=0.5)
    KX, KY = np.meshgrid(kx, kx, indexing="xy")
    k2 = KX**2 + KY**2
    small = k2 < 0.05
    assert np.allclose(S[small], Ssp[small], rtol=2e-2)


def test_static_solver_matches_hankel_reference():
    disp, w, X, Y, x, iy, ix = _static_problem()
    lc = disp.characteristic_length()
    radii = np.array([0.0, 0.5 * lc, 1.0 * lc, 2.0 * lc])
    ref = np.atleast_1d(hankel_gauss_deflection(radii, 4.36e6, 4.0, disp.rigidity))
    num = np.interp(radii, x[ix:], w[iy, ix:])
    assert np.max(np.abs(num - ref)) / abs(ref[0]) * 100.0 <= 5.0
    # and against the classical point-load value
    assert num[0] == pytest.approx(hetenyi_point_load(4.36e6, disp.rigidity, 0.0), rel=0.05)


def test_hankel_quadrature_convergence():
    D = FlexuralDispersion.from_geometry(9.0e9, 1.5, 0.3).rigidity
    a = hankel_gauss_deflection(np.array([0.0, 10.0]), 4.36e6, 4.0, D, n_nodes=20001)
    b = hankel_gauss_deflection(np.array([0.0, 10.0]), 4.36e6, 4.0, D, n_nodes=80001)
    assert np.max(np.abs(a - b)) < 1e-8
    # shrinking patch -> classical point-load centre deflection
    point = hetenyi_point_load(4.36e6, D, 0.0)
    for sigma, tol in ((0.5, 5e-3), (0.2, 5e-3)):
        assert hankel_gauss_deflection(0.0, 4.36e6, sigma, D) == pytest.approx(point, rel=tol)


def test_bottom_stress_sign_and_profile():
    """Downward load: bottom tension under the load, top tension further out."""
    disp, w, X, Y, x, iy, ix = _static_problem()
    out = hankel_gauss_stress_r(np.array([0.0, 0.5 * disp.characteristic_length()]),
                                4.36e6, 4.0, disp.rigidity, 9.0e9, 1.5)
    assert out["sigma_r_Pa"][0] > 0.0            # tensile at the bottom
    assert out["sigma_r_Pa"][0] == pytest.approx(out["sigma_t_Pa"][0], rel=1e-6)
    assert out["sigma_t_Pa"][1] > out["sigma_r_Pa"][1]
