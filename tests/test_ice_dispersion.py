"""Verification of the plate-fluid dispersion and the potential-flow added mass."""
import numpy as np
import pytest

from tensorphysics.ice.coupled import CoupledPlateFluid, IceWaterParameters
from tensorphysics.ice.dispersion import (
    FlexuralDispersion,
    added_mass_per_area,
    characteristic_length,
    depth_fem_added_mass,
    plate_rigidity,
)


def test_plate_rigidity_and_characteristic_length():
    D = plate_rigidity(9.0e9, 1.0, 0.3)
    assert D == pytest.approx(9.0e9 * 1.0**3 / (12.0 * (1.0 - 0.09)), rel=1e-12)
    lc = characteristic_length(D, 1025.0, 9.81)
    assert lc == pytest.approx((D / (1025.0 * 9.81)) ** 0.25, rel=1e-12)
    # static limit of the dispersion relation: D k^4 = rho_w g at k = 1/lc
    disp = FlexuralDispersion.from_geometry(9.0e9, 1.0, 0.3)
    assert disp.rigidity * (1.0 / lc) ** 4 == pytest.approx(1025.0 * 9.81, rel=1e-10)


@pytest.mark.parametrize("k", [0.02, 0.05, 0.1, 0.3])
def test_added_mass_matches_independent_depth_fem(k):
    """The analytic added mass is cross-checked against a 1-D FEM of Laplace."""
    analytic = added_mass_per_area(k, 1025.0, 50.0)
    fem = depth_fem_added_mass(k, 1025.0, 50.0, n_elem=400)
    assert fem == pytest.approx(analytic, rel=2e-3)


def test_added_mass_limits():
    k = np.array([0.0, 0.1])
    m = added_mass_per_area(k, 1025.0, None)
    assert np.isinf(m[0])
    assert m[1] == pytest.approx(1025.0 / 0.1)
    # shallow-water limit: m_a -> rho_w / (k^2 H)
    deep = added_mass_per_area(0.01, 1025.0, None)
    shallow = added_mass_per_area(0.01, 1025.0, 5.0)
    assert shallow == pytest.approx(1025.0 / (0.01 * np.tanh(0.05)), rel=1e-12)
    assert shallow > deep          # confinement increases the added mass


def test_dispersion_inversion_and_limits():
    disp = FlexuralDispersion.from_geometry(9.0e9, 1.0, 0.3, depth=50.0)
    for k in (0.02, 0.06, 0.2):
        w = float(disp.omega(k))
        assert disp.k_from_omega(w) == pytest.approx(k, rel=1e-8)
    # long-wave limit vanishes, short waves grow
    assert float(disp.omega(0.0)) == 0.0
    assert float(disp.omega(1e-9)) < 1e-4
    assert disp.omega(0.4) > disp.omega(0.1)
    c_min, k_min = disp.minimum_phase_speed()
    assert 10.0 < c_min < 25.0
    assert disp.static_wavelength() == pytest.approx(2.0 * np.pi * disp.characteristic_length())


def test_time_domain_frequency_matches_dispersion():
    """A single grid mode must oscillate at the analytic frequency."""
    params = IceWaterParameters(E=9.0e9, h=1.0, nu=0.3, depth=50.0, damping=0.0)
    nx, ny, lx, ly = 64, 8, 512.0, 16.0
    sim = CoupledPlateFluid(params, nx=nx, ny=ny, lx=lx, ly=ly)
    disp = FlexuralDispersion.from_geometry(9.0e9, 1.0, 0.3, depth=50.0)
    m_mode = 6
    k0 = 2.0 * np.pi * m_mode / lx
    x, _ = sim.grid()
    sim.set_deflection(0.02 * np.cos(k0 * x))
    e0 = sim.energies()["total_J"]
    dt, n = 0.01, 3000
    probe = []
    for _ in range(n):
        sim.step(dt, np.zeros_like(sim.W))
        probe.append(sim.value_at(0.0, 0.0, "w"))
    probe = np.asarray(probe)
    t = np.arange(1, n + 1) * dt
    crossings = np.where(np.diff(np.signbit(probe)))[0]
    half = t[crossings[1:]] - t[crossings[:-1]]
    half = half[half > 2 * dt]
    omega = np.pi / np.mean(half)
    assert omega == pytest.approx(float(disp.omega(k0)), rel=0.01)
    # non-dissipative integrator: energy is preserved
    assert abs(sim.energies()["total_J"] - e0) / e0 < 1e-9


def test_patch_load_normalisation():
    """The Gaussian patch must carry exactly the requested total force."""
    params = IceWaterParameters(E=9.0e9, h=1.0, depth=50.0)
    sim = CoupledPlateFluid(params, nx=64, ny=64, lx=256.0, ly=256.0)
    F = 1.0e6
    f_hat = sim.patch_load_hat(F, 10.0, -20.0, 3.0)
    pressure = np.real(np.fft.ifft2(f_hat))
    assert np.sum(pressure) * sim.dx * sim.dy == pytest.approx(F, rel=1e-4)
    assert sim.patch_pressure_peak(F, 3.0) == pytest.approx(F / (2 * np.pi * 9.0))
