"""Energy consistency and time-integration checks of the coupled plate-fluid solver."""
import numpy as np
import pytest

from tensorphysics.ice.coupled import CoupledPlateFluid, IceWaterParameters
from tensorphysics.ice.dispersion import FlexuralDispersion


def _sim(**kw):
    params = IceWaterParameters(E=9.0e9, h=1.5, nu=0.3, depth=50.0,
                                damping=kw.pop("damping", 0.0))
    return CoupledPlateFluid(params, nx=64, ny=64, lx=256.0, ly=256.0, **kw)


def test_free_vibration_conserves_energy():
    sim = _sim()
    x, y = sim.grid()
    lc = sim.characteristic_length()
    sim.set_deflection(0.03 * np.cos(x / lc) * np.cos(0.5 * y / lc))
    e0 = sim.energies()["total_J"]
    for _ in range(500):
        sim.step(0.005, np.zeros_like(sim.W))
        assert np.all(np.isfinite(sim.W))
    e1 = sim.energies()["total_J"]
    assert abs(e1 - e0) / e0 < 1e-9


def test_energy_components_are_positive_and_add_up():
    sim = _sim()
    x, y = sim.grid()
    sim.set_deflection(0.02 * np.cos(x / 20.0))
    e = sim.energies()
    for key in ("bending_J", "buoyancy_J", "ice_kinetic_J", "fluid_kinetic_J"):
        assert e[key] >= 0.0
    total = e["bending_J"] + e["buoyancy_J"] + e["ice_kinetic_J"] + e["fluid_kinetic_J"]
    assert e["total_J"] == pytest.approx(total, rel=1e-12)
    # a static deflection stores bending + buoyancy energy only
    assert e["ice_kinetic_J"] < 1e-12 and e["fluid_kinetic_J"] < 1e-12


def test_static_limit_of_the_dynamic_solver():
    """Heavily damped relaxation must reproduce the static FFT solution.

    The comparison is made on the *same* periodic box, so the (known) truncation
    error of the finite domain cancels and only the time-integration limit is
    tested.
    """
    from tensorphysics.ice.static_plate import solve_biharmonic_fft

    sim = _sim(damping=2.0e4)
    X, Y = sim.grid()
    F, sigma = 1.0e6, 12.0
    f_hat = sim.patch_load_hat(F, 0.5 * sim.lx, 0.5 * sim.ly, sigma)
    load = F / (2.0 * np.pi * sigma**2) * np.exp(
        -((X - 0.5 * sim.lx) ** 2 + (Y - 0.5 * sim.ly) ** 2) / (2.0 * sigma**2))
    w_static = solve_biharmonic_fft(load, sim.dx, sim.params.rigidity)
    for _ in range(3000):
        sim.step(0.02, f_hat)
    w_num = sim.value_at(0.5 * sim.lx, 0.5 * sim.ly, "w")
    iy, ix = int(round(0.5 * sim.ly / sim.dy)) % sim.ny, int(round(0.5 * sim.lx / sim.dx)) % sim.nx
    assert w_num == pytest.approx(float(w_static[iy, ix]), rel=0.02)


def test_damping_removes_energy():
    sim = _sim(damping=5.0e3)
    x, y = sim.grid()
    sim.set_deflection(0.02 * np.cos(x / 20.0))
    e0 = sim.energies()["total_J"]
    diss = 0.0
    p_prev = sim.dissipation_power()
    for _ in range(400):
        sim.step(0.01, np.zeros_like(sim.W))
        p = sim.dissipation_power()
        diss += 0.5 * (p_prev + p) * 0.01
        p_prev = p
    e1 = sim.energies()["total_J"]
    assert diss > 0.0
    assert abs((e0 - e1) - diss) / (e0 - e1) < 0.05


def test_underwater_pressure_is_finite_and_bounded():
    sim = _sim()
    x, y = sim.grid()
    sim.set_deflection(0.01 * np.cos(x / 25.0))
    sim.step(0.01, np.zeros_like(sim.W))
    p = sim.fluid_pressure()
    assert np.all(np.isfinite(p))
    assert np.max(np.abs(p)) > 0.0


def test_zero_wavenumber_mode_is_excluded():
    sim = _sim()
    sim.set_deflection(np.full((sim.ny, sim.nx), 0.01))
    assert abs(sim.W[0, 0]) < 1e-15
    f = sim.patch_load_hat(1.0e6, 0.0, 0.0, 5.0)
    assert abs(f[0, 0]) > 0.0
    sim.step(0.01, f)
    assert abs(sim.W[0, 0]) < 1e-15
