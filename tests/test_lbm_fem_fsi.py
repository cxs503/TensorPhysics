"""Verification tests for the LBM -> FEM conservative exchange (node p362w2-n09).

The tests are deliberately ordered from the primitives (equilibrium, wall
operator, Q4 element, Newmark identity) to the exchange (resultant-preserving
mapping, impulse/work conservation) so a failure points at one layer only.
Every test is fast enough for CI except the final coupled smoke test, which runs
two shedding periods on a D = 6 cylinder.
"""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.lbm_fem_fsi import coupling as cp
from tensorphysics.lbm_fem_fsi import interface as itf
from tensorphysics.lbm_fem_fsi.lattice import (EX, EY, W, LatticeFSI, Obstacle,
                                                equilibrium)
from tensorphysics.lbm_fem_fsi.structure import PlaneStressTether, q4_mass, q4_stiffness


# --------------------------------------------------------------------- lattice
def test_equilibrium_has_correct_moments():
    rng = np.random.default_rng(0)
    rho = 1.0 + 0.1 * rng.random((5, 4))
    ux = 0.05 * rng.standard_normal((5, 4))
    uy = 0.05 * rng.standard_normal((5, 4))
    feq = equilibrium(rho, ux, uy)
    assert np.allclose(feq.sum(axis=0), rho, rtol=1e-13, atol=1e-14)
    assert np.allclose((EX[:, None, None] * feq).sum(axis=0), rho * ux, atol=1e-14)
    assert np.allclose((EY[:, None, None] * feq).sum(axis=0), rho * uy, atol=1e-14)
    assert np.isclose(W.sum(), 1.0)


def test_uniform_flow_is_a_fixed_point():
    """Inlet/outlet/free-slip walls must preserve a uniform stream exactly."""
    lat = LatticeFSI(24, 16, tau=0.6, U=0.1, obstacle=Obstacle(6.0, -400.0, 2.0))
    _, ux, uy = lat.velocity_field()
    assert np.allclose(ux, 0.1, atol=1e-9)
    for _ in range(120):
        lat.step()
    rho, ux, uy = lat.velocity_field()
    assert np.allclose(ux[:, 2:-2], 0.1, atol=1e-6)
    assert np.allclose(uy[:, 2:-2], 0.0, atol=1e-6)
    assert np.allclose(rho[:, 2:-2], 1.0, atol=1e-6)


def test_mass_ledger_closes_to_roundoff():
    lat = LatticeFSI(40, 30, tau=0.6, U=0.1, obstacle=Obstacle(12.0, 15.0, 3.0))
    lat.perturb(0.02)
    lat.mass_initial = lat.mass_total()
    for _ in range(150):
        lat.step()
    mb = lat.mass_balance()
    assert abs(mb["closure_error"]) / lat.mass_initial < 1e-10


def test_ladd_exchange_matches_global_budget():
    """The two *independent* force measurements agree to round-off."""
    lat = LatticeFSI(48, 32, tau=0.6, U=0.1, obstacle=Obstacle(16.0, 16.0, 4.0))
    mea = np.zeros(2)
    bud = np.zeros(2)
    for _ in range(120):
        led = lat.step()
        mea += led.force_mea
        bud += led.force_budget
    assert np.linalg.norm(mea - bud) / np.linalg.norm(mea) < 1e-9


def test_halfway_wall_creates_no_mass_but_interpolated_does():
    """Half-way bounce-back is exactly mass conserving; the Bouzidi variant leaks."""
    lat = LatticeFSI(40, 30, tau=0.6, U=0.1, obstacle=Obstacle(12.0, 15.0, 3.0))
    lat.perturb(0.02)
    lat.mass_initial = lat.mass_total()
    for _ in range(80):
        lat.step()
    assert abs(lat.mass_body_exchange) < 1e-9

    lat2 = LatticeFSI(40, 30, tau=0.6, U=0.1, obstacle=Obstacle(12.0, 15.0, 3.0),
                      wall_mode="interp")
    lat2.perturb(0.02)
    lat2.mass_initial = lat2.mass_total()
    for _ in range(80):
        lat2.step()
    assert abs(lat2.mass_body_exchange) > 1e-6


def test_rasterised_body_area_matches_circle():
    """Geometry gate: the solid area reproduces pi R^2 to better than 2 %.

    The rasterised area is quantised in whole nodes, so the achievable mismatch
    is 0.5 % (D = 12, 24) to 1.7 % (D = 8, 16) in diameter; the naive
    ``R - 1/2`` mask without calibration gives 12 % (D = 12).
    """
    for D in (10, 12, 16):
        lat = LatticeFSI(6 * D, 5 * D, tau=0.6, U=0.1, obstacle=Obstacle(2 * D, 2.5 * D, D / 2))
        rel = abs(lat.effective_diameter() - D) / D
        assert rel < 2e-2, (D, lat.effective_diameter())


# --------------------------------------------------------------- FE structure
def test_q4_patch_constant_strain():
    """Patch test: a Q4 mesh under uniform strain must reproduce sigma = D eps."""
    xy = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]])
    E, nu, t = 210e9, 0.3, 0.01
    K = q4_stiffness(xy, E, nu, t)
    # uniform uniaxial strain eps_xx = e: u = e x
    e = 1e-3
    u = np.array([0.0, 0.0, e * 2.0, 0.0, e * 2.0, 0.0, 0.0, 0.0])
    f = K @ u
    # a uniform-strain state is self-equilibrated: the nodal forces must sum to 0
    assert np.allclose([f[0::2].sum(), f[1::2].sum()], 0.0, atol=1e-6 * np.abs(f).max())
    # ... and the strain energy must equal the analytic 1/2 sigma:eps V
    U = 0.5 * u @ K @ u
    sig = E / (1 - nu ** 2) * e                 # sigma_xx for eps_xx = e, eps_yy = 0
    U_analytic = 0.5 * sig * e * (2.0 * 1.0 * t)
    assert np.isclose(U, U_analytic, rtol=1e-10)
    M = q4_mass(xy, 1000.0, t)
    assert np.isclose(0.5 * M.sum(), 1000.0 * t * 2.0, rtol=1e-10)  # two DOFs per node


def test_tether_frequency_scales_with_stiffness():
    t = PlaneStressTether(length=32.0, height=4.0, n_s=12, n_w=3, E=100.0, rho=0.05)
    t.add_tip_mass(500.0)
    f1 = t.first_frequency()
    assert 0.0 < f1 < 1.0
    t.E *= 4.0
    t._assemble()
    t.add_tip_mass(500.0)
    assert np.isclose(t.first_frequency() / f1, 2.0, rtol=1e-6)


def test_newmark_energy_identity():
    """W_external = dKE + dStrain + E_damping for a sinusoidally driven tether."""
    t = PlaneStressTether(length=32.0, height=4.0, n_s=16, n_w=3, E=1.0, rho=0.05)
    t.add_tip_mass(600.0)
    f1 = t.first_frequency()
    assert np.isclose(f1, 0.0025, atol=2e-3) or f1 > 0
    t.set_rayleigh(0.01)
    adofs = 2 * t.tip_nodes + 1
    ndof = 2 * t.n_nodes
    dt, T = 4.0, 1.0 / t.first_frequency()
    for k in range(300):
        load = np.zeros(ndof)
        load[adofs] = 0.02 * np.sin(2 * np.pi * (k * dt) / T)
        t.advance(load, dt)
    residual = abs(t.external_work - (t.kinetic() + t.strain() + t.damping_dissipation))
    assert residual / abs(t.external_work) < 0.02


# ------------------------------------------------------------------ interface
def test_resultant_preserving_mapping_is_exact():
    rng = np.random.default_rng(1)
    ang = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    pos = np.column_stack([8.0 + 4 * np.cos(ang), 8.0 + 4 * np.sin(ang)])
    frc = rng.standard_normal((64, 2)) * 1e-3
    sl = itf.surface_load(pos, frc, np.array([8.0, 8.0]))
    # the attachment patch must span the moment arm (two node rows), exactly as
    # the FE free end does; a single row cannot transmit a moment
    nodes = np.column_stack([np.r_[np.full(4, 30.0), np.full(4, 32.0)],
                             np.linspace(-2, 2, 8)])
    nodal = itf.map_to_nodes(sl, nodes)
    aud = itf.audit_transfer(sl, nodes, nodal)
    assert aud["force_rel_error"] < 1e-12
    assert aud["moment_rel_error"] < 1e-12
    # the uniform (force-only) variant keeps the force and drops the moment
    uniform = np.full(8, sl.force[1] / 8)
    aud2 = itf.audit_transfer(sl, nodes, uniform)
    assert aud2["force_rel_error"] < 1e-12
    assert abs(uniform.sum() - sl.force[1]) < 1e-15


def test_pressure_friction_split_reproduces_resultant():
    ang = np.linspace(0, 2 * np.pi, 48, endpoint=False)
    pos = np.column_stack([5.0 + 2 * np.cos(ang), 5.0 + 2 * np.sin(ang)])
    n = np.column_stack([np.cos(ang), np.sin(ang)])
    frc = -0.01 * n + 0.002 * np.column_stack([-np.sin(ang), np.cos(ang)])
    sl = itf.surface_load(pos, frc, np.array([5.0, 5.0]))
    comp = itf.traction_components(sl)
    assert np.allclose(comp["F_total"], sl.force, atol=1e-15)


def test_strouhal_estimator_recovers_known_frequency():
    t = np.arange(0, 20000, 4.0)
    D, U, st_true = 12.0, 0.1, 0.17
    sig = np.sin(2 * np.pi * st_true * U / D * t) + 0.2 * np.sin(4 * np.pi * st_true * U / D * t)
    st = cp.strouhal_from_history(t, sig, D, U)
    assert abs(st["st"] - st_true) / st_true < 0.02
    assert abs(st["st_zero"] - st_true) / st_true < 0.02


# -------------------------------------------------------------- coupled system
@pytest.mark.parametrize("scheme", ["impulse", "lagged"])
def test_coupled_smoke_conserves_exchange(scheme):
    """Short coupled run: the exchange defects must be at round-off / small."""
    cfg = cp.FSIConfig(diameter=6, n_periods=2.0, ramp_periods=1.0, n_sub=4,
                       fem_substeps=4, scheme=scheme)
    res = cp.run_case(cfg)
    assert res.momentum_error < (1e-8 if scheme == "impulse" else 1.0)
    assert res.energy_exchange_error < (1e-8 if scheme == "impulse" else 1.0)
    assert res.mass_error < 1e-3
    assert res.momentum_error_mea_budget < 1e-9
    assert np.isfinite(res.strouhal)


def test_exchange_ledgers_are_reproducible():
    cfg = cp.FSIConfig(diameter=6, n_periods=1.5, ramp_periods=1.0, n_sub=8,
                       fem_substeps=4)
    a = cp.run_case(cfg)
    b = cp.run_case(cfg)
    assert abs(a.momentum_error - b.momentum_error) < 1e-6
    assert a.ledgers["impulse_lattice"] == pytest.approx(b.ledgers["impulse_lattice"],
                                                         rel=1e-12)