"""Wave field and rigid-box kinematics: the two analytic ingredients of the chain."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.chain_ledger.body import BoxBody, RigidPose, box_panels, rigid_velocity
from tensorphysics.chain_ledger.interfaces import face_a_force
from tensorphysics.chain_ledger.waves import StokesWave, dispersion_wavenumber


@pytest.fixture()
def wave() -> StokesWave:
    return StokesWave(height=1.0, period=8.0, depth=10.0)


def test_dispersion_is_solved_to_machine_precision(wave):
    assert wave.dispersion_residual() < 1e-12
    # monotone in depth: shallower water -> shorter wave for the same period
    shallow = StokesWave(height=1.0, period=8.0, depth=3.0)
    assert shallow.wavelength < wave.wavelength
    assert dispersion_wavenumber(wave.omega, wave.depth) == pytest.approx(wave.wavenumber, rel=1e-14)


def test_second_harmonic_reduces_to_the_deep_water_limit():
    deep = StokesWave(height=0.2, period=6.0, depth=400.0)
    expected = 0.5 * deep.wavenumber * deep.amplitude ** 2      # k a^2 / 2
    assert deep.second_harmonic_amplitude == pytest.approx(expected, rel=1e-3)
    # finite depth always gives a smaller second harmonic
    # finite depth amplifies the second harmonic: the amplification factor is
    # (3 - sigma^2)/(2 sigma^3) and must grow towards shallow water
    factors = []
    for depth in (60.0, 20.0, 8.0):
        w = StokesWave(height=0.2, period=6.0, depth=depth)
        sigma = w.sigma
        factor = w.second_harmonic_amplitude / (0.25 * w.wavenumber * w.amplitude ** 2)
        assert factor == pytest.approx((3.0 - sigma ** 2) / sigma ** 3, rel=1e-12)
        factors.append(factor)
    assert factors == sorted(factors)          # shallower -> stronger
    assert all(f > 2.0 for f in factors)       # 2 is the deep-water limit


def test_free_surface_is_crest_sharpened(wave):
    t = 0.0
    x = np.linspace(0.0, wave.wavelength, 2001)
    eta = wave.eta(x, t)
    assert np.max(eta) > -np.min(eta)
    # to first order the elevation is symmetric about the mean level
    eta1 = wave.eta_first_order(x, t)
    assert np.max(eta1) == pytest.approx(-np.min(eta1), rel=1e-12)


def test_energy_flux_matches_the_depth_integral(wave):
    """``< int p_dyn u dz >`` must equal ``E c_g``.

    The numerical flux carries the deliberate ``-rho |u|^2 / 2`` second-order
    pressure term, so the agreement is at the ``(ka)^2`` level, not to machine
    precision; the ``u = a omega cosh/sinh`` amplitude law is what the check
    really pins down (dropping the ``1/sinh(kh)`` factor breaks it by 40 %).
    """
    e1 = 0.5 * wave.rho * wave.g * wave.amplitude ** 2
    assert wave.numerical_energy_flux(0.0) == pytest.approx(e1 * wave.group_velocity, rel=1e-2)
    assert (wave.steepness ** 2) < 1e-2


def test_surface_pressure_residual_is_first_order_in_steepness(wave):
    """``p(z = eta_1)`` is O(ka) small - and the residual scales like ``ka``.

    Linear theory puts the pressure to zero on the *still-water* level; evaluated
    on the displaced surface the residual is ``rho g k a^2 cos(theta)``, i.e.
    first order in steepness.  Doubling the wave amplitude must double the
    relative residual - that is the documented accuracy limit of the pressure
    field used by interface A.
    """
    def residual_rel(w: StokesWave) -> float:
        x, t = 0.0, 0.0
        eta = float(w.eta_first_order(x, t))
        p = float(w.pressure(x, eta, t))
        u, ww = w.velocity(x, eta, t)
        second = -0.5 * w.rho * float(u ** 2 + ww ** 2)
        return abs(p - second) / (w.rho * w.g * w.amplitude)

    r1 = residual_rel(StokesWave(height=0.2, period=8.0, depth=10.0))
    r2 = residual_rel(StokesWave(height=0.4, period=8.0, depth=10.0))
    assert r2 / r1 == pytest.approx(2.0, rel=0.05)
    assert r1 < 0.02


def test_velocity_satisfies_the_kinematic_free_surface_condition(wave):
    """``w(z=0) = d eta / dt`` to first order."""
    dt = 1e-4
    for t in (0.0, 0.4 * wave.period):
        w = float(wave.velocity(0.0, 0.0, t)[1])
        detadt = float(
            (wave.eta_first_order(0.0, t + dt) - wave.eta_first_order(0.0, t - dt)) / (2 * dt)
        )
        assert w == pytest.approx(detadt, rel=5e-3, abs=1e-3)


def test_panel_areas_and_hydrostatic_restoring():
    body = BoxBody(half_beam=5.0, draught=0.15, width=1.0, mass=1537.5, depth=0.6)
    wetted = body.panels(n_edge=4)
    closed = body.panels(n_edge=4, wetted=False)
    assert wetted.total_area == pytest.approx(
        2 * body.half_beam * body.width + 2 * body.draught * body.width + 4 * body.half_beam * body.draught
    )
    assert closed.total_area > wetted.total_area

    # hydrostatic restoring: dF_z/dz from a numerical pressure integral must
    # equal -rho g A_wp for a small heave step
    def vertical_force(heave: float) -> float:
        pose = RigidPose(r=np.array([0.0, 0.0, heave]))
        panels = body.panels(pose, n_edge=16)
        p = -body.rho * body.g * panels.centroid[:, 2]
        return float(face_a_force(lambda pts: -body.rho * body.g * pts[:, 2], panels, np.zeros(3),
                                  method="gauss", order=3)[0][2])

    dz = 1e-4
    grad = (vertical_force(dz) - vertical_force(-dz)) / (2 * dz)
    assert grad == pytest.approx(-body.rho * body.g * body.waterplane_area, rel=1e-3)


def test_closed_surface_zero_flux_identity_and_its_negative_control():
    body = BoxBody(half_beam=5.0, draught=0.15, width=1.0, mass=1537.5, depth=0.6)
    pose = RigidPose(r=np.array([0.0, 0.0, 0.02]), theta=np.array([0.0, 0.03, 0.0]))
    panels = body.panels(pose, n_edge=16, wetted=False)
    u = rigid_velocity(pose, np.array([0.05, 0.0, 0.3]), np.array([0.0, 0.2, 0.0]), panels.centroid)
    flux = float(np.sum(np.einsum("ij,ij->i", u, panels.normal) * panels.area))
    scale = float(np.max(np.abs(u)) * panels.total_area)
    assert abs(flux) / scale < 1e-12
    # the defect (positions rotated, normals not) must break the identity
    flat = box_panels(body, RigidPose(), n_edge=16, wetted=False).normal
    broken = float(np.sum(np.einsum("ij,ij->i", u, flat) * panels.area))
    assert abs(broken) / scale > 1e-3