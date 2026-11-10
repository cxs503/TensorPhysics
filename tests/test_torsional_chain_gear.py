"""C2 - periodic gear mesh stiffness, backlash and the parametric response."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.torsional_chain.acceptance import c2_gear_parametric
from tensorphysics.torsional_chain.gear import (GearPair, MeshStiffness,
                                                ParametricMeshOscillator,
                                                mathieu_growth_rate,
                                                mathieu_instability_halfwidth)


@pytest.fixture(scope="module")
def c2():
    return c2_gear_parametric()


def test_mesh_stiffness_is_periodic_in_pinion_angle():
    mesh = MeshStiffness(k0=1.0e9, modulation=0.25, teeth=20)
    theta = 0.37
    period = 2.0 * math.pi / 20
    assert mesh.stiffness(theta) == pytest.approx(mesh.stiffness(theta + period), rel=1e-12)
    assert mesh.stiffness(0.0) == pytest.approx(1.25e9, rel=1e-12)
    assert mesh.stiffness(period / 2.0) == pytest.approx(0.75e9, rel=1e-12)
    # mean over a full mesh period equals k0
    grid = np.linspace(0.0, period, 4001)[:-1]
    assert float(np.mean(mesh.stiffness(grid))) == pytest.approx(1.0e9, rel=1e-3)


def test_mesh_stiffness_rejects_invalid_modulation():
    with pytest.raises(ValueError):
        MeshStiffness(1.0e9, modulation=1.0, teeth=20)
    with pytest.raises(ValueError):
        MeshStiffness(-1.0, modulation=0.1, teeth=20)
    with pytest.raises(ValueError):
        MeshStiffness(1.0e9, modulation=0.1, teeth=2)


def test_gear_conjugate_torques_do_no_net_work():
    """Power in through the pinion equals power out at the wheel (ideal mesh)."""
    mesh = MeshStiffness(1.0e9, 0.2, 21)
    gear = GearPair("g", 0, 1, ratio=4.0, pitch_radius=0.105, mesh=mesh)
    theta_p, theta_w = 3.0e-3, 3.0e-3 / 4.0 + 1.0e-4
    tp, tw = gear.torques(theta_p, theta_w)
    assert tp * 1.0 + tw * 0.25 == pytest.approx(0.0, abs=1e-6 * max(abs(tp), 1.0))


def test_gear_torque_ratio_equals_transmission_ratio():
    """Conjugate torques: ``T_wheel = -i * T_pinion`` (equal power, opposite signs)."""
    mesh = MeshStiffness(1.0e13, 0.0, 21)
    gear = GearPair("g", 0, 1, ratio=4.0, pitch_radius=0.105, mesh=mesh)
    tp, tw = gear.torques(0.02, 0.0)
    assert abs(tw + 4.0 * tp) < 1e-9 * max(abs(tp), 1.0)
    # and the transmitted power vanishes on the ideal mesh
    assert abs(tp * 1.0 + tw * 0.25) < 1e-6 * max(abs(tp), 1.0)


def test_backlash_dead_zone():
    mesh = MeshStiffness(1.0e9, 0.0, 21)
    small = GearPair("g", 0, 1, 4.0, 0.105, mesh, backlash=1.0e-3)
    free = GearPair("g", 0, 1, 4.0, 0.105, mesh, backlash=0.0)
    theta_p = 0.4e-3 / 0.105  # tooth-line deflection 0.4 mm, below half the 1 mm play
    assert small.mesh_force(theta_p, 0.0) == 0.0
    assert free.mesh_force(theta_p, 0.0) > 0.0
    theta_p2 = 1.2e-3 / 0.105
    assert small.mesh_force(theta_p2, 0.0) == pytest.approx(
        free.mesh_force(theta_p2 - 0.5e-3 / 0.105, 0.0), rel=1e-9)


def test_quasistatic_amplitude_matches_closed_form(c2):
    assert c2["amplitude_relative_error"] <= 0.05
    expected = 0.5 * (2.0e4 / (4.0e7 * 0.8) + 2.0e4 / (4.0e7 * 1.2))
    assert c2["amplitude_theory"] == pytest.approx(expected, rel=1e-12)


def test_peak_position_inside_mesh_cycle(c2):
    assert c2["peak_position_error_fraction"] <= 0.05
    # the displacement peaks where the mesh stiffness is minimum: half a mesh period
    assert c2["peak_phase_fraction_sim"] == pytest.approx(0.5, abs=0.05)


def test_mathieu_growth_rate_matches_two_timing_formula(c2):
    assert c2["mathieu_growth_error"] <= 0.10
    sigma = mathieu_growth_rate(2.0 * math.pi * 30.0, 0.2, 0.005)
    assert sigma == pytest.approx(0.05 * 2.0 * math.pi * 30.0 - 0.005 * 2.0 * math.pi * 30.0,
                                 rel=1e-12)


def test_mathieu_instability_halfwidth():
    assert mathieu_instability_halfwidth(0.2) == pytest.approx(0.05)
    assert mathieu_growth_rate(100.0, 0.2, 0.2) < 0.0  # heavily damped -> stable


def test_parametric_oscillator_rejects_bad_input():
    mesh = MeshStiffness(1.0e7, 0.1, 20)
    with pytest.raises(ValueError):
        ParametricMeshOscillator(-1.0, 0.0, mesh, 1.0)
    with pytest.raises(ValueError):
        ParametricMeshOscillator(1.0, -1.0, mesh, 1.0)
    with pytest.raises(ValueError):
        ParametricMeshOscillator(1.0, 0.0, mesh, 0.0)


def test_oscillator_off_resonance_tracks_quasistatic_response():
    """At mesh frequency = 2% of the natural frequency inertia is negligible."""
    mesh = MeshStiffness(4.0e7, 0.2, 21)
    omega_n = math.sqrt(4.0e7 / 40.0)
    osc = ParametricMeshOscillator(40.0, 2.0 * 0.02 * math.sqrt(4.0e7 * 40.0), mesh,
                                   omega_mean=0.02 * omega_n / 21)
    mesh_omega = osc.mesh_frequency
    hist = osc.integrate(4.0 * 2.0 * math.pi / mesh_omega,
                         (2.0 * math.pi / omega_n) / 200.0,
                         lambda t: 2.0e4 * math.cos(mesh_omega * t))
    window = hist["t"] >= 0.75 * hist["t"][-1]
    qs = 2.0e4 * np.cos(mesh_omega * hist["t"][window]) / hist["stiffness"][window]
    assert np.max(np.abs(hist["phi"][window] - qs)) < 0.02 * np.max(np.abs(qs))


def test_c2_acceptance_block(c2):
    assert c2["criterion"] == "C2"
    assert c2["pass"] is True
    assert c2["amplitude_relative_error"] <= 0.05
    assert c2["peak_position_error_fraction"] <= 0.05