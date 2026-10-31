"""Tests for Love-Weber homogenisation and Cauchy stress consistency (W5-N05)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorphysics.ice_wave import dem2d, homogenise


def _cell(n_particles: int = 80, steps: int = 300, shear_rate: float = 0.05) -> dem2d.ShearCell:
    cell = dem2d.ShearCell(n_particles=n_particles, shear_rate=shear_rate, kn=1.0e3, seed=3)
    for _ in range(steps):
        cell.step()
    return cell


def test_single_contact_analytic_stress():
    """One contact across the periodic boundary reproduces f (x) l / V exactly."""
    L = 10.0
    force = np.array([[3.0, -2.0]])
    branch = np.array([[0.5, -9.5]])  # wrapped across the periodic y boundary
    sigma = homogenise.love_weber_stress(force, branch, L * L)
    assert sigma == pytest.approx(np.outer(force[0], branch[0]) / (L * L), rel=1e-15)


def test_coordination_and_pressure_of_a_dense_cell():
    cell = _cell()
    snapshot = cell.snapshot()
    contacts = snapshot["contacts"]
    assert contacts["force"].shape[0] > snapshot["positions"].shape[0]  # Z > 1
    sigma = homogenise.love_weber_stress(
        contacts["force"], contacts["branch"], float(snapshot["box_length"][0]) ** 2
    )
    assert 0.5 * np.trace(sigma) > 0.0  # compressive-positive convention


def test_rotational_covariance_below_1e10():
    """Acceptance criterion (b): sigma(R x) = R sigma R^T to < 1e-10."""
    cell = _cell()
    contacts = cell.snapshot()["contacts"]
    volume = float(cell.L) ** 2
    for angle_deg in (1.0, 17.0, 45.0, 90.0, 137.0, 231.0):
        error = homogenise.rotational_covariance_error(
            contacts["force"], contacts["branch"], volume, np.deg2rad(angle_deg)
        )
        assert error < 1e-10


def test_inconsistent_rotation_breaks_covariance():
    """Negative control: rotating forces and branch vectors by *different* angles
    must break the covariance by O(1), proving the test has discriminating power."""
    cell = _cell()
    contacts = cell.snapshot()["contacts"]
    volume = float(cell.L) ** 2
    angle = np.deg2rad(37.0)
    R1 = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    angle2 = np.deg2rad(52.0)
    R2 = np.array([[np.cos(angle2), -np.sin(angle2)], [np.sin(angle2), np.cos(angle2)]])
    sigma = homogenise.love_weber_stress(contacts["force"], contacts["branch"], volume)
    inconsistent = homogenise.love_weber_stress(
        contacts["force"] @ R1.T, contacts["branch"] @ R2.T, volume
    )
    expected = R1 @ sigma @ R2.T
    rel = np.abs(inconsistent - expected).max() / np.abs(sigma).max()
    assert rel < 1e-12
    mismatch = np.abs(inconsistent - R1 @ sigma @ R1.T).max() / np.abs(sigma).max()
    assert mismatch > 1e-2


def test_homogenised_matches_per_particle_average():
    """Acceptance criterion (e): Love-Weber vs independent per-particle average."""
    cell = _cell()
    snapshot = cell.snapshot()
    contacts = snapshot["contacts"]
    box = float(snapshot["box_length"][0])
    volume = box**2
    sigma = homogenise.love_weber_stress(contacts["force"], contacts["branch"], volume)
    virial = homogenise.per_particle_virial_stress(
        snapshot["positions"],
        np.column_stack([contacts["i"], contacts["j"]]),
        contacts["force"],
        volume,
        box_length=box,
    )
    rel = np.abs(sigma - virial).max() / np.abs(sigma).max()
    assert rel < 0.10
    assert rel < 1e-12  # the two paths are algebraically identical


def test_periodic_wrap_does_not_leak_into_lever_arm():
    """A contact across the periodic boundary must use the wrapped lever arm for
    both particles (regression test for the j-side leaver-arm bug)."""
    L = 8.0
    positions = np.array([[0.2, 0.1], [0.2, 7.9]])
    forces = np.array([[0.0, 4.0]])
    index = np.array([[0, 1]])
    volume = L * L
    sigma = homogenise.per_particle_virial_stress(
        positions, index, forces, volume, box_length=L
    )
    # branch vector = (0, 0.2 - 8.0) = (0, -7.8)
    # positions[0].y = 0.1 and positions[1].y = 7.9 -> wrapped branch dy = 0.2
    expected = np.outer(forces[0], np.array([0.0, 0.1 - 7.9 + 8.0])) / volume
    assert sigma == pytest.approx(expected, rel=1e-15)


def test_homogenised_stress_from_snapshot_consistency():
    cell = _cell()
    result = homogenise.homogenised_stress_from_snapshot(cell.snapshot())
    assert result["relative_difference"] < 1e-12
    assert result["asymmetry"] >= 0.0
    assert result["n_contacts"] > 0


@pytest.mark.parametrize(
    "call",
    [
        lambda: homogenise.love_weber_stress([[1.0, 2.0]], [[1.0, 2.0], [3.0, 4.0]], 1.0),
        lambda: homogenise.love_weber_stress([[1.0, 2.0]], [[1.0, 2.0]], 0.0),
        lambda: homogenise.per_particle_virial_stress(
            [[0.0, 0.0]], [[0, 1]], [[1.0, 1.0]], 1.0
        ),
        lambda: homogenise.love_weber_stress([[np.nan, 1.0]], [[1.0, 1.0]], 1.0),
    ],
)
def test_invalid_inputs_raise(call):
    with pytest.raises(homogenise.HomogenisationError):
        call()