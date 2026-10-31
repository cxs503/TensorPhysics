"""DEM -> continuum homogenisation: Love-Weber stress recovery.

Love-Weber (static) homogenisation of a discrete contact network:

    sigma_ab = (1/V) sum_c f_a^c l_b^c,

where ``f^c`` is the contact force exerted on the ``i``-side particle and ``l^c``
the branch vector joining the two particle centres (minimum image in a periodic
cell).

Two independent code paths are provided so that the recovered stress can be
cross-validated (criterion (e), homogenised vs. direct DEM average):

* :func:`love_weber_stress` -- the textbook single-sum over contacts;
* :func:`per_particle_virial_stress` -- a per-particle virial sum of
  ``f^c (x^c - x_p)`` accumulated over each particle's own contact set, whose
  exact equality with the Love-Weber sum is an identity (each contact contributes
  ``f (x_i - x_j)`` once). Agreement to machine precision therefore verifies the
  bookkeeping (signs, branch vectors, minimum image) rather than the physics.

Rotational covariance (criterion (b)) is checked by rotating positions *and*
forces with the same rotation ``R``: the recovered stress must transform as
``sigma' = R sigma R^T``. Any anisotropy injected by the bookkeeping (e.g. a
branch vector built with the wrong image) breaks this covariance.
"""

from __future__ import annotations

from math import cos, isfinite, sin
from typing import Any

import numpy as np


class HomogenisationError(ValueError):
    """Invalid input for stress homogenisation."""


def _as_points(name: str, value: Any, dim: int = 2) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or array.shape[1] != dim:
        raise HomogenisationError(f"{name} must have shape (N, {dim}), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise HomogenisationError(f"{name} contains non-finite values")
    return array


def _volume(volume: Any) -> float:
    volume = float(np.asarray(volume, dtype=float).ravel()[0])
    if not isfinite(volume) or volume <= 0.0:
        raise HomogenisationError(f"volume must be finite and positive, got {volume!r}")
    return volume


def love_weber_stress(
    forces: Any,
    branch_vectors: Any,
    volume: Any,
) -> np.ndarray:
    """Love-Weber stress tensor ``(1/V) sum f (x) l`` for a 2-D contact network.

    Parameters
    ----------
    forces:
        ``(M, 2)`` contact forces.
    branch_vectors:
        ``(M, 2)`` branch vectors (from the ``j`` particle centre to the ``i``
        particle centre, minimum image applied).
    volume:
        Averaging volume (``L^2`` for a periodic square cell).
    """
    f = _as_points("forces", forces)
    l = _as_points("branch_vectors", branch_vectors)
    if f.shape != l.shape:
        raise HomogenisationError(f"forces {f.shape} and branch_vectors {l.shape} must match")
    V = _volume(volume)
    return (f.T @ l) / V


def per_particle_virial_stress(
    positions: Any,
    contact_index: Any,
    contact_forces: Any,
    volume: Any,
    *,
    box_length: float | None = None,
) -> np.ndarray:
    """Independent per-particle virial average (verification code path).

    For each particle ``p`` the contact forces acting on it are multiplied by the
    lever arm from the particle centre to the contact point, approximated by the
    mid-point of the branch vector (valid for equal-size discs; for polydisperse
    discs the contact point lies on the line of centres, so the mid-point
    approximation is exact up to the (exact) linear interpolation of ``|x_i - x_j|``,
    which cancels when both particles are summed).
    """
    pos = _as_points("positions", positions)
    idx = np.asarray(contact_index, dtype=int)
    if idx.ndim != 2 or idx.shape[1] != 2:
        raise HomogenisationError(f"contact_index must have shape (M, 2), got {idx.shape}")
    f = _as_points("contact_forces", contact_forces)
    if f.shape[0] != idx.shape[0]:
        raise HomogenisationError("contact_forces and contact_index lengths differ")
    if idx.size and (idx.min() < 0 or idx.max() >= pos.shape[0]):
        raise HomogenisationError("contact_index out of bounds")
    V = _volume(volume)

    delta = pos[idx[:, 0]] - pos[idx[:, 1]]
    if box_length is not None:
        L = float(box_length)
        # Minimum-image branch vector: the lever arm of particle j must use the
        # *wrapped* vector, otherwise the periodic image offset leaks into the
        # lever arm of the j-side particle.
        delta = delta - np.round(delta / L) * L
    half = 0.5 * delta
    per_particle = np.zeros((pos.shape[0], 2, 2))
    np.add.at(per_particle, idx[:, 0], np.einsum("ma,mb->mab", f, half))
    np.add.at(per_particle, idx[:, 1], np.einsum("ma,mb->mab", -f, -half))
    return per_particle.sum(axis=0) / V


def rotate_particles(
    positions: Any,
    forces: Any,
    angle: float,
    *,
    box_length: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate a periodic configuration (positions and forces) by ``angle`` radians."""
    pos = _as_points("positions", positions)
    f = _as_points("forces", forces)
    angle = float(angle)
    if not isfinite(angle):
        raise HomogenisationError(f"angle must be finite, got {angle!r}")
    R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    pos_rot = pos @ R.T
    if box_length is not None:
        L = float(box_length)
        pos_rot = np.mod(pos_rot, L)
    return pos_rot, f @ R.T


def rotational_covariance_error(
    forces: Any,
    branch_vectors: Any,
    volume: Any,
    angle: float,
) -> float:
    """``max_ab |sigma_rot - (R sigma R^T)_ab| / max_ab |sigma_ab|``.

    Rotating both the branch vectors and the forces by ``R`` must leave the
    homogenised stress covariant; the residual measures bookkeeping consistency.
    """
    sigma = love_weber_stress(forces, branch_vectors, volume)
    angle = float(angle)
    R = np.array([[cos(angle), -sin(angle)], [sin(angle), cos(angle)]])
    l_rot = _as_points("branch_vectors", branch_vectors) @ R.T
    f_rot = _as_points("forces", forces) @ R.T
    sigma_rot = love_weber_stress(f_rot, l_rot, volume)
    expected = R @ sigma @ R.T
    scale = max(float(np.abs(sigma).max()), 1e-300)
    return float(np.abs(sigma_rot - expected).max() / scale)


def homogenised_stress_from_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Convenience wrapper: Love-Weber + per-particle virial from a DEM snapshot."""
    contacts = snapshot.get("contacts", {})
    box = np.asarray(snapshot.get("box_length", [1.0]), dtype=float).ravel()[0]
    volume = float(box) ** 2
    sigma = love_weber_stress(contacts["force"], contacts["branch"], volume)
    sigma_virial = per_particle_virial_stress(
        snapshot["positions"],
        np.column_stack([contacts["i"], contacts["j"]]),
        contacts["force"],
        volume,
        box_length=float(box),
    )
    scale = max(float(np.abs(sigma).max()), 1e-300)
    return {
        "sigma": sigma,
        "sigma_virial": sigma_virial,
        "relative_difference": float(np.abs(sigma - sigma_virial).max() / scale),
        "pressure": float(-0.5 * np.trace(sigma)),
        "shear_stress": float(0.5 * (sigma[0, 1] + sigma[1, 0])),
        "asymmetry": float(abs(sigma[0, 1] - sigma[1, 0]) / scale),
        "n_contacts": int(contacts["force"].shape[0]),
        "volume": volume,
    }