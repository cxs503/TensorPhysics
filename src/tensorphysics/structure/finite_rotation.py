"""Planar finite-rotation kinematics for hull roll and co-rotational beam elements.

Reference frame
---------------
The transverse hull plane is spanned by ``(y, z)``: ``y`` is positive to starboard and
``z`` positive upwards, while ``x`` (the longitudinal axis, positive forward) is
orthogonal to the plane.  A *roll* is therefore a rotation **inside** the transverse
plane, i.e. an in-plane rotation of angle ``phi`` about the ship centreline, and the
corresponding rotation tensor is

    R(phi) = [[cos phi, -sin phi],
              [sin phi,  cos phi]]

applied to a vector stored as ``(y, z)``.  In 3-D ship coordinates this is a rotation
about the longitudinal axis, hence the naming ``roll_transform``.

All angles are in radians.  Rotations are kept as a single angle on the universal cover
of SO(2) (not reduced modulo 2*pi) so that large rotations can be incremented without
branch jumps; use :func:`wrap_angle` whenever an angle difference is compared.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = [
    "Rotation2D",
    "roll_transform",
    "wrap_angle",
    "chord_angle",
    "objective_local_rotation",
    "polar_decomposition_2x2",
    "section_rotation_from_displacements",
    "angular_velocity_field",
]


def wrap_angle(angle: float) -> float:
    """Wrap ``angle`` (radians) to the interval ``(-pi, pi]``."""
    wrapped = math.atan2(math.sin(angle), math.cos(angle))
    return float(wrapped)


@dataclass(frozen=True)
class Rotation2D:
    """A planar rotation by ``angle`` radians (kept un-wrapped)."""

    angle: float

    @property
    def matrix(self) -> np.ndarray:
        c, s = math.cos(self.angle), math.sin(self.angle)
        return np.array([[c, -s], [s, c]], dtype=float)

    def apply(self, points: np.ndarray) -> np.ndarray:
        """Rotate ``points`` of shape ``(..., 2)`` about the origin."""
        pts = np.asarray(points, dtype=float)
        return pts @ self.matrix.T

    def compose(self, other: "Rotation2D") -> "Rotation2D":
        """Return the rotation equivalent to ``self`` applied after ``other``."""
        return Rotation2D(self.angle + other.angle)

    def inverse(self) -> "Rotation2D":
        return Rotation2D(-self.angle)

    def rotate_about(self, points: np.ndarray, centre: np.ndarray) -> np.ndarray:
        """Rotate ``points`` about ``centre`` (finite rotation of a hull section)."""
        c = np.asarray(centre, dtype=float)
        return self.apply(np.asarray(points, dtype=float) - c) + c


def roll_transform(phi: float) -> np.ndarray:
    """Return the 2x2 roll tensor ``R(phi)`` acting on ``(y, z)`` components."""
    return Rotation2D(float(phi)).matrix


def chord_angle(p_i: np.ndarray, p_j: np.ndarray) -> float:
    """Angle of the chord ``p_i -> p_j`` measured from the +y axis."""
    d = np.asarray(p_j, dtype=float) - np.asarray(p_i, dtype=float)
    return float(math.atan2(d[1], d[0]))


def objective_local_rotation(theta_i: float, theta_i0: float,
                             beta: float, beta0: float) -> float:
    """Co-rotational (objective) local rotation of a node.

    ``theta_i`` is the current absolute nodal rotation and ``theta_i0`` its value in
    the reference configuration; ``beta`` / ``beta0`` are the current / reference chord
    angles of the element.  Subtracting the chord rotation removes the rigid-body part
    of the motion, which is exactly what keeps the strain measure frame indifferent: a
    pure rigid rotation of the element yields ``theta_local == 0``.
    """
    return wrap_angle((theta_i - theta_i0) - (beta - beta0))


def polar_decomposition_2x2(F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Polar decomposition ``F = R @ U`` of a 2x2 tensor with ``det(F) > 0``.

    The rotation angle of the rotation factor has the closed form

        theta = atan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])

    which is used here as an independent route (different from the co-rotational
    chord-angle update) to extract the finite rotation carried by a deformation
    gradient.  Raises ``ValueError`` for a degenerate / orientation-reversing tensor.
    """
    F = np.asarray(F, dtype=float)
    if F.shape != (2, 2):
        raise ValueError("F must be 2x2")
    det = float(np.linalg.det(F))
    if det <= 1e-14:
        raise ValueError("F must be orientation preserving (det > 0)")
    theta = math.atan2(F[1, 0] - F[0, 1], F[0, 0] + F[1, 1])
    R = Rotation2D(theta).matrix
    U = R.T @ F
    # enforce symmetry of the stretch (removes round-off asymmetry)
    U = 0.5 * (U + U.T)
    return R, U


def section_rotation_from_displacements(reference: np.ndarray, current: np.ndarray
                                        ) -> float:
    """Best-fit finite rotation of a hull section from nodal displacements.

    Uses the planar Kabsch (orthogonal Procrustes) route: with
    ``d_r = reference - centroid(reference)`` and ``d_c = current - centroid(current)``
    the rotation is ``atan2(sum(d_r x d_c), sum(d_r . d_c))``.  This is exact for a
    collinear node set (a beam strip has no lateral spread, so a full 2x2 polar
    decomposition of the covariance would be rank deficient).  Returns radians.  This is the
    kinematical measure that a roll angle is reported with; it is *not* a small-angle
    approximation (15 deg roll is ~26 % away from ``sin phi ~ phi`` in the transverse
    projection, so an arctan-based measure is mandatory).
    """
    r = np.asarray(reference, dtype=float)
    c = np.asarray(current, dtype=float)
    if r.shape != c.shape or r.ndim != 2 or r.shape[1] != 2:
        raise ValueError("reference/current must both be (n, 2)")
    d_r = r - r.mean(axis=0)
    d_c = c - c.mean(axis=0)
    dot = float(np.sum(d_r * d_c))
    cross = float(np.sum(d_r[:, 0] * d_c[:, 1] - d_r[:, 1] * d_c[:, 0]))
    if abs(dot) + abs(cross) <= 0.0:
        raise ValueError("degenerate section: no spread to determine a rotation")
    return float(math.atan2(cross, dot))


def angular_velocity_field(points: np.ndarray, omega: float,
                           centre: np.ndarray | None = None) -> np.ndarray:
    """Velocity field of a rigid rotation with rate ``omega`` (rad/s).

    ``v = omega x r`` in the plane reduces to ``v = omega * (-(z - z_c), (y - y_c))``.
    Used to convert an ice-impact impulse / roll rate into initial nodal velocities
    for the conservative large-rotation transient.
    """
    pts = np.asarray(points, dtype=float)
    c = np.zeros(2) if centre is None else np.asarray(centre, dtype=float)
    rel = pts - c
    v = np.empty_like(rel)
    v[..., 0] = -omega * rel[..., 1]
    v[..., 1] = omega * rel[..., 0]
    return v