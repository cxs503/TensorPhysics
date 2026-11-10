"""Periodic gear-mesh stiffness and the single-mesh parametric oscillator.

Physics
-------
A gear pair is described by the mesh stiffness ``k_m(theta)`` evaluated at the
pinion rotation angle.  The mesh stiffness of a real spur/helical pair pulsates
as the contact ratio changes between one and two tooth pairs, so the first
harmonic of the tooth-passing frequency dominates::

    k_m(theta) = k_m0 * (1 + eps * cos(Z1 * theta + phi0))

with ``Z1`` the pinion tooth count and ``eps = dk_m / k_m0`` the modulation
depth (typically 0.1-0.3 for spur gears).

Referred to the pinion shaft, the mesh is a torsional spring of stiffness
``k_m * r_p**2`` acting on the relative rotation ``theta_p - i * theta_w`` with
``i = Z2 / Z1 = omega_pinion / omega_wheel`` the reduction ratio (``i > 1`` for a
reduction gear).  The wheel side receives the conjugate mesh torque, so the
assembled stiffness matrix is symmetric and, in the rigid-mesh limit, the
high-speed (pinion) side sees ``J_w / i**2`` while the low-speed (wheel) side
sees ``J_p * i**2`` - the classical inertia-reflection law, obtained here without
a coordinate transformation.

Because the stiffness is a function of the *pinion angle* it is a parametric
(coefficient) excitation: the modal equation about a mean rotation
``theta_p = Omega * t`` becomes a Mathieu-type equation

    J phi'' + c phi' + k_m0 (1 + eps cos(Z1 Omega t)) phi = T_ext(t)

whose principal parametric resonance sits at the tooth-passing frequency equal
to twice the natural frequency, ``Z1 * Omega = 2 * omega_n``, with the
first-order growth exponent ``sigma = eps * omega_n / 4 - zeta * omega_n``.
This module provides both closed forms so the numerical response can be
compared against theory instead of against itself.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np

__all__ = [
    "MeshStiffness",
    "GearPair",
    "ParametricMeshOscillator",
    "mathieu_growth_rate",
    "mathieu_instability_halfwidth",
]


@dataclass(frozen=True)
class MeshStiffness:
    """Tooth-passing periodic mesh stiffness ``k_m(theta)`` [N/m]."""

    k0: float
    modulation: float = 0.2
    teeth: int = 20
    phase0: float = 0.0

    def __post_init__(self) -> None:
        if not (self.k0 > 0.0):
            raise ValueError("mean mesh stiffness k0 must be positive")
        if self.teeth < 3:
            raise ValueError("a gear with fewer than 3 teeth has no mesh periodicity")
        if not (0.0 <= self.modulation < 1.0):
            raise ValueError("modulation must satisfy 0 <= eps < 1 (stiffness stays positive)")

    @property
    def period(self) -> float:
        """Mesh period in pinion rotation angle [rad]."""
        return 2.0 * math.pi / self.teeth

    def stiffness(self, theta):
        """Mesh stiffness at pinion angle ``theta`` (scalar or array) [N/m]."""
        th = np.asarray(theta, dtype=float)
        return self.k0 * (1.0 + self.modulation * np.cos(self.teeth * th + self.phase0))


@dataclass(frozen=True)
class GearPair:
    """A gear pair coupling two torsional nodes.

    ``ratio = Z_wheel / Z_pinion``; ``pitch_radius`` is the pinion reference
    radius [m]; ``backlash`` is the optional linear backlash measured at the
    pitch circle [m] (zero = ideal mesh without play).
    """

    name: str
    pinion: int
    wheel: int
    ratio: float
    pitch_radius: float
    mesh: MeshStiffness
    backlash: float = 0.0

    def __post_init__(self) -> None:
        if not (self.ratio > 0.0):
            raise ValueError("gear ratio must be positive")
        if not (self.pitch_radius > 0.0):
            raise ValueError("pitch radius must be positive")
        if self.backlash < 0.0:
            raise ValueError("backlash cannot be negative")

    def torsional_stiffness(self, theta_pinion: float) -> float:
        """Mesh stiffness referred to the pinion shaft [N*m/rad]."""
        return float(self.mesh.stiffness(theta_pinion)) * self.pitch_radius ** 2

    def line_deflection(self, theta_p: float, theta_w: float) -> float:
        """Relative tooth-line deflection ``r_p (theta_p - i theta_w)`` [m]."""
        return self.pitch_radius * (theta_p - self.ratio * theta_w)

    def mesh_force(self, theta_p: float, theta_w: float) -> float:
        """Transmitted mesh force [N] including the optional backlash dead zone."""
        delta = self.line_deflection(theta_p, theta_w)
        if self.backlash > 0.0:
            play = 0.5 * self.backlash
            delta = math.copysign(max(abs(delta) - play, 0.0), delta)
        return self.torsional_stiffness(theta_p) / self.pitch_radius * delta

    def torques(self, theta_p: float, theta_w: float) -> tuple[float, float]:
        """Conjugate mesh torques on the pinion and the wheel [N*m]."""
        force = self.mesh_force(theta_p, theta_w)
        return -force * self.pitch_radius, force * self.ratio * self.pitch_radius


def mathieu_growth_rate(omega_n: float, modulation: float, zeta: float = 0.0) -> float:
    """First-order parametric growth exponent at ``Omega_mesh = 2 omega_n`` [1/s].

    Derived by two-timing the Mathieu equation
    ``phi'' + 2 zeta omega_n phi' + omega_n**2 (1 + eps cos(2 omega_n t)) phi = 0``:
    the envelope eigenvalues are ``-zeta omega_n +- eps omega_n / 4``.
    """
    return 0.25 * modulation * omega_n - zeta * omega_n


def mathieu_instability_halfwidth(modulation: float) -> float:
    """Relative half-width of the principal instability tongue (first order)."""
    return 0.25 * modulation


class ParametricMeshOscillator:
    """Single-mesh torsional oscillator with tooth-passing stiffness variation.

    ``J phi'' + c phi' + k_m(Omega t) phi = T_ext(t)``

    ``phi`` is the *perturbation* rotation about the mean operating point; the
    mean torque that maintains the mean speed is not part of this equation.
    """

    def __init__(self, inertia: float, damping: float, mesh: MeshStiffness,
                 omega_mean: float) -> None:
        if not (inertia > 0.0):
            raise ValueError("inertia must be positive")
        if damping < 0.0:
            raise ValueError("damping cannot be negative")
        if not (omega_mean > 0.0):
            raise ValueError("omega_mean must be positive")
        self.inertia = inertia
        self.damping = damping
        self.mesh = mesh
        self.omega_mean = omega_mean

    @property
    def mesh_frequency(self) -> float:
        """Tooth-passing angular frequency [rad/s]."""
        return self.mesh.teeth * self.omega_mean

    @property
    def natural_frequency(self) -> float:
        return math.sqrt(self.mesh.k0 / self.inertia)

    @property
    def damping_ratio(self) -> float:
        return self.damping / (2.0 * math.sqrt(self.mesh.k0 * self.inertia))

    def stiffness(self, t: float) -> float:
        return float(self.mesh.stiffness(self.omega_mean * t))

    def rhs(self, t: float, y: np.ndarray, torque: Callable[[float], float]) -> np.ndarray:
        phi, dphi = y
        return np.array([dphi,
                         (torque(t) - self.damping * dphi - self.stiffness(t) * phi) / self.inertia])

    def integrate(self, t_end: float, dt: float, torque: Callable[[float], float],
                  phi0: float = 0.0, dphi0: float = 0.0) -> Dict[str, np.ndarray]:
        """Classical RK4 integration; returns ``t``, ``phi``, ``dphi``, ``stiffness``."""
        n_steps = int(round(t_end / dt))
        t = np.arange(n_steps + 1) * dt
        phi = np.zeros(n_steps + 1)
        dphi = np.zeros(n_steps + 1)
        phi[0], dphi[0] = phi0, dphi0
        y = np.array([phi0, dphi0], dtype=float)
        for k in range(n_steps):
            tk = t[k]
            k1 = self.rhs(tk, y, torque)
            k2 = self.rhs(tk + 0.5 * dt, y + 0.5 * dt * k1, torque)
            k3 = self.rhs(tk + 0.5 * dt, y + 0.5 * dt * k2, torque)
            k4 = self.rhs(tk + dt, y + dt * k3, torque)
            y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            phi[k + 1], dphi[k + 1] = y
        return {"t": t, "phi": phi, "dphi": dphi,
                "stiffness": np.asarray(self.mesh.stiffness(self.omega_mean * t), dtype=float)}