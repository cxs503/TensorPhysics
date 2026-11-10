"""Lumped inertia / torsional-spring drivetrain chain with a gear pair.

Topology (the one required by the node)
---------------------------------------
    propeller -- shaft1 -- pinion ==mesh== wheel -- shaft2 -- motor

Each shaft is a torsional spring ``k`` with viscous damping ``c``; the gear pair
carries the tooth-passing periodic mesh stiffness described in
:mod:`tensorphysics.torsional_chain.gear`.  Nodes are pure inertias, so the mass
matrix is diagonal and the equations are

    ``J_i domega_i/dt = Q_i(t) - sum_springs T_ij - sum_gears T_i^mesh``
    ``dtheta_i/dt = omega_i``

The mesh contribution is evaluated from the tooth-line force (not from a
pre-assembled stiffness matrix) so that the optional backlash dead zone is a
genuine nonlinearity rather than a matrix approximation.

Stress recovery: for a shaft segment of outer/inner diameter ``(D_o, D_i)`` the
polar section modulus is ``W_p = pi (D_o**4 - D_i**4) / (16 D_o)`` and the outer
fibre shear stress is ``tau = |T| / W_p``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from .gear import GearPair, MeshStiffness

__all__ = [
    "InertiaNode",
    "ShaftSegment",
    "DrivetrainChain",
    "uniform_segment_stiffness",
    "polar_section_modulus",
    "standard_chain",
]


def polar_section_modulus(outer_diameter: float, inner_diameter: float = 0.0) -> float:
    """``W_p = pi (D_o**4 - D_i**4) / (16 D_o)`` [m**3]."""
    if not (outer_diameter > inner_diameter >= 0.0):
        raise ValueError("require outer_diameter > inner_diameter >= 0")
    return math.pi * (outer_diameter ** 4 - inner_diameter ** 4) / (16.0 * outer_diameter)


def uniform_segment_stiffness(length: float, outer_diameter: float,
                              inner_diameter: float, shear_modulus: float) -> float:
    """Torsional stiffness of a uniform shaft segment ``k = G J_p / L`` [N*m/rad]."""
    if not (length > 0.0):
        raise ValueError("segment length must be positive")
    if not (shear_modulus > 0.0):
        raise ValueError("shear modulus must be positive")
    jp = math.pi * (outer_diameter ** 4 - inner_diameter ** 4) / 32.0
    return shear_modulus * jp / length


@dataclass
class InertiaNode:
    """A lumped rotary inertia [kg*m**2]."""

    name: str
    inertia: float
    fixed: bool = False

    def __post_init__(self) -> None:
        if not (self.inertia > 0.0):
            raise ValueError(f"node {self.name!r}: inertia must be positive")


@dataclass
class ShaftSegment:
    """A torsional spring-damper between two nodes, with section geometry for stress."""

    name: str
    stiffness: float
    damping: float
    left: int
    right: int
    outer_diameter: float
    inner_diameter: float = 0.0

    def __post_init__(self) -> None:
        if not (self.stiffness > 0.0):
            raise ValueError(f"segment {self.name!r}: stiffness must be positive")
        if self.damping < 0.0:
            raise ValueError(f"segment {self.name!r}: damping cannot be negative")

    @property
    def section_modulus(self) -> float:
        return polar_section_modulus(self.outer_diameter, self.inner_diameter)

    def torque(self, theta: Sequence[float], omega: Sequence[float]) -> float:
        """Internal torque, positive when the right node leads the left one [N*m]."""
        return (self.stiffness * (theta[self.right] - theta[self.left])
                + self.damping * (omega[self.right] - omega[self.left]))

    def stress(self, theta: Sequence[float], omega: Sequence[float]) -> float:
        """Outer-fibre shear stress magnitude [Pa]."""
        return abs(self.torque(theta, omega)) / self.section_modulus

    def stress_amplitude(self, theta_hist: np.ndarray, omega_hist: np.ndarray,
                         fraction_from_start: float = 0.5) -> float:
        """Half peak-to-peak stress after discarding the transient."""
        n = theta_hist.shape[0]
        start = int(max(1, fraction_from_start * n))
        tau = [self.stress(theta_hist[k], omega_hist[k]) for k in range(start, n)]
        tau = np.asarray(tau)
        return 0.5 * (tau.max() - tau.min())


class DrivetrainChain:
    """Lumped torsional chain with optional gear pair(s)."""

    def __init__(self) -> None:
        self.nodes: List[InertiaNode] = []
        self.shafts: List[ShaftSegment] = []
        self.gears: List[GearPair] = []

    # ------------------------------------------------------------------ build
    def add_node(self, name: str, inertia: float, fixed: bool = False) -> int:
        self.nodes.append(InertiaNode(name, inertia, fixed))
        return len(self.nodes) - 1

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def _check_pair(self, left: int, right: int) -> None:
        for idx in (left, right):
            if not (0 <= idx < self.n_nodes):
                raise IndexError(f"node index {idx} outside chain of {self.n_nodes} nodes")
        if left == right:
            raise ValueError("a torsional element cannot connect a node to itself")

    def add_shaft(self, segment: ShaftSegment) -> ShaftSegment:
        self._check_pair(segment.left, segment.right)
        self.shafts.append(segment)
        return segment

    def add_uniform_shaft(self, name: str, length: float, outer_diameter: float,
                          inner_diameter: float, shear_modulus: float, left: int,
                          right: int, damping_ratio: float = 0.0,
                          reference_inertia: Optional[float] = None) -> ShaftSegment:
        """Add a uniform shaft segment; damping from ``zeta`` and a reference inertia."""
        k = uniform_segment_stiffness(length, outer_diameter, inner_diameter, shear_modulus)
        c = 0.0
        if damping_ratio > 0.0:
            ref = reference_inertia
            if ref is None:
                ref = min(self.nodes[left].inertia, self.nodes[right].inertia)
            c = 2.0 * damping_ratio * math.sqrt(k * ref)
        segment = ShaftSegment(name, k, c, left, right, outer_diameter, inner_diameter)
        return self.add_shaft(segment)

    def add_gear(self, name: str, pinion: int, wheel: int, ratio: float,
                 pitch_radius: float, mesh: MeshStiffness,
                 backlash: float = 0.0) -> GearPair:
        self._check_pair(pinion, wheel)
        gear = GearPair(name, pinion, wheel, ratio, pitch_radius, mesh, backlash)
        self.gears.append(gear)
        return gear

    # -------------------------------------------------------------- matrices
    @property
    def free_dofs(self) -> np.ndarray:
        return np.array([k for k, n in enumerate(self.nodes) if not n.fixed], dtype=int)

    @property
    def inertias(self) -> np.ndarray:
        return np.array([n.inertia for n in self.nodes], dtype=float)

    def matrices(self, theta: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        """Full ``M`` (diagonal), ``C`` and ``K`` matrices.

        ``theta=None`` uses the *mean* mesh stiffness (the modulation is dropped),
        which is the correct operator for a linear modal analysis.
        """
        n = self.n_nodes
        m = np.diag(self.inertias)
        c = np.zeros((n, n))
        k = np.zeros((n, n))
        for s in self.shafts:
            for (a, b, sign) in ((s.left, s.left, 1.0), (s.right, s.right, 1.0),
                                 (s.left, s.right, -1.0), (s.right, s.left, -1.0)):
                c[a, b] += sign * s.damping
                k[a, b] += sign * s.stiffness
        for g in self.gears:
            if theta is None:
                kp = g.mesh.k0 * g.pitch_radius ** 2
            else:
                kp = g.torsional_stiffness(float(theta[g.pinion]))
            r = g.ratio
            k[g.pinion, g.pinion] += kp
            k[g.wheel, g.wheel] += r * r * kp
            k[g.pinion, g.wheel] -= r * kp
            k[g.wheel, g.pinion] -= r * kp
        return {"M": m, "C": c, "K": k}

    def reduced(self, theta: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        """Free-DOF slice of the matrices (fixed nodes are grounded)."""
        mats = self.matrices(theta)
        f = self.free_dofs
        return {"M": mats["M"][np.ix_(f, f)], "C": mats["C"][np.ix_(f, f)],
                "K": mats["K"][np.ix_(f, f)], "free": f}

    def modal_frequencies(self, n_modes: Optional[int] = None) -> np.ndarray:
        """Natural frequencies [Hz] from ``det(K - omega**2 M) = 0`` (mean mesh)."""
        red = self.reduced()
        m_inv_sqrt = 1.0 / np.sqrt(np.diag(red["M"]))
        a = (red["K"] * np.outer(m_inv_sqrt, m_inv_sqrt))
        a = 0.5 * (a + a.T)
        lam = np.linalg.eigvalsh(a)
        lam = np.sort(lam)
        lam = lam[lam > max(1e-12 * lam.max(), 1e-9)]
        freqs = np.sqrt(lam) / (2.0 * math.pi)
        return freqs if n_modes is None else freqs[:n_modes]

    def modal_omega(self) -> np.ndarray:
        return 2.0 * math.pi * self.modal_frequencies()

    # ------------------------------------------------------------- dynamics
    def internal_torque(self, theta: np.ndarray, omega: np.ndarray) -> np.ndarray:
        """Sum of spring and mesh torques acting on every node [N*m].

        Sign convention: a positive internal torque accelerates the node, i.e. the
        equation of motion is ``J domega/dt = Q_external + internal_torque`` with
        the internal part restoring (``+k (theta_right - theta_left)`` on the left
        node).  The acceptance tests check the restoring sign through the modal
        frequencies and the energy balance.
        """
        q = np.zeros(self.n_nodes)
        for s in self.shafts:
            t = s.torque(theta, omega)
            q[s.left] += t
            q[s.right] -= t
        for g in self.gears:
            tp, tw = g.torques(float(theta[g.pinion]), float(theta[g.wheel]))
            q[g.pinion] += tp
            q[g.wheel] += tw
        return q

    def rhs(self, t: float, y: np.ndarray,
            torque: Callable[[float, np.ndarray, np.ndarray], np.ndarray]) -> np.ndarray:
        """State derivative for ``y = [theta_free, omega_free]``."""
        f = self.free_dofs
        nf = f.size
        theta = np.zeros(self.n_nodes)
        omega = np.zeros(self.n_nodes)
        theta[f] = y[:nf]
        omega[f] = y[nf:]
        q = np.asarray(torque(t, theta, omega), dtype=float) + self.internal_torque(theta, omega)
        alpha = np.zeros(self.n_nodes)
        alpha[f] = q[f] / self.inertias[f]
        return np.concatenate([omega[f], alpha[f]])

    def integrate(self, t_end: float, dt: float,
                  torque: Callable[[float, np.ndarray, np.ndarray], np.ndarray],
                  theta0: Optional[Sequence[float]] = None,
                  omega0: Optional[Sequence[float]] = None) -> Dict[str, np.ndarray]:
        """Classical RK4 integration; returns full-node ``theta``/``omega`` histories."""
        n_steps = int(round(t_end / dt))
        if n_steps < 1:
            raise ValueError("t_end must be at least one step")
        f = self.free_dofs
        nf = f.size
        y = np.zeros(2 * nf)
        if theta0 is not None:
            y[:nf] = np.asarray(theta0, dtype=float)[f]
        if omega0 is not None:
            y[nf:] = np.asarray(omega0, dtype=float)[f]
        theta_hist = np.zeros((n_steps + 1, self.n_nodes))
        omega_hist = np.zeros((n_steps + 1, self.n_nodes))
        theta_hist[0, f] = y[:nf]
        omega_hist[0, f] = y[nf:]
        t_hist = np.arange(n_steps + 1) * dt
        for step in range(n_steps):
            t = t_hist[step]
            k1 = self.rhs(t, y, torque)
            k2 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k1, torque)
            k3 = self.rhs(t + 0.5 * dt, y + 0.5 * dt * k2, torque)
            k4 = self.rhs(t + dt, y + dt * k3, torque)
            y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            theta_hist[step + 1, f] = y[:nf]
            omega_hist[step + 1, f] = y[nf:]
        return {"t": t_hist, "theta": theta_hist, "omega": omega_hist}

    def torque_history(self, segment_index: int, history: Dict[str, np.ndarray]) -> np.ndarray:
        """Internal torque of one shaft segment over a stored history [N*m]."""
        s = self.shafts[segment_index]
        return (s.stiffness * (history["theta"][:, s.right] - history["theta"][:, s.left])
                + s.damping * (history["omega"][:, s.right] - history["omega"][:, s.left]))

    def stress_history(self, segment_index: int, history: Dict[str, np.ndarray]) -> np.ndarray:
        """Outer-fibre shear stress magnitude of one segment [Pa]."""
        return np.abs(self.torque_history(segment_index, history)) / self.shafts[segment_index].section_modulus


def standard_chain(propeller_inertia: float = 1.2e3, motor_inertia: float = 8.0e2,
                   shaft_length: float = 6.0, outer_diameter: float = 0.42,
                   inner_diameter: float = 0.18, shear_modulus: float = 79.3e9,
                   pinion_teeth: int = 21, wheel_teeth: int = 84,
                   pitch_radius: float = 0.105, mesh_k0: float = 1.0e9,
                   mesh_modulation: float = 0.2, damping_ratio: float = 0.02,
                   gear_stiffness: float = 5.0e7) -> DrivetrainChain:
    """Propeller - shaft1 - wheel ==mesh== pinion - shaft2 - motor reference chain.

    The propeller sits on the low-speed (wheel) side and the motor on the
    high-speed (pinion) side, so a large ``gear_stiffness`` (pinion-to-motor
    coupling) makes the motor rotate with the pinion and demonstrates the
    reflected-inertia law ``J_motor * i**2`` referred to the propeller shaft.
    """
    chain = DrivetrainChain()
    prop = chain.add_node("propeller", propeller_inertia)
    wheel = chain.add_node("wheel", 0.05 * propeller_inertia)
    pinion = chain.add_node("pinion", 0.08 * propeller_inertia)
    motor = chain.add_node("motor", motor_inertia)
    chain.add_uniform_shaft("shaft1", shaft_length, outer_diameter, inner_diameter,
                            shear_modulus, prop, wheel, damping_ratio, propeller_inertia)
    chain.add_gear("reduction_gear", pinion, wheel, wheel_teeth / pinion_teeth,
                   pitch_radius, MeshStiffness(mesh_k0, mesh_modulation, pinion_teeth))
    chain.add_shaft(ShaftSegment("shaft2", gear_stiffness, 0.0, pinion, motor,
                                 outer_diameter, inner_diameter))
    return chain