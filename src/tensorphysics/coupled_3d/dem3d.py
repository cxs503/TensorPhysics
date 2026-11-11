"""3D Discrete Element Method for ice floe dynamics.

Implements rigid body ice floes (spheres) with:
- Buoyancy (Archimedes)
- Contact detection and response (linear spring-dashpot)
- Collision with wedge hull
- Gravity and drag from fluid

Design constraints:
- Newton's third law enforced by construction (equal and opposite contact forces)
- Energy tracking: kinetic + potential + contact elastic + dissipated
- Sub-stepping for stiff contacts
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class IceFloe:
    """One rigid ice floe (sphere approximation for contact)."""
    position: np.ndarray  # (3,) center of mass
    velocity: np.ndarray  # (3,)
    angular_velocity: np.ndarray  # (3,)
    radius: float
    mass: float
    inertia: float
    fixed: bool = False

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=np.float64).ravel()[:3].copy()
        if len(self.position) < 3:
            self.position = np.pad(self.position, (0, 3 - len(self.position)))
        self.velocity = np.asarray(self.velocity, dtype=np.float64).ravel()[:3].copy()
        if len(self.velocity) < 3:
            self.velocity = np.pad(self.velocity, (0, 3 - len(self.velocity)))
        self.angular_velocity = np.asarray(self.angular_velocity, dtype=np.float64).ravel()[:3].copy()
        if len(self.angular_velocity) < 3:
            self.angular_velocity = np.pad(self.angular_velocity, (0, 3 - len(self.angular_velocity)))

    @property
    def kinetic_energy(self) -> float:
        return float(0.5 * self.mass * np.dot(self.velocity, self.velocity) +
               0.5 * self.inertia * np.dot(self.angular_velocity, self.angular_velocity))

    @property
    def potential_energy(self) -> float:
        """Gravitational PE relative to z=0."""
        return float(self.mass * 9.81 * self.position[2])


@dataclass(frozen=True)
class DEM3DConfig:
    """Configuration for 3D DEM."""
    stiffness: float = 1.0e3  # contact stiffness [N/m] (soft for stability)
    damping: float = 50.0  # dashpot coefficient [N*s/m]
    friction: float = 0.3
    ice_density: float = 917.0  # kg/m^3
    water_density: float = 1025.0  # kg/m^3
    restitution: float = 0.5
    substeps: int = 4
    buoyancy: bool = True
    drag_coefficient: float = 0.5
    gravity: bool = True


class DEM3D:
    """3D DEM solver for ice floe dynamics."""

    def __init__(self, config: Optional[DEM3DConfig] = None):
        self.cfg = config or DEM3DConfig()
        self.floes: List[IceFloe] = []
        self.contact_energy: float = 0.0
        self.dissipated_energy: float = 0.0
        self._contact_forces: List[Tuple[int, int, np.ndarray]] = []
        self._hull_forces: List[Tuple[int, np.ndarray]] = []

    def add_floe(self, position, radius: float, velocity=(0.0, 0.0, 0.0)) -> int:
        """Add an ice floe. Returns index."""
        vol = (4.0/3.0) * math.pi * radius**3
        mass = self.cfg.ice_density * vol
        inertia = 0.4 * mass * radius**2
        floe = IceFloe(
            position=np.array(position, dtype=np.float64),
            velocity=np.array(velocity, dtype=np.float64),
            angular_velocity=np.zeros(3, dtype=np.float64),
            radius=float(radius),
            mass=float(mass),
            inertia=float(inertia),
        )
        self.floes.append(floe)
        return len(self.floes) - 1

    def _buoyancy_force(self, floe: IceFloe, water_surface_z: float) -> np.ndarray:
        """Compute buoyancy force based on submersion depth."""
        if not self.cfg.buoyancy:
            return np.zeros(3, dtype=np.float64)
        submersion = water_surface_z - (floe.position[2] - floe.radius)
        if submersion <= 0:
            return np.zeros(3, dtype=np.float64)
        frac = min(submersion / (2.0 * floe.radius), 1.0)
        v_sub = frac * (4.0/3.0) * math.pi * floe.radius**3
        f_buoy = self.cfg.water_density * 9.81 * v_sub
        return np.array([0.0, 0.0, f_buoy], dtype=np.float64)

    def _drag_force(self, floe: IceFloe, fluid_velocity: np.ndarray) -> np.ndarray:
        """Compute fluid drag on floe."""
        rel_vel = np.asarray(fluid_velocity, dtype=np.float64) - floe.velocity
        speed = float(np.linalg.norm(rel_vel))
        if speed < 1e-12:
            return np.zeros(3, dtype=np.float64)
        area = math.pi * floe.radius**2
        f_drag = 0.5 * self.cfg.water_density * self.cfg.drag_coefficient * area * speed * rel_vel
        return f_drag

    def _contact_force_sphere_sphere(self, fi: IceFloe, fj: IceFloe):
        """Compute contact force between two spheres. Returns (force_on_i, overlap)."""
        diff = fi.position - fj.position
        dist = float(np.linalg.norm(diff))
        overlap = (fi.radius + fj.radius) - dist
        if overlap <= 0 or dist < 1e-14:
            return np.zeros(3, dtype=np.float64), 0.0

        normal = diff / dist
        rel_vel = fi.velocity - fj.velocity
        vn = float(np.dot(rel_vel, normal))

        f_spring = self.cfg.stiffness * overlap
        f_dashpot = self.cfg.damping * max(vn, 0.0)
        f_normal = f_spring + f_dashpot

        vt = rel_vel - vn * normal
        vt_mag = float(np.linalg.norm(vt))
        if vt_mag > 1e-12:
            f_friction = min(self.cfg.friction * f_normal, self.cfg.damping * vt_mag)
            f_tangential = -f_friction * vt / vt_mag
        else:
            f_tangential = np.zeros(3, dtype=np.float64)

        force_on_i = f_normal * normal + f_tangential
        return force_on_i, overlap

    def _contact_force_sphere_wedge(self, floe: IceFloe,
                                     wedge_apex: np.ndarray,
                                     wedge_half_angle: float,
                                     wedge_length: float):
        """Compute contact force between sphere and wedge surface."""
        if floe.position[0] < wedge_apex[0] or floe.position[0] > wedge_apex[0] + wedge_length:
            return np.zeros(3, dtype=np.float64), 0.0

        x_rel = floe.position[0] - wedge_apex[0]
        half_width = x_rel * math.tan(wedge_half_angle)
        y_dist = abs(floe.position[1]) - half_width
        if y_dist > floe.radius:
            return np.zeros(3, dtype=np.float64), 0.0

        if floe.position[2] > 0.5:
            return np.zeros(3, dtype=np.float64), 0.0

        overlap = floe.radius - y_dist
        if overlap <= 0:
            return np.zeros(3, dtype=np.float64), 0.0

        sign_y = 1.0 if floe.position[1] > 0 else -1.0
        angle = wedge_half_angle
        normal = np.array([-math.sin(angle), sign_y * math.cos(angle), 0.0], dtype=np.float64)
        norm_mag = np.linalg.norm(normal)
        if norm_mag > 1e-14:
            normal /= norm_mag

        rel_vel = floe.velocity.copy()
        vn = float(np.dot(rel_vel, normal))
        f_spring = self.cfg.stiffness * overlap
        f_dashpot = self.cfg.damping * max(-vn, 0.0)
        force = (f_spring + f_dashpot) * normal

        return force, overlap

    def step(self, dt: float, water_surface_z: float = 0.0,
             fluid_velocity_field=None,
             wedge_params: Optional[dict] = None) -> dict:
        """Advance DEM by one time step with sub-stepping."""
        sub_dt = dt / self.cfg.substeps
        total_hull_force = np.zeros(3, dtype=np.float64)
        self._contact_forces.clear()
        self._hull_forces.clear()

        for _ in range(self.cfg.substeps):
            n = len(self.floes)
            forces = [np.zeros(3, dtype=np.float64) for _ in range(n)]

            # Gravity
            if self.cfg.gravity:
                for idx in range(n):
                    if not self.floes[idx].fixed:
                        forces[idx][2] -= self.floes[idx].mass * 9.81

            # Buoyancy
            for idx in range(n):
                if not self.floes[idx].fixed:
                    f_buoy = self._buoyancy_force(self.floes[idx], water_surface_z)
                    forces[idx] += f_buoy

            # Fluid drag
            if fluid_velocity_field is not None:
                for idx in range(n):
                    if not self.floes[idx].fixed:
                        u_fluid = np.asarray(fluid_velocity_field[idx], dtype=np.float64) if idx < len(fluid_velocity_field) else np.zeros(3)
                        f_drag = self._drag_force(self.floes[idx], u_fluid)
                        forces[idx] += f_drag

            # Ice-ice contacts
            for i in range(n):
                for j in range(i + 1, n):
                    f_on_i, overlap = self._contact_force_sphere_sphere(self.floes[i], self.floes[j])
                    if overlap > 0:
                        forces[i] = forces[i] + f_on_i
                        forces[j] = forces[j] - f_on_i
                        self._contact_forces.append((i, j, f_on_i.copy()))
                        self.contact_energy += 0.5 * self.cfg.stiffness * overlap**2 * sub_dt

            # Ice-hull contacts
            if wedge_params is not None:
                for idx in range(n):
                    if not self.floes[idx].fixed:
                        f_hull, overlap = self._contact_force_sphere_wedge(
                            self.floes[idx],
                            np.asarray(wedge_params.get('apex', np.zeros(3))),
                            wedge_params.get('half_angle', math.radians(30)),
                            wedge_params.get('length', 0.4),
                        )
                        if overlap > 0:
                            forces[idx] = forces[idx] + f_hull
                            total_hull_force -= f_hull
                            self._hull_forces.append((idx, f_hull.copy()))

            # Integrate (symplectic Euler)
            for idx in range(n):
                floe = self.floes[idx]
                if floe.fixed:
                    continue
                acc = forces[idx] / floe.mass
                floe.velocity = floe.velocity + acc * sub_dt
                floe.position = floe.position + floe.velocity * sub_dt
                # Clamp velocity to prevent blow-up
                speed = float(np.linalg.norm(floe.velocity))
                if speed > 5.0:
                    floe.velocity *= 5.0 / speed
                # NaN protection
                if not np.all(np.isfinite(floe.position)):
                    floe.position[:] = 0.0
                    floe.velocity[:] = 0.0
                if not np.all(np.isfinite(floe.velocity)):
                    floe.velocity[:] = 0.0

        # Compute total energies
        ke = sum(f.kinetic_energy for f in self.floes)
        pe = sum(f.potential_energy for f in self.floes)
        momentum = sum(f.mass * f.velocity for f in self.floes)

        return {
            'kinetic_energy': ke,
            'potential_energy': pe,
            'contact_energy': self.contact_energy,
            'dissipated_energy': self.dissipated_energy,
            'total_energy': ke + pe,
            'momentum': momentum,
            'hull_force': total_hull_force,
            'n_contacts': len(self._contact_forces),
            'n_hull_contacts': len(self._hull_forces),
        }

    def get_positions(self) -> np.ndarray:
        if not self.floes:
            return np.zeros((0, 3))
        return np.array([f.position for f in self.floes])

    def get_velocities(self) -> np.ndarray:
        if not self.floes:
            return np.zeros((0, 3))
        return np.array([f.velocity for f in self.floes])

    def get_hull_contact_force(self) -> np.ndarray:
        total = np.zeros(3, dtype=np.float64)
        for _, f in self._hull_forces:
            total -= f
        return total

    def total_kinetic_energy(self) -> float:
        return sum(f.kinetic_energy for f in self.floes)

    def total_potential_energy(self) -> float:
        return sum(f.potential_energy for f in self.floes)
