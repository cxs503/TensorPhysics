"""2D baseline for wedge icebreaking comparison.

Implements a 2D version of the wedge icebreaking problem using:
- D2Q9 LBM for flow
- 2D DEM for ice (circular discs)

This serves as the baseline for comparing 3D effects.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


# D2Q9 lattice
_C2 = np.array([
    [0, 0], [1, 0], [-1, 0], [0, 1], [0, -1],
    [1, 1], [-1, -1], [1, -1], [-1, 1],
], dtype=np.float64)
_W2 = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
_CS2_2D = 1.0 / 3.0
_Q2 = 9
_OPP2 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7])


@dataclass
class Baseline2DConfig:
    nx: int = 40
    ny: int = 20
    tau: float = 0.55
    rho0: float = 1.0
    dx: float = 1.0
    u_inlet: float = 0.05
    half_angle_deg: float = 30.0
    n_floes: int = 15
    floe_radius: float = 0.5
    dt: float = 1.0
    n_steps: int = 100
    lbm_substeps: int = 5
    seed: int = 42


class LBM2D:
    """Simplified D2Q9 LBM for 2D baseline."""

    def __init__(self, config: Baseline2DConfig):
        self.cfg = config
        self.nx = config.nx
        self.ny = config.ny
        self.tau = config.tau
        self.rho0 = config.rho0
        self.dx = config.dx

        shape = (_Q2, self.nx, self.ny)
        rho_init = np.full((self.nx, self.ny), self.rho0, dtype=np.float64)
        ux_init = np.full((self.nx, self.ny), config.u_inlet, dtype=np.float64)
        uy_init = np.zeros((self.nx, self.ny), dtype=np.float64)

        self.f = self._equilibrium(rho_init, ux_init, uy_init)
        self.f_tmp = np.empty_like(self.f)
        self.solid = np.zeros((self.nx, self.ny), dtype=bool)
        self.rho = rho_init.copy()
        self.ux = ux_init.copy()
        self.uy = uy_init.copy()
        self.wall_force = np.zeros(2, dtype=np.float64)

    def _equilibrium(self, rho, ux, uy):
        feq = np.empty((_Q2, self.nx, self.ny), dtype=np.float64)
        usq = ux**2 + uy**2
        for i in range(_Q2):
            ci_u = _C2[i, 0] * ux + _C2[i, 1] * uy
            feq[i] = _W2[i] * rho * (1.0 + ci_u / _CS2_2D + 0.5 * ci_u**2 / _CS2_2D**2 - 0.5 * usq / _CS2_2D)
        return feq

    def set_wedge(self, half_angle_deg: float):
        angle = math.radians(half_angle_deg)
        nx_w = int(0.4 * self.nx)
        ny_center = self.ny // 2
        for i in range(nx_w):
            x_frac = i / max(nx_w - 1, 1)
            half_width = x_frac * math.tan(angle) * self.ny / 2
            hw = max(1, int(half_width))
            for j in range(self.ny):
                if abs(j - ny_center) <= hw and j < self.ny // 2 + 5:
                    self.solid[i, j] = True

    def step(self):
        # Collision
        feq = self._equilibrium(self.rho, self.ux, self.uy)
        omega = 1.0 / self.tau
        self.f += omega * (feq - self.f)

        # Streaming
        self.f_tmp[:] = self.f
        for i in range(_Q2):
            cx, cy = int(_C2[i, 0]), int(_C2[i, 1])
            self.f[i] = np.roll(self.f_tmp[i], shift=(cx, cy), axis=(0, 1))

        # Bounce-back
        for i in range(_Q2):
            opp = _OPP2[i]
            self.f[i][self.solid] = self.f_tmp[opp][self.solid]

        # Boundaries
        for i in range(_Q2):
            self.f[i, -1, :] = self.f[i, -2, :]

        # Macroscopic
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.maximum(self.rho, 0.1 * self.rho0)
        self.ux = np.sum(self.f * _C2[:, 0].reshape(-1, 1, 1), axis=0) / self.rho
        self.uy = np.sum(self.f * _C2[:, 1].reshape(-1, 1, 1), axis=0) / self.rho

        # Force on solid
        self.wall_force[:] = 0.0
        for i in range(_Q2):
            cx, cy = _C2[i]
            df = self.f[i] - self.f_tmp[i]
            force = np.sum(df[self.solid])
            self.wall_force[0] += cx * force
            self.wall_force[1] += cy * force

        return {
            'kinetic_energy': float(0.5 * np.sum(self.rho * (self.ux**2 + self.uy**2))),
            'wall_force': self.wall_force.copy(),
            'max_velocity': float(np.max(np.sqrt(self.ux**2 + self.uy**2))),
        }

    def kinetic_energy(self):
        return float(0.5 * np.sum(self.rho * (self.ux**2 + self.uy**2)))


class DEM2D:
    """Simplified 2D DEM for ice discs."""

    def __init__(self, config: Baseline2DConfig):
        self.cfg = config
        self.positions: List[np.ndarray] = []
        self.velocities: List[np.ndarray] = []
        self.radii: List[float] = []
        self.masses: List[float] = []
        self.dissipated: float = 0.0
        self.stiffness = 1.0e2
        self.damping = 10.0

    def add_floe(self, x: float, y: float, radius: float, vx: float = 0, vy: float = 0):
        mass = 917.0 * math.pi * radius**2
        self.positions.append(np.array([x, y], dtype=np.float64))
        self.velocities.append(np.array([vx, vy], dtype=np.float64))
        self.radii.append(float(radius))
        self.masses.append(float(mass))

    def step(self, dt: float, fluid_vel=None):
        n = len(self.positions)
        forces = [np.zeros(2, dtype=np.float64) for _ in range(n)]

        # Floe-floe contacts
        for i in range(n):
            for j in range(i+1, n):
                diff = self.positions[i] - self.positions[j]
                dist = float(np.linalg.norm(diff))
                overlap = (self.radii[i] + self.radii[j]) - dist
                if overlap > 0 and dist > 1e-12:
                    normal = diff / dist
                    rel_vel = self.velocities[i] - self.velocities[j]
                    vn = float(np.dot(rel_vel, normal))
                    f_n = self.stiffness * overlap + self.damping * max(-vn, 0)
                    forces[i] = forces[i] + f_n * normal
                    forces[j] = forces[j] - f_n * normal

        # Integrate
        for i in range(n):
            acc = forces[i] / self.masses[i]
            self.velocities[i] = self.velocities[i] + acc * dt
            self.positions[i] = self.positions[i] + self.velocities[i] * dt

        ke = sum(0.5 * m * float(np.dot(v, v)) for m, v in zip(self.masses, self.velocities))
        return {'kinetic_energy': ke, 'n_contacts': 0}

    def total_kinetic_energy(self):
        return sum(0.5 * m * float(np.dot(v, v)) for m, v in zip(self.masses, self.velocities))


class Baseline2D:
    """2D baseline wedge icebreaking simulation."""

    def __init__(self, config: Optional[Baseline2DConfig] = None):
        self.cfg = config or Baseline2DConfig()
        self.lbm = LBM2D(self.cfg)
        self.lbm.set_wedge(self.cfg.half_angle_deg)
        self.dem = DEM2D(self.cfg)

        np.random.seed(self.cfg.seed)
        for _ in range(self.cfg.n_floes):
            x = 30 + np.random.uniform(0, 5)
            y = np.random.uniform(-5, 5)
            self.dem.add_floe(x, y, self.cfg.floe_radius)

        self.force_history: List[Tuple[float, np.ndarray]] = []
        self.energy_history: List[float] = []

    def run(self) -> dict:
        dt = self.cfg.dt
        time_val = 0.0

        for step in range(self.cfg.n_steps):
            for _ in range(self.cfg.lbm_substeps):
                lbm_diag = self.lbm.step()

            dem_diag = self.dem.step(dt)

            force = lbm_diag['wall_force']
            self.force_history.append((time_val, force.copy()))
            self.energy_history.append(lbm_diag['kinetic_energy'] + dem_diag['kinetic_energy'])

            time_val += dt

        force_array = np.array([f[1] for f in self.force_history]) if self.force_history else np.zeros((0, 2))
        time_array = np.array([f[0] for f in self.force_history]) if self.force_history else np.zeros(0)

        return {
            'status': 'completed',
            'n_steps': self.cfg.n_steps,
            'force_history': {
                'time': time_array.tolist(),
                'Fx': force_array[:, 0].tolist() if len(force_array) > 0 else [],
                'Fy': force_array[:, 1].tolist() if len(force_array) > 0 else [],
            },
            'energy_history': self.energy_history,
            'max_force': float(np.max(np.linalg.norm(force_array, axis=1))) if len(force_array) > 0 else 0.0,
            'mean_force': float(np.mean(np.linalg.norm(force_array, axis=1))) if len(force_array) > 0 else 0.0,
        }
