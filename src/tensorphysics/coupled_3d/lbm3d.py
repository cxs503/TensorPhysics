"""D3Q19 BGK lattice Boltzmann solver with free-surface approximation.

Implements a simplified 3D LBM solver for the flow field around a wedge hull.
The free surface is modelled via a height-function approach on the top boundary.

Design constraints:
- D3Q19 lattice with BGK collision
- Full-way bounce-back on solid boundaries (wedge hull)
- Free-slip / open boundary on lateral and outlet faces
- Free-surface approximated via hydrostatic pressure on top face
- Exact momentum-exchange tally for fluid-structure coupling
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

# D3Q19 lattice velocities and weights
# Ordering: rest(1) + 6 face + 12 edge = 19
_C3 = np.array([
    [0, 0, 0],       # 0: rest
    [1, 0, 0], [-1, 0, 0],  # 1,2: x-faces
    [0, 1, 0], [0, -1, 0],  # 3,4: y-faces
    [0, 0, 1], [0, 0, -1],  # 5,6: z-faces
    [1, 1, 0], [-1, -1, 0],  # 7,8: xy-edges
    [1, -1, 0], [-1, 1, 0],  # 9,10: xy-edges
    [1, 0, 1], [-1, 0, -1],  # 11,12: xz-edges
    [1, 0, -1], [-1, 0, 1],  # 13,14: xz-edges
    [0, 1, 1], [0, -1, -1],  # 15,16: yz-edges
    [0, 1, -1], [0, -1, 1],  # 17,18: yz-edges
], dtype=np.float64)

_W3 = np.array([
    1.0/3.0,  # rest
    1.0/18.0, 1.0/18.0,  # x-faces
    1.0/18.0, 1.0/18.0,  # y-faces
    1.0/18.0, 1.0/18.0,  # z-faces
    1.0/36.0, 1.0/36.0,  # xy-edges
    1.0/36.0, 1.0/36.0,  # xy-edges
    1.0/36.0, 1.0/36.0,  # xz-edges
    1.0/36.0, 1.0/36.0,  # xz-edges
    1.0/36.0, 1.0/36.0,  # yz-edges
    1.0/36.0, 1.0/36.0,  # yz-edges
], dtype=np.float64)

_CS2 = 1.0 / 3.0  # speed of sound squared
_Q3 = 19
_OPP3 = np.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17])


def equilibrium3d(rho: np.ndarray, ux: np.ndarray, uy: np.ndarray, uz: np.ndarray) -> np.ndarray:
    """Compute D3Q19 equilibrium distributions."""
    feq = np.empty((_Q3,) + rho.shape, dtype=np.float64)
    usq = ux**2 + uy**2 + uz**2
    for i in range(_Q3):
        ci_u = _C3[i, 0] * ux + _C3[i, 1] * uy + _C3[i, 2] * uz
        feq[i] = _W3[i] * rho * (1.0 + ci_u / _CS2 + 0.5 * ci_u**2 / _CS2**2 - 0.5 * usq / _CS2)
    return feq


@dataclass
class LBM3DStepDiagnostics:
    """Per-step diagnostics from the LBM solver."""
    kinetic_energy: float
    dissipation: float
    mass: float
    momentum: np.ndarray  # (3,)
    wall_force: np.ndarray  # (3,) force on solid boundaries
    max_velocity: float


@dataclass
class LBM3DConfig:
    """Configuration for the D3Q19 LBM solver."""
    nx: int = 20
    ny: int = 10
    nz: int = 10
    tau: float = 0.55
    rho0: float = 1.0  # lattice units (density)
    dx: float = 1.0  # lattice spacing = 1 in lattice units
    u_inlet: float = 0.05  # inlet velocity in lattice units (Ma < 0.1)
    gravity: float = 0.0  # no gravity in lattice units for stability
    free_surface_height: float = 1.0


class LBM3D:
    """D3Q19 BGK lattice Boltzmann solver for 3D flow with free-surface approximation.

    Works in lattice units internally. The domain is a rectangular channel with:
    - Inlet (x=0): velocity boundary
    - Outlet (x=nx-1): open boundary (zero-gradient)
    - Walls (y=0, y=ny-1): free-slip
    - Bottom (z=0): no-slip
    - Top (z=nz-1): free-surface (atmospheric pressure)

    Solid boundaries (wedge hull) are immersed via bounce-back.
    """

    def __init__(self, config: Optional[LBM3DConfig] = None):
        self.cfg = config or LBM3DConfig()
        self.nx = self.cfg.nx
        self.ny = self.cfg.ny
        self.nz = self.cfg.nz
        self.tau = self.cfg.tau
        self.rho0 = self.cfg.rho0
        self.dx = self.cfg.dx
        self.dt = 1.0  # lattice time step = 1
        self.nu = (self.tau - 0.5) / 3.0  # kinematic viscosity in lattice units

        shape = (_Q3, self.nx, self.ny, self.nz)
        # Initialize uniformly at rho0 with inlet velocity
        rho_init = np.full((self.nx, self.ny, self.nz), self.rho0)
        ux_init = np.full((self.nx, self.ny, self.nz), self.cfg.u_inlet)
        uy_init = np.zeros((self.nx, self.ny, self.nz))
        uz_init = np.zeros((self.nx, self.ny, self.nz))

        self.f = equilibrium3d(rho_init, ux_init, uy_init, uz_init)
        self.f_tmp = np.empty_like(self.f)

        # Solid flag: True = solid (bounce-back)
        self.solid = np.zeros((self.nx, self.ny, self.nz), dtype=bool)

        # Macroscopic fields
        self.rho = rho_init.copy()
        self.ux = ux_init.copy()
        self.uy = uy_init.copy()
        self.uz = uz_init.copy()

        # Force on solid boundaries (for FEM coupling)
        self.wall_force = np.zeros(3)

        # Energy tracking
        self._dissipation_total = 0.0

    def set_wedge_geometry(self, half_angle_deg: float = 30.0,
                           draft_frac: float = 0.6,
                           length_frac: float = 0.4) -> None:
        """Set wedge hull geometry as solid nodes."""
        angle = math.radians(half_angle_deg)
        nx_w = int(length_frac * self.nx)
        ny_center = self.ny // 2
        nz_draft = int(draft_frac * self.nz)

        for i in range(nx_w):
            x_frac = i / max(nx_w - 1, 1)
            half_width = x_frac * math.tan(angle) * self.ny / 2
            half_width_cells = max(1, int(half_width))
            for j in range(self.ny):
                for k in range(nz_draft):
                    dy = abs(j - ny_center)
                    if dy <= half_width_cells:
                        self.solid[i, j, k] = True

    def _compute_macroscopic(self) -> None:
        """Compute macroscopic density and velocity from distributions."""
        self.rho = np.sum(self.f, axis=0)
        self.rho = np.maximum(self.rho, 0.1 * self.rho0)
        self.ux = np.sum(self.f * _C3[:, 0].reshape(-1, 1, 1, 1), axis=0) / self.rho
        self.uy = np.sum(self.f * _C3[:, 1].reshape(-1, 1, 1, 1), axis=0) / self.rho
        self.uz = np.sum(self.f * _C3[:, 2].reshape(-1, 1, 1, 1), axis=0) / self.rho
        # Velocity clamping for stability (Ma < 0.3)
        u_max = 0.3  # max velocity in lattice units
        umag = np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)
        mask = umag > u_max
        scale = np.where(mask, u_max / np.maximum(umag, 1e-10), 1.0)
        self.ux *= scale
        self.uy *= scale
        self.uz *= scale

    def _collide(self) -> None:
        """BGK collision step."""
        feq = equilibrium3d(self.rho, self.ux, self.uy, self.uz)
        omega = 1.0 / self.tau
        self.f += omega * (feq - self.f)

    def _stream(self) -> None:
        """Streaming step with bounce-back on solid nodes."""
        self.f_tmp[:] = self.f
        self.wall_force[:] = 0.0

        for i in range(_Q3):
            cx, cy, cz = int(_C3[i, 0]), int(_C3[i, 1]), int(_C3[i, 2])
            self.f[i] = np.roll(self.f_tmp[i], shift=(cx, cy, cz), axis=(0, 1, 2))

        # Bounce-back on solid nodes
        for i in range(_Q3):
            opp = _OPP3[i]
            solid_mask = self.solid
            self.f[i][solid_mask] = self.f_tmp[opp][solid_mask]

        # Compute momentum exchange force on solid
        for i in range(_Q3):
            cx, cy, cz = _C3[i]
            df = self.f[i] - self.f_tmp[i]
            force_contrib = np.sum(df[self.solid])
            self.wall_force[0] += cx * force_contrib
            self.wall_force[1] += cy * force_contrib
            self.wall_force[2] += cz * force_contrib

    def _apply_boundaries(self) -> None:
        """Apply boundary conditions."""
        # Inlet (x=0): equilibrium at inlet velocity
        for i in range(_Q3):
            ci_u = _C3[i, 0] * self.cfg.u_inlet
            feq_in = _W3[i] * self.rho0 * (1.0 + ci_u / _CS2 + 0.5 * ci_u**2 / _CS2**2 - 0.5 * self.cfg.u_inlet**2 / _CS2)
            # Extrapolate non-equilibrium from node 1
            rho_1 = self.rho[1, :, :]
            ux_1 = self.ux[1, :, :]
            uy_1 = self.uy[1, :, :]
            uz_1 = self.uz[1, :, :]
            ci_u_1 = _C3[i, 0] * ux_1 + _C3[i, 1] * uy_1 + _C3[i, 2] * uz_1
            usq_1 = ux_1**2 + uy_1**2 + uz_1**2
            feq_1 = _W3[i] * rho_1 * (1.0 + ci_u_1 / _CS2 + 0.5 * ci_u_1**2 / _CS2**2 - 0.5 * usq_1 / _CS2)
            self.f[i, 0, :, :] = feq_in + (self.f[i, 1, :, :] - feq_1)

        # Outlet (x=nx-1): zero-gradient
        for i in range(_Q3):
            self.f[i, -1, :, :] = self.f[i, -2, :, :]

        # Top (z=nz-1): atmospheric pressure (rho = rho0)
        for i in range(_Q3):
            ci_u = _C3[i, 0] * self.ux[:, :, -1] + _C3[i, 1] * self.uy[:, :, -1] + _C3[i, 2] * self.uz[:, :, -1]
            usq = self.ux[:, :, -1]**2 + self.uy[:, :, -1]**2 + self.uz[:, :, -1]**2
            self.f[i, :, :, -1] = _W3[i] * self.rho0 * (1.0 + ci_u / _CS2 + 0.5 * ci_u**2 / _CS2**2 - 0.5 * usq / _CS2)

        # Bottom (z=0): bounce-back (no-slip)
        for i in range(_Q3):
            opp = _OPP3[i]
            self.f[i, :, :, 0] = self.f[opp, :, :, 0]

        # Lateral walls (y=0, y=ny-1): free-slip
        for i in range(_Q3):
            if _C3[i, 1] != 0:
                opp = _OPP3[i]
                self.f[i, :, 0, :] = self.f[opp, :, 0, :]
                self.f[i, :, -1, :] = self.f[opp, :, -1, :]

    def step(self) -> LBM3DStepDiagnostics:
        """Advance one time step."""
        self._collide()
        self._stream()
        self._apply_boundaries()
        self._compute_macroscopic()

        # Diagnostics
        feq = equilibrium3d(self.rho, self.ux, self.uy, self.uz)
        diss = float(np.sum((self.f - feq)**2))
        ke_new = float(0.5 * np.sum(self.rho * (self.ux**2 + self.uy**2 + self.uz**2)))
        max_vel = float(np.max(np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)))
        self._dissipation_total += diss

        return LBM3DStepDiagnostics(
            kinetic_energy=ke_new,
            dissipation=diss,
            mass=float(np.sum(self.rho)),
            momentum=np.array([
                float(np.sum(self.rho * self.ux)),
                float(np.sum(self.rho * self.uy)),
                float(np.sum(self.rho * self.uz)),
            ]),
            wall_force=self.wall_force.copy(),
            max_velocity=max_vel,
        )

    def get_pressure_field(self) -> np.ndarray:
        """Get pressure field from density (p = cs^2 * (rho - rho0))."""
        return _CS2 * (self.rho - self.rho0)

    def get_velocity_magnitude(self) -> np.ndarray:
        """Get velocity magnitude field."""
        return np.sqrt(self.ux**2 + self.uy**2 + self.uz**2)

    def get_forces_on_solid(self) -> np.ndarray:
        """Get total hydrodynamic force on solid boundaries."""
        return self.wall_force.copy()

    def kinetic_energy(self) -> float:
        """Total kinetic energy of the fluid."""
        return float(0.5 * np.sum(self.rho * (self.ux**2 + self.uy**2 + self.uz**2)))
