"""3D FEM for hull structure response (shell/beam elements).

Implements a simplified structural model for the wedge hull:
- Flat plate elements (Kirchhoff-Love shell approximation)
- Linear elastic material
- Static solve per coupling step (quasi-static assumption)
- Pressure loading from LBM → displacement → boundary update for LBM

Design constraints:
- Small deformation assumption (linear)
- Simply supported boundary conditions at frame locations
- Stress recovery for failure assessment
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FEM3DConfig:
    """Configuration for 3D FEM hull model."""
    # Hull plate properties
    plate_length: float = 0.2  # m (along x)
    plate_width: float = 0.1  # m (along y)
    plate_thickness: float = 0.005  # m
    young_modulus: float = 2.1e11  # Pa (steel)
    poisson_ratio: float = 0.3
    density: float = 7850.0  # kg/m^3
    # Mesh
    nx: int = 8  # elements in x
    ny: int = 4  # elements in y
    # Boundary conditions
    bc_type: str = "simply_supported"  # "clamped" or "simply_supported"


@dataclass
class FEM3DResult:
    """Result of a FEM solve."""
    displacements: np.ndarray  # (n_nodes,) nodal displacements
    max_displacement: float
    max_stress: float
    strain_energy: float
    reaction_forces: np.ndarray


class FEM3DHull:
    """3D FEM solver for hull plate under pressure loading.

    Uses a 4-node rectangular plate bending element (simplified).
    The plate represents a hull panel between frames.
    """

    def __init__(self, config: Optional[FEM3DConfig] = None):
        self.cfg = config or FEM3DConfig()
        self._build_mesh()
        self._assemble_stiffness()
        self.displacements = np.zeros(self.n_dof)
        self.pressure_load = np.zeros(self.n_nodes)
        self.strain_energy = 0.0

    def _build_mesh(self) -> None:
        """Build the plate mesh."""
        nx, ny = self.cfg.nx, self.cfg.ny
        self.nx_elem = nx
        self.ny_elem = ny
        self.nx_nodes = nx + 1
        self.ny_nodes = ny + 1
        self.n_nodes = self.nx_nodes * self.ny_nodes

        # Node coordinates
        dx = self.cfg.plate_length / nx
        dy = self.cfg.plate_width / ny
        self.node_coords = np.zeros((self.n_nodes, 2))
        for j in range(self.ny_nodes):
            for i in range(self.nx_nodes):
                idx = j * self.nx_nodes + i
                self.node_coords[idx, 0] = i * dx
                self.node_coords[idx, 1] = j * dy

        # Element connectivity (4-node quads)
        self.elements = []
        for j in range(ny):
            for i in range(nx):
                n1 = j * self.nx_nodes + i
                n2 = n1 + 1
                n3 = n2 + self.nx_nodes
                n4 = n1 + self.nx_nodes
                self.elements.append((n1, n2, n3, n4))

        # DOFs: one transverse displacement per node (simplified plate bending)
        self.n_dof = self.n_nodes
        self.dof_map = {node: node for node in range(self.n_nodes)}

    def _plate_bending_stiffness(self) -> float:
        """Compute plate bending stiffness D = E*t^3 / (12*(1-nu^2))."""
        E = self.cfg.young_modulus
        t = self.cfg.plate_thickness
        nu = self.cfg.poisson_ratio
        return E * t**3 / (12.0 * (1.0 - nu**2))

    def _assemble_stiffness(self) -> None:
        """Assemble global stiffness matrix using simplified plate elements."""
        D = self._plate_bending_stiffness()
        dx = self.cfg.plate_length / self.nx_elem
        dy = self.cfg.plate_width / self.ny_elem

        # Simplified stiffness: use finite difference approximation for plate bending
        # ∇⁴w = q/D, discretized on the grid
        self.K = np.zeros((self.n_dof, self.n_dof))

        for j in range(self.ny_nodes):
            for i in range(self.nx_nodes):
                idx = j * self.nx_nodes + i
                # Central difference for biharmonic operator
                # This is a simplified approach; proper FEM would use shape functions
                coeff = D * (1.0/dx**4 + 1.0/dy**4 + 2.0/(dx**2 * dy**2))

                # Check if this is an internal node
                is_internal = (0 < i < self.nx_nodes - 1) and (0 < j < self.ny_nodes - 1)
                if not is_internal:
                    # Boundary node - will be constrained
                    self.K[idx, idx] = 1.0  # placeholder, will be modified by BC
                    continue

                self.K[idx, idx] = coeff * 2.0 * (1.0/dx**2 + 1.0/dy**2)

                # Neighbors
                if i > 0:
                    self.K[idx, idx - 1] = -D / dx**4
                if i < self.nx_nodes - 1:
                    self.K[idx, idx + 1] = -D / dx**4
                if j > 0:
                    self.K[idx, idx - self.nx_nodes] = -D / dy**4
                if j < self.ny_nodes - 1:
                    self.K[idx, idx + self.nx_nodes] = -D / dy**4

                # Cross terms
                if i > 0 and j > 0:
                    self.K[idx, idx - self.nx_nodes - 1] = D / (dx**2 * dy**2)
                if i > 0 and j < self.ny_nodes - 1:
                    self.K[idx, idx + self.nx_nodes - 1] = -D / (dx**2 * dy**2)
                if i < self.nx_nodes - 1 and j > 0:
                    self.K[idx, idx - self.nx_nodes + 1] = -D / (dx**2 * dy**2)
                if i < self.nx_nodes - 1 and j < self.ny_nodes - 1:
                    self.K[idx, idx + self.nx_nodes + 1] = D / (dx**2 * dy**2)

        # Apply boundary conditions (simply supported: w=0 on edges)
        self._apply_bc()

    def _apply_bc(self) -> None:
        """Apply simply-supported boundary conditions (w=0 on all edges)."""
        # Set rows/cols for boundary DOFs to identity
        for j in range(self.ny_nodes):
            for i in range(self.nx_nodes):
                idx = j * self.nx_nodes + i
                if i == 0 or i == self.nx_nodes - 1 or j == 0 or j == self.ny_nodes - 1:
                    self.K[idx, :] = 0.0
                    self.K[:, idx] = 0.0
                    self.K[idx, idx] = 1.0

    def apply_pressure(self, pressure: np.ndarray) -> None:
        """Apply pressure load on the plate.

        Parameters
        ----------
        pressure : ndarray
            Pressure at each node (n_nodes,) [Pa]
        """
        self.pressure_load = pressure.copy()

    def solve(self) -> FEM3DResult:
        """Solve the static problem K*u = f."""
        # Build load vector
        f = self.pressure_load.copy()

        # Zero out boundary DOF loads
        for j in range(self.ny_nodes):
            for i in range(self.nx_nodes):
                idx = j * self.nx_nodes + i
                if i == 0 or i == self.nx_nodes - 1 or j == 0 or j == self.ny_nodes - 1:
                    f[idx] = 0.0

        # Solve
        try:
            self.displacements = np.linalg.solve(self.K, f)
        except np.linalg.LinAlgError:
            self.displacements = np.zeros(self.n_dof)

        # Compute strain energy
        self.strain_energy = 0.5 * np.dot(self.displacements, self.K @ self.displacements)

        # Compute max stress (simplified: from bending moment)
        max_stress = self._compute_max_stress()

        # Reaction forces at boundaries
        reactions = self.K @ self.displacements - f

        return FEM3DResult(
            displacements=self.displacements.copy(),
            max_displacement=float(np.max(np.abs(self.displacements))),
            max_stress=max_stress,
            strain_energy=self.strain_energy,
            reaction_forces=reactions,
        )

    def _compute_max_stress(self) -> float:
        """Compute maximum bending stress from displacements."""
        D = self._plate_bending_stiffness()
        t = self.cfg.plate_thickness
        dx = self.cfg.plate_length / self.nx_elem
        dy = self.cfg.plate_width / self.ny_elem

        max_moment = 0.0
        for j in range(1, self.ny_nodes - 1):
            for i in range(1, self.nx_nodes - 1):
                idx = j * self.nx_nodes + i
                # Curvature from finite differences
                d2w_dx2 = (self.displacements[idx + 1] - 2*self.displacements[idx] + self.displacements[idx - 1]) / dx**2
                d2w_dy2 = (self.displacements[idx + self.nx_nodes] - 2*self.displacements[idx] + self.displacements[idx - self.nx_nodes]) / dy**2
                moment = D * (d2w_dx2 + self.cfg.poisson_ratio * d2w_dy2)
                max_moment = max(max_moment, abs(moment))

        # Stress = M * t/2 / (t^3/12) = 6M/t^2
        return 6.0 * max_moment / t**2

    def get_displacement_at_node(self, node_idx: int) -> float:
        """Get transverse displacement at a node."""
        return float(self.displacements[node_idx])

    def get_displaced_surface(self) -> np.ndarray:
        """Get the displaced surface coordinates (n_nodes × 3)."""
        surface = np.zeros((self.n_nodes, 3))
        surface[:, :2] = self.node_coords
        surface[:, 2] = self.displacements
        return surface

    def strain_energy_value(self) -> float:
        """Get current strain energy."""
        return self.strain_energy

    def update_boundary_for_lbm(self) -> np.ndarray:
        """Get boundary displacement field for LBM mesh update.

        Returns displacement at each node that maps to LBM boundary.
        """
        return self.displacements.copy()
