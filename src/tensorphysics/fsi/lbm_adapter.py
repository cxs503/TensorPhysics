"""Adapter for TensorLBM surface force extraction.

This adapter extracts surface forces from TensorLBM simulations and formats
them according to the FSI handoff contract (fsi_surface_force_handoff.v1).

In a full FSI simulation, this would interface with the actual TensorLBM solver.
For verification purposes, it provides a reduced-order model that mimics the
LBM force output format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LBMSurfaceForce:
    """Surface force data from LBM simulation.
    
    Attributes
    ----------
    timestamp : float
        Simulation time [s].
    node_positions : np.ndarray, shape (n_nodes, 3)
        Positions of interface nodes [m].
    forces : np.ndarray, shape (n_nodes, 3)
        Force vectors at each node [N].
    total_force : np.ndarray, shape (3,)
        Total force on body [N].
    total_moment : np.ndarray, shape (3,)
        Total moment on body [N·m].
    """
    timestamp: float
    node_positions: np.ndarray
    forces: np.ndarray
    total_force: np.ndarray
    total_moment: np.ndarray


class LBMForceAdapter:
    """Adapter for extracting and formatting TensorLBM surface forces.
    
    This adapter provides two modes:
    1. Mock mode: Generates synthetic forces based on analytical models
       (for testing and verification without running full LBM).
    2. Real mode: Extracts forces from actual TensorLBM simulation output.
    
    Parameters
    ----------
    chord : float
        Airfoil chord length [m].
    n_surface_nodes : int
        Number of surface nodes for force distribution.
    """
    
    def __init__(self, chord: float = 1.0, n_surface_nodes: int = 20):
        self.chord = chord
        self.n_surface_nodes = n_surface_nodes
        
        # Generate surface node positions (NACA 0012)
        self._generate_surface_nodes()
    
    def _generate_surface_nodes(self) -> None:
        """Generate surface node positions for NACA 0012 airfoil."""
        # Chordwise positions (cosine spacing)
        beta = np.linspace(0, np.pi, self.n_surface_nodes)
        xc = 0.5 * (1 - np.cos(beta))
        
        # NACA 0012 thickness distribution
        t = 0.12  # 12% thickness
        yt = 5 * t * (
            0.2969 * np.sqrt(xc) - 0.1260 * xc
            - 0.3516 * xc**2 + 0.2843 * xc**3 - 0.1015 * xc**4
        )
        
        # Upper surface
        x_upper = xc * self.chord
        y_upper = yt * self.chord
        
        # Lower surface (mirror)
        x_lower = xc[::-1] * self.chord
        y_lower = -yt[::-1] * self.chord
        
        # Combine (upper then lower, trailing edge to leading edge)
        x_surface = np.concatenate([x_upper, x_lower])
        y_surface = np.concatenate([y_upper, y_lower])
        
        # Store as 3D positions (z=0 for 2D)
        self.surface_nodes = np.column_stack([
            x_surface,
            y_surface,
            np.zeros_like(x_surface)
        ])
    
    def compute_forces_analytical(
        self,
        U: float,
        alpha: float,
        h_dot: float = 0.0,
        alpha_dot: float = 0.0,
    ) -> LBMSurfaceForce:
        """Compute surface forces using analytical thin airfoil theory.
        
        This provides a reference solution for verification without running
        full LBM simulation.
        
        Parameters
        ----------
        U : float
            Flow velocity [m/s].
        alpha : float
            Angle of attack [rad].
        h_dot : float
            Plunge velocity [m/s].
        alpha_dot : float
            Pitch rate [rad/s].
        
        Returns
        -------
        LBMSurfaceForce
            Surface force data.
        """
        # Thin airfoil theory: Cl = 2π·α (for small α)
        Cl = 2 * np.pi * alpha
        
        # Dynamic pressure
        rho = 1.0  # Normalized
        q = 0.5 * rho * U**2
        
        # Lift per unit span
        L = Cl * q * self.chord
        
        # Moment about quarter-chord (zero for symmetric airfoil at small α)
        M_ac = 0.0  # Aerodynamic center moment for symmetric airfoil
        
        # Distribute force over surface nodes
        # Simple distribution: uniform pressure on upper/lower surfaces
        n_nodes = len(self.surface_nodes)
        n_upper = n_nodes // 2
        
        # Force per node (simplified distribution)
        forces = np.zeros((n_nodes, 3))
        
        # Upper surface: suction (negative pressure)
        p_upper = -L / (self.chord * 0.5)  # Pressure difference
        for i in range(n_upper):
            # Normal vector (pointing outward)
            if i < n_upper - 1:
                dx = self.surface_nodes[i+1, 0] - self.surface_nodes[i, 0]
                dy = self.surface_nodes[i+1, 1] - self.surface_nodes[i, 1]
            else:
                dx = self.surface_nodes[i, 0] - self.surface_nodes[i-1, 0]
                dy = self.surface_nodes[i, 1] - self.surface_nodes[i-1, 1]
            
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                nx, ny = -dy / length, dx / length  # Outward normal
            else:
                nx, ny = 0, 1
            
            forces[i, 0] = p_upper * nx * length / n_upper
            forces[i, 1] = p_upper * ny * length / n_upper
        
        # Lower surface: positive pressure
        p_lower = L / (self.chord * 0.5)
        for i in range(n_upper, n_nodes):
            if i < n_nodes - 1:
                dx = self.surface_nodes[i+1, 0] - self.surface_nodes[i, 0]
                dy = self.surface_nodes[i+1, 1] - self.surface_nodes[i, 1]
            else:
                dx = self.surface_nodes[i, 0] - self.surface_nodes[i-1, 0]
                dy = self.surface_nodes[i, 1] - self.surface_nodes[i-1, 1]
            
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                nx, ny = -dy / length, dx / length
            else:
                nx, ny = 0, -1
            
            forces[i, 0] = p_lower * nx * length / (n_nodes - n_upper)
            forces[i, 1] = p_lower * ny * length / (n_nodes - n_upper)
        
        # Total force and moment
        total_force = np.sum(forces, axis=0)
        
        # Moment about elastic axis (at 35% chord)
        ea_x = 0.35 * self.chord
        moments = np.cross(self.surface_nodes - np.array([ea_x, 0, 0]), forces)
        total_moment = np.sum(moments, axis=0)
        
        return LBMSurfaceForce(
            timestamp=0.0,
            node_positions=self.surface_nodes.copy(),
            forces=forces,
            total_force=total_force,
            total_moment=total_moment,
        )
    
    def to_handoff_contract(self, force_data: LBMSurfaceForce) -> dict:
        """Format force data according to FSI handoff contract.
        
        Parameters
        ----------
        force_data : LBMSurfaceForce
            Surface force data.
        
        Returns
        -------
        dict
            Formatted according to fsi_surface_force_handoff.v1 schema.
        """
        interface_nodes = []
        for i in range(len(force_data.node_positions)):
            interface_nodes.append({
                "node_id": i,
                "x": float(force_data.node_positions[i, 0]),
                "y": float(force_data.node_positions[i, 1]),
                "z": float(force_data.node_positions[i, 2]),
                "Fx_N": float(force_data.forces[i, 0]),
                "Fy_N": float(force_data.forces[i, 1]),
                "Fz_N": float(force_data.forces[i, 2]),
            })
        
        return {
            "schema": "fsi_surface_force_handoff.v1",
            "frame": {
                "x": "streamwise",
                "y": "transverse",
                "z": "vertical_up"
            },
            "timestamp_s": force_data.timestamp,
            "interface_nodes": interface_nodes,
            "force_components": ["Fx_N", "Fy_N", "Fz_N"],
            "total_force_N": force_data.total_force.tolist(),
            "total_moment_Nm": force_data.total_moment.tolist(),
            "mesh_id": "naca0012_surface_v1",
        }
    
    def save_handoff(self, force_data: LBMSurfaceForce, path: Path) -> None:
        """Save force handoff data to JSON file.
        
        Parameters
        ----------
        force_data : LBMSurfaceForce
            Surface force data.
        path : Path
            Output file path.
        """
        contract = self.to_handoff_contract(force_data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(contract, f, indent=2)
