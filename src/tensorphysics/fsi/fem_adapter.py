"""Adapter for TensorFEM structural response computation.

This adapter computes structural response using TensorFEM and formats
displacement data for the FSI coupling interface.

In a full FSI simulation, this would interface with the actual TensorFEM solver.
For verification purposes, it provides a reduced-order structural model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class FEMDisplacement:
    """Structural displacement data from FEM simulation.
    
    Attributes
    ----------
    timestamp : float
        Simulation time [s].
    node_positions : np.ndarray, shape (n_nodes, 3)
        Original positions of interface nodes [m].
    displacements : np.ndarray, shape (n_nodes, 3)
        Displacement vectors at each node [m].
    plunge : float
        Plunge displacement (vertical translation) [m].
    pitch : float
        Pitch angle (rotation about elastic axis) [rad].
    """
    timestamp: float
    node_positions: np.ndarray
    displacements: np.ndarray
    plunge: float
    pitch: float


class FEMStructuralAdapter:
    """Adapter for computing and formatting TensorFEM structural response.
    
    This adapter provides a 2-DOF structural model (plunge + pitch) that
    represents the essential dynamics of an airfoil section.
    
    Parameters
    ----------
    chord : float
        Airfoil chord length [m].
    mass : float
        Structural mass per unit span [kg/m].
    mass_moment_of_inertia : float
        Mass moment of inertia about elastic axis [kg·m].
    plunge_frequency : float
        Natural frequency in plunge [Hz].
    pitch_frequency : float
        Natural frequency in pitch [Hz].
    plunge_damping : float
        Damping ratio in plunge.
    pitch_damping : float
        Damping ratio in pitch.
    elastic_axis : float
        Elastic axis location as fraction of chord (0=LE, 1=TE).
    """
    
    def __init__(
        self,
        chord: float = 1.0,
        mass: float = 20.0,
        mass_moment_of_inertia: float = 5.0,
        plunge_frequency: float = 1.0,
        pitch_frequency: float = 2.0,
        plunge_damping: float = 0.01,
        pitch_damping: float = 0.01,
        elastic_axis: float = 0.35,
    ):
        self.chord = chord
        self.mass = mass
        self.Iα = mass_moment_of_inertia
        self.ωh = 2 * np.pi * plunge_frequency
        self.ωα = 2 * np.pi * pitch_frequency
        self.ch = 2 * mass * self.ωh * plunge_damping
        self.cα = 2 * self.Iα * self.ωα * pitch_damping
        self.kh = mass * self.ωh**2
        self.kα = self.Iα * self.ωα**2
        self.ea = elastic_axis
        
        # State variables
        self.h = 0.0  # Plunge displacement
        self.alpha = 0.0  # Pitch angle
        self.h_dot = 0.0  # Plunge velocity
        self.alpha_dot = 0.0  # Pitch rate
        
        # Generate structural mesh
        self._generate_mesh()
    
    def _generate_mesh(self) -> None:
        """Generate structural mesh nodes."""
        # Simple beam mesh along chord
        n_nodes = 21
        x = np.linspace(0, self.chord, n_nodes)
        y = np.zeros(n_nodes)
        z = np.zeros(n_nodes)
        
        self.mesh_nodes = np.column_stack([x, y, z])
    
    def compute_displacement(
        self,
        lift: float,
        moment: float,
        dt: float,
    ) -> FEMDisplacement:
        """Compute structural displacement under aerodynamic loads.
        
        Uses Newmark-beta time integration for the 2-DOF system.
        
        Parameters
        ----------
        lift : float
            Lift force [N/m].
        moment : float
            Moment about elastic axis [N·m/m].
        dt : float
            Time step [s].
        
        Returns
        -------
        FEMDisplacement
            Displacement data.
        """
        # Equations of motion:
        # m·ḧ + ch·ḣ + kh·h = L
        # Iα·α̈ + cα·α̇ + kα·α = M
        
        # Mass matrix
        M_mat = np.array([
            [self.mass, 0],
            [0, self.Iα]
        ])
        
        # Force vector
        F_vec = np.array([
            lift - self.ch * self.h_dot - self.kh * self.h,
            moment - self.cα * self.alpha_dot - self.kα * self.alpha
        ])
        
        # Acceleration
        acc = np.linalg.solve(M_mat, F_vec)
        h_ddot = acc[0]
        alpha_ddot = acc[1]
        
        # Newmark-beta integration (β=0.25, γ=0.5)
        beta = 0.25
        gamma = 0.5
        
        # Update velocities
        self.h_dot += dt * ((1 - gamma) * h_ddot + gamma * h_ddot)
        self.alpha_dot += dt * ((1 - gamma) * alpha_ddot + gamma * alpha_ddot)
        
        # Update displacements
        self.h += dt * self.h_dot + dt**2 * ((0.5 - beta) * h_ddot + beta * h_ddot)
        self.alpha += dt * self.alpha_dot + dt**2 * ((0.5 - beta) * alpha_ddot + beta * alpha_ddot)
        
        # Compute nodal displacements
        displacements = np.zeros_like(self.mesh_nodes)
        
        # Plunge: uniform vertical translation
        displacements[:, 1] = self.h
        
        # Pitch: rotation about elastic axis
        ea_x = self.ea * self.chord
        for i in range(len(self.mesh_nodes)):
            x_rel = self.mesh_nodes[i, 0] - ea_x
            # Small angle rotation
            displacements[i, 0] += -self.alpha * (self.mesh_nodes[i, 1])
            displacements[i, 1] += self.alpha * x_rel
        
        return FEMDisplacement(
            timestamp=0.0,
            node_positions=self.mesh_nodes.copy(),
            displacements=displacements,
            plunge=self.h,
            pitch=self.alpha,
        )
    
    def to_handoff_contract(self, disp_data: FEMDisplacement) -> dict:
        """Format displacement data for FSI coupling.
        
        Parameters
        ----------
        disp_data : FEMDisplacement
            Displacement data.
        
        Returns
        -------
        dict
            Formatted displacement data.
        """
        interface_nodes = []
        for i in range(len(disp_data.node_positions)):
            interface_nodes.append({
                "node_id": i,
                "x": float(disp_data.node_positions[i, 0]),
                "y": float(disp_data.node_positions[i, 1]),
                "z": float(disp_data.node_positions[i, 2]),
                "ux_m": float(disp_data.displacements[i, 0]),
                "uy_m": float(disp_data.displacements[i, 1]),
                "uz_m": float(disp_data.displacements[i, 2]),
            })
        
        return {
            "schema": "fsi_displacement_handoff.v1",
            "frame": {
                "x": "streamwise",
                "y": "transverse",
                "z": "vertical_up"
            },
            "timestamp_s": disp_data.timestamp,
            "interface_nodes": interface_nodes,
            "displacement_components": ["ux_m", "uy_m", "uz_m"],
            "rigid_body_motion": {
                "plunge_m": disp_data.plunge,
                "pitch_rad": disp_data.pitch,
            },
            "mesh_id": "naca0012_structure_v1",
        }
    
    def save_handoff(self, disp_data: FEMDisplacement, path: Path) -> None:
        """Save displacement handoff data to JSON file.
        
        Parameters
        ----------
        disp_data : FEMDisplacement
            Displacement data.
        path : Path
            Output file path.
        """
        contract = self.to_handoff_contract(disp_data)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(contract, f, indent=2)
    
    def reset(self, h0: float = 0.0, alpha0: float = 0.0) -> None:
        """Reset structural state.
        
        Parameters
        ----------
        h0 : float
            Initial plunge displacement [m].
        alpha0 : float
            Initial pitch angle [rad].
        """
        self.h = h0
        self.alpha = alpha0
        self.h_dot = 0.0
        self.alpha_dot = 0.0
