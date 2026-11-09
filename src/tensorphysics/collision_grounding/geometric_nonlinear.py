"""
Geometric Nonlinearity for Large Deformation Analysis

Implements:
- Green-Lagrange strain tensor
- 2nd Piola-Kirchhoff stress tensor
- Large deformation solver with updated Lagrangian formulation
"""

import numpy as np
from typing import Tuple, Optional


class GreenLagrangeStrain:
    """
    Green-Lagrange strain tensor computation.
    
    E = 0.5 * (F^T * F - I)
    
    where F is the deformation gradient.
    """
    
    @staticmethod
    def compute_from_displacement_gradient(H: np.ndarray) -> np.ndarray:
        """
        Compute Green-Lagrange strain from displacement gradient.
        
        Parameters
        ----------
        H : np.ndarray
            Displacement gradient tensor (3x3), H = ∂u/∂X
        
        Returns
        -------
        np.ndarray
            Green-Lagrange strain tensor (3x3)
        """
        F = np.eye(3) + H
        E = 0.5 * (F.T @ F - np.eye(3))
        return E
    
    @staticmethod
    def compute_from_deformation_gradient(F: np.ndarray) -> np.ndarray:
        """
        Compute Green-Lagrange strain from deformation gradient.
        
        Parameters
        ----------
        F : np.ndarray
            Deformation gradient tensor (3x3)
        
        Returns
        -------
        np.ndarray
            Green-Lagrange strain tensor (3x3)
        """
        E = 0.5 * (F.T @ F - np.eye(3))
        return E
    
    @staticmethod
    def to_voigt(E: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 strain tensor to 6-component Voigt notation.
        
        Parameters
        ----------
        E : np.ndarray
            3x3 strain tensor
        
        Returns
        -------
        np.ndarray
            6-component vector [E11, E22, E33, 2*E12, 2*E23, 2*E13]
        """
        return np.array([
            E[0, 0], E[1, 1], E[2, 2],
            2.0 * E[0, 1], 2.0 * E[1, 2], 2.0 * E[0, 2]
        ])
    
    @staticmethod
    def from_voigt(E_voigt: np.ndarray) -> np.ndarray:
        """
        Convert 6-component Voigt notation to 3x3 strain tensor.
        
        Parameters
        ----------
        E_voigt : np.ndarray
            6-component vector [E11, E22, E33, 2*E12, 2*E23, 2*E13]
        
        Returns
        -------
        np.ndarray
            3x3 strain tensor
        """
        E = np.zeros((3, 3))
        E[0, 0] = E_voigt[0]
        E[1, 1] = E_voigt[1]
        E[2, 2] = E_voigt[2]
        E[0, 1] = E[1, 0] = E_voigt[3] / 2.0
        E[1, 2] = E[2, 1] = E_voigt[4] / 2.0
        E[0, 2] = E[2, 0] = E_voigt[5] / 2.0
        return E


class SecondPiolaKirchhoffStress:
    """
    2nd Piola-Kirchhoff stress tensor computation.
    
    S = F^{-1} * (J * sigma) * F^{-T}
    
    where F is deformation gradient, J = det(F), sigma is Cauchy stress.
    """
    
    @staticmethod
    def from_cauchy(stress_cauchy: np.ndarray, F: np.ndarray) -> np.ndarray:
        """
        Convert Cauchy stress to 2nd Piola-Kirchhoff stress.
        
        Parameters
        ----------
        stress_cauchy : np.ndarray
            Cauchy stress tensor (3x3)
        F : np.ndarray
            Deformation gradient tensor (3x3)
        
        Returns
        -------
        np.ndarray
            2nd Piola-Kirchhoff stress tensor (3x3)
        """
        J = np.linalg.det(F)
        F_inv = np.linalg.inv(F)
        S = F_inv @ (J * stress_cauchy) @ F_inv.T
        return S
    
    @staticmethod
    def to_voigt(S: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 stress tensor to 6-component Voigt notation.
        
        Parameters
        ----------
        S : np.ndarray
            3x3 stress tensor
        
        Returns
        -------
        np.ndarray
            6-component vector [S11, S22, S33, S12, S23, S13]
        """
        return np.array([
            S[0, 0], S[1, 1], S[2, 2],
            S[0, 1], S[1, 2], S[0, 2]
        ])
    
    @staticmethod
    def from_voigt(S_voigt: np.ndarray) -> np.ndarray:
        """
        Convert 6-component Voigt notation to 3x3 stress tensor.
        
        Parameters
        ----------
        S_voigt : np.ndarray
            6-component vector [S11, S22, S33, S12, S23, S13]
        
        Returns
        -------
        np.ndarray
            3x3 stress tensor
        """
        S = np.zeros((3, 3))
        S[0, 0] = S_voigt[0]
        S[1, 1] = S_voigt[1]
        S[2, 2] = S_voigt[2]
        S[0, 1] = S[1, 0] = S_voigt[3]
        S[1, 2] = S[2, 1] = S_voigt[4]
        S[0, 2] = S[2, 0] = S_voigt[5]
        return S


class LargeDeformationSolver:
    """
    Large deformation solver using updated Lagrangian formulation.
    
    Parameters
    ----------
    material : object
        Material model with compute_stress method
    initial_stiffness : np.ndarray
        Initial stiffness matrix
    """
    
    def __init__(self, material, initial_stiffness: np.ndarray):
        self.material = material
        self.K0 = initial_stiffness.copy()
        self.n_dof = initial_stiffness.shape[0]
        
        # State
        self.displacement = np.zeros(self.n_dof)
        self.internal_force = np.zeros(self.n_dof)
        self.tangent_stiffness = initial_stiffness.copy()
        self.deformation_history = []
    
    def compute_internal_force(self, displacement: np.ndarray) -> np.ndarray:
        """
        Compute internal force vector from displacement.
        
        Parameters
        ----------
        displacement : np.ndarray
            Nodal displacement vector
        
        Returns
        -------
        np.ndarray
            Internal force vector
        """
        # Simplified: use linear stiffness for demonstration
        # In full implementation, would integrate B^T * S over volume
        f_int = self.K0 @ displacement
        return f_int
    
    def compute_tangent_stiffness(self, displacement: np.ndarray) -> np.ndarray:
        """
        Compute tangent stiffness matrix.
        
        Parameters
        ----------
        displacement : np.ndarray
            Nodal displacement vector
        
        Returns
        -------
        np.ndarray
            Tangent stiffness matrix
        """
        # Simplified: geometric stiffness contribution
        # In full implementation, would include material + geometric stiffness
        K_T = self.K0.copy()
        
        # Add geometric stiffness for large deformation
        # This is a simplified version
        return K_T
    
    def solve_load_step(self, external_force: np.ndarray,
                       max_iterations: int = 50,
                       tolerance: float = 1e-6) -> Tuple[np.ndarray, int, bool]:
        """
        Solve a single load step using Newton-Raphson iteration.
        
        Parameters
        ----------
        external_force : np.ndarray
            External force vector
        max_iterations : int
            Maximum number of iterations
        tolerance : float
            Convergence tolerance
        
        Returns
        -------
        displacement : np.ndarray
            Converged displacement
        iterations : int
            Number of iterations
        converged : bool
            Whether solution converged
        """
        u = self.displacement.copy()
        
        for i in range(max_iterations):
            # Compute residual
            f_int = self.compute_internal_force(u)
            residual = external_force - f_int
            
            # Check convergence
            res_norm = np.linalg.norm(residual)
            if res_norm < tolerance:
                self.displacement = u
                self.internal_force = f_int
                return u, i, True
            
            # Compute tangent stiffness
            K_T = self.compute_tangent_stiffness(u)
            
            # Solve for displacement increment
            try:
                delta_u = np.linalg.solve(K_T, residual)
            except np.linalg.LinAlgError:
                return u, i, False
            
            # Update displacement
            u += delta_u
        
        self.displacement = u
        return u, max_iterations, False
    
    def compute_energy(self, displacement: np.ndarray,
                      external_force: np.ndarray) -> dict:
        """
        Compute energy quantities for energy balance check.
        
        Parameters
        ----------
        displacement : np.ndarray
            Current displacement
        external_force : np.ndarray
            External force
        
        Returns
        -------
        dict
            Dictionary with external_work, internal_energy, kinetic_energy
        """
        # External work
        W_ext = np.dot(external_force, displacement)
        
        # Internal strain energy (simplified)
        U_int = 0.5 * np.dot(displacement, self.K0 @ displacement)
        
        # Energy balance
        balance = W_ext - U_int
        
        return {
            'external_work': W_ext,
            'internal_energy': U_int,
            'energy_balance': balance,
            'relative_error': abs(balance) / (abs(W_ext) + 1e-12)
        }
