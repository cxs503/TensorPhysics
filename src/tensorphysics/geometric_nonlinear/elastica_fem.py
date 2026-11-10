"""Elastica FEM solver using discrete elastica model.

Uses n rigid links connected by rotational springs.
Correctly implements the equilibrium for a cantilever with vertical tip load.

Reference: Frisch-Fay (1962). Flexible Bars.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

__all__ = ["ElasticaFEMSolver", "solve_elastica_fem"]


@dataclass
class ElasticaFEMResult:
    """Container for FEM elastica solution."""
    P: float
    nodes: np.ndarray
    tip_deflection: float
    strain_energy: float
    external_work: float
    converged: bool
    iterations: int


class ElasticaFEMSolver:
    """Discrete elastica: n rigid links with rotational springs.
    
    The beam is modeled as n rigid links of length l = L/n.
    Link i makes angle θ_i with the horizontal.
    Rotational spring at joint i has stiffness k = EI/l.
    
    For a cantilever clamped at the base (θ_0 = 0) with vertical tip load P:
    The bending moment at joint i (between link i and i+1) is:
    M_i = P * (horizontal distance from joint i to tip)
        = P * Σ_{j=i}^{n-1} l*cos(θ_j)
    
    The spring moment is: M_i = k*(θ_i - θ_{i-1})
    
    Equilibrium: k*(θ_i - θ_{i-1}) = P * Σ_{j=i}^{n-1} l*cos(θ_j)
    """
    
    def __init__(self, L: float, EI: float, n_elem: int = 20):
        self.L = L
        self.EI = EI
        self.n_elem = n_elem
        self.l = L / n_elem
        self.k = EI / self.l
    
    def _residual(self, theta: np.ndarray, P: float) -> np.ndarray:
        """Residual of equilibrium equations.
        
        theta: array of n angles [θ_1, θ_2, ..., θ_n]
        θ_0 = 0 (clamped base)
        
        For joint i (i=1,...,n):
        k*(θ_i - θ_{i-1}) - P*l*Σ_{j=i}^{n} cos(θ_j) = 0
        
        Note: θ_n is the angle of the last link, and the sum includes it.
        """
        n = self.n_elem
        k = self.k
        l = self.l
        
        # Full angle array including θ_0 = 0
        theta_full = np.concatenate([[0.0], theta])
        
        residual = np.zeros(n)
        for i in range(1, n+1):
            # Spring moment
            M_spring = k * (theta_full[i] - theta_full[i-1])
            
            # Moment from load: P * (horizontal distance to tip)
            M_load = P * l * np.sum(np.cos(theta_full[i:]))
            
            residual[i-1] = M_spring - M_load
        
        return residual
    
    def _jacobian(self, theta: np.ndarray, P: float) -> np.ndarray:
        """Jacobian of residual."""
        n = self.n_elem
        k = self.k
        l = self.l
        
        theta_full = np.concatenate([[0.0], theta])
        
        J = np.zeros((n, n))
        for i in range(1, n+1):
            # ∂residual[i-1]/∂theta[j-1]
            # residual[i-1] = k*(θ_i - θ_{i-1}) - P*l*Σ_{j=i}^{n} cos(θ_j)
            
            # Diagonal: ∂/∂θ_i
            J[i-1, i-1] = k + P * l * np.sin(theta_full[i])
            
            # Off-diagonal: ∂/∂θ_j for j < i (from the spring term)
            if i > 1:
                J[i-1, i-2] = -k
            
            # Off-diagonal: ∂/∂θ_j for j >= i (from the load sum)
            for j in range(i, n):
                J[i-1, j] = P * l * np.sin(theta_full[j+1])
        
        return J
    
    def solve(self, P_target: float, max_iter: int = 100, tol: float = 1e-10) -> ElasticaFEMResult:
        """Solve for a given tip load P."""
        n = self.n_elem
        
        # Initial guess: linear solution θ(s) = P*(L-s)²/(2*EI)
        theta = np.zeros(n)
        for i in range(n):
            s = (i + 1) * self.l  # position of joint i+1
            theta[i] = P_target * (self.L - s)**2 / (2 * self.EI)
        theta = np.clip(theta, 0, 1.5)
        
        converged = False
        n_iter = 0
        
        for iteration in range(max_iter):
            n_iter = iteration + 1
            
            R = self._residual(theta, P_target)
            res_norm = np.linalg.norm(R)
            
            if res_norm < tol:
                converged = True
                break
            
            J = self._jacobian(theta, P_target)
            
            try:
                delta_theta = np.linalg.solve(J, -R)
            except np.linalg.LinAlgError:
                break
            
            # Line search
            alpha = 1.0
            theta_new = theta + alpha * delta_theta
            for _ in range(20):
                R_new = self._residual(theta_new, P_target)
                if np.linalg.norm(R_new) < res_norm:
                    break
                alpha *= 0.5
                theta_new = theta + alpha * delta_theta
            
            theta = theta_new
        
        # Compute node positions
        nodes = np.zeros((n+1, 2))
        theta_full = np.concatenate([[0.0], theta])
        for i in range(n):
            nodes[i+1, 0] = nodes[i, 0] + self.l * np.cos(theta_full[i+1])
            nodes[i+1, 1] = nodes[i, 1] + self.l * np.sin(theta_full[i+1])
        
        tip_deflection = nodes[-1, 1]
        
        # Strain energy: U = Σ 1/2 * k * (Δθ)²
        strain_energy = 0.0
        for i in range(n):
            dtheta = theta_full[i+1] - theta_full[i]
            strain_energy += 0.5 * self.k * dtheta**2
        
        external_work = P_target * tip_deflection
        
        return ElasticaFEMResult(
            P=P_target,
            nodes=nodes,
            tip_deflection=tip_deflection,
            strain_energy=strain_energy,
            external_work=external_work,
            converged=converged,
            iterations=n_iter,
        )


def solve_elastica_fem(
    L: float = 1.0, EI: float = 1.0,
    P_values: np.ndarray = None,
    n_elem: int = 20,
) -> dict:
    """Solve elastica for multiple loads."""
    if P_values is None:
        P_values = np.array([0.5, 1.0, 2.0, 3.5, 6.0])
    
    solver = ElasticaFEMSolver(L=L, EI=EI, n_elem=n_elem)
    
    results = []
    for P in P_values:
        result = solver.solve(P)
        results.append(result)
    
    tip_deflections = np.array([r.tip_deflection for r in results])
    
    return {
        'P_values': P_values,
        'tip_deflections': tip_deflections,
        'results': results,
        'n_elem': n_elem,
    }
