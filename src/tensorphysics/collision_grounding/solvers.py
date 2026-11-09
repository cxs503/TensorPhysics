"""
Nonlinear Solvers for Collision Analysis

Implements:
- Newton-Raphson method
- Arc-length method for post-buckling and snap-through
"""

import numpy as np
from typing import Tuple, Callable, Optional, List


class NewtonRaphsonSolver:
    """
    Newton-Raphson solver for nonlinear equations.
    
    Solves: R(u) = F_ext - F_int(u) = 0
    
    Parameters
    ----------
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum number of iterations
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 50):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.convergence_history = []
    
    def solve(self, residual_func: Callable, tangent_func: Callable,
             u0: np.ndarray) -> Tuple[np.ndarray, int, bool]:
        """
        Solve nonlinear system.
        
        Parameters
        ----------
        residual_func : callable
            Function R(u) that returns residual vector
        tangent_func : callable
            Function K_T(u) that returns tangent stiffness matrix
        u0 : np.ndarray
            Initial guess
        
        Returns
        -------
        u : np.ndarray
            Solution
        iterations : int
            Number of iterations
        converged : bool
            Whether solution converged
        """
        u = u0.copy()
        self.convergence_history = []
        
        for i in range(self.max_iterations):
            # Compute residual
            R = residual_func(u)
            R_norm = np.linalg.norm(R)
            self.convergence_history.append(R_norm)
            
            # Check convergence
            if R_norm < self.tolerance:
                return u, i, True
            
            # Compute tangent
            K_T = tangent_func(u)
            
            # Solve for increment
            try:
                delta_u = np.linalg.solve(K_T, R)
            except np.linalg.LinAlgError:
                return u, i, False
            
            # Update
            u += delta_u
        
        return u, self.max_iterations, False


class ArcLengthMethod:
    """
    Arc-length method for tracing equilibrium paths through limit points.
    
    Implements Crisfield's spherical arc-length constraint:
    Δu^T Δu + Δλ² F_ref^T F_ref = Δl²
    
    Parameters
    ----------
    tolerance : float
        Convergence tolerance
    max_iterations : int
        Maximum iterations per step
    initial_arc_length : float
        Initial arc-length parameter
    """
    
    def __init__(self, tolerance: float = 1e-6, max_iterations: int = 50,
                 initial_arc_length: float = 1.0):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        self.initial_arc_length = initial_arc_length
        
        # State
        self.u = None
        self.lam = 0.0
        self.arc_length = initial_arc_length
        self.path_history = []
    
    def initialize(self, u0: np.ndarray, lam0: float = 0.0):
        """
        Initialize arc-length method.
        
        Parameters
        ----------
        u0 : np.ndarray
            Initial displacement
        lam0 : float
            Initial load parameter
        """
        self.u = u0.copy()
        self.lam = lam0
        self.path_history = [{'u': u0.copy(), 'lam': lam0}]
    
    def solve_step(self, residual_func: Callable, tangent_func: Callable,
                  F_ref: np.ndarray) -> Tuple[np.ndarray, float, int, bool]:
        """
        Solve one arc-length step.
        
        Parameters
        ----------
        residual_func : callable
            R(u, lam) = lam * F_ref - F_int(u)
        tangent_func : callable
            K_T(u) tangent stiffness
        F_ref : np.ndarray
            Reference load vector
        
        Returns
        -------
        u : np.ndarray
            Updated displacement
        lam : float
            Updated load parameter
        iterations : int
            Number of iterations
        converged : bool
            Whether converged
        """
        if self.u is None:
            raise ValueError("Must call initialize() first")
        
        u_n = self.u.copy()
        lam_n = self.lam
        
        # Predictor step
        K_T = tangent_func(u_n)
        
        try:
            delta_u_t = np.linalg.solve(K_T, F_ref)
        except np.linalg.LinAlgError:
            return self.u, self.lam, 0, False
        
        # Initial predictor
        delta_lam = self.arc_length / np.sqrt(1.0 + np.dot(delta_u_t, delta_u_t))
        delta_u = delta_lam * delta_u_t
        
        u_trial = u_n + delta_u
        lam_trial = lam_n + delta_lam
        
        # Corrector iterations
        u = u_trial.copy()
        lam = lam_trial
        
        for i in range(self.max_iterations):
            # Compute residual
            R = residual_func(u, lam)
            R_norm = np.linalg.norm(R)
            
            if R_norm < self.tolerance:
                self.u = u
                self.lam = lam
                self.path_history.append({'u': u.copy(), 'lam': lam})
                return u, lam, i, True
            
            # Compute tangent
            K_T = tangent_func(u)
            
            # Solve for corrections
            try:
                delta_u_r = np.linalg.solve(K_T, R)
                delta_u_t = np.linalg.solve(K_T, F_ref)
            except np.linalg.LinAlgError:
                return u, lam, i, False
            
            # Arc-length constraint
            du = u - u_n
            a1 = np.dot(delta_u_t, delta_u_t)
            a2 = np.dot(du + delta_u_r, delta_u_t)
            a3 = np.dot(du + delta_u_r, du + delta_u_r) - self.arc_length**2
            
            # Solve quadratic for delta_lam
            discriminant = a2**2 - a1 * a3
            if discriminant < 0:
                delta_lam_corr = -a2 / a1
            else:
                sign = 1.0 if (lam - lam_n) >= 0 else -1.0
                delta_lam_corr = (-a2 + sign * np.sqrt(discriminant)) / a1
            
            delta_u_corr = delta_u_r + delta_lam_corr * delta_u_t
            
            # Update
            u += delta_u_corr
            lam += delta_lam_corr
        
        self.u = u
        self.lam = lam
        self.path_history.append({'u': u.copy(), 'lam': lam})
        return u, lam, self.max_iterations, False
    
    def get_equilibrium_path(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the equilibrium path.
        
        Returns
        -------
        displacements : np.ndarray
            Displacement norms along path
        load_params : np.ndarray
            Load parameters along path
        """
        displacements = np.array([np.linalg.norm(p['u']) for p in self.path_history])
        load_params = np.array([p['lam'] for p in self.path_history])
        return displacements, load_params
