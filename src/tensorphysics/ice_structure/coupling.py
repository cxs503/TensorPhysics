"""Partitioned coupling schemes for ice-structure interaction.

Implements:
- Loose (one-way) coupling: no sub-iterations
- Tight (iterative) coupling with fixed relaxation
- Aitken dynamic relaxation for enhanced stability

The key challenge is the added-mass effect: when the fluid/ice domain
has significant added mass relative to the structure, partitioned schemes
can become unstable without proper relaxation.

The instability mechanism:
- Structure solves: (m_s + m_a) * a = F_ext + F_ice
- Ice solves: F_ice = -m_a * a (reaction to structural acceleration)
- In partitioned scheme, a is lagged, creating a feedback loop
- Spectral radius of this loop: rho ≈ |1 - omega*(1+mu)| for fixed relaxation
- Stability requires rho < 1, hence mu < (2-omega)/omega
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple
from enum import Enum


class CouplingScheme(Enum):
    """Coupling scheme types."""
    LOOSE = "loose"          # One-way, no sub-iterations
    TIGHT_FIXED = "tight_fixed"  # Iterative with fixed relaxation
    TIGHT_AITKEN = "tight_aitken"  # Iterative with Aitken relaxation


@dataclass
class CouplingResult:
    """Result from a partitioned coupling simulation.
    
    Attributes
    ----------
    t : ndarray
        Time array [s]
    x_s : ndarray
        Structural displacement history [m]
    v_s : ndarray
        Structural velocity history [m/s]
    F_ice : ndarray
        Ice force history [N]
    iterations : ndarray
        Number of sub-iterations per coupling step
    converged : bool
        Whether the coupling converged
    diverged : bool
        Whether the solution diverged
    spectral_radius : float
        Estimated spectral radius of the iteration operator
    scheme : str
        Coupling scheme used
    omega : float
        Relaxation factor used
    """
    t: np.ndarray
    x_s: np.ndarray
    v_s: np.ndarray
    F_ice: np.ndarray
    iterations: np.ndarray
    converged: bool
    diverged: bool
    spectral_radius: float
    scheme: str
    omega: float
    
    @property
    def max_iterations(self) -> int:
        """Maximum iterations in any coupling step."""
        return int(np.max(self.iterations)) if len(self.iterations) > 0 else 0
    
    @property
    def mean_iterations(self) -> float:
        """Mean iterations per coupling step."""
        return float(np.mean(self.iterations)) if len(self.iterations) > 0 else 0.0


@dataclass
class PartitionedCoupling:
    """Partitioned coupling solver for ice-structure interaction.
    
    This implements a simplified model that captures the essential
    added-mass instability mechanism:
    
    Structure equation:
        m_s * a_s = F_ext - k_s * x_s - c_s * v_s + F_ice
    
    Ice force (added mass reaction):
        F_ice = -m_a * a_s_lagged
    
    In a partitioned scheme, a_s is lagged from the previous iteration,
    creating a fixed-point iteration with spectral radius that depends
    on mu = m_a / m_s.
    
    Parameters
    ----------
    m_s : float
        Structural mass [kg]
    k_s : float
        Structural stiffness [N/m]
    c_s : float
        Structural damping [N·s/m]
    m_a : float
        Added mass from ice [kg]
    scheme : CouplingScheme
        Coupling scheme to use
    omega : float
        Fixed relaxation factor (for TIGHT_FIXED)
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum sub-iterations per coupling step
    """
    m_s: float
    k_s: float
    c_s: float
    m_a: float = 0.0
    scheme: CouplingScheme = CouplingScheme.LOOSE
    omega: float = 1.0
    tol: float = 1e-8
    max_iter: int = 50
    
    @property
    def mu(self) -> float:
        """Added mass ratio."""
        return self.m_a / self.m_s if self.m_s > 0 else 0.0
    
    def run(self, t_end: float, dt: float, 
            x0: float = 0.0, v0: float = 1.0) -> CouplingResult:
        """Run the partitioned coupling simulation.
        
        This implements a simplified fixed-point iteration that captures
        the added-mass instability:
        
        At each time step:
        1. Ice solver: compute F_ice = -m_a * a_lagged
        2. Structure solver: compute a = (F_ext - k*x - c*v + F_ice) / m_s
        3. Update: x, v using Newmark
        4. For tight coupling: iterate until convergence
        
        The instability arises because F_ice depends on a, which depends
        on F_ice, creating a feedback loop with gain mu = m_a/m_s.
        
        Parameters
        ----------
        t_end : float
            End time [s]
        dt : float
            Time step [s]
        x0 : float
            Initial displacement [m]
        v0 : float
            Initial velocity [m/s]
            
        Returns
        -------
        CouplingResult
            Simulation results
        """
        n_steps = int(t_end / dt)
        t = np.zeros(n_steps + 1)
        x_s = np.zeros(n_steps + 1)
        v_s = np.zeros(n_steps + 1)
        F_ice = np.zeros(n_steps + 1)
        iterations = np.zeros(n_steps + 1, dtype=int)
        
        # Initial state
        x = x0
        v = v0
        # Initial acceleration (without ice force for simplicity)
        a = (-self.k_s * x - self.c_s * v) / self.m_s
        
        x_s[0] = x
        v_s[0] = v
        
        diverged = False
        converged = True
        # Per-step spectral radius estimates
        step_spectral_radii = []
        
        # Newmark parameters (unconditionally stable)
        gamma = 0.5
        beta = 0.25
        
        for n in range(n_steps):
            t[n+1] = (n + 1) * dt
            
            # External force (impulse at start, then zero)
            F_ext = 0.0
            
            if self.scheme == CouplingScheme.LOOSE:
                # One-way coupling: use lagged acceleration
                F_ice_val = -self.m_a * a
                # Structure solve
                a_new = (F_ext - self.k_s * x - self.c_s * v + F_ice_val) / self.m_s
                # Newmark update
                x_new = x + dt * v + 0.5 * dt**2 * ((1 - 2*beta) * a + 2*beta * a_new)
                v_new = v + dt * ((1 - gamma) * a + gamma * a_new)
                
                x, v, a = x_new, v_new, a_new
                iterations[n+1] = 1
                step_spectral_radii.append(0.0)  # No iteration, rho=0
                
            else:
                # Iterative coupling
                x_old, v_old, a_old = x, v, a
                
                # Predictor (initial guess)
                x_pred = x_old + dt * v_old + 0.5 * dt**2 * a_old
                v_pred = v_old + dt * a_old
                
                x_trial = x_pred
                n_it = 0
                a_lagged = a_old  # Start with lagged acceleration
                
                # Per-step residual history for spectral radius estimation
                step_residuals = []
                
                for it in range(self.max_iter):
                    n_it = it + 1
                    
                    # Ice solver: force from lagged acceleration
                    F_ice_trial = -self.m_a * a_lagged
                    
                    # Structure solver
                    a_trial = (F_ext - self.k_s * x_trial - self.c_s * v_pred + F_ice_trial) / self.m_s
                    
                    # Newmark: compute x from a
                    x_new = x_old + dt * v_old + dt**2 * ((0.5 - beta) * a_old + beta * a_trial)
                    
                    # Residual
                    resid = x_new - x_trial
                    resid_norm = abs(resid)
                    step_residuals.append(resid_norm)
                    
                    # Check convergence
                    if resid_norm < self.tol * max(abs(x_trial), 1e-14):
                        break
                    
                    # Update with relaxation
                    if self.scheme == CouplingScheme.TIGHT_FIXED:
                        x_trial = x_trial + self.omega * resid
                    else:  # AITKEN
                        # Simple adaptive relaxation
                        if len(step_residuals) >= 2:
                            r_prev = step_residuals[-2]
                            r_curr = step_residuals[-1]
                            delta_r = r_curr - r_prev
                            if abs(delta_r) > 1e-14:
                                omega_ait = -self.omega * r_prev / delta_r
                                omega_ait = np.clip(omega_ait, 0.05, 1.5)
                                x_trial = x_trial + omega_ait * resid
                            else:
                                x_trial = x_trial + self.omega * resid
                        else:
                            x_trial = x_trial + self.omega * resid
                    
                    # Update lagged acceleration for next iteration
                    v_trial = v_old + dt * ((1 - gamma) * a_old + gamma * a_trial)
                    a_lagged = a_trial
                
                iterations[n+1] = n_it
                
                if n_it >= self.max_iter:
                    converged = False
                
                # Estimate spectral radius for this step
                step_rho = self._estimate_spectral_radius_from_step(step_residuals)
                step_spectral_radii.append(step_rho)
                
                # Final update
                x = x_trial
                v = v_old + dt * ((1 - gamma) * a_old + gamma * a_trial)
                a = a_trial
            
            x_s[n+1] = x
            v_s[n+1] = v
            F_ice[n+1] = -self.m_a * a
            
            # Check for divergence
            if not np.isfinite(x) or abs(x) > 1e10:
                diverged = True
                converged = False
                break
        
        # Overall spectral radius: max across all steps
        spectral_radius = max(step_spectral_radii) if step_spectral_radii else 0.0
        
        return CouplingResult(
            t=t,
            x_s=x_s,
            v_s=v_s,
            F_ice=F_ice,
            iterations=iterations,
            converged=converged,
            diverged=diverged,
            spectral_radius=spectral_radius,
            scheme=self.scheme.value,
            omega=self.omega,
        )
    
    @staticmethod
    def _estimate_spectral_radius_from_step(residuals: List[float]) -> float:
        """Estimate spectral radius from residuals within a single coupling step.
        
        Uses the ratio of consecutive residuals to estimate the
        asymptotic convergence rate.
        
        Parameters
        ----------
        residuals : list
            Residual norms from consecutive iterations within one step
            
        Returns
        -------
        float
            Estimated spectral radius (0 if not enough data)
        """
        if len(residuals) < 2:
            return 0.0
        
        # Filter out zero/tiny residuals
        non_zero = [r for r in residuals if r > 1e-14]
        if len(non_zero) < 2:
            return 0.0
        
        # Compute ratios of consecutive residuals
        ratios = []
        for i in range(1, len(non_zero)):
            if non_zero[i-1] > 1e-14:
                ratios.append(non_zero[i] / non_zero[i-1])
        
        if len(ratios) == 0:
            return 0.0
        
        # Use median to be robust to outliers
        return float(np.median(ratios))


def run_partitioned_coupling(
    m_s: float,
    k_s: float,
    c_s: float,
    m_a: float,
    scheme: str = "loose",
    omega: float = 1.0,
    t_end: float = 1.0,
    dt: float = 0.001,
    x0: float = 0.0,
    v0: float = 1.0,
    tol: float = 1e-8,
    max_iter: int = 50,
) -> CouplingResult:
    """Convenience function to run partitioned coupling.
    
    Parameters
    ----------
    m_s : float
        Structural mass [kg]
    k_s : float
        Structural stiffness [N/m]
    c_s : float
        Structural damping [N·s/m]
    m_a : float
        Added mass [kg]
    scheme : str
        Coupling scheme: "loose", "tight_fixed", or "tight_aitken"
    omega : float
        Relaxation factor
    t_end : float
        End time [s]
    dt : float
        Time step [s]
    x0 : float
        Initial displacement [m]
    v0 : float
        Initial velocity [m/s]
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum sub-iterations
        
    Returns
    -------
    CouplingResult
        Simulation results
    """
    coupling = PartitionedCoupling(
        m_s=m_s,
        k_s=k_s,
        c_s=c_s,
        m_a=m_a,
        scheme=CouplingScheme(scheme),
        omega=omega,
        tol=tol,
        max_iter=max_iter,
    )
    return coupling.run(t_end, dt, x0, v0)
