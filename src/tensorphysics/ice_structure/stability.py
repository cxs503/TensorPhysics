"""Stability analysis for partitioned ice-structure coupling.

Provides tools to:
- Scan the stability domain (mu vs omega)
- Compute analytical spectral radius
- Determine critical added mass ratio
- Generate stability maps

The key result is that for a partitioned coupling with relaxation factor omega,
the spectral radius of the fixed-point iteration is:

    rho = |1 - omega * (1 + mu)|

where mu = m_a / m_s is the added mass ratio.

The stability condition rho < 1 gives:
    mu < (2 - omega) / omega = mu_c

For omega = 1 (no relaxation): mu_c = 1
For omega = 0.5: mu_c = 3
For omega → 0: mu_c → ∞ (but convergence is slow)
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
import json


@dataclass
class StabilityResult:
    """Result from stability analysis at a single (mu, omega) point.
    
    Attributes
    ----------
    mu : float
        Added mass ratio
    omega : float
        Relaxation factor
    spectral_radius : float
        Spectral radius of iteration operator
    converged : bool
        Whether the coupling converged
    iterations : int
        Number of iterations to converge (or max_iter if diverged)
    stable : bool
        Whether the scheme is stable (rho < 1)
    """
    mu: float
    omega: float
    spectral_radius: float
    converged: bool
    iterations: int
    stable: bool
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "mu": self.mu,
            "omega": self.omega,
            "spectral_radius": self.spectral_radius,
            "converged": self.converged,
            "iterations": self.iterations,
            "stable": self.stable,
        }


@dataclass
class StabilityAnalyzer:
    """Analyzer for partitioned coupling stability.
    
    Parameters
    ----------
    m_s : float
        Structural mass [kg]
    k_s : float
        Structural stiffness [N/m]
    c_s : float
        Structural damping [N·s/m]
    dt : float
        Time step [s]
    t_end : float
        Simulation time for stability test [s]
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations per coupling step
    """
    m_s: float = 1.0
    k_s: float = 100.0
    c_s: float = 1.0
    dt: float = 0.001
    t_end: float = 0.1
    tol: float = 1e-8
    max_iter: int = 100
    
    def test_stability(self, mu: float, omega: float, 
                       scheme: str = "tight_fixed") -> StabilityResult:
        """Test stability at a single (mu, omega) point.
        
        Parameters
        ----------
        mu : float
            Added mass ratio
        omega : float
            Relaxation factor
        scheme : str
            Coupling scheme
            
        Returns
        -------
        StabilityResult
            Stability test result
        """
        from .coupling import PartitionedCoupling, CouplingScheme
        
        m_a = mu * self.m_s
        
        coupling = PartitionedCoupling(
            m_s=self.m_s,
            k_s=self.k_s,
            c_s=self.c_s,
            m_a=m_a,
            scheme=CouplingScheme(scheme),
            omega=omega,
            tol=self.tol,
            max_iter=self.max_iter,
        )
        
        result = coupling.run(self.t_end, self.dt, x0=0.0, v0=1.0)
        
        return StabilityResult(
            mu=mu,
            omega=omega,
            spectral_radius=result.spectral_radius,
            converged=result.converged,
            iterations=result.max_iterations,
            stable=result.converged and not result.diverged,
        )
    
    def scan_1d(self, mu_values: np.ndarray, omega: float = 1.0,
                scheme: str = "tight_fixed") -> List[StabilityResult]:
        """Scan stability along mu axis at fixed omega.
        
        Parameters
        ----------
        mu_values : ndarray
            Array of mu values to test
        omega : float
            Fixed relaxation factor
        scheme : str
            Coupling scheme
            
        Returns
        -------
        list
            List of StabilityResult for each mu
        """
        results = []
        for mu in mu_values:
            result = self.test_stability(mu, omega, scheme)
            results.append(result)
        return results
    
    def scan_2d(self, mu_values: np.ndarray, omega_values: np.ndarray,
                scheme: str = "tight_fixed") -> List[List[StabilityResult]]:
        """Scan stability in 2D (mu, omega) space.
        
        Parameters
        ----------
        mu_values : ndarray
            Array of mu values
        omega_values : ndarray
            Array of omega values
        scheme : str
            Coupling scheme
            
        Returns
        -------
        list of list
            2D array of StabilityResult
        """
        results = []
        for omega in omega_values:
            row = []
            for mu in mu_values:
                result = self.test_stability(mu, omega, scheme)
                row.append(result)
            results.append(row)
        return results
    
    def find_critical_mu(self, omega: float, tol: float = 0.01,
                         scheme: str = "tight_fixed") -> float:
        """Find critical added mass ratio using bisection.
        
        Parameters
        ----------
        omega : float
            Relaxation factor
        tol : float
            Bisection tolerance
        scheme : str
            Coupling scheme
            
        Returns
        -------
        float
            Critical mu (mu_c)
        """
        # Start with bounds
        mu_low = 0.0
        mu_high = 10.0
        
        # Find upper bound where it diverges
        while self.test_stability(mu_high, omega, scheme).stable:
            mu_high *= 2.0
            if mu_high > 1000:
                return mu_high  # Very stable
        
        # Bisection
        while (mu_high - mu_low) > tol:
            mu_mid = 0.5 * (mu_low + mu_high)
            if self.test_stability(mu_mid, omega, scheme).stable:
                mu_low = mu_mid
            else:
                mu_high = mu_mid
        
        return 0.5 * (mu_low + mu_high)


def compute_analytical_spectral_radius(mu: float, omega: float) -> float:
    """Compute analytical spectral radius for partitioned coupling.
    
    For a partitioned scheme with relaxation omega and added mass ratio mu,
    the fixed-point iteration has spectral radius:
    
        rho = |1 - omega * (1 + mu)|
    
    Derivation:
    - Structure: m_s * a = F - m_a * a_lagged
    - Fixed-point: a_new = C - mu * a_lagged (where C = F/m_s)
    - With relaxation: a_new = (1-omega)*a_lagged + omega*(C - mu*a_lagged)
    - Spectral radius: |d(a_new)/d(a_lagged)| = |1 - omega - omega*mu| = |1 - omega*(1+mu)|
    
    Stability condition rho < 1 gives:
        mu < (2 - omega) / omega
    
    Parameters
    ----------
    mu : float
        Added mass ratio
    omega : float
        Relaxation factor
        
    Returns
    -------
    float
        Spectral radius
    """
    if omega <= 0:
        return 1.0
    
    return abs(1.0 - omega * (1.0 + mu))


def compute_critical_mu(omega: float) -> float:
    """Compute analytical critical added mass ratio.
    
    The critical mu where the scheme becomes unstable is:
        mu_c = (2 - omega) / omega
    
    This comes from setting rho = 1:
        |1 - omega*(1+mu)| = 1
        1 - omega*(1+mu) = -1  (for the unstable branch)
        omega*(1+mu) = 2
        mu = (2-omega)/omega
    
    Parameters
    ----------
    omega : float
        Relaxation factor
        
    Returns
    -------
    float
        Critical added mass ratio
    """
    if omega <= 0:
        return float('inf')
    return (2.0 - omega) / omega


def scan_stability_domain(
    mu_range: Tuple[float, float] = (0.0, 5.0),
    omega_range: Tuple[float, float] = (0.1, 1.5),
    n_mu: int = 20,
    n_omega: int = 15,
    use_analytical: bool = True,
    analyzer: Optional[StabilityAnalyzer] = None,
) -> Dict[str, Any]:
    """Scan the full stability domain.
    
    Parameters
    ----------
    mu_range : tuple
        (mu_min, mu_max)
    omega_range : tuple
        (omega_min, omega_max)
    n_mu : int
        Number of mu points
    n_omega : int
        Number of omega points
    use_analytical : bool
        If True, use analytical formula; otherwise run simulations
    analyzer : StabilityAnalyzer, optional
        Analyzer for numerical tests
        
    Returns
    -------
    dict
        Stability domain data with keys:
        - 'mu': array of mu values
        - 'omega': array of omega values
        - 'spectral_radius': 2D array of spectral radii
        - 'stable': 2D boolean array
        - 'critical_curve': analytical critical mu vs omega
    """
    mu_values = np.linspace(mu_range[0], mu_range[1], n_mu)
    omega_values = np.linspace(omega_range[0], omega_range[1], n_omega)
    
    if use_analytical:
        # Use analytical formula
        MU, OMEGA = np.meshgrid(mu_values, omega_values)
        RHO = np.zeros_like(MU)
        STABLE = np.zeros_like(MU, dtype=bool)
        
        for i in range(n_omega):
            for j in range(n_mu):
                RHO[i, j] = compute_analytical_spectral_radius(MU[i, j], OMEGA[i, j])
                STABLE[i, j] = RHO[i, j] < 1.0
        
    else:
        # Run numerical tests
        if analyzer is None:
            analyzer = StabilityAnalyzer()
        
        RHO = np.zeros((n_omega, n_mu))
        STABLE = np.zeros((n_omega, n_mu), dtype=bool)
        
        for i, omega in enumerate(omega_values):
            for j, mu in enumerate(mu_values):
                result = analyzer.test_stability(mu, omega)
                RHO[i, j] = result.spectral_radius
                STABLE[i, j] = result.stable
    
    # Analytical critical curve
    mu_critical = [compute_critical_mu(omega) for omega in omega_values]
    
    return {
        'mu': mu_values,
        'omega': omega_values,
        'spectral_radius': RHO,
        'stable': STABLE,
        'critical_curve': mu_critical,
    }


def generate_stability_report(
    stability_data: Dict[str, Any],
    output_path: str = "stability_report.json"
) -> None:
    """Generate a JSON report of stability analysis.
    
    Parameters
    ----------
    stability_data : dict
        Output from scan_stability_domain
    output_path : str
        Path to save the report
    """
    report = {
        "title": "Ice-Structure Partitioned Coupling Stability Analysis",
        "mu_range": [float(stability_data['mu'][0]), float(stability_data['mu'][-1])],
        "omega_range": [float(stability_data['omega'][0]), float(stability_data['omega'][-1])],
        "n_mu": len(stability_data['mu']),
        "n_omega": len(stability_data['omega']),
        "stability_boundary": {
            "omega": stability_data['omega'].tolist(),
            "mu_critical": stability_data['critical_curve'],
        },
        "summary": {
            "max_stable_mu": float(np.max(stability_data['mu'][stability_data['stable']])),
            "analytical_formula": "mu_c = (2 - omega) / omega",
        },
    }
    
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2)
