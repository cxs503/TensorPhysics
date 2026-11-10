"""Cantilever beam elastica: large deflection under end load.

Exact solution via numerical quadrature of the elliptic integrals.
Verification of geometric nonlinearity by comparing with linear theory.

Physics
-------
A slender cantilever beam (length L, flexural rigidity EI) is clamped
horizontally at one end. A concentrated vertical load P is applied at the
free end. For large deflections (tip deflection > 50% of L), the linear
Euler-Bernoulli formula δ = PL³/(3EI) is invalid; the exact solution
requires solving the nonlinear elastica equation:

    EI θ''(s) + P cos θ(s) = 0

where θ(s) is the angle of the beam tangent with the horizontal at arc
length s from the clamp.

The first integral gives:
    (θ')² = (2P/EI)(sin θ_B - sin θ)

where θ_B is the angle at the free end. This leads to elliptic integrals
for the beam length and tip coordinates.

Verification approach:
1. Exact solution: numerical quadrature of the elliptic integrals
2. Compare with linear theory for small deflections (validation)
3. Demonstrate geometric nonlinearity for large deflections

Verification targets:
- Small deflection: δ/(PL³/3EI) → 1.0 as P → 0
- Large deflection: δ/L > 0.5 for θ_B > 0.8
- Load-deflection curve is monotonically increasing
"""

from __future__ import annotations

import numpy as np
from scipy import integrate, optimize
from dataclasses import dataclass
from typing import Tuple, Optional

__all__ = ["ElasticaExact", "solve_elastica_problem"]


@dataclass
class ElasticaSolution:
    """Container for elastica solution results."""
    P: float              # Applied load [N]
    L: float              # Beam length [m]
    EI: float             # Flexural rigidity [N·m²]
    theta_B: float        # Angle at free end [rad]
    x_tip: float          # Horizontal coordinate of tip [m]
    y_tip: float          # Vertical deflection of tip [m] (positive down)
    x_array: np.ndarray   # x-coordinates along beam
    y_array: np.ndarray   # y-coordinates along beam
    theta_array: np.ndarray  # angle along beam
    s_array: np.ndarray   # arc length parameter


class ElasticaExact:
    """Exact elastica solution using numerical quadrature.
    
    The cantilever beam is parameterized by the free-end angle θ_B.
    For each θ_B, we compute the corresponding load P and tip deflections
    by evaluating the elliptic integrals that arise from the first integral
    of the elastica equation.
    """
    
    def __init__(self, L: float = 1.0, EI: float = 1.0):
        self.L = L
        self.EI = EI
    
    def _compute_integrals(self, theta_B: float, theta_upper: float = None):
        """Compute the three key integrals for the elastica.
        
        Uses substitution θ = θ_upper * t² to remove the integrable
        singularity at the upper limit.
        """
        if theta_upper is None:
            theta_upper = theta_B
        
        if theta_B < 1e-12:
            return 0.0, 0.0, 0.0
        
        def integrand_I_L(t):
            theta = theta_upper * t**2
            dtheta = 2 * theta_upper * t
            val = np.sin(theta_B) - np.sin(theta)
            if val <= 1e-15:
                return 0.0
            return dtheta / np.sqrt(val)
        
        def integrand_I_x(t):
            theta = theta_upper * t**2
            dtheta = 2 * theta_upper * t
            val = np.sin(theta_B) - np.sin(theta)
            if val <= 1e-15:
                return 0.0
            return np.cos(theta) * dtheta / np.sqrt(val)
        
        def integrand_I_y(t):
            theta = theta_upper * t**2
            dtheta = 2 * theta_upper * t
            val = np.sin(theta_B) - np.sin(theta)
            if val <= 1e-15:
                return 0.0
            return np.sin(theta) * dtheta / np.sqrt(val)
        
        # Gauss-Legendre quadrature (100 points for accuracy)
        n_quad = 100
        xi, wi = np.polynomial.legendre.leggauss(n_quad)
        t_pts = 0.5 * (xi + 1)
        w_pts = 0.5 * wi
        
        I_L = np.sum(w_pts * [integrand_I_L(t) for t in t_pts])
        I_x = np.sum(w_pts * [integrand_I_x(t) for t in t_pts])
        I_y = np.sum(w_pts * [integrand_I_y(t) for t in t_pts])
        
        return I_L, I_x, I_y
    
    def compute_from_theta_B(self, theta_B: float) -> ElasticaSolution:
        """Compute exact solution for a given free-end angle θ_B.
        
        Parameters
        ----------
        theta_B : float
            Angle at the free end [rad], measured from horizontal.
        
        Returns
        -------
        ElasticaSolution
            Complete solution including load, deflections, and shape.
        """
        if theta_B < 1e-10:
            return ElasticaSolution(
                P=0.0, L=self.L, EI=self.EI, theta_B=0.0,
                x_tip=self.L, y_tip=0.0,
                x_array=np.linspace(0, self.L, 50),
                y_array=np.zeros(50),
                theta_array=np.zeros(50),
                s_array=np.linspace(0, self.L, 50),
            )
        
        I_L, I_x, I_y = self._compute_integrals(theta_B)
        
        # From I_L: sqrt(2P/EI) * L = I_L → P = EI * I_L² / (2 L²)
        P = self.EI * I_L**2 / (2.0 * self.L**2)
        
        # Tip coordinates
        x_tip = self.L * I_x / I_L
        y_tip = self.L * I_y / I_L
        
        # Generate the deformed shape
        n_pts = 100
        theta_vals = np.linspace(0, theta_B, n_pts)
        x_arr = np.zeros(n_pts)
        y_arr = np.zeros(n_pts)
        s_arr = np.zeros(n_pts)
        
        for i in range(1, n_pts):
            th_i = theta_vals[i]
            il, ix, iy = self._compute_integrals(theta_B, th_i)
            s_arr[i] = self.L * il / I_L
            x_arr[i] = self.L * ix / I_L
            y_arr[i] = self.L * iy / I_L
        
        return ElasticaSolution(
            P=P, L=self.L, EI=self.EI, theta_B=theta_B,
            x_tip=x_tip, y_tip=y_tip,
            x_array=x_arr, y_array=y_arr,
            theta_array=theta_vals,
            s_array=s_arr,
        )
    
    def compute_load_deflection_curve(
        self, n_points: int = 20, max_theta_B: float = 1.4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the load-deflection curve P*L²/(EI) vs δ/L.
        
        Returns
        -------
        P_normalized : ndarray
            Normalized load P*L²/(EI).
        delta_over_L : ndarray
            Normalized tip deflection δ/L.
        theta_B_array : ndarray
            Free-end angles.
        """
        theta_B_array = np.linspace(0.05, max_theta_B, n_points)
        P_norm = np.zeros(n_points)
        delta_L = np.zeros(n_points)
        
        for i, tb in enumerate(theta_B_array):
            sol = self.compute_from_theta_B(tb)
            P_norm[i] = sol.P * self.L**2 / self.EI
            delta_L[i] = sol.y_tip / self.L
        
        return P_norm, delta_L, theta_B_array
    
    def compute_shape_at_load(self, P_target: float) -> ElasticaSolution:
        """Find the exact solution for a given load P."""
        def residual(theta_B):
            if theta_B < 1e-10:
                return -P_target
            I_L, _, _ = self._compute_integrals(theta_B)
            P = self.EI * I_L**2 / (2.0 * self.L**2)
            return P - P_target
        
        theta_B_init = min(np.sqrt(3.0 * P_target * self.L**2 / self.EI), 1.5)
        theta_B_init = max(theta_B_init, 0.05)
        
        try:
            theta_B_sol = optimize.brentq(residual, 1e-6, min(3.0, 10.0 * theta_B_init))
        except ValueError:
            res = optimize.minimize_scalar(lambda t: residual(t)**2, 
                                          bounds=(1e-6, 3.0), method='bounded')
            theta_B_sol = res.x
        
        return self.compute_from_theta_B(theta_B_sol)
    
    def verify_linear_limit(self) -> dict:
        """Verify that the exact solution matches linear theory for small deflections.
        
        For small deflections: δ = PL³/(3EI)
        So δ/(PL³/3EI) should approach 1.0 as P → 0.
        
        Returns
        -------
        dict with verification results.
        """
        # Test at very small deflection
        theta_B_small = 0.01
        sol = self.compute_from_theta_B(theta_B_small)
        
        P = sol.P
        delta_exact = sol.y_tip
        delta_linear = P * self.L**3 / (3 * self.EI)
        
        ratio = delta_exact / delta_linear
        
        return {
            'theta_B': theta_B_small,
            'P': P,
            'delta_exact': delta_exact,
            'delta_linear': delta_linear,
            'ratio': ratio,
            'error_pct': abs(ratio - 1.0) * 100,
        }


# Alias for backward compatibility
ElasticaFEM = ElasticaExact


def solve_elastica_problem(
    L: float = 1.0, EI: float = 1.0, 
    n_points: int = 20,
    max_theta_B: float = 1.4,
) -> dict:
    """Solve the elastica problem for a range of deflections.
    
    Returns
    -------
    dict with:
        - P_values: array of loads
        - P_normalized: P*L²/(EI)
        - delta_over_L: δ/L
        - theta_B_array: free-end angles
        - linear_verification: results of linear limit check
    """
    exact = ElasticaExact(L=L, EI=EI)
    
    P_norm, delta_L, theta_B = exact.compute_load_deflection_curve(
        n_points=n_points, max_theta_B=max_theta_B
    )
    
    P_values = P_norm * EI / L**2
    
    linear_check = exact.verify_linear_limit()
    
    return {
        'P_values': P_values,
        'P_normalized': P_norm,
        'delta_over_L': delta_L,
        'theta_B_array': theta_B,
        'linear_verification': linear_check,
    }
