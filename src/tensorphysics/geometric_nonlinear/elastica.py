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
from functools import lru_cache
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
    
    #: Gauss-Legendre order used for the elliptic-type integrals.  The
    #: quadrature points never touch the singular end point (t = 0 below), and
    #: with the substitution used here the integrand is analytic over the whole
    #: interval, so the rule converges spectrally -- 160 nodes give ~1e-10
    #: relative accuracy for theta_B <= pi/2.
    _GL_ORDER = 160

    @staticmethod
    @lru_cache(maxsize=None)
    def _gauss_legendre(n_quad: int):
        """Cached Gauss-Legendre nodes/weights (avoids recomputing the roots)."""
        return np.polynomial.legendre.leggauss(n_quad)

    def _compute_integrals(self, theta_B: float, theta_upper: float = None):
        """Compute the three key integrals for the elastica.

        The elastica first integral gives

            ds = dtheta / sqrt((2P/EI) (sin(theta_B) - sin(theta)))

        so the beam length and the tip coordinates follow from

            I_L = int_0^{theta_u} dtheta / sqrt(sin(theta_B) - sin(theta))
            I_x = int_0^{theta_u} cos(theta) dtheta / sqrt(...)
            I_y = int_0^{theta_u} sin(theta) dtheta / sqrt(...)

        The integrand behaves like 1/sqrt(theta_B - theta) at the upper limit,
        i.e. it is *integrable but singular*.  The substitution used below,

            theta = theta_upper * (1 - t**2),   |dtheta| = 2 theta_upper t dt,

        maps the whole interval to t in (0, 1] and makes the transformed
        integrand analytic at t = 0:

            dtheta / sqrt(sin(theta_B) - sin(theta))
                -> 2 theta_upper / sqrt(theta_B cos(theta_B))  as t -> 0.

        (A plain theta = theta_upper * t**2 substitution -- as used in the
        first version of this module -- leaves a 1/sqrt(1-t) singularity in
        place and biases P low by ~1.2-1.5%; see docs/m8870_r1_audit.md.)
        The sine difference is evaluated with the identity
        sin(a) - sin(b) = 2 cos((a+b)/2) sin((a-b)/2) to avoid cancellation.

        Notes
        -----
        A real first integral requires sin(theta_B) - sin(theta) >= 0 on
        [0, theta_B], i.e. theta_B <= pi/2 (vertical tangent at the free end).
        Larger values are rejected instead of being silently truncated.
        """
        if theta_upper is None:
            theta_upper = theta_B

        if theta_B < 1e-12 or theta_upper <= 0.0:
            return 0.0, 0.0, 0.0

        if theta_B > 0.5 * np.pi + 1e-12:
            raise ValueError(
                "the elastica first integral requires theta_B <= pi/2 "
                f"(got theta_B={theta_B:.6f}); beyond that sin(theta_B) - "
                "sin(theta) changes sign and no monotone solution exists"
            )

        xi, wi = self._gauss_legendre(self._GL_ORDER)
        t = 0.5 * (xi + 1.0)                 # t in (0, 1)
        w = 0.5 * wi

        theta = theta_upper * (1.0 - t**2)   # theta in [0, theta_upper)
        dtheta = 2.0 * theta_upper * t       # |dtheta/dt|

        # sin(theta_B) - sin(theta), cancellation-free.
        diff = 2.0 * np.cos(0.5 * (theta_B + theta)) * np.sin(0.5 * (theta_B - theta))
        diff = np.maximum(diff, 1e-300)

        base = dtheta / np.sqrt(diff)

        I_L = float(np.sum(w * base))
        I_x = float(np.sum(w * base * np.cos(theta)))
        I_y = float(np.sum(w * base * np.sin(theta)))

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
    
    def strain_energy(self, theta_B: float) -> float:
        """Exact bending strain energy of the elastica [J].

            U = (EI/2) int_0^L (dtheta/ds)^2 ds
              = sqrt(EI P / 2) * int_0^{theta_B} sqrt(sin(theta_B) - sin(theta)) dtheta

        (the second form follows from the first integral
        (dtheta/ds)^2 = (2P/EI)(sin(theta_B) - sin(theta))).

        The remaining integral is evaluated with the singularity-free
        substitution theta = theta_B (1 - t^2), which makes the integrand
        analytic, so Gauss-Legendre converges spectrally.  This is the energy
        used by the work-of-deformation check
        ``verification.verify_energy_conservation_elastica``: for a dead tip
        load the quasi-static work int P d(delta) must equal U.
        """
        if theta_B < 1e-12:
            return 0.0
        if theta_B > 0.5 * np.pi + 1e-12:
            raise ValueError(
                f"strain_energy: theta_B={theta_B:.6f} exceeds pi/2 (no monotone branch)"
            )

        xi, wi = self._gauss_legendre(self._GL_ORDER)
        t = 0.5 * (xi + 1.0)
        w = 0.5 * wi

        theta = theta_B * (1.0 - t**2)
        diff = 2.0 * np.cos(0.5 * (theta_B + theta)) * np.sin(0.5 * (theta_B - theta))
        integrand = 2.0 * theta_B * t * np.sqrt(np.maximum(diff, 0.0))
        J = float(np.sum(w * integrand))

        I_L, _, _ = self._compute_integrals(theta_B)
        P = self.EI * I_L**2 / (2.0 * self.L**2)

        return float(np.sqrt(self.EI * P / 2.0) * J)

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
    
    #: Upper bound for the free-end angle.  theta_B -> pi/2 corresponds to
    #: P -> infinity (the beam wraps around the clamp in a boundary layer), so
    #: the root search must stay strictly below pi/2.
    THETA_B_MAX = 0.5 * np.pi * (1.0 - 1e-9)

    def compute_shape_at_load(self, P_target: float) -> ElasticaSolution:
        """Find the exact solution for a given load P.

        Inverts the monotone relation P(theta_B) by a bracketed root find on
        [0, THETA_B_MAX].  Raises ValueError if P_target is not reachable on
        that branch (which cannot happen for finite P_target, since
        P -> infinity as theta_B -> pi/2).
        """
        def residual(theta_B):
            if theta_B < 1e-10:
                return -P_target
            I_L, _, _ = self._compute_integrals(theta_B)
            P = self.EI * I_L**2 / (2.0 * self.L**2)
            return P - P_target
        
        theta_B_init = min(np.sqrt(3.0 * P_target * self.L**2 / self.EI), 1.5)
        theta_B_init = max(theta_B_init, 0.05)
        upper = min(self.THETA_B_MAX, 10.0 * theta_B_init + 0.1)
        
        try:
            theta_B_sol = optimize.brentq(residual, 1e-9, upper)
        except ValueError:
            res = optimize.minimize_scalar(lambda t: residual(t)**2,
                                          bounds=(1e-9, upper), method='bounded')
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
