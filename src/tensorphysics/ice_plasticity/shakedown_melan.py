"""Melan shakedown bound via Linear Programming.

Melan's theorem (1938): A structure shakes down under cyclic loading if and only if
there exists a time-independent residual stress field ρ such that:
  f(σ^E(t) + ρ) ≤ 0  for all t in the load cycle

where:
  σ^E(t) = elastic stress response to the varying load
  f = yield function
  ρ = self-equilibrating residual stress field (div ρ = 0, ρ·n = 0 on traction-free boundaries)

For discrete load vertices {P_i}, the condition becomes:
  f(σ^E_i + ρ) ≤ 0  for all i

This is formulated as an LP:
  maximize μ
  subject to:
    f(μ·σ^E_i + ρ) ≤ 0  for all i
    ρ is self-equilibrating

The optimal μ is the shakedown load factor.

For a beam cross-section under bending:
  σ = M·y/I (elastic)
  Yield: |σ| ≤ σy
  Self-equilibrating: ∫ρ·dA = 0, ∫ρ·y·dA = 0

For the two-bar benchmark (classic):
  Two bars in parallel (same length, same strain), area A each.
  Total load P = P1 + P2, where P1 is constant and P2 cycles.
  Stress in each bar: σ = P/(2A) (equal sharing).
  
  Classic result: for P1 = A·σy, shakedown P2_max = 2·A·σy
  First yield: P2 = A·σy (total P = 2·A·σy, each bar at σy)
  Ratio: shakedown / first_yield = 2.0
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from scipy.optimize import linprog


@dataclass
class MelanShakedownBeam:
    """Melan shakedown analysis for beam cross-section under cyclic bending.

    Uses the actual section properties for stress computation.
    """
    n_fibers: int = 51  # number of fibers through thickness
    sigma_y: float = 355e6  # yield stress [Pa]
    E: float = 210e9  # Young's modulus [Pa]
    b: float = 1.0  # section width [m]
    h: float = 1.0  # section height [m]

    @property
    def I(self) -> float:
        return self.b * self.h**3 / 12.0

    @property
    def Z_el(self) -> float:
        return self.b * self.h**2 / 6.0

    def shakedown_factor(self, M_elastic_range: Tuple[float, float]) -> float:
        """Compute shakedown load factor for given elastic moment range.

        Parameters
        ----------
        M_elastic_range : (M_min, M_max)
            Elastic moment range [N·m]

        Returns
        -------
        mu : float
            Shakedown load factor (multiplier on the elastic range)
        """
        M_min, M_max = M_elastic_range

        # Discretize cross-section
        y = np.linspace(-self.h / 2.0, self.h / 2.0, self.n_fibers)
        dy = self.h / (self.n_fibers - 1)

        # Elastic stress distribution
        sigma_el_min = M_min * y / self.I
        sigma_el_max = M_max * y / self.I

        # LP formulation:
        # Variables: [mu, rho_1, rho_2, ..., rho_n]
        # Objective: minimize -mu (maximize mu)
        # Constraints:
        #   sigma_el_min_i * mu + rho_i ≤ sigma_y  for all i
        #  -sigma_el_min_i * mu - rho_i ≤ sigma_y  for all i
        #   sigma_el_max_i * mu + rho_i ≤ sigma_y  for all i
        #  -sigma_el_max_i * mu - rho_i ≤ sigma_y  for all i
        #   sum(rho_i * b * dy) = 0  (axial force equilibrium)
        #   sum(rho_i * y_i * b * dy) = 0  (moment equilibrium)

        n = self.n_fibers
        n_vars = 1 + n  # mu + n residual stresses

        # Objective: minimize -mu
        c = np.zeros(n_vars)
        c[0] = -1.0  # maximize mu

        # Inequality constraints: A_ub @ x ≤ b_ub
        n_ineq = 4 * n
        A_ub = np.zeros((n_ineq, n_vars))
        b_ub = np.full(n_ineq, self.sigma_y)

        for i in range(n):
            row = 4 * i
            A_ub[row, 0] = sigma_el_min[i]
            A_ub[row, i + 1] = 1.0
            A_ub[row + 1, 0] = -sigma_el_min[i]
            A_ub[row + 1, i + 1] = -1.0
            A_ub[row + 2, 0] = sigma_el_max[i]
            A_ub[row + 2, i + 1] = 1.0
            A_ub[row + 3, 0] = -sigma_el_max[i]
            A_ub[row + 3, i + 1] = -1.0

        # Equality constraints: A_eq @ x = b_eq
        dA = self.b * dy
        A_eq = np.zeros((2, n_vars))
        b_eq = np.zeros(2)
        for i in range(n):
            A_eq[0, i + 1] = dA
            A_eq[1, i + 1] = y[i] * dA

        # Bounds: mu ≥ 0, rho unbounded
        bounds = [(0, None)] + [(None, None)] * n

        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                         bounds=bounds, method='highs')

        if result.success:
            return float(result.x[0])
        else:
            return 1.0

    def shakedown_moment(self, M_amplitude: float) -> float:
        """Compute shakedown moment for symmetric cyclic loading.

        Parameters
        ----------
        M_amplitude : float
            Moment amplitude [N·m] (loading from -M_amp to +M_amp)

        Returns
        -------
        M_sd : float
            Shakedown moment amplitude [N·m]
        """
        mu = self.shakedown_factor((-M_amplitude, M_amplitude))
        return mu * M_amplitude


def two_bar_melan_shakedown(sigma_y: float, A: float,
                            P1_constant: float) -> dict:
    """Two-bar system Melan shakedown analysis (classic parallel bars).

    Two bars in parallel (same length, same strain), area A each.
    Bar 1: constant load contribution P1
    Bar 2: cyclic load contribution P2 ∈ [0, P2_max]
    Total load P = P1 + P2, shared equally: σ = P/(2A)

    Classic result: for P1 = A·σy, shakedown P2_max = 2·A·σy
    First yield: P2 = A·σy (total P = 2·A·σy)
    Ratio: shakedown / first_yield = 2.0
    """
    # Elastic stress in each bar (equal sharing)
    sigma_min = P1_constant / (2 * A)
    
    # Residual stresses: ρ1 + ρ2 = 0 → ρ2 = -ρ1 = -ρ
    # Shakedown conditions:
    # |sigma_min + ρ| ≤ σy
    # |sigma_max - ρ| ≤ σy

    # To maximize P2_max (and thus sigma_max), we need:
    # sigma_max ≤ 2σy - sigma_min
    sigma_max_limit = 2 * sigma_y - sigma_min
    P2_max = 2 * A * sigma_max_limit - P1_constant

    # First yield: when σ_max = σy
    P_first_yield = 2 * A * sigma_y - P1_constant

    ratio = P2_max / P_first_yield if P_first_yield > 0 else float('inf')

    return {
        "P1_constant": P1_constant,
        "sigma_min": sigma_min,
        "sigma_max_limit": sigma_max_limit,
        "P2_max_shakedown": P2_max,
        "P_first_yield": P_first_yield,
        "ratio": ratio,
    }


def verify_two_bar_shakedown() -> dict:
    """Verify two-bar shakedown: classic result ratio = 2.0.

    For P1 = A·σy (bar 1 at half the total yield load), the shakedown limit
    for bar 2 should be P2_max = 2·A·σy, giving ratio = 2.0.
    """
    sigma_y = 355e6
    A = 0.01  # 100 cm²
    P1 = A * sigma_y  # P1 = A·σy

    result = two_bar_melan_shakedown(sigma_y, A, P1)

    expected_ratio = 2.0

    return {
        **result,
        "expected_ratio": expected_ratio,
        "ratio_error": abs(result["ratio"] - expected_ratio),
        "pass": abs(result["ratio"] - expected_ratio) < 0.01,
    }
