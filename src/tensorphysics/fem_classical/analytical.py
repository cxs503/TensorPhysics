"""Analytical reference solutions for classical FEM verification.

All formulas are closed-form (no numerical approximation) unless explicitly
noted.  Functions return SI quantities (metres, radians, Pascals).
"""

from __future__ import annotations

import math
from typing import Dict

import numpy as np

__all__ = [
    "cantilever_tip_deflection_point_load",
    "cantilever_tip_rotation_point_load",
    "cantilever_tip_deflection_udl",
    "cantilever_tip_rotation_udl",
    "navier_square_plate_center_deflection",
    "classical_cylindrical_shell_buckling_stress",
]


# ---------------------------------------------------------------------------
# Euler-Bernoulli cantilever beam
# ---------------------------------------------------------------------------
def cantilever_tip_deflection_point_load(P: float, L: float, E: float, I: float) -> float:
    """Tip deflection of a cantilever under end point load.

    δ = P L³ / (3 E I)

    Parameters
    ----------
    P : float
        Point load at free end [N].
    L : float
        Beam length [m].
    E : float
        Young's modulus [Pa].
    I : float
        Second moment of area [m⁴].

    Returns
    -------
    float
        Tip deflection [m] (positive downward).
    """
    return P * L ** 3 / (3.0 * E * I)


def cantilever_tip_rotation_point_load(P: float, L: float, E: float, I: float) -> float:
    """Tip rotation of a cantilever under end point load.

    θ = P L² / (2 E I)

    Returns
    -------
    float
        Tip rotation [rad] (positive clockwise).
    """
    return P * L ** 2 / (2.0 * E * I)


def cantilever_tip_deflection_udl(q: float, L: float, E: float, I: float) -> float:
    """Tip deflection of a cantilever under uniform distributed load.

    δ = q L⁴ / (8 E I)

    Parameters
    ----------
    q : float
        Distributed load intensity [N/m].
    """
    return q * L ** 4 / (8.0 * E * I)


def cantilever_tip_rotation_udl(q: float, L: float, E: float, I: float) -> float:
    """Tip rotation of a cantilever under uniform distributed load.

    θ = q L³ / (6 E I)
    """
    return q * L ** 3 / (6.0 * E * I)


# ---------------------------------------------------------------------------
# Kirchhoff simply-supported square plate (Navier solution)
# ---------------------------------------------------------------------------
def navier_square_plate_center_deflection(
    a: float, t: float, E: float, nu: float, q: float,
    n_terms: int = 25
) -> float:
    """Center deflection of a simply-supported square plate under uniform load.

    Navier double Fourier series solution (Timoshenko & Woinowsky-Krieger,
    *Theory of Plates and Shells*, 2nd ed., §44):

        w_max = (16 q a⁴) / (π⁶ D) · ΣΣ [(-1)^((m+n-2)/2) / (m n (m²+n²)²)]

    summed over odd m, n = 1, 3, 5, ...

    Parameters
    ----------
    a : float
        Plate side length [m] (square plate, a = b).
    t : float
        Plate thickness [m].
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    q : float
        Uniform pressure [Pa].
    n_terms : int
        Number of odd terms per direction (default 25 → 625 terms,
        converged to 8 significant digits).

    Returns
    -------
    float
        Center deflection [m] (positive downward).
    """
    D = E * t ** 3 / (12.0 * (1.0 - nu ** 2))
    prefactor = 16.0 * q * a ** 4 / (math.pi ** 6 * D)

    odd = np.arange(1, 2 * n_terms, 2)
    m, n = np.meshgrid(odd, odd, indexing="ij")
    sign = (-1.0) ** ((m + n - 2) // 2)
    term = sign / (m * n * (m ** 2 + n ** 2) ** 2)
    return float(prefactor * term.sum())


# ---------------------------------------------------------------------------
# Cylindrical shell axial-compression buckling (classical formula)
# ---------------------------------------------------------------------------
def classical_cylindrical_shell_buckling_stress(
    E: float, nu: float, t: float, R: float
) -> float:
    """Classical critical buckling stress for a thin cylindrical shell under axial compression.

    σ_cr = E / √(3(1 − ν²)) · t / R

    Reference: Timoshenko & Gere, *Theory of Elastic Stability*, 2nd ed.,
    §11.2; also Donnell (1933), von Mises (1929).

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    t : float
        Shell thickness [m].
    R : float
        Shell mid-surface radius [m].

    Returns
    -------
    float
        Critical buckling stress [Pa] (positive = compressive).
    """
    return E / math.sqrt(3.0 * (1.0 - nu ** 2)) * t / R
