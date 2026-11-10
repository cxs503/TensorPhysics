"""Haskind relation for radiation-diffraction inter-verification.

The Haskind relation connects the diffraction force (from the incident wave
acting on the body) with the radiation potential (from the body oscillating
in still water):

    F_diff,j = -iωρ ∮ φ_R,j · n · φ_I dS

where:
- F_diff,j is the diffraction force in mode j
- φ_R,j is the radiation potential for mode j
- φ_I is the incident wave potential
- n is the outward normal (body → fluid)
- the integral is over the wetted surface

This relation is exact for linear potential flow and provides a powerful
cross-check: the diffraction force computed directly from the diffraction
problem should match the Haskind integral of the radiation potential.

The incident wave potential for a regular wave of amplitude A, frequency ω,
and direction β is:

    φ_I = (gA/ω) e^{kz} e^{ik(x cosβ + y sinβ)}

where k = ω²/g is the wave number.

References
----------
- Haskind, M.D. (1957) "The hydrodynamic theory of the oscillation of a
  ship in waves"
- Newman, J.N. (1977) "Marine Hydrodynamics", §6.7
- Faltinsen, O.M. (1990) "Sea Loads on Ships and Offshore Structures"
"""

from __future__ import annotations

import numpy as np
from typing import Tuple

from .green_function import RHO_WATER, GRAVITY

__all__ = [
    "haskind_relation",
    "haskind_residual",
]


def incident_wave_potential(x: np.ndarray, y: np.ndarray, z: np.ndarray,
                            omega: float, A: float = 1.0, beta: float = 0.0,
                            g: float = GRAVITY) -> np.ndarray:
    """Compute the incident wave potential φ_I at given points.

    φ_I = (gA/ω) e^{kz} e^{ik(x cosβ + y sinβ)}

    Parameters
    ----------
    x, y, z : np.ndarray
        Point coordinates.
    omega : float
        Angular frequency.
    A : float
        Wave amplitude.
    beta : float
        Wave direction (rad), 0 = head sea.
    g : float
        Gravity.

    Returns
    -------
    np.ndarray
        Complex incident wave potential.
    """
    k = omega ** 2 / g
    phi_I = (g * A / omega) * np.exp(k * z) * np.exp(1j * k * (x * np.cos(beta) + y * np.sin(beta)))
    return phi_I


def haskind_relation(omega: float, phi_rad: np.ndarray, n_z: np.ndarray,
                     panel_x: np.ndarray, panel_y: np.ndarray, panel_z: np.ndarray,
                     panel_area: np.ndarray, A: float = 1.0, beta: float = 0.0,
                     rho: float = RHO_WATER, g: float = GRAVITY) -> complex:
    """Compute the diffraction force via the Haskind relation.

    F_diff = -iωρ ∮ φ_R · n_z · φ_I dS

    Parameters
    ----------
    omega : float
        Angular frequency.
    phi_rad : np.ndarray
        Radiation potential on panels (complex).
    n_z : np.ndarray
        Normal z-component on panels.
    panel_x, panel_y, panel_z : np.ndarray
        Panel centroid coordinates.
    panel_area : np.ndarray
        Panel areas.
    A : float
        Wave amplitude.
    beta : float
        Wave direction.
    rho : float
        Water density.
    g : float
        Gravity.

    Returns
    -------
    complex
        Diffraction force from Haskind relation.
    """
    phi_I = incident_wave_potential(panel_x, panel_y, panel_z, omega, A, beta, g)
    integrand = phi_rad * n_z * phi_I * panel_area
    F_diff = -1j * omega * rho * np.sum(integrand)
    return F_diff


def haskind_residual(F_diff_direct: complex, F_diff_haskind: complex) -> float:
    """Compute the relative residual between direct and Haskind diffraction forces.

    residual = |F_diff_direct - F_diff_haskind| / |F_diff_direct|

    Parameters
    ----------
    F_diff_direct : complex
        Diffraction force computed directly from the diffraction problem.
    F_diff_haskind : complex
        Diffraction force computed via the Haskind relation.

    Returns
    -------
    float
        Relative residual.
    """
    denom = abs(F_diff_direct)
    if denom < 1e-15:
        return abs(F_diff_direct - F_diff_haskind)
    return abs(F_diff_direct - F_diff_haskind) / denom


def compute_diffraction_force_direct(omega: float, phi_diff: np.ndarray,
                                     n_z: np.ndarray, panel_area: np.ndarray,
                                     rho: float = RHO_WATER) -> complex:
    """Compute the diffraction force directly from the diffraction potential.

    F_diff = -iωρ ∮ (φ_I + φ_D) n_z dS

    where φ_D is the diffraction potential (scattered wave).

    Parameters
    ----------
    omega : float
        Angular frequency.
    phi_diff : np.ndarray
        Total potential (incident + diffraction) on panels.
    n_z : np.ndarray
        Normal z-component on panels.
    panel_area : np.ndarray
        Panel areas.
    rho : float
        Water density.

    Returns
    -------
    complex
        Direct diffraction force.
    """
    F_diff = -1j * omega * rho * np.sum(phi_diff * n_z * panel_area)
    return F_diff
