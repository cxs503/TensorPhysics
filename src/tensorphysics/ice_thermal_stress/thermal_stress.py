"""Thermal Stress in Ice Sheets.

Implements closed-form solutions for thermal stress development in ice sheets
due to temperature gradients from surface cooling.

Physical Model:
  When an ice sheet cools from the surface, the temperature field T(z,t) creates
  differential thermal contraction. For a thin plate constrained against in-plane
  deformation (plane strain), the thermal stress is:
  
    σ_xx(z,t) = -E*α_T / (1-ν) * [T(z,t) - <T>]
  
  where <T> is the thickness-averaged temperature:
    <T> = (1/H) ∫₀ᴴ T(z,t) dz
  
  The stress is tensile (positive) where T < <T> (cold surface) and compressive
  where T > <T> (warm interior).

Key Relations:
  - Thermal stress: σ = E*α_T*ΔT/(1-ν) for fully constrained plate
  - Stress intensity factor: K_I = σ*sqrt(π*a)*Y(a/W)
  - Maximum stress occurs at surface for early times

Validation:
  Numerical integration of stress profile validated against analytical estimates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import integrate

from .thermal_conduction import IceThermalConduction, ALPHA_ICE


# ── Ice mechanical properties ──────────────────────────────────────────
E_ICE = 9.0e9        # Young's modulus (Pa) - typical for ice at -10°C
NU_ICE = 0.33        # Poisson's ratio
ALPHA_T_ICE = 50e-6  # Coefficient of thermal expansion (1/K)
SIGMA_T_ICE = 1.5e6  # Tensile strength (Pa) - typical range 1-2 MPa
K_IC_ICE = 0.1e6     # Fracture toughness (Pa√m) - typical range 0.05-0.2


@dataclass
class ThermalStressSolver:
    """Thermal stress solver for ice sheets.
    
    Computes thermal stress from temperature field using plane-strain
    constrained plate theory.
    
    Parameters
    ----------
    thermal_solver : IceThermalConduction
        Thermal conduction solver with current temperature field.
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_T : float
        Coefficient of thermal expansion (1/K).
    """
    
    thermal_solver: IceThermalConduction
    E: float = E_ICE
    nu: float = NU_ICE
    alpha_T: float = ALPHA_T_ICE
    
    def __post_init__(self):
        self.H = self.thermal_solver.H
        self.z = self.thermal_solver.z
        self.nz = self.thermal_solver.nz
    
    def thickness_averaged_temperature(self) -> float:
        """Compute thickness-averaged temperature <T> = (1/H)∫T dz."""
        T = self.thermal_solver.temperature_profile()
        return np.trapezoid(T, self.z) / self.H
    
    def thermal_stress_profile(self) -> np.ndarray:
        """Compute thermal stress profile σ(z).
        
        σ(z) = -E*α_T/(1-ν) * [T(z) - <T>]
        
        Returns
        -------
        sigma : array
            Thermal stress (Pa). Positive = tensile, negative = compressive.
        """
        T = self.thermal_solver.temperature_profile()
        T_avg = self.thickness_averaged_temperature()
        
        # Plane strain thermal stress
        prefactor = -self.E * self.alpha_T / (1.0 - self.nu)
        sigma = prefactor * (T - T_avg)
        
        return sigma
    
    def max_tensile_stress(self) -> float:
        """Find maximum tensile stress (typically at surface)."""
        sigma = self.thermal_stress_profile()
        return np.max(sigma)
    
    def max_compressive_stress(self) -> float:
        """Find maximum compressive stress (typically at mid-depth)."""
        sigma = self.thermal_stress_profile()
        return np.min(sigma)
    
    def stress_at_surface(self) -> float:
        """Return thermal stress at surface (z=0)."""
        sigma = self.thermal_stress_profile()
        return sigma[0]
    
    def stress_at_bottom(self) -> float:
        """Return thermal stress at bottom (z=H)."""
        sigma = self.thermal_stress_profile()
        return sigma[-1]


def thermal_stress_profile(
    z: np.ndarray,
    T_profile: np.ndarray,
    E: float = E_ICE,
    nu: float = NU_ICE,
    alpha_T: float = ALPHA_T_ICE,
) -> np.ndarray:
    """Compute thermal stress from temperature profile.
    
    σ(z) = -E*α_T/(1-ν) * [T(z) - <T>]
    
    Parameters
    ----------
    z : array
        Depth coordinates (m).
    T_profile : array
        Temperature profile (°C).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_T : float
        Thermal expansion coefficient (1/K).
    
    Returns
    -------
    sigma : array
        Thermal stress (Pa).
    """
    T_avg = np.trapezoid(T_profile, z) / (z[-1] - z[0])
    prefactor = -E * alpha_T / (1.0 - nu)
    return prefactor * (T_profile - T_avg)


def max_thermal_stress(
    z: np.ndarray,
    T_profile: np.ndarray,
    E: float = E_ICE,
    nu: float = NU_ICE,
    alpha_T: float = ALPHA_T_ICE,
) -> float:
    """Find maximum tensile thermal stress."""
    sigma = thermal_stress_profile(z, T_profile, E, nu, alpha_T)
    return np.max(sigma)


def stress_intensity_factor(
    sigma: float,
    a: float,
    W: float,
    K_IC: float = K_IC_ICE,
) -> float:
    """Compute stress intensity factor K_I for edge crack.
    
    K_I = σ * sqrt(π*a) * Y(a/W)
    
    where Y is geometry correction factor:
    Y(ξ) = 1.12 - 0.23*ξ + 10.6*ξ² - 21.7*ξ³ + 30.4*ξ⁴
    with ξ = a/W.
    
    Parameters
    ----------
    sigma : float
        Applied stress (Pa).
    a : float
        Crack length (m).
    W : float
        Characteristic width (m) - ice thickness or crack spacing.
    K_IC : float
        Fracture toughness (Pa√m).
    
    Returns
    -------
    K_I : float
        Stress intensity factor (Pa√m).
    """
    if a <= 0 or W <= 0:
        return 0.0
    
    xi = a / W
    # Geometry correction factor (Tada et al.)
    Y = 1.12 - 0.23 * xi + 10.6 * xi**2 - 21.7 * xi**3 + 30.4 * xi**4
    
    return sigma * math.sqrt(math.pi * a) * Y
