"""Thermal Crack Initiation in Ice Sheets.

Implements models for predicting when and where thermal cracks initiate in
ice sheets due to thermal stress from surface cooling.

Physical Model:
  Thermal cracks initiate when the thermal stress exceeds the tensile strength
  of ice. For a cooling ice sheet:
  
  1. Surface cooling creates temperature gradient
  2. Thermal stress develops: σ(z,t) = -E*α_T/(1-ν) * [T(z,t) - <T>]
  3. Maximum tensile stress occurs at surface (coldest region)
  4. Crack initiates when σ_max(t) ≥ σ_tensile
  
  Critical Time:
    The time t_c when crack initiates is found by solving:
      σ_max(t_c) = σ_tensile
    
    For early times (semi-infinite approximation):
      σ_max(t) ≈ E*α_T/(1-ν) * (T_init - T_surface) * [1 - 2*sqrt(alpha*t)/(sqrt(pi)*H)]
    
    Solving for t_c gives an estimate of crack initiation time.

Validation:
  Crack initiation time validated against analytical estimates for simple cases.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.optimize import brentq

from .thermal_conduction import IceThermalConduction, ALPHA_ICE
from .thermal_stress import ThermalStressSolver, SIGMA_T_ICE, E_ICE, NU_ICE, ALPHA_T_ICE


@dataclass
class CrackInitiationSolver:
    """Solver for thermal crack initiation time and location.
    
    Parameters
    ----------
    thermal_solver : IceThermalConduction
        Thermal conduction solver.
    sigma_tensile : float
        Tensile strength of ice (Pa).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_T : float
        Thermal expansion coefficient (1/K).
    """
    
    thermal_solver: IceThermalConduction
    sigma_tensile: float = SIGMA_T_ICE
    E: float = E_ICE
    nu: float = NU_ICE
    alpha_T: float = ALPHA_T_ICE
    
    def __post_init__(self):
        self.stress_solver = ThermalStressSolver(
            self.thermal_solver,
            E=self.E,
            nu=self.nu,
            alpha_T=self.alpha_T,
        )
    
    def crack_initiation_time(self, tol: float = 1e-3) -> float:
        """Find time t_c when thermal stress first reaches tensile strength.
        
        Solves: σ_max(t_c) = σ_tensile
        
        Uses time-stepping to find when stress crosses threshold.
        
        Parameters
        ----------
        tol : float
            Relative tolerance for time (fraction of total time).
        
        Returns
        -------
        t_c : float
            Crack initiation time (s).
        """
        # Save initial state
        T_init_saved = self.thermal_solver.T_init
        T_surface_saved = self.thermal_solver.T_surface
        
        # Reset to uniform initial temperature (no surface cooling yet)
        self.thermal_solver.T[:] = T_init_saved
        self.thermal_solver.time = 0.0
        
        # Check initial stress (should be zero for uniform temperature)
        sigma_max_0 = self.stress_solver.max_tensile_stress()
        if sigma_max_0 >= self.sigma_tensile:
            return 0.0
        
        # Apply surface cooling
        self.thermal_solver.T[0] = T_surface_saved
        
        # Estimate upper bound for t_c
        delta_T = T_init_saved - T_surface_saved
        sigma_max_instant = self.E * self.alpha_T * delta_T / (1.0 - self.nu)
        
        if sigma_max_instant < self.sigma_tensile:
            # Stress never reaches tensile strength
            return float('inf')
        
        # Estimate t_c from linearized relation
        ratio = 1.0 - self.sigma_tensile / sigma_max_instant
        if ratio <= 0:
            return 0.0
        
        t_estimate = (ratio * math.sqrt(math.pi) * self.thermal_solver.H / 2.0)**2 / ALPHA_ICE
        
        # Time-stepping to find t_c
        dt = self.thermal_solver.dt
        max_time = max(t_estimate * 2.0, 3600.0 * 24)  # At least 1 day
        
        t = 0.0
        while t < max_time:
            # Advance one time step
            self.thermal_solver.step(1)
            t = self.thermal_solver.time
            
            # Check if cracked
            sigma_max = self.stress_solver.max_tensile_stress()
            if sigma_max >= self.sigma_tensile:
                return t
        
        # If not found, return estimate
        return t_estimate
    
    def crack_initiation_location(self) -> float:
        """Find depth where crack initiates (typically surface).
        
        Returns
        -------
        z_c : float
            Crack initiation depth (m).
        """
        sigma = self.stress_solver.thermal_stress_profile()
        idx_max = np.argmax(sigma)
        return self.thermal_solver.z[idx_max]
    
    def is_cracked(self) -> bool:
        """Check if thermal stress has exceeded tensile strength."""
        sigma_max = self.stress_solver.max_tensile_stress()
        return sigma_max >= self.sigma_tensile


def crack_initiation_time(
    H: float,
    T_surface: float,
    T_init: float,
    sigma_tensile: float = SIGMA_T_ICE,
    E: float = E_ICE,
    nu: float = NU_ICE,
    alpha_T: float = ALPHA_T_ICE,
    alpha: float = ALPHA_ICE,
) -> float:
    """Analytical estimate of crack initiation time.
    
    For semi-infinite solid with surface cooling:
      σ_max(t) ≈ E*α_T/(1-ν) * (T_init - T_surface) * [1 - 2*sqrt(alpha*t)/(sqrt(pi)*H)]
    
    Setting σ_max(t_c) = σ_tensile and solving:
      t_c ≈ [sqrt(pi)*H/(2*sqrt(alpha))]² * [1 - σ_tensile*(1-ν)/(E*α_T*ΔT)]²
    
    Parameters
    ----------
    H : float
        Ice thickness (m).
    T_surface : float
        Surface temperature (°C).
    T_init : float
        Initial temperature (°C).
    sigma_tensile : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_T : float
        Thermal expansion coefficient (1/K).
    alpha : float
        Thermal diffusivity (m²/s).
    
    Returns
    -------
    t_c : float
        Crack initiation time (s).
    """
    delta_T = T_init - T_surface
    
    # Maximum instantaneous stress
    sigma_max_instant = E * alpha_T * delta_T / (1.0 - nu)
    
    if sigma_max_instant < sigma_tensile:
        return float('inf')
    
    # Solve for t_c
    ratio = 1.0 - sigma_tensile / sigma_max_instant
    if ratio <= 0:
        return 0.0
    
    t_c = (math.sqrt(math.pi) * H / (2.0 * math.sqrt(alpha)))**2 * ratio**2
    
    return t_c


def critical_temperature_difference(
    H: float,
    sigma_tensile: float = SIGMA_T_ICE,
    E: float = E_ICE,
    nu: float = NU_ICE,
    alpha_T: float = ALPHA_T_ICE,
) -> float:
    """Minimum temperature difference to cause cracking.
    
    For instantaneous cooling:
      ΔT_crit = σ_tensile * (1-ν) / (E*α_T)
    
    Parameters
    ----------
    H : float
        Ice thickness (m) - not used in this simple estimate.
    sigma_tensile : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa).
    nu : float
        Poisson's ratio.
    alpha_T : float
        Thermal expansion coefficient (1/K).
    
    Returns
    -------
    delta_T_crit : float
        Critical temperature difference (°C).
    """
    return sigma_tensile * (1.0 - nu) / (E * alpha_T)
