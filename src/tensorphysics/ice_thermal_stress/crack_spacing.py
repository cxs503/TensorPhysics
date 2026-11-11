"""Thermal Crack Spacing Scaling Law.

Implements models for predicting the spacing between thermal cracks in ice sheets.

Physical Model:
  When thermal stress exceeds tensile strength, multiple cracks form. The spacing
  between cracks is determined by stress redistribution and fracture mechanics:
  
  1. First crack forms at location of maximum stress (typically surface)
  2. Crack relieves stress in a zone of width ~H (ice thickness)
  3. Stress rebuilds outside this zone
  4. Second crack forms when stress again reaches tensile strength
  5. Process repeats, establishing characteristic spacing L ~ c*H
  
  Scaling Law:
    L = c * H * f(σ_max/σ_tensile, material properties)
    
  Observations and theory suggest:
    - c ~ 1-3 for thermal cracking
    - Higher stress ratio -> smaller spacing
    - L ~ H^1 for stress-driven cracking (not energy balance)

References:
  - van der Veen, C.J. (1998). "Fracture surface topography of ice."
  - Weiss, J. & Schulson, E.M. (1995). "Ice cracking and spalling."
  - Schulson, E.M. (2001). "Brittle failure of ice." Eng. Fract. Mech.
  - Mellor, M. (1986). "Mechanical behavior of ice." CRREL Monograph.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .thermal_stress import E_ICE, SIGMA_T_ICE, K_IC_ICE


# ── Fracture energy ────────────────────────────────────────────────────
# G_c = K_IC² / E' where E' = E for plane stress, E/(1-ν²) for plane strain
NU_ICE = 0.33
G_C_ICE = K_IC_ICE**2 / (E_ICE / (1.0 - NU_ICE**2))  # ~1 J/m²


@dataclass
class CrackSpacingModel:
    """Model for thermal crack spacing in ice sheets.
    
    Parameters
    ----------
    H : float
        Ice thickness (m).
    sigma_max : float
        Maximum thermal stress (Pa).
    sigma_tensile : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa).
    G_c : float
        Fracture energy (J/m²).
    nu : float
        Poisson's ratio.
    """
    
    H: float
    sigma_max: float
    sigma_tensile: float = SIGMA_T_ICE
    E: float = E_ICE
    G_c: float = G_C_ICE
    nu: float = NU_ICE
    
    def __post_init__(self):
        if self.sigma_max < self.sigma_tensile:
            raise ValueError("sigma_max must exceed sigma_tensile for cracking")
    
    def crack_spacing_scaling(self) -> float:
        """Compute characteristic crack spacing.
        
        For thermal cracking, observations and stress-redistribution theory give:
          L ~ c * H * (σ_tensile/σ_max)^0.5
        
        where c ~ 2 is a geometric constant.
        
        The scaling L ~ H^1 comes from stress redistribution (not energy balance).
        Higher stress ratio leads to more cracks and smaller spacing.
        
        Returns
        -------
        L : float
            Characteristic crack spacing (m).
        """
        # Base spacing proportional to ice thickness
        # c ~ 2 from stress redistribution theory
        c = 2.0
        
        # Stress ratio effect: higher stress -> smaller spacing
        stress_ratio = self.sigma_tensile / self.sigma_max
        f_stress = math.sqrt(stress_ratio)
        
        L = c * self.H * f_stress
        
        return L
    
    def crack_spacing_distribution(self, n_samples: int = 1000) -> np.ndarray:
        """Generate sample crack spacings from statistical distribution.
        
        Crack spacing follows a gamma distribution:
          p(L) = (L/μ)^(k-1) * exp(-L/μ) / (μ * Γ(k))
        
        where μ is mean spacing and k is shape parameter (typically 2-4).
        
        Parameters
        ----------
        n_samples : int
            Number of samples.
        
        Returns
        -------
        spacings : array
            Sample crack spacings (m).
        """
        L_mean = self.crack_spacing_scaling()
        
        # Shape parameter k (higher = more uniform spacing)
        k = 3.0
        theta = L_mean / k  # Scale parameter
        
        # Gamma distribution
        spacings = np.random.gamma(k, theta, n_samples)
        
        return spacings
    
    def number_of_cracks(self, width: float) -> int:
        """Estimate number of cracks for given domain width.
        
        Parameters
        ----------
        width : float
            Domain width (m).
        
        Returns
        -------
        n_cracks : int
            Estimated number of cracks.
        """
        L = self.crack_spacing_scaling()
        return max(1, int(width / L))


def crack_spacing_scaling(
    H: float,
    sigma_max: float,
    sigma_tensile: float = SIGMA_T_ICE,
    E: float = E_ICE,
    G_c: float = G_C_ICE,
) -> float:
    """Compute characteristic crack spacing.
    
    L ~ 2*H * sqrt(σ_tensile/σ_max)
    
    Parameters
    ----------
    H : float
        Ice thickness (m).
    sigma_max : float
        Maximum thermal stress (Pa).
    sigma_tensile : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa) - not used in this formula.
    G_c : float
        Fracture energy (J/m²) - not used in this formula.
    
    Returns
    -------
    L : float
        Characteristic crack spacing (m).
    """
    model = CrackSpacingModel(H, sigma_max, sigma_tensile, E, G_c)
    return model.crack_spacing_scaling()


def crack_spacing_distribution(
    H: float,
    sigma_max: float,
    sigma_tensile: float = SIGMA_T_ICE,
    E: float = E_ICE,
    G_c: float = G_C_ICE,
    n_samples: int = 1000,
) -> np.ndarray:
    """Generate sample crack spacings.
    
    Parameters
    ----------
    H : float
        Ice thickness (m).
    sigma_max : float
        Maximum thermal stress (Pa).
    sigma_tensile : float
        Tensile strength (Pa).
    E : float
        Young's modulus (Pa).
    G_c : float
        Fracture energy (J/m²).
    n_samples : int
        Number of samples.
    
    Returns
    -------
    spacings : array
        Sample crack spacings (m).
    """
    model = CrackSpacingModel(H, sigma_max, sigma_tensile, E, G_c)
    return model.crack_spacing_distribution(n_samples)
