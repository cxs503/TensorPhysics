"""Floe Size Distribution (FSD) Interface from Thermal Cracking.

Provides the interface between thermal crack spacing and the floe size
distribution (f_FSD), which is a key input for ice-ocean interaction models.

Physical Connection:
  Thermal cracks in ice sheets create an initial network of fractures that
  determines the initial floe size distribution. This distribution evolves
  through mechanical breaking, but the thermal cracking provides the "seed"
  distribution.

FSD Model:
  The floe size distribution f(d) gives the number of floes per unit area
  with diameter d. From thermal cracking:
  
  1. Crack spacing L determines characteristic floe size d_0 ~ L
  2. Distribution shape follows from crack network statistics
  3. Typical form: f(d) ~ d^(-α) * exp(-d/d_0)
  
  where α is the power-law exponent (typically 1.5-2.5) and d_0 is the
  characteristic size from thermal crack spacing.

Interface to f_FSD:
  This module provides functions to convert thermal crack spacing into
  FSD parameters that can be used by ice-ocean interaction models.

References:
  - Rothrock, D.A. & Thorndike, A.S. (1984). "Measuring the sea ice floe size distribution."
  - Perovich, D.K. et al. (2008). "Scale dependence in sea ice floe size distributions."
  - Stern, H.L. et al. (2018). "Floe size distribution in the Arctic."
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .crack_spacing import CrackSpacingModel, G_C_ICE
from .thermal_stress import E_ICE, SIGMA_T_ICE


@dataclass
class FSDParameters:
    """Parameters for floe size distribution.
    
    Attributes
    ----------
    d_0 : float
        Characteristic floe diameter (m).
    alpha : float
        Power-law exponent.
    n_total : float
        Total number of floes per unit area (1/m²).
    d_min : float
        Minimum floe diameter (m).
    d_max : float
        Maximum floe diameter (m).
    """
    d_0: float
    alpha: float = 2.0
    n_total: float = 1.0
    d_min: float = 0.1
    d_max: float = 1000.0


@dataclass
class FSDInterface:
    """Interface between thermal cracking and floe size distribution.
    
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
    """
    
    H: float
    sigma_max: float
    sigma_tensile: float = SIGMA_T_ICE
    E: float = E_ICE
    G_c: float = G_C_ICE
    
    def __post_init__(self):
        self.crack_model = CrackSpacingModel(
            self.H, self.sigma_max, self.sigma_tensile, self.E, self.G_c
        )
    
    def thermal_fsd_parameters(self) -> FSDParameters:
        """Compute FSD parameters from thermal crack spacing.
        
        The characteristic floe size d_0 is related to crack spacing L by:
          d_0 = c * L
        
        where c ~ 1.0 (geometric factor for 2D crack network).
        
        The power-law exponent α depends on crack network topology:
          - α ≈ 2.0 for random crack networks
          - α ≈ 1.5-1.8 for more ordered patterns
        
        Returns
        -------
        params : FSDParameters
            FSD parameters.
        """
        L = self.crack_model.crack_spacing_scaling()
        
        # Characteristic floe size (geometric factor ~1.0)
        d_0 = 1.0 * L
        
        # Power-law exponent (typical for thermal cracking)
        alpha = 2.0
        
        # Total number density (from crack spacing)
        # n_total ~ 1/L² for 2D network
        n_total = 1.0 / L**2
        
        # Size range
        d_min = 0.1 * d_0
        d_max = 10.0 * d_0
        
        return FSDParameters(
            d_0=d_0,
            alpha=alpha,
            n_total=n_total,
            d_min=d_min,
            d_max=d_max,
        )
    
    def fsd_from_thermal_cracking(
        self,
        d: np.ndarray,
    ) -> np.ndarray:
        """Compute floe size distribution f(d) from thermal cracking.
        
        f(d) = n_total * (d/d_0)^(-alpha) * exp(-d/d_0) / Z
        
        where Z is normalization constant.
        
        Parameters
        ----------
        d : array
            Floe diameters (m).
        
        Returns
        -------
        f : array
            Number density of floes (1/m² per m).
        """
        params = self.thermal_fsd_parameters()
        
        # Power-law with exponential cutoff
        f = params.n_total * (d / params.d_0)**(-params.alpha) * np.exp(-d / params.d_0)
        
        # Normalize
        Z = np.trapezoid(f, d)
        if Z > 0:
            f /= Z
        
        return f


def initial_crack_spacing_to_fsd(
    L: float,
    d: np.ndarray,
    alpha: float = 2.0,
) -> np.ndarray:
    """Convert crack spacing to floe size distribution.
    
    Parameters
    ----------
    L : float
        Characteristic crack spacing (m).
    d : array
        Floe diameters (m).
    alpha : float
        Power-law exponent.
    
    Returns
    -------
    f : array
        Floe size distribution (1/m² per m).
    """
    d_0 = L  # Characteristic size
    n_total = 1.0 / L**2  # Number density
    
    f = n_total * (d / d_0)**(-alpha) * np.exp(-d / d_0)
    
    # Normalize
    Z = np.trapezoid(f, d)
    if Z > 0:
        f /= Z
    
    return f


def thermal_fsd_parameters(
    H: float,
    sigma_max: float,
    sigma_tensile: float = SIGMA_T_ICE,
    E: float = E_ICE,
    G_c: float = G_C_ICE,
) -> FSDParameters:
    """Compute FSD parameters from thermal cracking conditions.
    
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
    
    Returns
    -------
    params : FSDParameters
        FSD parameters.
    """
    interface = FSDInterface(H, sigma_max, sigma_tensile, E, G_c)
    return interface.thermal_fsd_parameters()
