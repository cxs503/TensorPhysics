"""
Plate bending theory for thin plates (Kirchhoff-Love theory).

Implements classical plate theory for dispersion analysis of elastic waves
in homogeneous plates.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class PlateProperties:
    """Material and geometric properties of a plate."""
    E: float  # Young's modulus [Pa]
    nu: float  # Poisson's ratio [-]
    rho: float  # Density [kg/m³]
    h: float  # Thickness [m]
    
    @property
    def D(self) -> float:
        """Flexural rigidity D = Eh³/(12(1-ν²)) [N·m]"""
        return self.E * self.h**3 / (12 * (1 - self.nu**2))
    
    @property
    def rho_h(self) -> float:
        """Mass per unit area ρh [kg/m²]"""
        return self.rho * self.h


class PlateTheory:
    """Base class for plate theory implementations."""
    
    def __init__(self, properties: PlateProperties):
        self.properties = properties
    
    def dispersion_relation(self, k: np.ndarray) -> np.ndarray:
        """
        Compute angular frequency ω for given wavenumber k.
        
        Parameters
        ----------
        k : np.ndarray
            Wavenumber(s) [rad/m]
        
        Returns
        -------
        np.ndarray
            Angular frequency ω [rad/s]
        """
        raise NotImplementedError
    
    def phase_velocity(self, k: np.ndarray) -> np.ndarray:
        """Compute phase velocity c_p = ω/k [m/s]"""
        omega = self.dispersion_relation(k)
        return omega / k
    
    def group_velocity(self, k: np.ndarray) -> np.ndarray:
        """Compute group velocity c_g = dω/dk [m/s]"""
        dk = 1e-6 * k
        omega_plus = self.dispersion_relation(k + dk)
        omega_minus = self.dispersion_relation(k - dk)
        return (omega_plus - omega_minus) / (2 * dk)


class ThinPlateTheory(PlateTheory):
    """
    Kirchhoff-Love thin plate theory.
    
    Dispersion relation: ω = √(D/ρh) · k²
    
    This is exact for thin plates (h/L < 0.1) and provides the baseline
    for validating periodic structure analysis.
    """
    
    def dispersion_relation(self, k: np.ndarray) -> np.ndarray:
        """
        Thin plate dispersion: ω = √(D/ρh) · k²
        
        Parameters
        ----------
        k : np.ndarray
            Wavenumber(s) [rad/m]
        
        Returns
        -------
        np.ndarray
            Angular frequency ω [rad/s]
        """
        D = self.properties.D
        rho_h = self.properties.rho_h
        return np.sqrt(D / rho_h) * k**2
    
    def wavelength(self, k: np.ndarray) -> np.ndarray:
        """Compute wavelength λ = 2π/k [m]"""
        return 2 * np.pi / k
    
    def critical_frequency(self, L: float) -> float:
        """
        Compute critical frequency for plate of characteristic length L.
        
        Parameters
        ----------
        L : float
            Characteristic length [m]
        
        Returns
        -------
        float
            Critical frequency [Hz]
        """
        k_critical = np.pi / L
        omega_critical = self.dispersion_relation(k_critical)
        return omega_critical / (2 * np.pi)
