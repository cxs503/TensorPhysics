"""
Metamaterial plate with local resonators.

Implements acoustic metamaterial plates with attached mass-spring resonators
that create bandgaps through local resonance mechanisms.

Uses the effective mass density approach to predict bandgaps:
ρ_eff(ω) = ρ_base + Σ m_r · ω_r² / (ω_r² - ω²) / A_cell

Bandgaps occur where ρ_eff < 0 (negative effective mass).
"""

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import brentq
from dataclasses import dataclass
from typing import List, Tuple, Optional
from .plate_theory import PlateProperties
from .bloch_floquet import BlochFloquetAnalyzer


@dataclass
class Resonator:
    """
    Local mass-spring resonator attached to the plate.
    
    Attributes
    ----------
    m_r : float
        Resonator mass [kg]
    k_r : float
        Spring stiffness [N/m]
    position : tuple
        (x, y) position within unit cell [m]
    """
    m_r: float
    k_r: float
    position: Tuple[float, float]
    
    @property
    def resonance_frequency(self) -> float:
        """Natural frequency of the resonator f_r = √(k_r/m_r)/(2π) [Hz]"""
        return np.sqrt(self.k_r / self.m_r) / (2 * np.pi)
    
    @property
    def resonance_angular_frequency(self) -> float:
        """Natural angular frequency ω_r = √(k_r/m_r) [rad/s]"""
        return np.sqrt(self.k_r / self.m_r)


class MetamaterialPlate:
    """
    Acoustic metamaterial plate with local resonators.
    
    Uses the effective mass density approach:
    - The plate has base mass density ρ_base = ρ·h
    - Each resonator adds a frequency-dependent contribution
    - Bandgaps occur where ρ_eff < 0
    
    This captures the essential physics of local resonance bandgaps.
    """
    
    def __init__(self, plate_props: PlateProperties, L: float, 
                 resonators: List[Resonator], nx: int = 10, ny: int = 10):
        """
        Initialize metamaterial plate.
        
        Parameters
        ----------
        plate_props : PlateProperties
            Host plate properties
        L : float
            Unit cell size [m]
        resonators : list of Resonator
            List of resonators in the unit cell
        nx, ny : int
            Number of points for discretization
        """
        self.plate_props = plate_props
        self.L = L
        self.resonators = resonators
        self.nx = nx
        self.ny = ny
        
        # Create base analyzer for plate-only part
        self.analyzer = BlochFloquetAnalyzer(plate_props, L, nx, ny)
        
        # Effective plate mass per unit cell
        self.m_plate = plate_props.rho_h * L**2
    
    @property
    def total_resonator_mass(self) -> float:
        """Total mass of all resonators [kg]"""
        return sum(r.m_r for r in self.resonators)
    
    @property
    def plate_mass(self) -> float:
        """Mass of the unit cell plate [kg]"""
        return self.m_plate
    
    @property
    def mass_ratio(self) -> float:
        """Ratio of total resonator mass to plate mass"""
        return self.total_resonator_mass / self.m_plate
    
    @property
    def mean_resonance_frequency(self) -> float:
        """Mean resonance frequency of all resonators [Hz]"""
        if not self.resonators:
            return 0.0
        return np.mean([r.resonance_frequency for r in self.resonators])
    
    def effective_mass_density(self, freq: np.ndarray) -> np.ndarray:
        """
        Compute effective mass density as function of frequency.
        
        ρ_eff(ω) = ρ_base + Σ m_r · ω_r² / (ω_r² - ω²) / A_cell
        
        Parameters
        ----------
        freq : np.ndarray
            Frequency array [Hz]
        
        Returns
        -------
        rho_eff : np.ndarray
            Effective mass per unit area [kg/m²]
        """
        omega = 2 * np.pi * freq
        rho_base = self.plate_props.rho_h  # mass per unit area
        A_cell = self.L**2
        
        rho_eff = np.zeros_like(freq, dtype=float)
        
        for i, w in enumerate(omega):
            F = rho_base
            
            for resonator in self.resonators:
                wr = resonator.resonance_angular_frequency
                denom = wr**2 - w**2
                if abs(denom) > 1e-10:
                    F += resonator.m_r / A_cell * wr**2 / denom
            
            rho_eff[i] = F
        
        return rho_eff
    
    def _find_bandgap_edges(self, f_min: float, f_max: float, n_points: int = 1000) -> List[Tuple[float, float]]:
        """
        Find bandgap edges by locating where ρ_eff crosses zero.
        
        Parameters
        ----------
        f_min, f_max : float
            Frequency range to search [Hz]
        n_points : int
            Number of points for initial scan
        
        Returns
        -------
        bandgaps : list
            List of (f_lower, f_upper) bandgap boundaries [Hz]
        """
        freq = np.linspace(f_min, f_max, n_points)
        rho_eff = self.effective_mass_density(freq)
        
        # Find zero crossings
        bandgaps = []
        for i in range(len(rho_eff) - 1):
            if rho_eff[i] * rho_eff[i+1] < 0:
                # Zero crossing found
                def func(f):
                    return self.effective_mass_density(np.array([f]))[0]
                
                try:
                    f_zero = brentq(func, freq[i], freq[i+1], xtol=1e-6)
                    bandgaps.append(f_zero)
                except:
                    pass
        
        # Pair up zero crossings to form bandgaps
        result = []
        for i in range(0, len(bandgaps) - 1, 2):
            result.append((bandgaps[i], bandgaps[i+1]))
        
        return result
    
    def solve_dispersion(self, kx: float, ky: float, n_modes: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the dispersion problem using effective mass density.
        
        The dispersion relation is modified by the effective mass:
        ω² = (D/ρ_eff) · k⁴
        
        Parameters
        ----------
        kx, ky : float
            Bloch wavenumbers [rad/m]
        n_modes : int
            Number of modes to compute
        
        Returns
        -------
        omega : np.ndarray
            Angular frequencies [rad/s]
        modes : np.ndarray
            Dummy mode shapes
        """
        k = np.sqrt(kx**2 + ky**2)
        
        if k < 1e-10:
            # At k=0, return plate modes
            omega = np.array([0.0])
            modes = np.ones((1, 1))
            return omega[:n_modes], modes
        
        # Scan frequency range to find where dispersion relation is satisfied
        f_min = 1.0
        f_max = 5000.0
        n_freq = 2000
        freq_array = np.linspace(f_min, f_max, n_freq)
        
        omega_list = []
        
        for freq in freq_array:
            rho_eff = self.effective_mass_density(np.array([freq]))[0]
            
            if rho_eff > 0:
                # Dispersion relation: ω² = (D/ρ_eff) · k⁴
                omega_sq = self.plate_props.D / rho_eff * k**4
                if omega_sq > 0:
                    omega_calc = np.sqrt(omega_sq)
                    freq_calc = omega_calc / (2 * np.pi)
                    
                    # Check if this is a solution (freq_calc ≈ freq)
                    if abs(freq_calc - freq) / freq < 0.01:
                        omega_list.append(2 * np.pi * freq)
        
        # Remove duplicates
        if omega_list:
            omega_list = sorted(list(set([round(w, 2) for w in omega_list])))
            omega = np.array(omega_list)
        else:
            omega = np.array([])
        
        modes = np.ones((1, len(omega))) if len(omega) > 0 else np.ones((1, 1))
        
        return omega[:n_modes], modes
    
    def compute_dispersion_curves(self, n_points: int = 50, n_modes: int = 15) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dispersion curves along irreducible Brillouin zone path.
        
        Parameters
        ----------
        n_points : int
            Points per segment of BZ path
        n_modes : int
            Number of modes to track
        
        Returns
        -------
        k_cumulative : np.ndarray
            Cumulative distance along BZ path
        omega_bands : np.ndarray
            Frequency bands (n_total_points x n_modes)
        """
        k_path = self.analyzer.irreducible_brillouin_zone_path(n_points)
        n_total = len(k_path)
        omega_bands = np.zeros((n_total, n_modes))
        k_cumulative = np.zeros(n_total)
        
        for i, (kx, ky) in enumerate(k_path):
            omega, _ = self.solve_dispersion(kx, ky, n_modes)
            n_found = min(len(omega), n_modes)
            omega_bands[i, :n_found] = omega[:n_found]
            
            if i > 0:
                dk = np.sqrt((kx - k_path[i-1, 0])**2 + (ky - k_path[i-1, 1])**2)
                k_cumulative[i] = k_cumulative[i-1] + dk
        
        return k_cumulative, omega_bands
    
    def detect_bandgaps(self, k_cumulative: np.ndarray = None, omega_bands: np.ndarray = None,
                        freq_range: Tuple[float, float] = None) -> List[Tuple[float, float]]:
        """
        Detect bandgaps from effective mass density analysis.
        
        Parameters
        ----------
        k_cumulative, omega_bands : np.ndarray, optional
            Dispersion data (not used in this implementation)
        freq_range : tuple, optional
            Frequency range to search [Hz]
        
        Returns
        -------
        bandgaps : list
            List of (f_lower, f_upper) bandgap boundaries [Hz]
        """
        if freq_range is None:
            f_r = self.analytical_resonance_frequency()
            freq_range = (0.5 * f_r, 3.0 * f_r)
        
        return self._find_bandgap_edges(freq_range[0], freq_range[1])
    
    def analytical_resonance_frequency(self) -> float:
        """
        Compute analytical local resonance frequency.
        
        For a single resonator: f_r = √(k_r/m_r)/(2π)
        For multiple resonators: uses effective mass and stiffness.
        
        Returns
        -------
        f_r : float
            Analytical resonance frequency [Hz]
        """
        if len(self.resonators) == 1:
            return self.resonators[0].resonance_frequency
        
        # Effective properties for multiple resonators
        total_k = sum(r.k_r for r in self.resonators)
        total_m = sum(r.m_r for r in self.resonators)
        return np.sqrt(total_k / total_m) / (2 * np.pi)
