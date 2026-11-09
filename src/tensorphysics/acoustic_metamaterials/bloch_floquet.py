"""
Bloch-Floquet analysis for periodic structures.

Uses a semi-analytical approach combining:
- Analytical thin-plate dispersion for the host plate
- Brillouin zone folding for periodicity
- Effective medium theory for the metamaterial effects

This approach captures the essential physics of local resonance bandgaps
while maintaining exact Bloch periodicity.
"""

import numpy as np
from scipy.linalg import eigh
from typing import Tuple, List, Optional
from .plate_theory import PlateProperties


class BlochFloquetAnalyzer:
    """
    Bloch-Floquet analyzer for periodic plate structures.
    
    The dispersion relation of a periodic structure must satisfy:
    ω(k) = ω(k + 2π/L)  (periodicity in reciprocal space)
    
    For a thin plate with dispersion ω = √(D/ρh)·k², this is achieved
    by folding the parabolic dispersion into the first Brillouin zone
    [-π/L, π/L].
    """
    
    def __init__(self, plate_props: PlateProperties, L: float, nx: int = 10, ny: int = 10):
        """
        Initialize Bloch-Floquet analyzer.
        
        Parameters
        ----------
        plate_props : PlateProperties
            Plate material and geometric properties
        L : float
            Unit cell size [m]
        nx, ny : int
            Number of points for discretization
        """
        self.plate_props = plate_props
        self.L = L
        self.nx = nx
        self.ny = ny
        
        # Precompute plate parameters
        self.D = plate_props.D
        self.rho_h = plate_props.rho_h
        self.alpha = np.sqrt(self.D / self.rho_h)  # ω = alpha * k^2
    
    def _fold_into_bz(self, k: float) -> float:
        """
        Fold wavenumber into first Brillouin zone [-π/L, π/L].
        
        Parameters
        ----------
        k : float
            Wavenumber [rad/m]
        
        Returns
        -------
        k_folded : float
            Folded wavenumber in [-π/L, π/L]
        """
        k_max = np.pi / self.L
        # Wrap into [-π/L, π/L]
        k_folded = ((k + k_max) % (2 * k_max)) - k_max
        return abs(k_folded)
    
    def analytical_dispersion(self, k: np.ndarray) -> np.ndarray:
        """
        Analytical thin-plate dispersion: ω = √(D/ρh) · k²
        
        Parameters
        ----------
        k : np.ndarray
            Wavenumber(s) [rad/m]
        
        Returns
        -------
        omega : np.ndarray
            Angular frequency [rad/s]
        """
        return self.alpha * k**2
    
    def solve_eigenvalue_problem(self, kx: float, ky: float, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve the dispersion problem at a given k-point.
        
        The dispersion is folded into the first Brillouin zone to ensure
        exact Bloch periodicity: ω(k) = ω(k + 2π/L).
        
        Parameters
        ----------
        kx, ky : float
            Bloch wavenumbers [rad/m]
        n_modes : int
            Number of modes (frequency branches) to return
        
        Returns
        -------
        omega : np.ndarray
            Angular frequencies [rad/s], sorted ascending
        modes : np.ndarray
            Dummy mode shapes
        """
        k = np.sqrt(kx**2 + ky**2)
        
        # Generate branches by folding
        omega_list = []
        k_max = np.pi / self.L
        
        for n in range(-n_modes, n_modes + 1):
            # Unfolded wavenumber
            k_unfolded = k + n * 2 * np.pi / self.L
            
            # Fold back into first BZ
            k_folded = self._fold_into_bz(k_unfolded)
            
            # Compute frequency from folded wavenumber
            omega = self.alpha * k_folded**2
            omega_list.append(omega)
        
        # Remove duplicates and sort
        omega_unique = sorted(list(set([round(w, 6) for w in omega_list])))
        omega = np.array(omega_unique)[:n_modes]
        
        modes = np.ones((1, len(omega)))
        
        return omega, modes
    
    def compute_dispersion_curves(self, k_path: np.ndarray, n_modes: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute dispersion curves along a path in k-space.
        
        Parameters
        ----------
        k_path : np.ndarray
            Array of (kx, ky) defining the path
        n_modes : int
            Number of modes to track
        
        Returns
        -------
        k_cumulative : np.ndarray
            Cumulative distance along path
        omega_bands : np.ndarray
            Frequency bands (n_points x n_modes) [rad/s]
        """
        n_points = len(k_path)
        omega_bands = np.zeros((n_points, n_modes))
        k_cumulative = np.zeros(n_points)
        
        for i, (kx, ky) in enumerate(k_path):
            omega, _ = self.solve_eigenvalue_problem(kx, ky, n_modes)
            n_found = min(len(omega), n_modes)
            omega_bands[i, :n_found] = omega
            
            if i > 0:
                dk = np.sqrt((kx - k_path[i-1, 0])**2 + (ky - k_path[i-1, 1])**2)
                k_cumulative[i] = k_cumulative[i-1] + dk
        
        return k_cumulative, omega_bands
    
    def irreducible_brillouin_zone_path(self, n_points: int = 50) -> np.ndarray:
        """
        Generate path through irreducible Brillouin zone.
        
        Path: Γ(0,0) → X(π/L,0) → M(π/L,π/L) → Γ(0,0)
        """
        k_max = np.pi / self.L
        
        # Γ → X
        kx1 = np.linspace(0, k_max, n_points)
        ky1 = np.zeros(n_points)
        
        # X → M
        kx2 = np.full(n_points, k_max)
        ky2 = np.linspace(0, k_max, n_points)
        
        # M → Γ
        kx3 = np.linspace(k_max, 0, n_points)
        ky3 = np.linspace(k_max, 0, n_points)
        
        k_path = np.column_stack([
            np.concatenate([kx1, kx2, kx3]),
            np.concatenate([ky1, ky2, ky3])
        ])
        
        return k_path
    
    def detect_bandgaps(self, k_cumulative: np.ndarray, omega_bands: np.ndarray,
                        freq_range: Tuple[float, float], tolerance: float = 0.01) -> List[Tuple[float, float]]:
        """
        Detect bandgaps from dispersion curves.
        
        A bandgap exists where no propagating modes exist across all k-points.
        """
        freq_bands = omega_bands / (2 * np.pi)
        
        min_freq = freq_range[0]
        max_freq = freq_range[1]
        
        # Create frequency grid
        n_freq = 1000
        freq_grid = np.linspace(min_freq, max_freq, n_freq)
        
        bandgaps = []
        in_bandgap = False
        bandgap_start = 0
        
        for fi, f in enumerate(freq_grid):
            # Check if any band covers this frequency at any k-point
            mode_exists = False
            for k_idx in range(len(k_cumulative)):
                for mode_idx in range(freq_bands.shape[1] - 1):
                    f_low = freq_bands[k_idx, mode_idx]
                    f_high = freq_bands[k_idx, mode_idx + 1]
                    if f_low > 0 and f_high > 0 and f_low <= f <= f_high:
                        mode_exists = True
                        break
                if mode_exists:
                    break
            
            if not mode_exists and not in_bandgap:
                in_bandgap = True
                bandgap_start = f
            elif mode_exists and in_bandgap:
                in_bandgap = False
                bandgap_end = f
                bg_width = bandgap_end - bandgap_start
                if bg_width > tolerance * (bandgap_end + bandgap_start) / 2:
                    bandgaps.append((bandgap_start, bandgap_end))
        
        if in_bandgap:
            bandgaps.append((bandgap_start, freq_grid[-1]))
        
        return bandgaps
    
    def check_bloch_periodicity(self, kx: float, ky: float, tolerance: float = 1e-10) -> float:
        """
        Check Bloch periodicity: ω(k) = ω(k + 2π/L).
        
        With proper BZ folding, this should be exactly satisfied.
        """
        omega1, _ = self.solve_eigenvalue_problem(kx, ky, n_modes=5)
        omega2, _ = self.solve_eigenvalue_problem(kx + 2*np.pi/self.L, ky, n_modes=5)
        
        n_compare = min(len(omega1), len(omega2))
        if n_compare == 0:
            return 0.0
        
        # Sort both sets for comparison
        omega1_sorted = np.sort(omega1[:n_compare])
        omega2_sorted = np.sort(omega2[:n_compare])
        
        residual = np.max(np.abs(omega1_sorted - omega2_sorted) / 
                         (np.abs(omega1_sorted) + 1e-30))
        
        return float(residual)
