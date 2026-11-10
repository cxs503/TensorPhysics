"""
Bilinear Traction-Separation Law for Ice-Steel Interface

Implements the constitutive relation:
    T = K * δ  (elastic, δ < δ_0)
    T = σ_max * (δ_f - δ) / (δ_f - δ_0)  (softening, δ_0 ≤ δ < δ_f)
    T = 0  (failed, δ ≥ δ_f)

where:
    K = initial stiffness [Pa/m]
    σ_max = strength [Pa]
    δ_0 = σ_max / K = onset of damage [m]
    δ_f = 2*G_c / σ_max = complete failure [m]
    G_c = fracture energy [J/m²]

The area under the traction-separation curve equals G_c (energy conservation).
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class TractionSeparationState:
    """State of traction-separation law at a given separation.
    
    Attributes:
        delta: Separation [m]
        traction: Traction [Pa]
        damage: Damage variable D ∈ [0, 1]
        stiffness: Current tangent stiffness [Pa/m]
        energy_dissipated: Energy dissipated per unit area [J/m²]
        energy_stored: Elastic energy stored per unit area [J/m²]
        failed: Whether fully failed (D ≥ 1)
    """
    delta: float = 0.0
    traction: float = 0.0
    damage: float = 0.0
    stiffness: float = 0.0
    energy_dissipated: float = 0.0
    energy_stored: float = 0.0
    failed: bool = False


class BilinearTractionSeparation:
    """Bilinear traction-separation law with linear softening.
    
    The constitutive law is defined by:
    - Initial stiffness K (penalty stiffness for numerical stability)
    - Strength σ_max (peak traction before damage)
    - Fracture energy G_c (area under softening curve)
    
    Critical separations:
    - δ_0 = σ_max / K (onset of damage)
    - δ_f = 2*G_c / σ_max (complete failure)
    
    The bilinear law ensures:
    1. Energy conservation: ∫T dδ = G_c
    2. Irreversibility: damage cannot heal (D is monotonically increasing)
    3. Thermodynamic consistency: dissipation ≥ 0
    """
    
    def __init__(
        self,
        K: float,
        sigma_max: float,
        G_c: float,
    ):
        """Initialize bilinear traction-separation law.
        
        Args:
            K: Initial stiffness [Pa/m]
            sigma_max: Strength (peak traction) [Pa]
            G_c: Fracture energy [J/m²]
        
        Raises:
            ValueError: If parameters are non-physical
        """
        if K <= 0:
            raise ValueError(f"Stiffness K must be positive, got {K}")
        if sigma_max <= 0:
            raise ValueError(f"Strength sigma_max must be positive, got {sigma_max}")
        if G_c <= 0:
            raise ValueError(f"Fracture energy G_c must be positive, got {G_c}")
        
        self.K = K
        self.sigma_max = sigma_max
        self.G_c = G_c
        
        # Critical separations
        self.delta_0 = sigma_max / K  # Onset of damage [m]
        self.delta_f = 2.0 * G_c / sigma_max  # Complete failure [m]
        
        # Verify energy consistency: area under bilinear curve = G_c
        # Triangle area = 0.5 * sigma_max * delta_f = 0.5 * sigma_max * (2*G_c/sigma_max) = G_c ✓
        self._verify_energy_consistency()
    
    def _verify_energy_consistency(self):
        """Verify that the area under the traction-separation curve equals G_c."""
        # For bilinear law: area = 0.5 * sigma_max * delta_f
        area = 0.5 * self.sigma_max * self.delta_f
        rel_error = abs(area - self.G_c) / self.G_c
        if rel_error > 1e-10:
            raise ValueError(
                f"Energy consistency violated: area={area}, G_c={self.G_c}, "
                f"rel_error={rel_error}"
            )
    
    def compute_traction(
        self,
        delta: float,
        delta_max_prev: float = 0.0,
        damage_prev: float = 0.0,
    ) -> TractionSeparationState:
        """Compute traction for given separation (monotonic loading).
        
        Args:
            delta: Current separation [m]
            delta_max_prev: Previous maximum separation [m]
            damage_prev: Previous damage variable
        
        Returns:
            TractionSeparationState with updated state
        """
        # Irreversibility: track maximum separation
        delta_max = max(delta, delta_max_prev)
        
        # Elastic regime
        if delta_max <= self.delta_0:
            traction = self.K * delta
            damage = 0.0
            # Tangent stiffness
            stiffness = self.K
            energy_stored = 0.5 * self.K * delta**2
            energy_dissipated = 0.0
        
        # Softening regime
        elif delta_max < self.delta_f:
            # Damage variable (linear softening)
            damage = (delta_max - self.delta_0) / (self.delta_f - self.delta_0)
            damage = max(damage, damage_prev)  # No healing
            
            # Traction: linear softening from sigma_max to 0
            # T = sigma_max * (delta_f - delta_max) / (delta_f - delta_0)
            # But we need to account for unloading/reloading
            # For monotonic loading: use delta_max
            # For unloading: use secant stiffness
            if delta <= delta_max:
                # Unloading or reloading: secant stiffness
                if delta_max > 1e-30:
                    T_at_delta_max = self.sigma_max * (self.delta_f - delta_max) / (self.delta_f - self.delta_0)
                    secant_K = T_at_delta_max / delta_max
                    traction = secant_K * delta
                    stiffness = secant_K
                else:
                    traction = 0.0
                    stiffness = 0.0
            else:
                # Continuing to load: follow softening curve
                traction = self.sigma_max * (self.delta_f - delta) / (self.delta_f - self.delta_0)
                # Tangent stiffness (negative in softening)
                stiffness = -self.sigma_max / (self.delta_f - self.delta_0)
            
            # Energy components
            energy_stored = 0.5 * stiffness * delta**2 if stiffness > 0 else 0.0
            # Total area under curve up to delta_max
            if delta_max <= self.delta_0:
                total_area = 0.5 * self.K * delta_max**2
            else:
                # Triangle to peak + trapezoid in softening
                area_peak = 0.5 * self.sigma_max * self.delta_0
                # Softening part: trapezoid from delta_0 to delta_max
                T_at_delta_max = self.sigma_max * (self.delta_f - delta_max) / (self.delta_f - self.delta_0)
                area_softening = 0.5 * (self.sigma_max + T_at_delta_max) * (delta_max - self.delta_0)
                total_area = area_peak + area_softening
            
            energy_dissipated = total_area - energy_stored
        
        # Fully failed
        else:
            damage = 1.0
            traction = 0.0
            stiffness = 0.0
            energy_stored = 0.0
            energy_dissipated = self.G_c  # All energy dissipated
        
        failed = damage >= 1.0 - 1e-10
        
        return TractionSeparationState(
            delta=delta,
            traction=traction,
            damage=damage,
            stiffness=stiffness,
            energy_dissipated=energy_dissipated,
            energy_stored=energy_stored,
            failed=failed,
        )
    
    def compute_path(
        self,
        delta_path: np.ndarray,
    ) -> list[TractionSeparationState]:
        """Compute traction-separation response along a loading path.
        
        Args:
            delta_path: Array of separations [m]
        
        Returns:
            List of TractionSeparationState at each point
        """
        states = []
        delta_max = 0.0
        damage = 0.0
        
        for delta in delta_path:
            state = self.compute_traction(delta, delta_max, damage)
            states.append(state)
            delta_max = max(delta, delta_max)
            damage = state.damage
        
        return states
    
    def get_critical_separations(self) -> dict[str, float]:
        """Get critical separations for visualization/analysis.
        
        Returns:
            Dictionary with delta_0 and delta_f
        """
        return {
            "delta_0": self.delta_0,
            "delta_f": self.delta_f,
        }
    
    def __repr__(self) -> str:
        return (
            f"BilinearTractionSeparation(K={self.K:.3e}, "
            f"sigma_max={self.sigma_max:.3e}, G_c={self.G_c:.3e})"
        )
