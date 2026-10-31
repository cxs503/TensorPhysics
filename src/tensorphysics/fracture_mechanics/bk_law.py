"""Benzeggagh-Kenane (BK) mixed-mode fracture criterion.

The BK law provides an empirical relationship for mixed-mode fracture toughness:

    G_c(ψ) = G_Ic + (G_IIc - G_Ic) * ψ^η

where:
    G_Ic  = Mode I fracture toughness [J/m²]
    G_IIc = Mode II fracture toughness [J/m²]
    ψ     = G_II / (G_I + G_II) is the mixed-mode ratio
    η     = Material parameter (typically 1.0 - 2.5)

References:
    Benzeggagh, M.L., Kenane, M. (1996). "Measurement of mixed-mode delamination
    fracture toughness of unidirectional glass/epoxy composites with mixed-mode
    bending apparatus." Composites Science and Technology, 56(4), 439-449.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class BKLaw:
    """Benzeggagh-Kenane mixed-mode fracture criterion.
    
    Attributes:
        G_Ic: Mode I fracture toughness [J/m²]
        G_IIc: Mode II fracture toughness [J/m²]
        eta: BK exponent (material parameter)
    """
    G_Ic: float
    G_IIc: float
    eta: float = 1.5
    
    def __post_init__(self):
        """Validate parameters."""
        if self.G_Ic <= 0:
            raise ValueError(f"G_Ic must be positive, got {self.G_Ic}")
        if self.G_IIc <= 0:
            raise ValueError(f"G_IIc must be positive, got {self.G_IIc}")
        if self.eta <= 0:
            raise ValueError(f"eta must be positive, got {self.eta}")
    
    def effective_toughness(self, mixed_mode_ratio: float) -> float:
        """Compute effective fracture toughness for given mixed-mode ratio.
        
        Args:
            mixed_mode_ratio: ψ = G_II / (G_I + G_II), range [0, 1]
            
        Returns:
            G_c: Effective fracture toughness [J/m²]
        """
        if not 0 <= mixed_mode_ratio <= 1:
            raise ValueError(f"mixed_mode_ratio must be in [0, 1], got {mixed_mode_ratio}")
        
        psi = mixed_mode_ratio
        G_c = self.G_Ic + (self.G_IIc - self.G_Ic) * (psi ** self.eta)
        return G_c
    
    def effective_toughness_array(self, mixed_mode_ratios: np.ndarray) -> np.ndarray:
        """Compute effective toughness for array of mixed-mode ratios.
        
        Args:
            mixed_mode_ratios: Array of ψ values
            
        Returns:
            Array of G_c values
        """
        psi = np.asarray(mixed_mode_ratios)
        if np.any(psi < 0) or np.any(psi > 1):
            raise ValueError("All mixed_mode_ratios must be in [0, 1]")
        
        G_c = self.G_Ic + (self.G_IIc - self.G_Ic) * (psi ** self.eta)
        return G_c


def compute_mixed_mode_ratio(G_I: float, G_II: float) -> float:
    """Compute mixed-mode ratio ψ = G_II / (G_I + G_II).
    
    Args:
        G_I: Mode I energy release rate [J/m²]
        G_II: Mode II energy release rate [J/m²]
        
    Returns:
        ψ: Mixed-mode ratio in [0, 1]
    """
    G_total = G_I + G_II
    if G_total <= 0:
        return 0.0
    return G_II / G_total


def compute_effective_fracture_toughness(
    G_I: float,
    G_II: float,
    G_Ic: float,
    G_IIc: float,
    eta: float = 1.5
) -> Tuple[float, float]:
    """Compute effective fracture toughness using BK law.
    
    Args:
        G_I: Mode I energy release rate [J/m²]
        G_II: Mode II energy release rate [J/m²]
        G_Ic: Mode I fracture toughness [J/m²]
        G_IIc: Mode II fracture toughness [J/m²]
        eta: BK exponent
        
    Returns:
        Tuple of (G_c_effective, mixed_mode_ratio)
    """
    psi = compute_mixed_mode_ratio(G_I, G_II)
    bk = BKLaw(G_Ic, G_IIc, eta)
    G_c = bk.effective_toughness(psi)
    return G_c, psi
