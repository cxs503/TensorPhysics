"""Interface contract: SPH Aeration → Ship Resistance Correction

Defines the data exchange protocol between TensorSPH aeration module
and ship-performance-platform resistance correction module.

Interface specification:
- Input: void fraction field from SPH solver
- Output: mean void fraction per hull patch (10 patches along LOA)
- Units: dimensionless [0,1]
- Temporal: steady-state or time-averaged (T > 10 wave periods)
- Spatial: patch-averaged over hull panels
- Tolerance: ≤2% deviation between SPH output and resistance module input

Contract version: 1.0
Last validated: 2026-11-10
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class AerationInterfaceContract:
    """Interface contract for SPH aeration → resistance correction.
    
    Attributes:
        version: Contract version string
        n_patches: Number of hull patches (default 10)
        alpha_units: Void fraction units (dimensionless)
        temporal_mode: "steady" or "time_averaged"
        spatial_mode: "patch_averaged"
        tolerance_pct: Maximum allowed deviation (%)
    """
    version: str = "1.0"
    n_patches: int = 10
    alpha_units: str = "dimensionless"
    temporal_mode: str = "steady"
    spatial_mode: str = "patch_averaged"
    tolerance_pct: float = 2.0
    
    def validate_alpha_range(self, alpha_array: np.ndarray) -> bool:
        """Check void fraction is in valid range [0, 1]."""
        return bool(np.all(alpha_array >= 0) and np.all(alpha_array <= 1))
    
    def validate_patch_count(self, alpha_array: np.ndarray) -> bool:
        """Check array has correct number of patches."""
        return len(alpha_array) == self.n_patches


class SPHToResistanceInterface:
    """Adapter: SPH aeration output → resistance correction input.
    
    This class transforms the SPH void fraction field into the format
    expected by the ship-performance-platform resistance correction module.
    
    Transformation:
    1. Extract hull-surface void fraction from SPH particles
    2. Average over hull patches (n_patches along LOA)
    3. Apply temporal averaging if transient
    4. Validate against interface contract
    
    Usage:
        interface = SPHToResistanceInterface()
        alpha_hull = interface.transform(sph_result)
        correction = resistance_module.apply(alpha_hull)
    """
    
    def __init__(self, contract: Optional[AerationInterfaceContract] = None):
        self.contract = contract or AerationInterfaceContract()
        
    def transform(
        self,
        sph_void_field,
        hull_geometry: Optional[Dict] = None,
    ) -> np.ndarray:
        """Transform SPH void field to hull-patch void fractions.
        
        Args:
            sph_void_field: VoidFractionField from SPH solver
            hull_geometry: Optional hull geometry dict with 'patches' key
            
        Returns:
            alpha_hull: Mean void fraction per patch [n_patches]
        """
        x = sph_void_field.x
        alpha = sph_void_field.alpha
        
        n_patches = self.contract.n_patches
        L = np.max(x[:, 0]) - np.min(x[:, 0])
        x_min = np.min(x[:, 0])
        
        alpha_hull = np.zeros(n_patches)
        
        for i in range(n_patches):
            # Patch boundaries
            x_lo = x_min + i * L / n_patches
            x_hi = x_min + (i + 1) * L / n_patches
            
            # Select particles in this patch (near hull surface)
            patch_mask = (x[:, 0] >= x_lo) & (x[:, 0] < x_hi)
            
            # Near-surface particles (y < 0.1*L above nominal waterline)
            surface_mask = x[:, 1] < 0.1 * L
            
            combined_mask = patch_mask & surface_mask
            
            if np.any(combined_mask):
                alpha_hull[i] = np.mean(alpha[combined_mask])
            else:
                alpha_hull[i] = 0.0
        
        return alpha_hull
    
    def validate(self, alpha_hull: np.ndarray) -> Tuple[bool, str]:
        """Validate transformed data against contract.
        
        Returns:
            (is_valid, message)
        """
        if not self.contract.validate_alpha_range(alpha_hull):
            return False, "Void fraction out of range [0, 1]"
        
        if not self.contract.validate_patch_count(alpha_hull):
            return False, f"Expected {self.contract.n_patches} patches, got {len(alpha_hull)}"
        
        return True, "Interface contract satisfied"


def validate_interface_consistency(
    sph_output: np.ndarray,
    resistance_input: np.ndarray,
    tolerance_pct: float = 2.0,
) -> Dict:
    """Validate consistency between SPH output and resistance module input.
    
    Checks that the data passed through the interface has not been corrupted
    and meets the ≤2% deviation requirement.
    
    Args:
        sph_output: Void fraction from SPH solver [n_patches]
        resistance_input: Void fraction received by resistance module [n_patches]
        tolerance_pct: Maximum allowed relative deviation (%)
        
    Returns:
        Dict with validation results:
            - consistent: bool
            - max_deviation_pct: float
            - mean_deviation_pct: float
            - patch_deviations: array of per-patch deviations
    """
    sph_output = np.asarray(sph_output)
    resistance_input = np.asarray(resistance_input)
    
    if len(sph_output) != len(resistance_input):
        return {
            "consistent": False,
            "max_deviation_pct": float("inf"),
            "mean_deviation_pct": float("inf"),
            "patch_deviations": None,
            "error": "Array length mismatch",
        }
    
    # Compute relative deviations
    # Avoid division by zero: use absolute deviation for near-zero values
    mask = sph_output > 1e-6
    deviations = np.zeros_like(sph_output)
    
    if np.any(mask):
        deviations[mask] = np.abs(
            (resistance_input[mask] - sph_output[mask]) / sph_output[mask]
        ) * 100.0
    
    # For near-zero values, use absolute tolerance
    deviations[~mask] = np.abs(resistance_input[~mask] - sph_output[~mask]) * 100.0
    
    max_dev = float(np.max(deviations))
    mean_dev = float(np.mean(deviations))
    consistent = max_dev <= tolerance_pct
    
    return {
        "consistent": consistent,
        "max_deviation_pct": max_dev,
        "mean_deviation_pct": mean_dev,
        "patch_deviations": deviations.tolist(),
        "tolerance_pct": tolerance_pct,
    }
