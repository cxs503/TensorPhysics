"""Interface contract: Phase-Field Fracture → Structural Integrity Assessment

Defines the data exchange protocol between TensorFEM phase-field fracture module
and ship-performance-platform structural integrity evaluation module.

Interface specification:
- Input: crack path and crack length from phase-field solver
- Output: crack length a(t) at specified time/load points
- Units: meters
- Temporal: quasi-static load steps
- Spatial: crack tip position [x, y] or [x, y, z]
- Tolerance: ≤2% deviation between FEM output and structural module input

Contract version: 1.0
Last validated: 2026-11-10
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple


@dataclass
class FractureInterfaceContract:
    """Interface contract for FEM fracture → structural integrity.
    
    Attributes:
        version: Contract version string
        crack_length_units: Length units (meters)
        temporal_mode: "quasi_static" or "dynamic"
        output_format: "crack_length_history" or "crack_path"
        tolerance_pct: Maximum allowed deviation (%)
    """
    version: str = "1.0"
    crack_length_units: str = "meters"
    temporal_mode: str = "quasi_static"
    output_format: str = "crack_length_history"
    tolerance_pct: float = 2.0
    
    def validate_crack_length(self, crack_length: float) -> bool:
        """Check crack length is physically reasonable."""
        return crack_length >= 0 and crack_length < 100.0  # < 100m for ships
    
    def validate_history(self, crack_lengths: np.ndarray) -> bool:
        """Check crack length history is monotonically increasing."""
        # Allow small numerical noise
        return bool(np.all(np.diff(crack_lengths) >= -1e-10))


class FEMToStructuralInterface:
    """Adapter: FEM fracture output → structural integrity input.
    
    This class transforms the phase-field fracture result into the format
    expected by the ship-performance-platform structural integrity module.
    
    Transformation:
    1. Extract crack length history from phase-field evolution
    2. Identify crack tip position at each load step
    3. Compute energy release rate history
    4. Validate against interface contract
    
    Usage:
        interface = FEMToStructuralInterface()
        crack_data = interface.transform(fracture_result)
        integrity = structural_module.assess(crack_data)
    """
    
    def __init__(self, contract: Optional[FractureInterfaceContract] = None):
        self.contract = contract or FractureInterfaceContract()
        
    def transform(
        self,
        fracture_result,
        critical_G: Optional[float] = None,
    ) -> Dict:
        """Transform fracture result to structural integrity input.
        
        Args:
            fracture_result: FractureResult from phase-field solver
            critical_G: Critical energy release rate for failure [J/m²]
            
        Returns:
            Dict with:
                - crack_length: Final crack length [m]
                - crack_length_history: Array of crack lengths [n_steps]
                - crack_tip: Final crack tip position [2] or [3]
                - load_steps: Applied load levels [n_steps]
                - G_history: Energy release rate history [n_steps]
                - failure_index: max(G/G_c) over history
        """
        crack_length = fracture_result.crack_path.length
        crack_length_history = fracture_result.crack_length_history
        crack_tip = fracture_result.crack_path.tip_position
        load_steps = fracture_result.load_steps
        G_history = fracture_result.G_history
        
        # Compute failure index
        if critical_G is not None and critical_G > 0:
            failure_index = float(np.max(G_history) / critical_G)
        else:
            failure_index = 0.0
        
        return {
            "crack_length": crack_length,
            "crack_length_history": crack_length_history,
            "crack_tip": crack_tip,
            "load_steps": load_steps,
            "G_history": G_history,
            "failure_index": failure_index,
            "convergence_status": fracture_result.convergence_status,
        }
    
    def validate(self, crack_data: Dict) -> Tuple[bool, str]:
        """Validate transformed data against contract.
        
        Returns:
            (is_valid, message)
        """
        crack_length = crack_data.get("crack_length", -1)
        
        if not self.contract.validate_crack_length(crack_length):
            return False, f"Crack length {crack_length} out of physical range"
        
        crack_history = crack_data.get("crack_length_history")
        if crack_history is not None:
            if not self.contract.validate_history(crack_history):
                return False, "Crack length history not monotonically increasing"
        
        return True, "Interface contract satisfied"


def validate_crack_interface_consistency(
    fem_output: np.ndarray,
    structural_input: np.ndarray,
    tolerance_pct: float = 2.0,
) -> Dict:
    """Validate consistency between FEM output and structural module input.
    
    Checks that crack length data passed through the interface has not been
    corrupted and meets the ≤2% deviation requirement.
    
    Args:
        fem_output: Crack length history from FEM solver [n_steps]
        structural_input: Crack length history received by structural module [n_steps]
        tolerance_pct: Maximum allowed relative deviation (%)
        
    Returns:
        Dict with validation results:
            - consistent: bool
            - max_deviation_pct: float
            - mean_deviation_pct: float
            - step_deviations: array of per-step deviations
    """
    fem_output = np.asarray(fem_output)
    structural_input = np.asarray(structural_input)
    
    if len(fem_output) != len(structural_input):
        return {
            "consistent": False,
            "max_deviation_pct": float("inf"),
            "mean_deviation_pct": float("inf"),
            "step_deviations": None,
            "error": "Array length mismatch",
        }
    
    # Compute relative deviations
    # Avoid division by zero: use absolute deviation for near-zero values
    mask = fem_output > 1e-6
    deviations = np.zeros_like(fem_output)
    
    if np.any(mask):
        deviations[mask] = np.abs(
            (structural_input[mask] - fem_output[mask]) / fem_output[mask]
        ) * 100.0
    
    # For near-zero values, use absolute tolerance
    deviations[~mask] = np.abs(structural_input[~mask] - fem_output[~mask]) * 100.0
    
    max_dev = float(np.max(deviations))
    mean_dev = float(np.mean(deviations))
    consistent = max_dev <= tolerance_pct
    
    return {
        "consistent": consistent,
        "max_deviation_pct": max_dev,
        "mean_deviation_pct": mean_dev,
        "step_deviations": deviations.tolist(),
        "tolerance_pct": tolerance_pct,
    }
