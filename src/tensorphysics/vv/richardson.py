"""
Richardson Extrapolation Engine with GCI (Grid Convergence Index).

Implements ASME V&V 20 standard methodology for grid convergence assessment.
Reference: Roache, P.J., "Perspective: A Method for Uniform Reporting of
Grid Refinement Studies", ASME J. Fluids Eng., 116(3), 1994, pp. 405-413.
"""

import numpy as np
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple


class ConvergenceStatus(Enum):
    """Convergence determination status."""
    PASS = "PASS"           # Converged with acceptable GCI
    FAIL = "FAIL"           # Not converged or divergent
    UNRESOLVED = "UNRESOLVED"  # Cannot determine (oscillatory, etc.)


@dataclass
class GCIResult:
    """Result of Richardson extrapolation and GCI calculation."""
    # Input data
    f_coarse: float         # Coarsest grid solution
    f_medium: float         # Medium grid solution
    f_fine: float           # Finest grid solution
    r_coarse: float         # Refinement ratio medium->coarse
    r_fine: float           # Refinement ratio fine->medium
    
    # Computed quantities
    p: float                # Observed order of convergence
    f_ext: float            # Extrapolated value (zero mesh spacing)
    epsilon_fine: float     # Relative error fine-medium (%)
    epsilon_coarse: float   # Relative error coarse-medium (%)
    gci_fine: float         # GCI for fine grid (%)
    gci_coarse: float       # GCI for coarse grid (%)
    gci_ratio: float        # GCI_coarse / (r^p * GCI_fine) - should be ~1.0
    
    # Assessment
    status: ConvergenceStatus
    asymptotic: bool        # Whether in asymptotic range
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'f_coarse': self.f_coarse,
            'f_medium': self.f_medium,
            'f_fine': self.f_fine,
            'r_coarse': self.r_coarse,
            'r_fine': self.r_fine,
            'p': self.p,
            'f_ext': self.f_ext,
            'epsilon_fine': self.epsilon_fine,
            'epsilon_coarse': self.epsilon_coarse,
            'gci_fine': self.gci_fine,
            'gci_coarse': self.gci_coarse,
            'gci_ratio': self.gci_ratio,
            'status': self.status.value,
            'asymptotic': self.asymptotic,
        }


class RichardsonExtrapolation:
    """
    Richardson extrapolation with GCI calculation.
    
    Parameters
    ----------
    safety_factor : float, optional
        Safety factor Fs for GCI calculation. Default is 1.25 for 3 grids.
    gci_threshold : float, optional
        GCI threshold (%) for PASS/FAIL determination. Default is 5.0%.
    """
    
    def __init__(self, safety_factor: float = 1.25, gci_threshold: float = 5.0):
        self.safety_factor = safety_factor
        self.gci_threshold = gci_threshold
    
    def compute(
        self,
        f_fine: float,
        f_medium: float,
        f_coarse: float,
        r_fine: float = 2.0,
        r_coarse: float = 2.0,
    ) -> GCIResult:
        """
        Compute Richardson extrapolation and GCI.
        
        Parameters
        ----------
        f_fine : float
            Solution on finest grid
        f_medium : float
            Solution on medium grid
        f_coarse : float
            Solution on coarsest grid
        r_fine : float
            Refinement ratio between fine and medium grids (h_medium/h_fine)
        r_coarse : float
            Refinement ratio between medium and coarse grids (h_coarse/h_medium)
            
        Returns
        -------
        GCIResult
            Complete result with extrapolated value, GCI, and convergence status
        """
        # Compute differences
        eps1 = f_fine - f_medium  # Fine - Medium
        eps2 = f_medium - f_coarse  # Medium - Coarse
        
        # Check for oscillatory convergence
        if eps1 * eps2 <= 0:
            # Oscillatory - cannot determine convergence order reliably
            return self._handle_oscillatory(f_fine, f_medium, f_coarse, r_fine, r_coarse)
        
        # Compute observed order of convergence p
        # Using the formula: p = ln|eps2/eps1| / ln(r)
        r = r_fine  # Assume uniform refinement for simplicity
        if r_coarse != r_fine:
            # Non-uniform refinement - use generalized formula
            p = self._compute_p_nonuniform(eps1, eps2, r_fine, r_coarse)
        else:
            p = np.log(abs(eps2 / eps1)) / np.log(r)
        
        # Compute extrapolated value
        f_ext = f_fine + eps1 / (r**p - 1)
        
        # Compute relative errors
        if abs(f_fine) > 1e-14:
            epsilon_fine = 100.0 * abs(eps1 / f_fine)
        else:
            epsilon_fine = 100.0 * abs(eps1)
            
        if abs(f_coarse) > 1e-14:
            epsilon_coarse = 100.0 * abs(eps2 / f_coarse)
        else:
            epsilon_coarse = 100.0 * abs(eps2)
        
        # Compute GCI
        gci_fine = self.safety_factor * epsilon_fine / (r**p - 1)
        gci_coarse = self.safety_factor * epsilon_coarse / (r**p - 1)
        
        # GCI ratio check (should be ~1.0 if in asymptotic range)
        gci_ratio = gci_coarse / (r**p * gci_fine) if gci_fine > 1e-14 else float('inf')
        
        # Determine asymptotic range
        asymptotic = 0.9 <= gci_ratio <= 1.1
        
        # Determine convergence status
        status = self._determine_status(gci_fine, gci_coarse, p, asymptotic)
        
        return GCIResult(
            f_coarse=f_coarse,
            f_medium=f_medium,
            f_fine=f_fine,
            r_coarse=r_coarse,
            r_fine=r_fine,
            p=p,
            f_ext=f_ext,
            epsilon_fine=epsilon_fine,
            epsilon_coarse=epsilon_coarse,
            gci_fine=gci_fine,
            gci_coarse=gci_coarse,
            gci_ratio=gci_ratio,
            status=status,
            asymptotic=asymptotic,
        )
    
    def _compute_p_nonuniform(
        self, eps1: float, eps2: float, r_fine: float, r_coarse: float
    ) -> float:
        """Compute observed order for non-uniform refinement."""
        # Generalized formula for non-uniform grids
        # p = ln|eps2/eps1| / ln(r_eff) where r_eff is effective refinement
        r_eff = np.sqrt(r_fine * r_coarse)
        p = np.log(abs(eps2 / eps1)) / np.log(r_eff)
        return max(p, 0.1)  # Ensure positive
    
    def _handle_oscillatory(
        self, f_fine: float, f_medium: float, f_coarse: float,
        r_fine: float, r_coarse: float
    ) -> GCIResult:
        """Handle oscillatory convergence case."""
        eps1 = f_fine - f_medium
        eps2 = f_medium - f_coarse
        
        # Conservative estimates
        if abs(f_fine) > 1e-14:
            epsilon_fine = 100.0 * abs(eps1 / f_fine)
        else:
            epsilon_fine = 100.0 * abs(eps1)
            
        if abs(f_coarse) > 1e-14:
            epsilon_coarse = 100.0 * abs(eps2 / f_coarse)
        else:
            epsilon_coarse = 100.0 * abs(eps2)
        
        # Use p=1 as conservative estimate for oscillatory
        p = 1.0
        r = r_fine
        f_ext = f_fine  # Cannot extrapolate reliably
        
        gci_fine = self.safety_factor * epsilon_fine
        gci_coarse = self.safety_factor * epsilon_coarse
        gci_ratio = float('nan')
        
        return GCIResult(
            f_coarse=f_coarse,
            f_medium=f_medium,
            f_fine=f_fine,
            r_coarse=r_coarse,
            r_fine=r_fine,
            p=p,
            f_ext=f_ext,
            epsilon_fine=epsilon_fine,
            epsilon_coarse=epsilon_coarse,
            gci_fine=gci_fine,
            gci_coarse=gci_coarse,
            gci_ratio=gci_ratio,
            status=ConvergenceStatus.UNRESOLVED,
            asymptotic=False,
        )
    
    def _determine_status(
        self, gci_fine: float, gci_coarse: float, p: float, asymptotic: bool
    ) -> ConvergenceStatus:
        """Determine convergence status based on GCI values."""
        # Check for divergence
        if p < 0.1 or np.isnan(p):
            return ConvergenceStatus.FAIL
        
        # Check GCI threshold
        if gci_fine > self.gci_threshold:
            return ConvergenceStatus.FAIL
        
        # Check if in asymptotic range
        if not asymptotic:
            # Not in asymptotic range - may need more refinement
            if gci_fine > self.gci_threshold * 0.5:
                return ConvergenceStatus.FAIL
        
        return ConvergenceStatus.PASS
    
    def assess_grid_convergence(
        self,
        solutions: list,
        refinement_ratios: list,
        quantity_name: str = "quantity"
    ) -> dict:
        """
        Assess grid convergence for a sequence of solutions.
        
        Parameters
        ----------
        solutions : list
            List of solutions from coarsest to finest: [f_coarse, f_medium, f_fine]
        refinement_ratios : list
            List of refinement ratios: [r_coarse, r_fine]
        quantity_name : str
            Name of the quantity being assessed
            
        Returns
        -------
        dict
            Assessment report
        """
        if len(solutions) != 3:
            raise ValueError("Exactly 3 solutions required (coarse, medium, fine)")
        if len(refinement_ratios) != 2:
            raise ValueError("Exactly 2 refinement ratios required")
        
        result = self.compute(
            f_fine=solutions[2],
            f_medium=solutions[1],
            f_coarse=solutions[0],
            r_fine=refinement_ratios[1],
            r_coarse=refinement_ratios[0],
        )
        
        return {
            'quantity': quantity_name,
            'result': result.to_dict(),
            'recommendation': self._generate_recommendation(result),
        }
    
    def _generate_recommendation(self, result: GCIResult) -> str:
        """Generate recommendation based on GCI result."""
        if result.status == ConvergenceStatus.PASS:
            if result.asymptotic:
                return "Grid converged. Solution is in asymptotic range."
            else:
                return "Grid converged but not in asymptotic range. Consider further refinement for higher accuracy."
        elif result.status == ConvergenceStatus.UNRESOLVED:
            return "Oscillatory convergence detected. Check for numerical instabilities or insufficient grid quality."
        else:
            if result.p < 0.5:
                return "Very low convergence order. Check for bugs or inappropriate discretization."
            else:
                return "Grid not converged. Refine mesh further."
