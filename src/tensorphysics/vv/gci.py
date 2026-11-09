"""
GCI (Grid Convergence Index) calculation and utilities.
"""

from __future__ import annotations

from typing import List, Callable, Tuple
import math

# Status constants
PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"


def gci(
    f_fine: float,
    f_medium: float,
    r: float,
    p: float,
    safety_factor: float = 1.25,
) -> float:
    """
    Compute GCI for a pair of grid levels.
    
    Parameters
    ----------
    f_fine : float
        Solution on finer grid
    f_medium : float
        Solution on coarser grid
    r : float
        Refinement ratio (h_coarse / h_fine)
    p : float
        Observed order of convergence
    safety_factor : float
        Safety factor Fs (default 1.25 for 3+ grids)
    
    Returns
    -------
    float
        GCI value (relative)
    """
    eps = abs(f_fine - f_medium)
    
    if abs(f_fine) < 1e-14:
        return safety_factor * eps
    
    epsilon = eps / abs(f_fine)
    
    if abs(r**p - 1) < 1e-14:
        return safety_factor * epsilon
    
    return safety_factor * epsilon / (r**p - 1)


def sweep_refinement_ratio(
    func: Callable[[float], float],
    h_coarse: float,
    p: float,
    ratios: Tuple[float, ...],
    safety_factor: float = 1.25,
) -> List[Tuple[float, float]]:
    """
    Sweep refinement ratios and compute GCI for each.
    
    Parameters
    ----------
    func : callable
        Function f(h) that returns solution at grid spacing h
    h_coarse : float
        Coarsest grid spacing
    p : float
        Order of convergence
    ratios : tuple of float
        Refinement ratios to test
    safety_factor : float
        Safety factor
    
    Returns
    -------
    list of (ratio, gci_value)
    """
    results = []
    
    for r in ratios:
        # Generate three levels with this refinement ratio
        h_medium = h_coarse / r
        h_fine = h_medium / r
        
        f_coarse = func(h_coarse)
        f_medium = func(h_medium)
        f_fine = func(h_fine)
        
        # Compute GCI for fine-medium pair
        gci_val = gci(f_fine, f_medium, r, p, safety_factor)
        results.append((r, gci_val))
    
    return results
