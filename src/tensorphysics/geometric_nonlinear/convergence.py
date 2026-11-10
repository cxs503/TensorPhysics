"""Convergence and verification studies for geometric nonlinear problems.

Verifies:
1. Elastica exact solution matches linear theory for small deflections
2. Von Karman plate shows correct nonlinear stiffening
3. Load-deflection curves are physically correct
"""

from __future__ import annotations

import numpy as np
from typing import List

from .elastica import ElasticaExact
from .von_karman_plate import VonKarmanPlateExact

__all__ = ["convergence_study_elastica", "convergence_study_plate"]


def convergence_study_elastica(
    L: float = 1.0, EI: float = 1.0,
    theta_B_range: List[float] = None,
) -> dict:
    """Verify the elastica solution converges to linear theory for small deflections.
    
    For small deflections (θ_B → 0):
    - δ/(PL³/3EI) → 1.0
    - The error should decrease as θ_B decreases
    
    Parameters
    ----------
    L : float
        Beam length.
    EI : float
        Flexural rigidity.
    theta_B_range : list of float
        Free-end angles to test.
    
    Returns
    -------
    dict with verification data.
    """
    if theta_B_range is None:
        theta_B_range = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    exact = ElasticaExact(L=L, EI=EI)
    
    ratios = []
    errors = []
    
    for theta_B in theta_B_range:
        sol = exact.compute_from_theta_B(theta_B)
        P = sol.P
        delta = sol.y_tip
        delta_linear = P * L**3 / (3 * EI)
        
        ratio = delta / delta_linear if delta_linear > 1e-15 else 1.0
        error = abs(ratio - 1.0) * 100
        
        ratios.append(ratio)
        errors.append(error)
    
    return {
        'theta_B_range': theta_B_range,
        'ratios': np.array(ratios),
        'errors_pct': np.array(errors),
        'expected_limit': 1.0,
    }


def convergence_study_plate(
    a: float = 1.0, h: float = 0.01, E: float = 210e9, nu: float = 0.3,
    w0_h_range: List[float] = None,
) -> dict:
    """Verify the von Karman plate solution.
    
    For small deflections (w₀/h → 0):
    - P*a²/(16πDh) / (w₀/h) → 1.0
    - For large deflections, the cubic term dominates
    
    Parameters
    ----------
    a : float
        Plate radius.
    h : float
        Plate thickness.
    E : float
        Young's modulus.
    nu : float
        Poisson's ratio.
    w0_h_range : list of float
        Normalized deflections to test.
    
    Returns
    -------
    dict with verification data.
    """
    if w0_h_range is None:
        w0_h_range = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0]
    
    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)
    w0_h_arr = np.array(w0_h_range)
    w0_h_ref, P_norm_ref, P_ref = ref.compute_load_deflection_curve(w0_h_arr)
    
    # For linear theory: P_norm = w0_h
    # For nonlinear: P_norm = w0_h + c * w0_h³
    # Ratio: P_norm / w0_h = 1 + c * w0_h²
    
    ratios = P_norm_ref / w0_h_ref
    
    return {
        'w0_h_range': w0_h_arr,
        'P_normalized': P_norm_ref,
        'ratios': ratios,
        'expected_linear_limit': 1.0,
    }
