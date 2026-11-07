"""Aitken and dynamic relaxation methods for partitioned coupling.

The Aitken dynamic relaxation method adaptively updates the relaxation
factor based on the residual history, significantly improving convergence
for problems with strong added-mass effects.

References
----------
- Küttler, U., Wall, W.A. (2008). Vector and matrix relaxation techniques
  for partitioned FSI.
- Irons, B.M., Tuck, R.C. (1969). A version of the Aitken accelerator for
  computer iteration.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class AitkenRelaxation:
    """Aitken dynamic relaxation for partitioned coupling.
    
    The Aitken method updates the relaxation factor as:
        omega_{k+1} = -omega_k * ||r_{k-1}|| / (||r_k|| - ||r_{k-1}||)
    
    where r_k is the residual at iteration k.
    
    Parameters
    ----------
    omega_init : float
        Initial relaxation factor
    omega_min : float
        Minimum allowed relaxation factor
    omega_max : float
        Maximum allowed relaxation factor
    """
    omega_init: float = 1.0
    omega_min: float = 0.05
    omega_max: float = 1.5
    
    # Internal state
    _residual_history: List[float] = field(default_factory=list)
    _omega_history: List[float] = field(default_factory=list)
    _omega_current: float = 1.0
    
    def reset(self) -> None:
        """Reset the relaxation state."""
        self._residual_history = []
        self._omega_history = []
        self._omega_current = self.omega_init
    
    @property
    def omega(self) -> float:
        """Current relaxation factor."""
        return self._omega_current
    
    def update(self, residual_norm: float) -> float:
        """Update relaxation factor based on new residual.
        
        Parameters
        ----------
        residual_norm : float
            Current residual norm
            
        Returns
        -------
        float
            Updated relaxation factor
        """
        self._residual_history.append(residual_norm)
        
        if len(self._residual_history) >= 2:
            r_prev = self._residual_history[-2]
            r_curr = self._residual_history[-1]
            delta_r = r_curr - r_prev
            
            if abs(delta_r) > 1e-14:
                # Aitken update
                omega_new = -self._omega_current * r_prev / delta_r
                # Clip to bounds
                omega_new = np.clip(omega_new, self.omega_min, self.omega_max)
                self._omega_current = float(omega_new)
        
        self._omega_history.append(self._omega_current)
        return self._omega_current
    
    def get_convergence_rate(self) -> float:
        """Estimate convergence rate from residual history.
        
        Returns
        -------
        float
            Estimated convergence rate (rho < 1 means converging)
        """
        if len(self._residual_history) < 2:
            return 0.0
        
        # Compute ratios of consecutive residuals
        ratios = []
        for i in range(1, len(self._residual_history)):
            if self._residual_history[i-1] > 1e-14:
                ratio = self._residual_history[i] / self._residual_history[i-1]
                ratios.append(ratio)
        
        if len(ratios) == 0:
            return 0.0
        
        return float(np.mean(ratios))


@dataclass
class DynamicRelaxation:
    """Dynamic relaxation with multiple strategy options.
    
    Supports:
    - Fixed relaxation
    - Aitken dynamic relaxation
    - Adaptive relaxation based on convergence history
    
    Parameters
    ----------
    strategy : str
        Relaxation strategy: "fixed", "aitken", "adaptive"
    omega_init : float
        Initial relaxation factor
    """
    strategy: str = "aitken"
    omega_init: float = 1.0
    
    _aitken: AitkenRelaxation = field(default_factory=AitkenRelaxation)
    _iteration_count: int = 0
    
    def __post_init__(self):
        self._aitken.omega_init = self.omega_init
        self._aitken._omega_current = self.omega_init
    
    def reset(self) -> None:
        """Reset the relaxation state."""
        self._aitken.reset()
        self._iteration_count = 0
    
    @property
    def omega(self) -> float:
        """Current relaxation factor."""
        if self.strategy == "fixed":
            return self.omega_init
        else:
            return self._aitken.omega
    
    def update(self, residual_norm: float) -> float:
        """Update relaxation factor.
        
        Parameters
        ----------
        residual_norm : float
            Current residual norm
            
        Returns
        -------
        float
            Updated relaxation factor
        """
        self._iteration_count += 1
        
        if self.strategy == "fixed":
            return self.omega_init
        elif self.strategy == "aitken":
            return self._aitken.update(residual_norm)
        elif self.strategy == "adaptive":
            return self._adaptive_update(residual_norm)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _adaptive_update(self, residual_norm: float) -> float:
        """Adaptive relaxation based on convergence behavior.
        
        Uses Aitken as base, but adds:
        - Increase omega if converging fast
        - Decrease omega if oscillating
        """
        # Start with Aitken update
        omega_aitken = self._aitken.update(residual_norm)
        
        # Check for oscillation (sign changes in residual)
        if len(self._aitken._residual_history) >= 3:
            r = self._aitken._residual_history[-3:]
            # Check if residuals are oscillating
            if (r[1] - r[0]) * (r[2] - r[1]) < 0:
                # Oscillating: reduce omega
                omega_aitken *= 0.8
        
        return omega_aitken
    
    def get_statistics(self) -> dict:
        """Get relaxation statistics.
        
        Returns
        -------
        dict
            Statistics including omega history and convergence rate
        """
        return {
            "strategy": self.strategy,
            "omega_init": self.omega_init,
            "omega_final": self.omega,
            "omega_history": list(self._aitken._omega_history),
            "convergence_rate": self._aitken.get_convergence_rate(),
            "iterations": self._iteration_count,
        }


def compute_optimal_relaxation(mu: float, scheme: str = "aitken") -> float:
    """Compute optimal relaxation factor for given added mass ratio.
    
    For a simple partitioned scheme, the optimal relaxation factor is:
        omega_opt = 1 / (1 + mu)
    
    Parameters
    ----------
    mu : float
        Added mass ratio
    scheme : str
        Relaxation scheme
        
    Returns
    -------
    float
        Optimal relaxation factor
    """
    if scheme == "optimal":
        return 1.0 / (1.0 + mu)
    elif scheme == "aitken":
        # Aitken will adapt, but start with a reasonable value
        return min(1.0, 1.0 / (1.0 + 0.5 * mu))
    else:
        return 1.0
