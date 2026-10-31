"""Cohesive zone model for mixed-mode fracture.

Implements a bilinear traction-separation law with power-law or BK-type
mixed-mode damage evolution. The model supports non-proportional loading paths
where the ratio of mode I to mode II loading changes during the simulation.

Traction-separation law:
    - Linear elastic ramp to peak traction (σ_max or τ_max)
    - Softening with damage variable d ∈ [0, 1]
    - Complete failure when energy release rate reaches G_c(ψ)

Non-proportional path handling:
    - Tracks maximum effective displacement δ_max at each load step
    - Uses Benzeggagh-Kenane criterion for mixed-mode damage evolution
    - Accounts for path-dependent energy dissipation
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from .bk_law import BKLaw, compute_mixed_mode_ratio


@dataclass
class CohesiveProperties:
    """Material properties for cohesive element.
    
    Attributes:
        K_n: Normal (mode I) stiffness [Pa/m]
        K_s: Shear (mode II) stiffness [Pa/m]
        sigma_max: Maximum normal traction (mode I strength) [Pa]
        tau_max: Maximum shear traction (mode II strength) [Pa]
        G_Ic: Mode I fracture toughness [J/m²]
        G_IIc: Mode II fracture toughness [J/m²]
        eta: BK exponent for mixed-mode criterion
        thickness: Cohesive element thickness [m] (for numerical regularization)
    """
    K_n: float = 1e10       # Pa/m
    K_s: float = 1e10       # Pa/m
    sigma_max: float = 5e6  # Pa
    tau_max: float = 5e6    # Pa
    G_Ic: float = 200.0     # J/m²
    G_IIc: float = 600.0    # J/m²
    eta: float = 1.5
    thickness: float = 1e-4 # m


@dataclass
class CohesiveState:
    """State of a cohesive element at a given load step.
    
    Attributes:
        delta_n: Normal separation [m]
        delta_s: Shear separation [m]
        sigma_n: Normal traction [Pa]
        tau_s: Shear traction [Pa]
        damage: Damage variable d ∈ [0, 1]
        delta_max: Maximum effective displacement reached [m]
        G_I: Mode I energy release rate [J/m²]
        G_II: Mode II energy release rate [J/m²]
        G_total: Total energy release rate [J/m²]
        mixed_mode_ratio: ψ = G_II / (G_I + G_II)
        failed: Whether element has fully failed
    """
    delta_n: float = 0.0
    delta_s: float = 0.0
    sigma_n: float = 0.0
    tau_s: float = 0.0
    damage: float = 0.0
    delta_max: float = 0.0
    G_I: float = 0.0
    G_II: float = 0.0
    G_total: float = 0.0
    mixed_mode_ratio: float = 0.0
    failed: bool = False


class MixedModeCohesiveModel:
    """Mixed-mode cohesive zone model with BK damage evolution.
    
    Implements a bilinear traction-separation law with:
    - Quadratic stress criterion for damage initiation
    - BK law for mixed-mode damage evolution
    - Support for non-proportional loading paths
    
    The model tracks the history of maximum effective displacement to ensure
    irreversible damage evolution (no healing).
    """
    
    def __init__(self, props: Optional[CohesiveProperties] = None):
        """Initialize cohesive model.
        
        Args:
            props: Material properties (uses defaults if None)
        """
        self.props = props or CohesiveProperties()
        self.bk = BKLaw(self.props.G_Ic, self.props.G_IIc, self.props.eta)
        
        # Compute critical separations at damage initiation
        self.delta_n0 = self.props.sigma_max / self.props.K_n  # m
        self.delta_s0 = self.props.tau_max / self.props.K_s    # m
        
        # Compute final separations at complete failure (pure modes)
        # For bilinear law: G = 0.5 * sigma_max * delta_f → delta_f = 2*G/sigma_max
        self.delta_nf = 2.0 * self.props.G_Ic / self.props.sigma_max   # m
        self.delta_sf = 2.0 * self.props.G_IIc / self.props.tau_max    # m
    
    def compute_traction(
        self,
        delta_n: float,
        delta_s: float,
        delta_max_prev: float = 0.0,
        damage_prev: float = 0.0,
    ) -> CohesiveState:
        """Compute traction for given separations (monotonic loading step).
        
        Args:
            delta_n: Normal separation [m] (≥ 0 for opening)
            delta_s: Shear separation [m]
            delta_max_prev: Previous maximum effective displacement [m]
            damage_prev: Previous damage variable
            
        Returns:
            CohesiveState with updated tractions and damage
        """
        p = self.props
        
        # Effective displacement (using a mixed-mode measure)
        # Macaulay bracket: only positive delta_n contributes to opening
        delta_n_eff = max(delta_n, 0.0)
        delta_eff = np.sqrt(delta_n_eff**2 + delta_s**2)
        
        # Effective stiffness (weighted)
        if delta_eff > 1e-30:
            beta = delta_n_eff / delta_eff  # mode mixity parameter
        else:
            beta = 0.0
        
        # Effective peak displacement (quadratic criterion)
        K_eff = np.sqrt(p.K_n**2 * beta**2 + p.K_s**2 * (1 - beta**2))
        if K_eff < 1e-10:
            K_eff = p.K_n
        
        # Effective peak traction
        sigma_eff_peak = np.sqrt(p.sigma_max**2 * beta**2 + p.tau_max**2 * (1 - beta**2))
        if sigma_eff_peak < 1e-10:
            sigma_eff_peak = p.sigma_max
        
        delta_eff_0 = sigma_eff_peak / K_eff
        
        # BK effective toughness for current mixity
        # Use the current displacement ratio to determine mixity
        if delta_eff > 1e-30:
            # Mode I contribution
            G_I_potential = 0.5 * p.K_n * delta_n_eff**2
            # Mode II contribution
            G_II_potential = 0.5 * p.K_s * delta_s**2
            G_total_potential = G_I_potential + G_II_potential
            
            if G_total_potential > 1e-30:
                psi = G_II_potential / G_total_potential
            else:
                psi = 0.0
        else:
            psi = 0.0
        
        G_c = self.bk.effective_toughness(psi)
        
        # Effective final displacement
        delta_eff_f = 2.0 * G_c / sigma_eff_peak
        
        # Update maximum effective displacement (irreversibility)
        delta_max = max(delta_eff, delta_max_prev)
        
        # Compute damage
        if delta_max <= delta_eff_0:
            damage = 0.0
        elif delta_max >= delta_eff_f:
            damage = 1.0
        else:
            # Linear softening in effective space
            damage = (delta_max - delta_eff_0) / (delta_eff_f - delta_eff_0)
            damage = max(damage_prev, damage)  # no healing
        
        # Compute tractions
        failed = damage >= 1.0 - 1e-10
        
        if delta_n_eff > 0:
            sigma_n = (1.0 - damage) * p.K_n * delta_n_eff
        else:
            # Contact: no tension, but compression stiffness
            sigma_n = p.K_n * delta_n  # negative = compression
        
        tau_s = (1.0 - damage) * p.K_s * delta_s
        
        # Compute energy release rates from damage and fracture toughness
        # The ERR is the energy dissipated by damage
        # For a bilinear law: G = damage * G_c(ψ)
        # But we need to decompose into mode I and mode II
        
        # Mode I ERR: fraction of total based on displacement mixity
        if delta_eff > 1e-30:
            # Weight by displacement components
            w_I = delta_n_eff**2 / (delta_n_eff**2 + delta_s**2) if (delta_n_eff**2 + delta_s**2) > 0 else 0.0
            w_II = delta_s**2 / (delta_n_eff**2 + delta_s**2) if (delta_n_eff**2 + delta_s**2) > 0 else 0.0
        else:
            w_I = 1.0
            w_II = 0.0
        
        # Total ERR dissipated
        G_total_dissipated = damage * G_c
        
        # Decompose based on displacement weights
        G_I = w_I * G_total_dissipated
        G_II = w_II * G_total_dissipated
        G_total = G_I + G_II
        
        # Actual mixed-mode ratio
        if G_total > 1e-30:
            psi_actual = G_II / G_total
        else:
            psi_actual = psi
        
        return CohesiveState(
            delta_n=delta_n,
            delta_s=delta_s,
            sigma_n=sigma_n,
            tau_s=tau_s,
            damage=damage,
            delta_max=delta_max,
            G_I=G_I,
            G_II=G_II,
            G_total=G_total,
            mixed_mode_ratio=psi_actual,
            failed=failed,
        )
    
    def compute_traction_path(
        self,
        delta_n_path: np.ndarray,
        delta_s_path: np.ndarray,
    ) -> List[CohesiveState]:
        """Compute traction-separation response along a loading path.
        
        Args:
            delta_n_path: Array of normal separations [m]
            delta_s_path: Array of shear separations [m]
            
        Returns:
            List of CohesiveState at each load step
        """
        states = []
        delta_max = 0.0
        damage = 0.0
        
        for dn, ds in zip(delta_n_path, delta_s_path):
            state = self.compute_traction(dn, ds, delta_max, damage)
            states.append(state)
            delta_max = state.delta_max
            damage = state.damage
        
        return states


class CohesiveElement:
    """Single cohesive element with state tracking.
    
    Wraps MixedModeCohesiveModel with history tracking for use in
    finite element simulations.
    """
    
    def __init__(self, props: Optional[CohesiveProperties] = None):
        self.model = MixedModeCohesiveModel(props)
        self.state_history: List[CohesiveState] = []
        self._delta_max = 0.0
        self._damage = 0.0
    
    def update(self, delta_n: float, delta_s: float) -> CohesiveState:
        """Update element state for new separations.
        
        Args:
            delta_n: Normal separation [m]
            delta_s: Shear separation [m]
            
        Returns:
            Updated CohesiveState
        """
        state = self.model.compute_traction(
            delta_n, delta_s, self._delta_max, self._damage
        )
        self.state_history.append(state)
        self._delta_max = state.delta_max
        self._damage = state.damage
        return state
    
    def reset(self):
        """Reset element to undamaged state."""
        self.state_history.clear()
        self._delta_max = 0.0
        self._damage = 0.0
    
    @property
    def current_state(self) -> Optional[CohesiveState]:
        """Get current state (None if no updates yet)."""
        if self.state_history:
            return self.state_history[-1]
        return None
    
    def total_dissipated_energy(self) -> float:
        """Compute total energy dissipated by damage (area under softening curve).
        
        Returns:
            Total dissipated energy per unit area [J/m²]
        """
        if len(self.state_history) < 2:
            return 0.0
        
        energy = 0.0
        for i in range(1, len(self.state_history)):
            s0 = self.state_history[i-1]
            s1 = self.state_history[i]
            # Trapezoidal integration of traction × separation increment
            d_delta_n = s1.delta_n - s0.delta_n
            d_delta_s = s1.delta_s - s0.delta_s
            avg_sigma = 0.5 * (s0.sigma_n + s1.sigma_n)
            avg_tau = 0.5 * (s0.tau_s + s1.tau_s)
            energy += avg_sigma * d_delta_n + avg_tau * d_delta_s
        
        return energy
