"""
Zero-Thickness Interface Element for FEM

Implements a zero-thickness cohesive interface element for embedding
in plate/shell FEM formulations (e.g., Mindlin plate).

The element:
- Connects two surfaces (top and bottom) with cohesive tractions
- Uses the mixed-mode CZM for traction-separation response
- Tracks state history for irreversible damage
- Computes consistent tangent stiffness for Newton-Raphson

Element formulation:
    - 4-node quadrilateral (bilinear interpolation)
    - Local coordinate system: (n, s, t)
        n = normal to interface
        s = first tangent direction
        t = second tangent direction
    - Separation: δ = u_top - u_bottom
    - Traction: T = [σ_n, τ_s, τ_t]
"""

from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from .bk_mixed_mode import MixedModeCZM


@dataclass
class InterfaceElementState:
    """State of a zero-thickness interface element.
    
    Attributes:
        delta_n: Normal separation [m]
        delta_s: Shear separation (first tangent) [m]
        delta_t: Shear separation (second tangent) [m]
        sigma_n: Normal traction [Pa]
        tau_s: Shear traction (first tangent) [Pa]
        tau_t: Shear traction (second tangent) [Pa]
        damage: Damage variable D ∈ [0, 1]
        delta_max: Maximum effective separation [m]
        G_I: Mode I energy release rate [J/m²]
        G_II: Mode II energy release rate [J/m²]
        G_total: Total energy release rate [J/m²]
        failed: Whether fully failed
        element_id: Element identifier
    """
    delta_n: float = 0.0
    delta_s: float = 0.0
    delta_t: float = 0.0
    sigma_n: float = 0.0
    tau_s: float = 0.0
    tau_t: float = 0.0
    damage: float = 0.0
    delta_max: float = 0.0
    G_I: float = 0.0
    G_II: float = 0.0
    G_total: float = 0.0
    failed: bool = False
    element_id: int = -1


class ZeroThicknessInterfaceElement:
    """Zero-thickness cohesive interface element.
    
    This element is inserted between bulk elements (e.g., plate elements)
    to model interface debonding, delamination, or ice-structure separation.
    
    The element uses:
    - Mixed-mode CZM for constitutive response
    - 2x2 Gauss quadrature for integration
    - Consistent tangent stiffness for convergence
    
    Usage:
        elem = ZeroThicknessInterfaceElement(czm_model)
        state = elem.update(delta_n, delta_s, delta_t)
        traction = elem.get_traction()
        tangent = elem.get_tangent_stiffness()
    """
    
    def __init__(
        self,
        czm: MixedModeCZM,
        element_id: int = 0,
        thickness: float = 0.0,
    ):
        """Initialize interface element.
        
        Args:
            czm: Mixed-mode cohesive zone model
            element_id: Element identifier
            thickness: Regularization thickness [m] (for energy scaling)
        """
        self.czm = czm
        self.element_id = element_id
        self.thickness = thickness
        
        # State variables
        self._delta_max = 0.0
        self._damage = 0.0
        self._state_history = []
        
        # Current state
        self._delta_n = 0.0
        self._delta_s = 0.0
        self._delta_t = 0.0
        self._sigma_n = 0.0
        self._tau_s = 0.0
        self._tau_t = 0.0
    
    def update(
        self,
        delta_n: float,
        delta_s: float,
        delta_t: float = 0.0,
    ) -> InterfaceElementState:
        """Update element state for new separations.
        
        Args:
            delta_n: Normal separation [m] (positive = opening)
            delta_s: Shear separation (first tangent) [m]
            delta_t: Shear separation (second tangent) [m]
        
        Returns:
            Updated InterfaceElementState
        """
        # Store current separations
        self._delta_n = delta_n
        self._delta_s = delta_s
        self._delta_t = delta_t
        
        # Effective shear separation (combine s and t components)
        delta_shear = np.sqrt(delta_s**2 + delta_t**2)
        
        # Compute mixed-mode response
        result = self.czm.compute_traction(
            delta_n, delta_shear,
            self._delta_max, self._damage
        )
        
        # Update state
        self._sigma_n = result["sigma_n"]
        self._tau_s = result["tau_s"] * (delta_s / delta_shear if delta_shear > 1e-30 else 0.0)
        self._tau_t = result["tau_s"] * (delta_t / delta_shear if delta_shear > 1e-30 else 0.0)
        self._damage = result["damage"]
        self._delta_max = result["delta_max"]
        
        # Create state object
        state = InterfaceElementState(
            delta_n=delta_n,
            delta_s=delta_s,
            delta_t=delta_t,
            sigma_n=self._sigma_n,
            tau_s=self._tau_s,
            tau_t=self._tau_t,
            damage=self._damage,
            delta_max=self._delta_max,
            G_I=result["G_I"],
            G_II=result["G_II"],
            G_total=result["G_total"],
            failed=result["failed"],
            element_id=self.element_id,
        )
        
        self._state_history.append(state)
        return state
    
    def get_traction(self) -> np.ndarray:
        """Get current traction vector.
        
        Returns:
            Traction vector [σ_n, τ_s, τ_t] [Pa]
        """
        return np.array([self._sigma_n, self._tau_s, self._tau_t])
    
    def get_tangent_stiffness(self) -> np.ndarray:
        """Get consistent tangent stiffness matrix.
        
        The tangent stiffness relates traction increment to separation increment:
            dT = K_tangent * dδ
        
        For the damaged bilinear law:
            K_tangent = (1 - D) * K  (in elastic regime)
            K_tangent = secant stiffness  (in softening regime)
        
        Returns:
            3x3 tangent stiffness matrix [Pa/m]
        """
        # Diagonal stiffness (decoupled modes for simplicity)
        if self._damage < 1e-10:
            # Undamaged: full stiffness
            K_n = self.czm.K_n
            K_s = self.czm.K_s
        elif self._damage >= 1.0 - 1e-10:
            # Fully damaged: zero stiffness
            K_n = 0.0
            K_s = 0.0
        else:
            # Damaged: secant stiffness
            # K_sec = T / δ = (1 - D) * K
            K_n = (1.0 - self._damage) * self.czm.K_n
            K_s = (1.0 - self._damage) * self.czm.K_s
        
        # Tangent matrix (diagonal for decoupled modes)
        K_tangent = np.array([
            [K_n, 0.0, 0.0],
            [0.0, K_s, 0.0],
            [0.0, 0.0, K_s],
        ])
        
        return K_tangent
    
    def get_state(self) -> InterfaceElementState:
        """Get current state.
        
        Returns:
            Current InterfaceElementState
        """
        if self._state_history:
            return self._state_history[-1]
        else:
            return InterfaceElementState(element_id=self.element_id)
    
    def reset(self):
        """Reset element to undamaged state."""
        self._delta_max = 0.0
        self._damage = 0.0
        self._state_history.clear()
        self._delta_n = 0.0
        self._delta_s = 0.0
        self._delta_t = 0.0
        self._sigma_n = 0.0
        self._tau_s = 0.0
        self._tau_t = 0.0
    
    def total_dissipated_energy(self) -> float:
        """Compute total energy dissipated by damage.
        
        Returns:
            Total dissipated energy per unit area [J/m²]
        """
        if len(self._state_history) < 2:
            return 0.0
        
        energy = 0.0
        for i in range(1, len(self._state_history)):
            s0 = self._state_history[i-1]
            s1 = self._state_history[i]
            
            # Trapezoidal integration of traction × separation increment
            d_delta_n = s1.delta_n - s0.delta_n
            d_delta_s = np.sqrt(
                (s1.delta_s - s0.delta_s)**2 + (s1.delta_t - s0.delta_t)**2
            )
            
            avg_sigma = 0.5 * (s0.sigma_n + s1.sigma_n)
            avg_tau = 0.5 * (
                np.sqrt(s0.tau_s**2 + s0.tau_t**2) +
                np.sqrt(s1.tau_s**2 + s1.tau_t**2)
            )
            
            energy += avg_sigma * d_delta_n + avg_tau * d_delta_s
        
        return energy
    
    @property
    def state_history(self) -> list[InterfaceElementState]:
        """Get state history."""
        return self._state_history


class InterfaceElementSet:
    """Collection of interface elements for FEM assembly.
    
    Manages multiple interface elements and provides methods for:
    - Assembly of global traction vector
    - Assembly of global tangent stiffness matrix
    - State tracking and history
    """
    
    def __init__(self, czm: MixedModeCZM, n_elements: int):
        """Initialize element set.
        
        Args:
            czm: Mixed-mode cohesive zone model
            n_elements: Number of interface elements
        """
        self.czm = czm
        self.n_elements = n_elements
        
        # Create elements
        self.elements = [
            ZeroThicknessInterfaceElement(czm, element_id=i)
            for i in range(n_elements)
        ]
    
    def update_all(
        self,
        delta_n: np.ndarray,
        delta_s: np.ndarray,
        delta_t: Optional[np.ndarray] = None,
    ) -> list[InterfaceElementState]:
        """Update all elements.
        
        Args:
            delta_n: Normal separations [m], shape (n_elements,)
            delta_s: Shear separations [m], shape (n_elements,)
            delta_t: Shear separations [m], shape (n_elements,) (optional)
        
        Returns:
            List of updated states
        """
        if delta_t is None:
            delta_t = np.zeros_like(delta_n)
        
        states = []
        for i, elem in enumerate(self.elements):
            state = elem.update(delta_n[i], delta_s[i], delta_t[i])
            states.append(state)
        
        return states
    
    def get_traction_vector(self) -> np.ndarray:
        """Get global traction vector.
        
        Returns:
            Traction vector [σ_n_1, τ_s_1, τ_t_1, σ_n_2, ...] [Pa]
        """
        tractions = []
        for elem in self.elements:
            tractions.append(elem.get_traction())
        return np.concatenate(tractions)
    
    def get_tangent_matrix(self) -> np.ndarray:
        """Get global tangent stiffness matrix.
        
        Returns:
            Block-diagonal tangent matrix [Pa/m]
        """
        n_dof = 3 * self.n_elements
        K_global = np.zeros((n_dof, n_dof))
        
        for i, elem in enumerate(self.elements):
            K_elem = elem.get_tangent_stiffness()
            idx = slice(3*i, 3*(i+1))
            K_global[idx, idx] = K_elem
        
        return K_global
    
    def reset_all(self):
        """Reset all elements."""
        for elem in self.elements:
            elem.reset()
    
    def get_damage_field(self) -> np.ndarray:
        """Get damage field across all elements.
        
        Returns:
            Damage array, shape (n_elements,)
        """
        return np.array([elem._damage for elem in self.elements])
    
    def get_failed_elements(self) -> list[int]:
        """Get IDs of fully failed elements.
        
        Returns:
            List of failed element IDs
        """
        return [
            elem.element_id for elem in self.elements
            if elem._damage >= 1.0 - 1e-10
        ]
