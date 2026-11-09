"""
ISO18072 Damage Length Criterion for Ship Collision

Implements ISO 18072:2018 guidelines for damage assessment in ship collision,
including damage length calculation and safety margin evaluation.
"""

import numpy as np
from typing import Dict, Tuple, Optional


class ISO18072DamageLength:
    """
    ISO18072 damage length assessment for ship collision scenarios.
    
    The damage length depends on:
    - Ship dimensions (L, B, D)
    - Collision energy
    - Structural arrangement
    - Material properties
    
    Parameters
    ----------
    ship_length : float
        Ship length L [m]
    ship_breadth : float
        Ship breadth B [m]
    ship_depth : float
        Ship depth D [m]
    collision_energy : float
        Collision energy E [J]
    """
    
    def __init__(self, ship_length: float, ship_breadth: float,
                 ship_depth: float, collision_energy: float):
        self.L = ship_length
        self.B = ship_breadth
        self.D = ship_depth
        self.E_collision = collision_energy
    
    def damage_length_empirical(self) -> float:
        """
        Compute empirical damage length according to ISO18072.
        
        l_d = 0.12 * L * (E_collision / (0.5 * Δ * V_ref²))^(1/3)
        
        where Δ is displacement and V_ref is reference velocity.
        
        Returns
        -------
        float
            Damage length [m]
        """
        # Simplified empirical formula
        # Reference: ISO 18072:2018 Section 6.3
        E_ref = 1e6  # Reference energy [J]
        
        l_d = 0.12 * self.L * (self.E_collision / E_ref) ** (1.0/3.0)
        
        # Limit to reasonable values
        l_d = min(l_d, 0.3 * self.L)
        l_d = max(l_d, 0.01 * self.L)
        
        return l_d
    
    def damage_length(self) -> float:
        """Alias for damage_length_empirical."""
        return self.damage_length_empirical()
    
    def damage_height(self) -> float:
        """
        Compute damage height.
        
        Returns
        -------
        float
            Damage height [m]
        """
        # Typically 10-20% of ship depth
        h_d = 0.15 * self.D
        return h_d
    
    def damage_volume(self) -> float:
        """
        Compute damaged volume.
        
        Returns
        -------
        float
            Damaged volume [m³]
        """
        l_d = self.damage_length()
        h_d = self.damage_height()
        
        # Trapezoidal cross-section approximation
        V_d = 0.5 * l_d * h_d * self.B
        
        return V_d
    
    def residual_strength_factor(self) -> float:
        """
        Compute residual strength factor.
        
        Returns
        -------
        float
            Residual strength factor (0-1)
        """
        l_d = self.damage_length()
        
        # Normalized damage
        damage_ratio = l_d / self.L
        
        # Residual strength (simplified)
        RSF = 1.0 - 0.8 * damage_ratio
        
        return max(0.1, RSF)


def calculate_damage_length(energy: float, ship_length: float,
                           ship_breadth: float, ship_depth: float,
                           method: str = 'ISO18072') -> float:
    """
    Calculate damage length using specified method.
    
    Parameters
    ----------
    energy : float
        Collision energy [J]
    ship_length : float
        Ship length [m]
    ship_breadth : float
        Ship breadth [m]
    ship_depth : float
        Ship depth [m]
    method : str
        Calculation method ('ISO18072', 'empirical', 'energy_based')
    
    Returns
    -------
    float
        Damage length [m]
    """
    iso = ISO18072DamageLength(ship_length, ship_breadth, ship_depth, energy)
    
    if method == 'ISO18072':
        return iso.damage_length_empirical()
    elif method == 'empirical':
        # Alternative empirical formula
        return 0.1 * ship_length * (energy / 1e6) ** 0.4
    elif method == 'energy_based':
        # Energy-based approach
        sigma_y = 350e6  # Typical yield stress [Pa]
        t = 0.015  # Typical plate thickness [m]
        
        # Energy per unit length
        e_l = energy / ship_breadth
        
        # Damage length
        l_d = e_l / (sigma_y * t * 0.5)
        
        return min(l_d, 0.3 * ship_length)
    else:
        raise ValueError(f"Unknown method: {method}")


def safety_margin(applied_load: float, capacity: float,
                 partial_factor_load: float = 1.3,
                 partial_factor_material: float = 1.1) -> float:
    """
    Compute safety margin according to ISO18072.
    
    η = R_d / S_d
    
    where:
    R_d = R_k / γ_m (design resistance)
    S_d = γ_f * S_k (design load effect)
    
    Parameters
    ----------
    applied_load : float
        Characteristic applied load S_k
    capacity : float
        Characteristic capacity R_k
    partial_factor_load : float
        Load partial factor γ_f
    partial_factor_material : float
        Material partial factor γ_m
    
    Returns
    -------
    float
        Safety margin η
    """
    R_d = capacity / partial_factor_material
    S_d = applied_load * partial_factor_load
    
    eta = R_d / S_d
    
    return eta


def collision_assessment(ship_length: float, ship_breadth: float,
                        ship_depth: float, displacement: float,
                        collision_velocity: float, collision_mass: float,
                        structural_capacity: float) -> Dict:
    """
    Perform complete collision assessment according to ISO18072.
    
    Parameters
    ----------
    ship_length : float
        Ship length [m]
    ship_breadth : float
        Ship breadth [m]
    ship_depth : float
        Ship depth [m]
    displacement : float
        Ship displacement [kg]
    collision_velocity : float
        Collision velocity [m/s]
    collision_mass : float
        Striking mass [kg]
    structural_capacity : float
        Structural capacity [N]
    
    Returns
    -------
    dict
        Assessment results
    """
    # Collision energy
    E_collision = 0.5 * collision_mass * collision_velocity**2
    
    # Added mass (simplified)
    added_mass = 0.1 * displacement
    E_total = 0.5 * (collision_mass + added_mass) * collision_velocity**2
    
    # Damage assessment
    iso = ISO18072DamageLength(ship_length, ship_breadth, ship_depth, E_total)
    
    l_d = iso.damage_length_empirical()
    h_d = iso.damage_height()
    V_d = iso.damage_volume()
    RSF = iso.residual_strength_factor()
    
    # Applied load (simplified)
    F_applied = E_total / (l_d * 0.5)
    
    # Safety margin
    eta = safety_margin(F_applied, structural_capacity)
    
    return {
        'collision_energy': E_total,
        'damage_length': l_d,
        'damage_height': h_d,
        'damage_volume': V_d,
        'residual_strength_factor': RSF,
        'applied_force': F_applied,
        'structural_capacity': structural_capacity,
        'safety_margin': eta,
        'acceptable': eta >= 1.0
    }
