"""Propeller wake model: axial induction and wake velocity field.

Physics
-------
The propeller accelerates the flow through the disk. The wake behind the propeller
has a reduced velocity (relative to the ship) due to this acceleration.

Actuator disk theory gives:
    - Axial induction factor: a = 1 - V_disk/V_inflow
    - Wake velocity at disk: V_disk = V_inflow * (1 - a)
    - Far wake velocity: V_wake = V_inflow * (1 - 2a)
    
The wake modulation affects ice loads: the relative velocity between ship and ice
is reduced by the wake, leading to lower ice resistance.

Wake fraction (Taylor wake fraction):
    w = 1 - V_inflow/V_ship
    V_inflow = V_ship * (1 - w)

The wake fraction 'w' depends on hull form and propeller position. Typical values:
    - Single screw: w = 0.15 - 0.35
    - Twin screw: w = 0.05 - 0.20

References
----------
- Carlton, J.S. (2007) "Marine Propellers and Propulsion", 2nd ed.
- Harvath, J. (1999) "Wake fraction and thrust deduction factor", MARIC.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

__all__ = [
    "WakeModel",
    "WakeField",
]


@dataclass
class WakeModel:
    """Propeller wake model based on actuator disk theory.
    
    The wake reduces the effective inflow velocity to the propeller (wake fraction)
    and accelerates the flow behind the propeller (axial induction).
    """
    
    wake_fraction: float = 0.25  # Taylor wake fraction w [-]
    
    def inflow_velocity(self, V_ship: float) -> float:
        """Effective inflow velocity to propeller [m/s].
        
        V_inflow = V_ship * (1 - w)
        """
        return V_ship * (1.0 - self.wake_fraction)
    
    def thrust_deduction(self, wake_fraction: Optional[float] = None) -> float:
        """Thrust deduction factor t [-].
        
        Empirical relation: t ≈ 0.5 * w (for single screw ships)
        or t ≈ 0.7 * w (for twin screw ships)
        
        Using the simple relation t = 0.5 * w for single screw.
        """
        w = wake_fraction if wake_fraction is not None else self.wake_fraction
        return 0.5 * w


@dataclass
class WakeField:
    """Wake velocity field behind the propeller.
    
    Computes the axial velocity in the propeller wake as a function of
    the axial induction factor.
    
    V_wake = V_inflow * (1 - 2a)
    
    where 'a' is the axial induction factor from actuator disk theory.
    """
    
    V_inflow: float      # Inflow velocity to propeller [m/s]
    axial_induction: float  # Axial induction factor a [-]
    
    @property
    def V_disk(self) -> float:
        """Velocity at the propeller disk [m/s]."""
        return self.V_inflow * (1.0 - self.axial_induction)
    
    @property
    def V_wake_far(self) -> float:
        """Far wake velocity (slipstream) [m/s].
        
        This is the velocity far behind the propeller where the pressure
        has recovered to ambient.
        """
        return self.V_inflow * (1.0 - 2.0 * self.axial_induction)
    
    def wake_velocity_at(self, x: float, r: float, x_disk: float = 0.0) -> float:
        """Wake velocity at axial position x and radial position r.
        
        Simplified model: uniform wake within the slipstream, potential flow outside.
        
        Parameters
        ----------
        x : float
            Axial position behind propeller [m] (x > x_disk for wake)
        r : float
            Radial position from propeller axis [m]
        x_disk : float
            Axial position of propeller disk [m]
            
        Returns
        -------
        float
            Axial wake velocity [m/s]
        """
        if x <= x_disk:
            # Upstream of propeller, no wake effect
            return self.V_inflow
        
        # Simplified: uniform wake within slipstream radius
        # In reality, the wake contracts and then expands
        # Here we use a simple model with constant slipstream radius
        R_slipstream = self.V_disk / self.V_wake_far  # Contracted slipstream radius ratio
        # Actually, by continuity: A_disk * V_disk = A_slipstream * V_wake_far
        # So R_slipstream = R * sqrt(V_disk / V_wake_far)
        # But for simplicity, we assume the wake fills the disk area
        # and use the far-wake velocity as the characteristic value
        
        # For this coupling model, we return the far-wake velocity
        # as the characteristic velocity for ice interaction
        return self.V_wake_far
    
    def effective_ice_velocity(self, V_ship: float) -> float:
        """Effective velocity of ship relative to ice, accounting for wake.
        
        The wake accelerates the water behind the ship, reducing the relative
        velocity between the hull/propeller and the ice.
        
        V_ice_effective = V_ship - V_wake_induced
        
        where V_wake_induced is the wake-induced velocity reduction.
        
        For ice directly behind the propeller:
        V_ice_effective = V_ship - (V_inflow - V_wake_far)
                        = V_ship - V_inflow * 2a
                        = V_ship - 2 * a * V_inflow
        
        Parameters
        ----------
        V_ship : float
            Ship speed [m/s]
            
        Returns
        -------
        float
            Effective ice encounter velocity [m/s]
        """
        # Wake-induced velocity reduction
        V_wake_induced = self.V_inflow - self.V_wake_far  # = 2 * a * V_inflow
        V_ice_eff = V_ship - V_wake_induced
        return max(V_ice_eff, 0.0)  # Cannot be negative
