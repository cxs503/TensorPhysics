"""Propeller models: actuator disk (momentum theory) and blade element method (BEM).

Physics
-------
Actuator disk theory (momentum theory):
    - Axial induction factor: a = 1 - V_disk/V_inflow
    - Thrust: T = 2 * rho * A * V_inflow^2 * a * (1-a)
    - Thrust coefficient: K_T = T / (rho * n^2 * D^4)
    - Wake velocity: V_wake = V_inflow * (1 - 2a)

Blade Element Method (BEM):
    - Radial discretization into blade elements
    - Local inflow angle: phi = arctan(V_a / V_t)
    - Lift/drag from airfoil data or linear theory
    - Thrust/torque by radial integration

References
----------
- Carlton, J.S. (2007) "Marine Propellers and Propulsion", 2nd ed., Butterworth-Heinemann.
- Lewis, E.V. (ed.) (1988) "Principles of Naval Architecture, Vol. II", SNAME.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

__all__ = [
    "ActuatorDisk",
    "BladeElementMethod",
    "PropellerGeometry",
]

RHO_WATER = 1025.0  # kg/m^3


@dataclass
class PropellerGeometry:
    """Propeller geometry parameters."""
    D: float           # Diameter [m]
    Z: int             # Number of blades [-]
    pitch_ratio: float # Pitch/Diameter ratio P/D [-]
    A_E_A_0: float = 0.7  # Expanded blade area ratio [-]
    
    @property
    def R(self) -> float:
        return self.D / 2.0
    
    @property
    def A_disk(self) -> float:
        """Disk area [m^2]."""
        return np.pi * self.R**2
    
    @property
    def pitch(self) -> float:
        return self.pitch_ratio * self.D


class ActuatorDisk:
    """Actuator disk propeller model (momentum theory).
    
    The actuator disk is an idealized propeller that imparts uniform axial
    acceleration to the flow through the disk. The thrust is related to the
    axial induction factor 'a' by momentum theory.
    
    Key relations:
        V_disk = V_inflow * (1 - a)
        V_wake_far = V_inflow * (1 - 2a)
        T = 2 * rho * A * V_inflow^2 * a * (1-a)
        K_T = T / (rho * n^2 * D^4)
    """
    
    def __init__(self, geom: PropellerGeometry, K_T_design: float = 0.25):
        """
        Parameters
        ----------
        geom : PropellerGeometry
            Propeller geometry
        K_T_design : float
            Design thrust coefficient [-]
        """
        self.geom = geom
        self.K_T_design = K_T_design
    
    def thrust_coefficient(self, J: float) -> float:
        """Compute thrust coefficient K_T at advance coefficient J.
        
        Uses a simple linear K_T-J relation:
            K_T(J) = K_T_design * (1 - J/J_0)
        
        where J_0 is the zero-thrust advance coefficient.
        
        Parameters
        ----------
        J : float
            Advance coefficient J = V/(n*D) [-]
            
        Returns
        -------
        float
            Thrust coefficient K_T [-]
        """
        # Zero-thrust advance coefficient (typical value)
        J_0 = 1.2 * self.geom.pitch_ratio
        
        if J >= J_0:
            return 0.0
        
        K_T = self.K_T_design * (1.0 - J / J_0)
        return max(K_T, 0.0)
    
    def thrust_from_KT(self, K_T: float, n_rps: float) -> float:
        """Compute thrust from thrust coefficient.
        
        T = K_T * rho * n^2 * D^4
        
        Parameters
        ----------
        K_T : float
            Thrust coefficient [-]
        n_rps : float
            Rotation rate [rev/s]
            
        Returns
        -------
        float
            Thrust [N]
        """
        return K_T * RHO_WATER * n_rps**2 * self.geom.D**4
    
    def torque_coefficient(self, J: float) -> float:
        """Compute torque coefficient K_Q at advance coefficient J.
        
        Uses a simple relation:
            K_Q(J) = K_Q_design * (1 - 0.5*J/J_0)
        
        Parameters
        ----------
        J : float
            Advance coefficient [-]
            
        Returns
        -------
        float
            Torque coefficient K_Q [-]
        """
        J_0 = 1.2 * self.geom.pitch_ratio
        K_Q_design = self.K_T_design / (2.0 * np.pi)  # Typical efficiency relation
        
        if J >= J_0:
            return 0.0
        
        K_Q = K_Q_design * (1.0 - 0.5 * J / J_0)
        return max(K_Q, 0.0)
    
    def torque_from_KT(self, K_T: float, n_rps: float) -> float:
        """Compute torque from thrust coefficient (approximate).
        
        Q ~ T * D / (2*pi*eta) where eta ~ 0.6
        
        Parameters
        ----------
        K_T : float
            Thrust coefficient [-]
        n_rps : float
            Rotation rate [rev/s]
            
        Returns
        -------
        float
            Torque [N*m]
        """
        T = self.thrust_from_KT(K_T, n_rps)
        eta = 0.6  # Typical open water efficiency
        Q = T * self.geom.D / (2.0 * np.pi * eta)
        return Q
    
    def axial_induction_from_thrust(self, T: float, V_inflow: float) -> float:
        """Compute axial induction factor from thrust.
        
        From momentum theory:
            T = 2 * rho * A * V_inflow^2 * a * (1-a)
        
        Solving for a:
            a = 0.5 * (1 - sqrt(1 - T/(0.5*rho*A*V_inflow^2)))
        
        Parameters
        ----------
        T : float
            Thrust [N]
        V_inflow : float
            Inflow velocity [m/s]
            
        Returns
        -------
        float
            Axial induction factor a [-]
        """
        # Thrust coefficient based on dynamic pressure
        T_ref = 0.5 * RHO_WATER * self.geom.A_disk * V_inflow**2
        
        if T_ref <= 0:
            return 0.0
        
        # Non-dimensional thrust
        T_nd = T / T_ref
        
        # Clamp to physical range (T_nd <= 2 for a <= 0.5)
        # But also ensure a < 0.5 to avoid V_wake = 0
        T_nd = min(T_nd, 1.5)  # This gives a ~ 0.37
        
        # Solve for a
        discriminant = 1.0 - T_nd
        if discriminant < 0:
            # Heavy loading, use empirical relation
            a = 0.35  # Typical value for heavily loaded propeller
        else:
            a = 0.5 * (1.0 - np.sqrt(discriminant))
        
        # Ensure a is in reasonable range (0.05 - 0.4)
        a = np.clip(a, 0.05, 0.4)
        
        return a
    
    def thrust_derivative_wrt_velocity(self, V_inflow: float, n_rps: float) -> float:
        """Compute dT/dV at given operating point.
        
        Parameters
        ----------
        V_inflow : float
            Inflow velocity [m/s]
        n_rps : float
            Rotation rate [rev/s]
            
        Returns
        -------
        float
            dT/dV [N/(m/s)]
        """
        J = V_inflow / (n_rps * self.geom.D)
        K_T = self.thrust_coefficient(J)
        
        # T = K_T * rho * n^2 * D^4
        # dT/dV = dK_T/dJ * dJ/dV * rho * n^2 * D^4
        # dJ/dV = 1/(n*D)
        
        # dK_T/dJ (linear relation)
        J_0 = 1.2 * self.geom.pitch_ratio
        dK_T_dJ = -self.K_T_design / J_0
        
        dT_dV = dK_T_dJ * (1.0 / (n_rps * self.geom.D)) * RHO_WATER * n_rps**2 * self.geom.D**4
        
        return dT_dV


class BladeElementMethod:
    """Blade Element Method (BEM) propeller model.
    
    The blade is discretized into radial elements. At each element:
    - Local inflow angle: phi = arctan(V_a / V_t)
    - Angle of attack: alpha = beta - phi (beta = blade pitch angle)
    - Lift: dL = 0.5 * rho * W^2 * c * CL * dr
    - Thrust contribution: dT = dL * cos(phi)
    - Torque contribution: dQ = dL * sin(phi) * r
    
    Total thrust and torque by radial integration.
    """
    
    def __init__(self, geom: PropellerGeometry, n_elements: int = 5):
        """
        Parameters
        ----------
        geom : PropellerGeometry
            Propeller geometry
        n_elements : int
            Number of radial elements
        """
        self.geom = geom
        self.n_elements = n_elements
        
        # Radial stations (from hub to tip)
        r_hub = 0.2 * geom.R  # Hub radius
        self.r = np.linspace(r_hub, geom.R, n_elements)
        self.dr = np.gradient(self.r)
        
        # Chord distribution (elliptic-like)
        c_mean = geom.D * geom.A_E_A_0 / (np.pi * geom.Z * 0.7)  # Mean chord
        self.c = c_mean * np.sqrt(1.0 - ((self.r - 0.5*(r_hub + geom.R)) / (0.5*(geom.R - r_hub)))**2)
        self.c = np.maximum(self.c, 0.1 * c_mean)  # Minimum chord
        
        # Pitch distribution (constant pitch ratio)
        self.beta = np.arctan(geom.pitch_ratio * geom.D / (2.0 * np.pi * self.r))
        
        # Calibrate to match actuator disk at design point
        self._calibration_factor = 0.85  # Reduce BEM thrust to match actuator disk
    
    def compute(self, J: float, n_rps: float) -> Dict[str, float]:
        """Compute thrust and torque coefficients.
        
        Parameters
        ----------
        J : float
            Advance coefficient [-]
        n_rps : float
            Rotation rate [rev/s]
            
        Returns
        -------
        Dict[str, float]
            K_T, K_Q, eta_0 (open water efficiency)
        """
        V_a = J * n_rps * self.geom.D
        omega = 2.0 * np.pi * n_rps
        
        T_total = 0.0
        Q_total = 0.0
        
        for i in range(self.n_elements):
            r = self.r[i]
            dr = self.dr[i]
            c = self.c[i]
            beta = self.beta[i]
            
            # Local velocities
            V_t = omega * r  # Tangential velocity
            W = np.sqrt(V_a**2 + V_t**2)  # Resultant velocity
            
            # Inflow angle
            phi = np.arctan2(V_a, V_t)
            
            # Angle of attack
            alpha = beta - phi
            
            # Lift coefficient (linear theory with stall limit)
            CL = 2.0 * np.pi * alpha
            CL = np.clip(CL, -1.5, 1.5)
            
            # Drag coefficient (simple model)
            CD = 0.01 + 0.05 * CL**2
            
            # Forces per unit span
            dL = 0.5 * RHO_WATER * W**2 * c * CL
            dD = 0.5 * RHO_WATER * W**2 * c * CD
            
            # Thrust and torque contributions
            dT = (dL * np.cos(phi) - dD * np.sin(phi)) * dr
            dQ = (dL * np.sin(phi) + dD * np.cos(phi)) * r * dr
            
            T_total += dT
            Q_total += dQ
        
        # Multiply by number of blades and calibration factor
        T_total *= self.geom.Z * self._calibration_factor
        Q_total *= self.geom.Z * self._calibration_factor
        
        # Non-dimensional coefficients
        K_T = T_total / (RHO_WATER * n_rps**2 * self.geom.D**4)
        K_Q = Q_total / (RHO_WATER * n_rps**2 * self.geom.D**5)
        
        # Open water efficiency
        if K_Q > 0 and J > 0:
            eta_0 = J * K_T / (2.0 * np.pi * K_Q)
        else:
            eta_0 = 0.0
        
        return {
            "K_T": float(K_T),
            "K_Q": float(K_Q),
            "eta_0": float(eta_0),
            "T": float(T_total),
            "Q": float(Q_total),
        }
