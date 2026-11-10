"""Hull structural model: multi-DOF ship motions (heave, pitch, vertical bending).

Physics
-------
The ship hull is modeled as a multi-degree-of-freedom system with:
1. Heave (vertical translation) - 1 DOF
2. Pitch (rotation about transverse axis) - 1 DOF  
3. Vertical bending (flexible hull girder) - N modes

The equations of motion in matrix form:
    M * x'' + C * x' + K * x = F(t)

where:
    M = mass matrix (including added mass)
    C = damping matrix (radiation + viscous)
    K = stiffness matrix (hydrostatic + structural)
    F(t) = excitation forces (ice loads, waves)

Natural frequencies:
    Heave: omega_heave = sqrt(K33 / M33)
    Pitch: omega_pitch = sqrt(K55 / M55)
    Bending: omega_n = (n*pi/L)^2 * sqrt(EI / m)

References
----------
- Newman, J.N. (1977) "Marine Hydrodynamics", MIT Press.
- Bishop, R.E.D. & Price, W.G. (1979) "Hydroelasticity of Ships", CUP.
- Senjanović, I. (2010) "Advanced Ship Design for Green Operation", FAMENA.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

__all__ = [
    "HullStructure",
    "MultiDOFResponse",
    "compute_natural_frequencies",
    "frequency_response_function",
]

RHO_WATER = 1025.0  # kg/m^3
G = 9.80665  # m/s^2


@dataclass
class HullStructure:
    """Hull structural properties for multi-DOF analysis.
    
    Attributes
    ----------
    L : float
        Length between perpendiculars [m]
    B : float
        Beam [m]
    T : float
        Draft [m]
    Cb : float
        Block coefficient [-]
    mass : float
        Ship mass (displacement) [kg]
    Iyy : float
        Mass moment of inertia about transverse axis [kg*m^2]
    EI : float
        Vertical bending stiffness of hull girder [N*m^2]
    added_mass_heave : float
        Added mass in heave (as fraction of displaced mass) [-]
    added_mass_pitch : float
        Added inertia in pitch (as fraction of Iyy) [-]
    damping_ratio_heave : float
        Damping ratio in heave [-]
    damping_ratio_pitch : float
        Damping ratio in pitch [-]
    n_bend_modes : int
        Number of vertical bending modes to include
    """
    L: float = 100.0
    B: float = 20.0
    T: float = 7.0
    Cb: float = 0.75
    mass: Optional[float] = None  # If None, computed from displacement
    Iyy: Optional[float] = None   # If None, computed from geometry
    EI: float = 1.0e12            # Vertical bending stiffness [N*m^2]
    added_mass_heave: float = 0.5  # Added mass fraction [-]
    added_mass_pitch: float = 0.3  # Added inertia fraction [-]
    damping_ratio_heave: float = 0.05
    damping_ratio_pitch: float = 0.03
    damping_ratio_bending: float = 0.02
    n_bend_modes: int = 3
    
    @property
    def displacement(self) -> float:
        """Displaced volume [m^3]."""
        return self.L * self.B * self.T * self.Cb
    
    @property
    def mass_ship(self) -> float:
        """Ship mass [kg]."""
        if self.mass is not None:
            return self.mass
        return RHO_WATER * self.displacement
    
    @property
    def Iyy_ship(self) -> float:
        """Mass moment of inertia about transverse axis [kg*m^2]."""
        if self.Iyy is not None:
            return self.Iyy
        # Approximation: Iyy ~ mass * (L/2)^2 * kyy^2
        # where kyy is the radius of gyration, typically kyy/L ~ 0.25
        kyy = 0.25 * self.L
        return self.mass_ship * kyy**2
    
    def hydrostatic_stiffness(self) -> Tuple[float, float]:
        """Hydrostatic stiffness in heave and pitch [N/m, N*m/rad].
        
        K33 = rho * g * A_wp  (waterplane area)
        K55 = rho * g * V * BM_L - rho * g * V * KG
        
        where BM_L = I_L / V, I_L = longitudinal second moment of waterplane area
        """
        # Waterplane area approximation
        A_wp = 0.85 * self.L * self.B  # Typical waterplane coefficient
        
        # Heave stiffness
        K33 = RHO_WATER * G * A_wp
        
        # Pitch stiffness (simplified)
        # Longitudinal second moment of waterplane area
        I_L = (1.0 / 12.0) * self.L**3 * self.B * 0.8  # With waterplane coeff
        BM_L = I_L / self.displacement
        
        # KG approximation (center of gravity above keel)
        KG = 0.55 * self.T  # Typical for cargo ships
        
        # KB (center of buoyancy above keel)
        KB = 0.5 * self.T
        
        K55 = RHO_WATER * G * self.displacement * (BM_L + KB - KG)
        
        return K33, K55


def compute_natural_frequencies(hull: HullStructure) -> Dict[str, float]:
    """Compute natural frequencies of the hull structure.
    
    Returns
    -------
    Dict[str, float]
        Natural frequencies in Hz:
        - f_heave: heave natural frequency
        - f_pitch: pitch natural frequency
        - f_bend: list of vertical bending natural frequencies
    """
    K33, K55 = hull.hydrostatic_stiffness()
    
    # Heave natural frequency
    M33 = hull.mass_ship * (1.0 + hull.added_mass_heave)
    omega_heave = np.sqrt(K33 / M33)
    f_heave = omega_heave / (2.0 * np.pi)
    
    # Pitch natural frequency
    M55 = hull.Iyy_ship * (1.0 + hull.added_mass_pitch)
    omega_pitch = np.sqrt(K55 / M55)
    f_pitch = omega_pitch / (2.0 * np.pi)
    
    # Vertical bending natural frequencies (simply-supported beam)
    # omega_n = (n*pi/L)^2 * sqrt(EI / m)
    m_per_length = hull.mass_ship / hull.L
    f_bend = []
    for n in range(1, hull.n_bend_modes + 1):
        omega_n = (n * np.pi / hull.L)**2 * np.sqrt(hull.EI / m_per_length)
        f_bend.append(omega_n / (2.0 * np.pi))
    
    return {
        "f_heave": float(f_heave),
        "f_pitch": float(f_pitch),
        "f_bend": f_bend,
        "omega_heave": float(omega_heave),
        "omega_pitch": float(omega_pitch),
    }


@dataclass
class MultiDOFResponse:
    """Multi-DOF ship motion response to excitation.
    
    Computes the frequency-domain response of the hull to ice load excitation.
    """
    
    hull: HullStructure
    
    def frequency_response(
        self,
        freq: np.ndarray,
        F_heave: np.ndarray,
        F_pitch: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        """Compute frequency-domain response to excitation forces.
        
        Parameters
        ----------
        freq : np.ndarray
            Frequency array [Hz]
        F_heave : np.ndarray
            Heave excitation force spectrum [N] (complex or magnitude)
        F_pitch : np.ndarray
            Pitch excitation moment spectrum [N*m] (complex or magnitude)
            
        Returns
        -------
        Dict[str, np.ndarray]
            Response spectra:
            - x3: heave response [m]
            - x5: pitch response [rad]
            - H_heave: heave FRF [m/N]
            - H_pitch: pitch FRF [rad/(N*m)]
        """
        K33, K55 = self.hull.hydrostatic_stiffness()
        
        # Heave FRF
        M33 = self.hull.mass_ship * (1.0 + self.hull.added_mass_heave)
        omega_n_heave = np.sqrt(K33 / M33)
        zeta_heave = self.hull.damping_ratio_heave
        
        omega = 2.0 * np.pi * freq
        H_heave = 1.0 / (K33 * (1.0 - (omega/omega_n_heave)**2 + 2j*zeta_heave*omega/omega_n_heave))
        
        # Pitch FRF
        M55 = self.hull.Iyy_ship * (1.0 + self.hull.added_mass_pitch)
        omega_n_pitch = np.sqrt(K55 / M55)
        zeta_pitch = self.hull.damping_ratio_pitch
        
        H_pitch = 1.0 / (K55 * (1.0 - (omega/omega_n_pitch)**2 + 2j*zeta_pitch*omega/omega_n_pitch))
        
        # Response spectra
        x3 = H_heave * F_heave
        x5 = H_pitch * F_pitch
        
        return {
            "x3": x3,
            "x5": x5,
            "H_heave": H_heave,
            "H_pitch": H_pitch,
            "freq": freq,
        }
    
    def motion_due_to_ice(
        self,
        freq: np.ndarray,
        S_ice: np.ndarray,
        x_cg_to_prop: float = 45.0,
    ) -> Dict[str, np.ndarray]:
        """Compute ship motions due to ice load excitation.
        
        The ice load excites heave and pitch motions. The vertical velocity
        at the propeller location (due to heave + pitch) modulates the
        propeller inflow and causes thrust pulsation.
        
        Parameters
        ----------
        freq : np.ndarray
            Frequency array [Hz]
        S_ice : np.ndarray
            Ice load power spectral density [N^2/Hz]
        x_cg_to_prop : float
            Distance from CG to propeller (positive aft) [m]
            
        Returns
        -------
        Dict[str, np.ndarray]
            Motion spectra at propeller location:
            - S_heave: heave PSD [m^2/Hz]
            - S_pitch: pitch PSD [rad^2/Hz]
            - S_vel_prop: vertical velocity PSD at propeller [(m/s)^2/Hz]
        """
        # Ice load distribution: assume uniform over forward half of ship
        # Heave excitation: total ice force
        F_heave = np.sqrt(S_ice + 0j)  # Convert to complex amplitude
        
        # Pitch excitation: ice force * lever arm (forward of CG)
        x_ice_to_cg = 0.3 * self.hull.L  # Ice contact ~30% L forward of CG
        F_pitch = F_heave * x_ice_to_cg
        
        # Get frequency response
        resp = self.frequency_response(freq, F_heave, F_pitch)
        
        # Response PSDs
        S_heave = np.abs(resp["x3"])**2
        S_pitch = np.abs(resp["x5"])**2
        
        # Vertical velocity at propeller due to heave + pitch
        # v_prop = x3_dot + x_cg_to_prop * x5_dot
        # In frequency domain: v_prop(omega) = j*omega * (x3 + x_cg_to_prop * x5)
        omega = 2.0 * np.pi * freq
        x_prop = resp["x3"] + x_cg_to_prop * resp["x5"]
        v_prop = 1j * omega * x_prop
        S_vel_prop = np.abs(v_prop)**2
        
        return {
            "S_heave": S_heave,
            "S_pitch": S_pitch,
            "S_vel_prop": S_vel_prop,
            "x3": resp["x3"],
            "x5": resp["x5"],
            "freq": freq,
        }


def frequency_response_function(
    freq: np.ndarray,
    f_n: float,
    zeta: float,
    K: float,
) -> np.ndarray:
    """Compute frequency response function for a single DOF oscillator.
    
    H(omega) = 1 / (K * (1 - (omega/omega_n)^2 + 2j*zeta*omega/omega_n))
    
    Parameters
    ----------
    freq : np.ndarray
        Frequency array [Hz]
    f_n : float
        Natural frequency [Hz]
    zeta : float
        Damping ratio [-]
    K : float
        Stiffness [N/m or N*m/rad]
        
    Returns
    -------
    np.ndarray
        Complex FRF values
    """
    omega = 2.0 * np.pi * freq
    omega_n = 2.0 * np.pi * f_n
    
    H = 1.0 / (K * (1.0 - (omega/omega_n)**2 + 2j*zeta*omega/omega_n))
    return H
