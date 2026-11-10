"""Propeller-hull-ice coupling: iterative solution and transfer function.

Physics
-------
The three-field coupling involves:
1. Propeller → Wake: Thrust determines wake velocity (actuator disk theory)
2. Wake → Ice: Wake reduces effective ship-ice relative velocity
3. Ice → Hull: Ice load excites hull motions (heave, pitch)
4. Hull → Propeller: Hull motions modulate propeller inflow → thrust pulsation

Coupling iteration:
    Given: V_ship, propeller characteristics, ice conditions
    Iterate:
    1. V_inflow = V_ship * (1 - w)  [wake fraction]
    2. Compute thrust T from propeller model
    3. Compute axial induction a from T
    4. V_wake = V_inflow * (1 - 2a)
    5. V_ice_eff = V_ship - (V_inflow - V_wake)  [wake modulation]
    6. R_ice = f(V_ice_eff)  [ice resistance]
    7. Hull response: heave/pitch from R_ice excitation
    8. δV_inflow = heave_dot + x_prop * pitch_dot  [inflow modulation]
    9. δT = dT/dV * δV_inflow  [thrust pulsation]
    10. Check convergence: |δT/δT_prev - 1| < tol
    
Transfer function:
    H_thrust(f) = S_thrust(f) / S_ice(f)
    
    where S_thrust is the thrust pulsation PSD and S_ice is the ice load PSD.

References
----------
- Carlton, J.S. (2007) "Marine Propellers and Propulsion", 2nd ed.
- Bose, D.J. (1991) "Ice-induced propeller torque fluctuations", POAC'91.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .propeller import ActuatorDisk, PropellerGeometry
from .wake import WakeModel, WakeField
from .ice_load import IceLoadModel, HullGeometry, IceProperties
from .hull_structure import HullStructure, MultiDOFResponse, compute_natural_frequencies

__all__ = [
    "CouplingSolver",
    "CouplingResult",
    "compute_transfer_function",
]


@dataclass
class CouplingResult:
    """Result of the propeller-hull-ice coupling analysis."""
    
    # Convergence history
    n_iterations: int
    converged: bool
    
    # Propeller operating point
    T_mean: float           # Mean thrust [N]
    Q_mean: float           # Mean torque [N*m]
    K_T: float              # Thrust coefficient [-]
    a_axial: float          # Axial induction factor [-]
    
    # Wake
    V_inflow: float         # Inflow velocity [m/s]
    V_wake: float           # Wake velocity [m/s]
    V_wake_induced: float   # Wake-induced velocity reduction [m/s]
    
    # Ice loads
    R_ice_no_wake: float    # Ice resistance without wake [N]
    R_ice_with_wake: float  # Ice resistance with wake [N]
    delta_R_ice: float      # Reduction in ice resistance [N]
    
    # Hull response
    heave_amplitude: float  # Heave amplitude [m]
    pitch_amplitude: float  # Pitch amplitude [rad]
    
    # Thrust pulsation
    delta_T_no_vibration: float  # Thrust pulsation without hull vibration [N]
    delta_T_with_vibration: float  # Thrust pulsation with hull vibration [N]
    delta_delta_T: float    # Increase due to vibration [N]
    
    # Transfer function
    freq: np.ndarray        # Frequency array [Hz]
    H_thrust: np.ndarray    # Transfer function H_thrust(f) [-]
    S_ice: np.ndarray       # Ice load PSD [N^2/Hz]
    S_thrust: np.ndarray    # Thrust pulsation PSD [N^2/Hz]


@dataclass
class CouplingSolver:
    """Iterative solver for propeller-hull-ice coupling.
    
    The solver iterates between:
    1. Propeller performance (thrust, torque, wake)
    2. Ice resistance (modulated by wake)
    3. Hull response (heave, pitch from ice excitation)
    4. Thrust pulsation (from inflow modulation due to hull motions)
    """
    
    # Ship and propeller parameters
    prop_geom: PropellerGeometry
    hull_geom: HullGeometry
    ice_props: IceProperties
    hull_struct: HullStructure
    
    # Operating conditions
    V_ship: float = 3.0     # Ship speed [m/s]
    n_rps: float = 2.0      # Propeller rotation rate [rev/s]
    
    # Wake model
    wake_fraction: float = 0.25  # Taylor wake fraction [-]
    
    # Solver settings
    max_iterations: int = 20
    tolerance: float = 0.01  # 1% convergence criterion
    
    def solve(self) -> CouplingResult:
        """Run the coupling iteration to convergence.
        
        Returns
        -------
        CouplingResult
            Complete coupling analysis result.
        """
        # Initialize models
        prop = ActuatorDisk(self.prop_geom)
        wake_model = WakeModel(wake_fraction=self.wake_fraction)
        ice_model = IceLoadModel(self.hull_geom, self.ice_props, wake_modulation=True)
        hull_resp = MultiDOFResponse(self.hull_struct)
        
        # Initial guess: no wake modulation, no vibration
        V_wake_induced = 0.0
        delta_T_prev = 0.0
        
        # Ice resistance without wake (for comparison)
        R_ice_no_wake = ice_model.ice_resistance(self.V_ship, V_wake_induced=0.0)["R_total"]
        
        converged = False
        n_iter = 0
        
        for i in range(self.max_iterations):
            n_iter = i + 1
            
            # 1. Compute inflow velocity
            V_inflow = wake_model.inflow_velocity(self.V_ship)
            
            # 2. Compute propeller thrust (with current wake estimate)
            # The thrust depends on advance coefficient J = V_inflow / (n * D)
            J = V_inflow / (self.n_rps * self.prop_geom.D)
            K_T = prop.thrust_coefficient(J)
            T = prop.thrust_from_KT(K_T, self.n_rps)
            
            # 3. Compute axial induction factor
            a = prop.axial_induction_from_thrust(T, V_inflow)
            
            # 4. Compute wake velocity
            wake_field = WakeField(V_inflow=V_inflow, axial_induction=a)
            V_wake = wake_field.V_wake_far
            V_wake_induced_new = V_inflow - V_wake  # = 2 * a * V_inflow
            
            # 5. Compute ice resistance with wake modulation
            ice_result = ice_model.ice_resistance(self.V_ship, V_wake_induced_new)
            R_ice_with_wake = ice_result["R_total"]
            
            # 6. Compute hull response to ice excitation
            # Generate ice load spectrum
            freq = np.linspace(0.01, 2.0, 200)
            ice_spectrum = ice_model.ice_load_spectrum(
                self.V_ship, V_wake_induced_new, n_freq=len(freq), f_max=2.0
            )
            
            # Hull motion response
            motion_resp = hull_resp.motion_due_to_ice(freq, ice_spectrum["S_ice"])
            
            # Characteristic heave/pitch amplitudes (from RMS of response)
            df = freq[1] - freq[0]
            heave_rms = np.sqrt(np.sum(motion_resp["S_heave"]) * df)
            pitch_rms = np.sqrt(np.sum(motion_resp["S_pitch"]) * df)
            
            # 7. Compute thrust pulsation due to hull vibration
            # δV_inflow = heave_dot + x_prop * pitch_dot
            # In frequency domain: δV(ω) = jω * (x3 + x_prop * x5)
            x_prop = 0.45 * self.hull_struct.L  # Propeller location aft of CG
            omega = 2.0 * np.pi * freq
            v_inflow_spectrum = np.abs(1j * omega * (motion_resp["x3"] + x_prop * motion_resp["x5"]))**2
            
            # Thrust pulsation: δT = (dT/dV) * δV
            # dT/dV from propeller characteristics
            dT_dV = prop.thrust_derivative_wrt_velocity(V_inflow, self.n_rps)
            
            # Thrust pulsation spectrum (without vibration - just from wake non-uniformity)
            # This is a baseline from propeller operating in non-uniform wake
            delta_T_no_vib = 0.05 * T  # 5% of mean thrust as baseline
            
            # Thrust pulsation spectrum (with vibration)
            S_delta_T = dT_dV**2 * v_inflow_spectrum
            delta_T_with_vib = np.sqrt(np.sum(S_delta_T) * df)
            
            # Check convergence
            if delta_T_prev > 0:
                rel_change = abs(delta_T_with_vib / delta_T_prev - 1.0)
                if rel_change < self.tolerance:
                    converged = True
                    break
            
            delta_T_prev = delta_T_with_vib
            V_wake_induced = V_wake_induced_new
        
        # Compute transfer function
        S_thrust = dT_dV**2 * v_inflow_spectrum
        S_ice = ice_spectrum["S_ice"]
        H_thrust = compute_transfer_function(freq, S_thrust, S_ice)
        
        return CouplingResult(
            n_iterations=n_iter,
            converged=converged,
            T_mean=T,
            Q_mean=prop.torque_from_KT(K_T, self.n_rps),
            K_T=K_T,
            a_axial=a,
            V_inflow=V_inflow,
            V_wake=V_wake,
            V_wake_induced=V_wake_induced,
            R_ice_no_wake=R_ice_no_wake,
            R_ice_with_wake=R_ice_with_wake,
            delta_R_ice=R_ice_no_wake - R_ice_with_wake,
            heave_amplitude=heave_rms,
            pitch_amplitude=pitch_rms,
            delta_T_no_vibration=delta_T_no_vib,
            delta_T_with_vibration=delta_T_with_vib,
            delta_delta_T=delta_T_with_vib - delta_T_no_vib,
            freq=freq,
            H_thrust=H_thrust,
            S_ice=S_ice,
            S_thrust=S_thrust,
        )


def compute_transfer_function(
    freq: np.ndarray,
    S_thrust: np.ndarray,
    S_ice: np.ndarray,
) -> np.ndarray:
    """Compute thrust pulsation transfer function.
    
    H_thrust(f) = S_thrust(f) / S_ice(f)
    
    This is the ratio of thrust pulsation PSD to ice load PSD,
    representing how ice load fluctuations are transmitted to thrust.
    
    Parameters
    ----------
    freq : np.ndarray
        Frequency array [Hz]
    S_thrust : np.ndarray
        Thrust pulsation power spectral density [N^2/Hz]
    S_ice : np.ndarray
        Ice load power spectral density [N^2/Hz]
        
    Returns
    -------
    np.ndarray
        Transfer function magnitude |H_thrust(f)| [-]
    """
    # Avoid division by zero
    S_ice_safe = np.maximum(S_ice, 1e-30)
    H = S_thrust / S_ice_safe
    
    # Return magnitude (transfer function is real for this formulation)
    return np.sqrt(np.abs(H))
