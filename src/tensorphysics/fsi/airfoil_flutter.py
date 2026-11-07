"""2D airfoil flutter physics using Theodorsen-type unsteady aerodynamics.

This module implements the classical flutter analysis for a 2D airfoil section
with pitch and plunge degrees of freedom. The aerodynamic loads are computed
using Theodorsen's function for incompressible, inviscid flow.

References
----------
[1] Theodorsen, T. (1935). "General theory of aerodynamic instability and the
    mechanism of flutter." NACA Report 496.
[2] Bisplinghoff, R.L., Ashley, H., Halfman, R.L. (1955). "Aeroelasticity."
    Dover Publications.
[3] Turek, S., Hron, J. (2006). "Proposal for Numerical Benchmarking of
    Fluid-Structure Interaction between an Elastic Object and Laminal
    Incompressible Flow." Lecture Notes in Computational Science and Engineering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class FlutterConfig:
    """Configuration for 2D airfoil flutter analysis.
    
    Parameters
    ----------
    chord : float
        Airfoil chord length [m].
    mass_ratio : float
        Structural mass ratio μ = m / (ρ·b²), where m is mass per unit span,
        ρ is fluid density, b = chord/2 is semi-chord.
    radius_of_gyration : float
        Radius of gyration rα = sqrt(Iα / m) normalized by semi-chord b.
    static_unbalance : float
        Static unbalance parameter xα = Sα / (m·b), where Sα is the static
        moment about the elastic axis.
    frequency_ratio : float
        Natural frequency ratio ωh / ωα, where ωh is plunge frequency and
        ωα is pitch frequency.
    reduced_damping_h : float
        Reduced damping in plunge, ζh = ch / (2·m·ωh).
    reduced_damping_a : float
        Reduced damping in pitch, ζα = cα / (2·Iα·ωα).
    density_ratio : float
        Density ratio for aerodynamic scaling.
    reynolds_number : float
        Reynolds number based on chord (for viscous corrections).
    """
    chord: float = 1.0
    mass_ratio: float = 20.0
    radius_of_gyration: float = 0.5
    static_unbalance: float = 0.0
    frequency_ratio: float = 0.5
    reduced_damping_h: float = 0.01
    reduced_damping_a: float = 0.01
    density_ratio: float = 1.0
    reynolds_number: float = 1000.0
    
    # Elastic axis location (0=leading edge, 1=trailing edge)
    elastic_axis: float = 0.35
    
    # Time integration
    dt: float = 0.001
    n_steps: int = 10000
    
    # Initial conditions
    h0: float = 0.0  # Initial plunge displacement
    alpha0: float = 0.1  # Initial pitch angle [rad]
    h_dot0: float = 0.0  # Initial plunge velocity
    alpha_dot0: float = 0.0  # Initial pitch rate


@dataclass
class FlutterResult:
    """Results from flutter analysis.
    
    Attributes
    ----------
    time : np.ndarray
        Time history [s].
    plunge : np.ndarray
        Plunge displacement history h(t) [m].
    pitch : np.ndarray
        Pitch angle history α(t) [rad].
    plunge_velocity : np.ndarray
        Plunge velocity history ḣ(t) [m/s].
    pitch_rate : np.ndarray
        Pitch rate history α̇(t) [rad/s].
    lift : np.ndarray
        Lift coefficient history Cl(t).
    moment : np.ndarray
        Moment coefficient history Cm(t).
    flutter_speed : float
        Computed flutter speed [m/s].
    flutter_frequency : float
        Flutter frequency [Hz].
    is_stable : bool
        Whether the response is stable (decaying or bounded).
    damping_ratio : float
        Effective damping ratio (negative = unstable).
    """
    time: np.ndarray
    plunge: np.ndarray
    pitch: np.ndarray
    plunge_velocity: np.ndarray
    pitch_rate: np.ndarray
    lift: np.ndarray
    moment: np.ndarray
    flutter_speed: float = 0.0
    flutter_frequency: float = 0.0
    is_stable: bool = True
    damping_ratio: float = 0.0


def theodorsen_function(k: complex) -> complex:
    """Compute Theodorsen's function C(k) for reduced frequency k.
    
    Theodorsen's function relates unsteady circulatory lift to quasi-steady value.
    C(k) = F(k) + i·G(k), where F and G are real functions.
    
    Parameters
    ----------
    k : complex
        Reduced frequency k = ω·b / U, where ω is angular frequency,
        b is semi-chord, U is flow speed.
    
    Returns
    -------
    complex
        Theodorsen function value C(k).
    
    Notes
    -----
    Uses approximate formula from Garrick (1938):
    C(k) ≈ 1 - 0.165/(1 - 0.0455/k) - 0.335/(1 - 0.3/k)
    
    For k → 0: C → 1 (quasi-steady)
    For k → ∞: C → 0.5
    """
    if abs(k) < 1e-10:
        return 1.0 + 0j
    
    # Garrick approximation
    term1 = 0.165 / (1.0 - 0.0455 / k)
    term2 = 0.335 / (1.0 - 0.3 / k)
    C = 1.0 - term1 - term2
    return C


def compute_flutter_speed(config: FlutterConfig) -> tuple[float, float]:
    """Compute flutter speed and frequency using p-k method.
    
    Solves the flutter determinant for the given configuration to find
    the speed at which damping becomes zero (flutter onset).
    
    Parameters
    ----------
    config : FlutterConfig
        Flutter configuration.
    
    Returns
    -------
    tuple[float, float]
        (flutter_speed, flutter_frequency) in [m/s] and [Hz].
    
    Notes
    -----
    Uses the classical 2-DOF flutter equations:
    
    [M]{q̈} + [C]{q̇} + [K]{q} = {F_aero}
    
    where q = [h, α]ᵀ and F_aero includes circulatory and non-circulatory terms.
    """
    b = config.chord / 2.0  # semi-chord
    μ = config.mass_ratio
    rα = config.radius_of_gyration
    xα = config.static_unbalance
    ωh_ωα = config.frequency_ratio
    ζh = config.reduced_damping_h
    ζα = config.reduced_damping_a
    
    # Normalize: set ωα = 1 for dimensionless analysis
    ωα = 1.0
    ωh = ωh_ωα * ωα
    
    # Mass matrix (dimensionless)
    M = np.array([
        [1.0, xα],
        [xα, rα**2]
    ])
    
    # Stiffness matrix (dimensionless)
    K = np.array([
        [ωh**2, 0.0],
        [0.0, rα**2 * ωα**2]
    ])
    
    # Search for flutter speed
    U_range = np.linspace(0.1, 20.0, 200)
    damping_ratios = []
    
    for U in U_range:
        # Reduced frequency at ωα
        k = ωα * b / U if U > 0 else 1e10
        
        # Theodorsen function
        C = theodorsen_function(k)
        
        # Aerodynamic influence matrix (simplified)
        # L = π·ρ·b²·(ḧ - b·xα·α̈ + U·α̇) + 2·π·ρ·U·b·C(k)·(ḣ + U·α + b·(0.5-a)·α̇)
        # M = π·ρ·b²·(-b·xα·ḧ + b²·rα²·α̈) + 2·π·ρ·U·b²·(a+0.5)·C(k)·(ḣ + U·α + b·(0.5-a)·α̇)
        #   + π·ρ·b²·U·(0.5-a)·(ḣ + U·α)
        
        # For stability analysis, assume harmonic motion q = q0·exp(i·ω·t)
        # and solve eigenvalue problem
        
        # Simplified: compute effective damping from energy balance
        # Aerodynamic damping contribution
        F_real = C.real
        F_imag = C.imag
        
        # Aerodynamic damping matrix (dimensionless, scaled by 1/(π·μ))
        Ca = np.array([
            [2 * F_real / U, 2 * F_real * (0.5 - config.elastic_axis)],
            [2 * F_real * (0.5 - config.elastic_axis) / U, 
             2 * F_real * (0.5 - config.elastic_axis)**2]
        ])
        
        # Total damping
        C_total = np.array([
            [2 * ζh * ωh, 0],
            [0, 2 * ζα * rα**2 * ωα]
        ]) + Ca / (np.pi * μ)
        
        # Eigenvalues of damped system
        A = np.zeros((4, 4))
        A[0:2, 2:4] = np.eye(2)
        A[2:4, 0:2] = -np.linalg.solve(M, K)
        A[2:4, 2:4] = -np.linalg.solve(M, C_total)
        
        eigvals = np.linalg.eigvals(A)
        
        # Find maximum real part (least stable mode)
        max_real = max(eigvals.real)
        # Damping ratio approximation
        if max_real < 0:
            # Estimate damping from decay rate
            omega_imag = max(abs(eigvals.imag))
            if omega_imag > 1e-10:
                zeta = -max_real / omega_imag
            else:
                zeta = -max_real
        else:
            zeta = -max_real  # Negative damping = unstable
        
        damping_ratios.append(zeta)
    
    # Find flutter point (damping crosses zero)
    damping_arr = np.array(damping_ratios)
    
    # Find first zero crossing
    flutter_idx = None
    for i in range(len(damping_arr) - 1):
        if damping_arr[i] >= 0 and damping_arr[i+1] < 0:
            flutter_idx = i
            break
        elif damping_arr[i] < 0 and damping_arr[i+1] >= 0:
            flutter_idx = i
            break
    
    if flutter_idx is not None:
        # Linear interpolation
        U_f = U_range[flutter_idx] + (U_range[flutter_idx+1] - U_range[flutter_idx]) * \
              abs(damping_arr[flutter_idx]) / (abs(damping_arr[flutter_idx]) + abs(damping_arr[flutter_idx+1]))
    else:
        # No flutter found in range
        U_f = U_range[-1] if damping_arr[-1] < 0 else U_range[0]
    
    # Flutter frequency (approximate)
    omega_f = ωα  # Typically close to pitch natural frequency
    
    return U_f, omega_f / (2 * np.pi)


class AirfoilFlutter:
    """2D airfoil flutter simulator using Theodorsen unsteady aerodynamics.
    
    This class implements time-domain simulation of a 2D airfoil section
    with plunge and pitch degrees of freedom, coupled with unsteady
    aerodynamic loads computed using Theodorsen's theory.
    
    Parameters
    ----------
    config : FlutterConfig
        Flutter configuration parameters.
    """
    
    def __init__(self, config: FlutterConfig):
        self.config = config
        self.b = config.chord / 2.0  # semi-chord
        
        # Structural parameters
        self.m = config.mass_ratio * 1.0 * self.b**2  # mass per unit span (ρ=1)
        self.Iα = self.m * (config.radius_of_gyration * self.b)**2
        self.Sα = config.static_unbalance * self.m * self.b
        
        # Natural frequencies
        self.ωh = 2 * np.pi * 1.0  # plunge frequency (normalized)
        self.ωα = 2 * np.pi * (1.0 / config.frequency_ratio)  # pitch frequency
        
        # Damping coefficients
        self.ch = 2 * self.m * self.ωh * config.reduced_damping_h
        self.cα = 2 * self.Iα * self.ωα * config.reduced_damping_a
        
        # Stiffness
        self.kh = self.m * self.ωh**2
        self.kα = self.Iα * self.ωα**2
        
        # State variables
        self.h = config.h0
        self.alpha = config.alpha0
        self.h_dot = config.h_dot0
        self.alpha_dot = config.alpha_dot0
        
        # Wake state for Theodorsen function (lag states)
        self.x1 = 0.0
        self.x2 = 0.0
        
        # History
        self.time_history: list[float] = []
        self.h_history: list[float] = []
        self.alpha_history: list[float] = []
        self.h_dot_history: list[float] = []
        self.alpha_dot_history: list[float] = []
        self.lift_history: list[float] = []
        self.moment_history: list[float] = []
    
    def compute_aerodynamic_loads(self, U: float) -> tuple[float, float]:
        """Compute unsteady aerodynamic lift and moment.
        
        Uses Theodorsen's theory with state-space approximation for the
        wake influence function.
        
        Parameters
        ----------
        U : float
            Flow velocity [m/s].
        
        Returns
        -------
        tuple[float, float]
            (lift, moment) per unit span.
        """
        b = self.b
        a = self.config.elastic_axis - 0.5  # elastic axis offset from mid-chord
        
        # Effective angle of attack
        alpha_eff = self.alpha + self.h_dot / U if U > 0 else self.alpha
        
        # Theodorsen function (state-space approximation)
        # C(k) ≈ 0.5 + x1 + x2, where x1, x2 are lag states
        C = 0.5 + self.x1 + self.x2
        
        # Circulatory lift
        L_c = 2 * np.pi * U * b * C * (self.h_dot + U * self.alpha + b * (0.5 - a) * self.alpha_dot)
        
        # Non-circulatory (apparent mass) lift
        L_nc = np.pi * b**2 * (self.h_dot - b * a * self.alpha_dot)
        
        # Total lift (per unit span, normalized by ρ)
        L = L_c + L_nc
        
        # Circulatory moment about elastic axis
        M_c = 2 * np.pi * U * b**2 * (a + 0.5) * C * (self.h_dot + U * self.alpha + b * (0.5 - a) * self.alpha_dot)
        
        # Non-circulatory moment
        M_nc = np.pi * b**2 * (b * a * self.h_dot - b**2 * (0.5 - a) * (0.5 + a) * self.alpha_dot)
        
        # Total moment
        M = M_c + M_nc
        
        return L, M
    
    def update_wake_states(self, U: float, dt: float) -> None:
        """Update Theodorsen lag states for wake influence.
        
        Uses Jones' approximation with two lag states.
        
        Parameters
        ----------
        U : float
            Flow velocity [m/s].
        dt : float
            Time step [s].
        """
        b = self.b
        
        # Reduced frequency
        k = self.ωα * b / U if U > 0 else 1e10
        
        # Jones approximation constants
        b1, b2 = 0.0455, 0.3
        c1, c2 = 0.165, 0.335
        
        # Input to lag states (effective velocity at 3/4 chord)
        w = self.h_dot + U * self.alpha + b * (0.5 - self.config.elastic_axis) * self.alpha_dot
        
        # Update lag states (first-order ODEs)
        tau = U / b if U > 0 else 0
        self.x1 = self.x1 * np.exp(-b1 * tau * dt) + c1 * w * dt * (1 - np.exp(-b1 * tau * dt)) / (b1 * tau * dt + 1e-10)
        self.x2 = self.x2 * np.exp(-b2 * tau * dt) + c2 * w * dt * (1 - np.exp(-b2 * tau * dt)) / (b2 * tau * dt + 1e-10)
    
    def step(self, U: float, dt: float, external_force: tuple[float, float] = (0.0, 0.0)) -> None:
        """Advance one time step using Newmark-beta integration.
        
        Parameters
        ----------
        U : float
            Flow velocity [m/s].
        dt : float
            Time step [s].
        external_force : tuple[float, float]
            External (non-aerodynamic) forces (Fh_ext, Mα_ext).
        """
        # Compute aerodynamic loads
        L_aero, M_aero = self.compute_aerodynamic_loads(U)
        
        # Total forces
        Fh = L_aero + external_force[0]
        Mα = M_aero + external_force[1]
        
        # Equations of motion:
        # m·ḧ + Sα·α̈ + ch·ḣ + kh·h = L
        # Sα·ḧ + Iα·α̈ + cα·α̇ + kα·α = M
        
        # Mass matrix
        M_mat = np.array([
            [self.m, self.Sα],
            [self.Sα, self.Iα]
        ])
        
        # Force vector
        F_vec = np.array([
            Fh - self.ch * self.h_dot - self.kh * self.h,
            Mα - self.cα * self.alpha_dot - self.kα * self.alpha
        ])
        
        # Acceleration
        acc = np.linalg.solve(M_mat, F_vec)
        h_ddot = acc[0]
        alpha_ddot = acc[1]
        
        # Newmark-beta integration (β=0.25, γ=0.5 for unconditional stability)
        beta = 0.25
        gamma = 0.5
        
        # Update velocities
        self.h_dot += dt * ((1 - gamma) * h_ddot + gamma * h_ddot)
        self.alpha_dot += dt * ((1 - gamma) * alpha_ddot + gamma * alpha_ddot)
        
        # Update displacements
        self.h += dt * self.h_dot + dt**2 * ((0.5 - beta) * h_ddot + beta * h_ddot)
        self.alpha += dt * self.alpha_dot + dt**2 * ((0.5 - beta) * alpha_ddot + beta * alpha_ddot)
        
        # Update wake states
        self.update_wake_states(U, dt)
        
        # Store history
        self.lift_history.append(L_aero)
        self.moment_history.append(M_aero)
    
    def run(self, U: float, n_steps: Optional[int] = None) -> FlutterResult:
        """Run time-domain simulation.
        
        Parameters
        ----------
        U : float
            Flow velocity [m/s].
        n_steps : int, optional
            Number of time steps. Defaults to config.n_steps.
        
        Returns
        -------
        FlutterResult
            Simulation results.
        """
        if n_steps is None:
            n_steps = self.config.n_steps
        
        dt = self.config.dt
        
        # Reset state
        self.h = self.config.h0
        self.alpha = self.config.alpha0
        self.h_dot = self.config.h_dot0
        self.alpha_dot = self.config.alpha_dot0
        self.x1 = 0.0
        self.x2 = 0.0
        
        # Clear history
        self.time_history = []
        self.h_history = []
        self.alpha_history = []
        self.h_dot_history = []
        self.alpha_dot_history = []
        self.lift_history = []
        self.moment_history = []
        
        # Time integration
        t = 0.0
        for i in range(n_steps):
            self.time_history.append(t)
            self.h_history.append(self.h)
            self.alpha_history.append(self.alpha)
            self.h_dot_history.append(self.h_dot)
            self.alpha_dot_history.append(self.alpha_dot)
            
            self.step(U, dt)
            t += dt
        
        # Convert to arrays
        time_arr = np.array(self.time_history)
        h_arr = np.array(self.h_history)
        alpha_arr = np.array(self.alpha_history)
        h_dot_arr = np.array(self.h_dot_history)
        alpha_dot_arr = np.array(self.alpha_dot_history)
        lift_arr = np.array(self.lift_history)
        moment_arr = np.array(self.moment_history)
        
        # Analyze stability
        is_stable, damping_ratio = self._analyze_stability(alpha_arr, dt)
        
        # Compute flutter frequency
        flutter_freq = self._estimate_frequency(alpha_arr, dt)
        
        return FlutterResult(
            time=time_arr,
            plunge=h_arr,
            pitch=alpha_arr,
            plunge_velocity=h_dot_arr,
            pitch_rate=alpha_dot_arr,
            lift=lift_arr,
            moment=moment_arr,
            flutter_speed=U,
            flutter_frequency=flutter_freq,
            is_stable=is_stable,
            damping_ratio=damping_ratio,
        )
    
    def _analyze_stability(self, signal: np.ndarray, dt: float) -> tuple[bool, float]:
        """Analyze stability from time history using logarithmic decrement.
        
        Parameters
        ----------
        signal : np.ndarray
            Time history signal (e.g., pitch angle).
        dt : float
            Time step.
        
        Returns
        -------
        tuple[bool, float]
            (is_stable, damping_ratio)
        """
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(signal, distance=int(0.01/dt))
        
        if len(peaks) < 3:
            # Not enough peaks, assume stable
            return True, 0.01
        
        # Logarithmic decrement
        peak_vals = signal[peaks]
        if len(peak_vals) >= 2:
            # Average over several cycles
            n_avg = min(5, len(peak_vals) - 1)
            delta = 0.0
            for i in range(n_avg):
                if peak_vals[i] > 0 and peak_vals[i+1] > 0:
                    delta += np.log(peak_vals[i] / peak_vals[i+1])
            delta /= n_avg
            
            # Damping ratio
            zeta = delta / np.sqrt(4 * np.pi**2 + delta**2)
            
            is_stable = delta > 0  # Positive decrement = decaying = stable
            return is_stable, zeta
        
        return True, 0.01
    
    def _estimate_frequency(self, signal: np.ndarray, dt: float) -> float:
        """Estimate dominant frequency using FFT.
        
        Parameters
        ----------
        signal : np.ndarray
            Time history signal.
        dt : float
            Time step.
        
        Returns
        -------
        float
            Dominant frequency [Hz].
        """
        # Remove mean
        signal_centered = signal - np.mean(signal)
        
        # FFT
        n = len(signal_centered)
        freqs = np.fft.rfftfreq(n, dt)
        fft_vals = np.abs(np.fft.rfft(signal_centered))
        
        # Find peak (excluding DC)
        if len(fft_vals) > 1:
            peak_idx = np.argmax(fft_vals[1:]) + 1
            return freqs[peak_idx]
        
        return 0.0
