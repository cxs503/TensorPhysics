"""
Nonlinear Shaft Torsional Vibration with Random Ice Loading
============================================================
Extension of m763 linear model to include:
1. Backlash nonlinearity (gear/coupling clearance)
2. Random ice load spectrum (Shinozuka-Jan harmonic superposition)
3. Multi-segment variable cross-section shaft (transfer matrix assembly)
4. Nonlinear transient solver (Newmark-beta + Newton-Raphson)
5. Monte Carlo simulation (N>=100 realizations)
6. Fatigue damage (Miner's rule, DNVGL S-N curve)

Physical model
--------------
Backlash: T(theta) = k*(theta - delta)   for theta > delta
                   = 0                    for |theta| <= delta
                   = k*(theta + delta)    for theta < -delta

Random ice load PSD:
  - BPF harmonics (blade passing frequency)
  - Broadband random component
  - Low-frequency impact component

Multi-segment shaft:
  Transfer matrix method assembling global M/K from segments with
  different diameter, length, material properties.

References
----------
- Nayfeh & Mook (1979): Nonlinear Oscillations - subharmonic resonance
- Shinozuka & Jan (1972): Digital simulation of random processes
- DNVGL-OS-C101: S-N curves for fatigue analysis
- Miner (1945): Linear cumulative damage rule
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import warnings


# =============================================================
# 1. BACKLASH NONLINEARITY MODEL
# =============================================================

@dataclass
class BacklashModel:
    """Gear/coupling backlash (deadband) nonlinearity.
    
    Torque-deflection relationship:
        T(theta) = k * (theta - delta)   for theta > delta
        T(theta) = 0                      for |theta| <= delta
        T(theta) = k * (theta + delta)    for theta < -delta
    
    Parameters
    ----------
    stiffness : float
        Linear stiffness outside deadband [N·m/rad]
    backlash : float
        Half-width of deadband [rad]
    """
    stiffness: float = 1e5  # N·m/rad
    backlash: float = 0.0   # rad (deadband half-width)
    
    def torque(self, theta: np.ndarray) -> np.ndarray:
        """Compute torque for given angular deflection."""
        theta = np.asarray(theta)
        delta = self.backlash
        k = self.stiffness
        
        result = np.zeros_like(theta, dtype=float)
        mask_pos = theta > delta
        mask_neg = theta < -delta
        
        result[mask_pos] = k * (theta[mask_pos] - delta)
        result[mask_neg] = k * (theta[mask_neg] + delta)
        
        return result
    
    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """Compute tangent stiffness (dT/dtheta) for Newton-Raphson."""
        theta = np.asarray(theta)
        delta = self.backlash
        
        result = np.zeros_like(theta, dtype=float)
        mask_active = np.abs(theta) > delta
        result[mask_active] = self.stiffness
        
        return result
    
    def is_linear(self) -> bool:
        """Check if model degenerates to linear (zero backlash)."""
        return self.backlash <= 1e-15


# =============================================================
# 2. MULTI-SEGMENT VARIABLE CROSS-SECTION SHAFT
# =============================================================

@dataclass
class ShaftSegment:
    """A single shaft segment with uniform properties."""
    name: str
    length: float        # [m]
    outer_diameter: float  # [m]
    inner_diameter: float = 0.0  # [m] (0 for solid)
    G: float = 79.3e9   # Shear modulus [Pa]
    rho: float = 7850.0  # Density [kg/m³]
    
    @property
    def J_polar(self) -> float:
        """Polar moment of area [m⁴]."""
        D = self.outer_diameter
        d = self.inner_diameter
        return np.pi * (D**4 - d**4) / 32.0
    
    @property
    def torsional_stiffness(self) -> float:
        """k = G*J/L [N·m/rad]."""
        return self.G * self.J_polar / self.length
    
    @property
    def mass_inertia(self) -> float:
        """I = rho*J*L [kg·m²]."""
        return self.rho * self.J_polar * self.length
    
    @property
    def wave_speed(self) -> float:
        """Torsional wave speed c = sqrt(G/rho) [m/s]."""
        return np.sqrt(self.G / self.rho)


class MultiSegmentShaft:
    """Multi-segment shaft assembled via transfer matrix method.
    
    Each segment is discretized into elements. Global M, K matrices
    are assembled from segment contributions.
    """
    
    def __init__(self, segments: List[ShaftSegment], 
                 elements_per_segment: int = 10):
        self.segments = segments
        self.n_elem_per_seg = elements_per_segment
        self._build_model()
    
    def _build_model(self):
        """Build global M, K matrices from segments."""
        nodes = []  # (position, J_inertia, name)
        springs = []  # (k, left_node, right_node, name)
        
        current_pos = 0.0
        node_idx = 0
        
        for seg_idx, seg in enumerate(self.segments):
            h = seg.length / self.n_elem_per_seg
            k_elem = seg.G * seg.J_polar / h
            I_elem = seg.rho * seg.J_polar * h
            
            # First node of segment (if not first segment, share with previous)
            if seg_idx == 0:
                nodes.append((current_pos, I_elem / 2.0, f"{seg.name}_start"))
                node_idx = 1
            
            # Interior nodes
            for i in range(1, self.n_elem_per_seg):
                pos = current_pos + i * h
                nodes.append((pos, I_elem, f"{seg.name}_{i}"))
                node_idx += 1
            
            # End node of segment
            pos = current_pos + seg.length
            if seg_idx < len(self.segments) - 1:
                # Shared with next segment
                nodes.append((pos, I_elem / 2.0, f"{seg.name}_end"))
            else:
                # Last segment end
                nodes.append((pos, I_elem / 2.0, f"{seg.name}_end"))
            node_idx += 1
            
            # Springs for this segment
            base_node = sum(self.n_elem_per_seg for s in self.segments[:seg_idx])
            for i in range(self.n_elem_per_seg):
                springs.append((k_elem, base_node + i, base_node + i + 1,
                               f"{seg.name}_spring_{i}"))
            
            current_pos += seg.length
        
        self.nodes = nodes
        self.springs = springs
        self.n_nodes = len(nodes)
        self.total_length = current_pos
    
    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global M (diagonal) and K matrices."""
        n = self.n_nodes
        M = np.zeros(n)
        K = np.zeros((n, n))
        
        for i, (pos, J, name) in enumerate(self.nodes):
            M[i] = J
        
        for k, left, right, name in self.springs:
            K[left, left] += k
            K[right, right] += k
            K[left, right] -= k
            K[right, left] -= k
        
        return M, K
    
    def solve_eigenvalue(self, fixed_node: int = 0, 
                         n_modes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Solve eigenvalue problem with one end fixed.
        
        Returns (frequencies_Hz, mode_shapes).
        """
        M, K = self.build_matrices()
        
        free = [i for i in range(self.n_nodes) if i != fixed_node]
        M_f = np.diag(M)[free][:, free]
        K_f = K[np.ix_(free, free)]
        
        # Solve via symmetric transformation
        M_inv_half = np.diag(1.0 / np.sqrt(np.diag(M_f)))
        A = M_inv_half @ K_f @ M_inv_half
        eig_vals, eig_vecs = np.linalg.eigh(A)
        
        phi = M_inv_half @ eig_vecs
        omega_sq = np.maximum(eig_vals, 0.0)
        freqs = np.sqrt(omega_sq) / (2.0 * np.pi)
        
        if n_modes is not None:
            freqs = freqs[:n_modes]
            phi = phi[:, :n_modes]
        
        return freqs, phi


# =============================================================
# 3. RANDOM ICE LOAD SPECTRUM (Shinozuka-Jan Method)
# =============================================================

@dataclass
class IceLoadPSD:
    """Ice load power spectral density model.
    
    Components:
    - BPF harmonics: discrete peaks at blade passing frequency and harmonics
    - Broadband: continuous spectrum representing ice-structure interaction
    - Low-freq impact: occasional large-amplitude low-frequency events
    """
    BPF: float = 10.0          # Blade passing frequency [Hz]
    BPF_amplitude: float = 1.0  # Relative amplitude of BPF component
    n_harmonics: int = 3        # Number of BPF harmonics
    broadband_level: float = 0.1  # Broadband PSD level relative to BPF
    f_low: float = 0.1         # Lower frequency bound [Hz]
    f_high: float = 50.0       # Upper frequency bound [Hz]
    impact_freq: float = 1.0   # Low-frequency impact center [Hz]
    impact_level: float = 0.5  # Impact component level
    
    def target_psd(self, freqs: np.ndarray) -> np.ndarray:
        """Compute target PSD at given frequencies."""
        psd = np.zeros_like(freqs)
        
        # Broadband component (flat spectrum)
        mask_bb = (freqs >= self.f_low) & (freqs <= self.f_high)
        psd[mask_bb] += self.broadband_level
        
        # BPF harmonics (narrow peaks modeled as Gaussians)
        for h in range(1, self.n_harmonics + 1):
            f_h = h * self.BPF
            sigma = 0.5  # Hz width
            psd += self.BPF_amplitude * np.exp(-0.5 * ((freqs - f_h) / sigma)**2)
        
        # Low-frequency impact component
        sigma_impact = 0.5
        psd += self.impact_level * np.exp(-0.5 * ((freqs - self.impact_freq) / sigma_impact)**2)
        
        return psd


class ShinozukaJanSimulator:
    """Generate random time series from target PSD using Shinozuka-Jan method.
    
    The method superposes N cosine waves with:
    - Evenly spaced frequencies covering the target band
    - Random phases uniformly distributed in [0, 2*pi]
    - Amplitudes derived from target PSD
    
    Reference: Shinozuka & Jan (1972), J. Sound Vib.
    """
    
    def __init__(self, psd_model: IceLoadPSD, n_components: int = 256,
                 seed: Optional[int] = None):
        self.psd_model = psd_model
        self.n_comp = n_components
        self.rng = np.random.RandomState(seed)
        
        # Frequency discretization
        self.freqs = np.linspace(psd_model.f_low, psd_model.f_high, n_components)
        self.df = self.freqs[1] - self.freqs[0]
        
        # Target PSD values
        self.target_psd_vals = psd_model.target_psd(self.freqs)
        
        # Amplitudes: A_i = sqrt(2 * S(f_i) * df)
        self.amplitudes = np.sqrt(2.0 * self.target_psd_vals * self.df)
    
    def generate(self, duration: float, dt: float, 
                 mean_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random time series.
        
        Parameters
        ----------
        duration : float
            Total duration [s]
        dt : float
            Time step [s]
        mean_value : float
            Mean value to add to signal
            
        Returns
        -------
        t : ndarray
            Time array
        signal : ndarray
            Random time series
        """
        t = np.arange(0, duration + dt/2, dt)
        n_t = len(t)
        
        # Random phases
        phases = self.rng.uniform(0, 2 * np.pi, self.n_comp)
        
        # Superpose cosine waves (vectorized)
        omega = 2 * np.pi * self.freqs
        # Vectorized: signal[j] = sum_i amp[i] * cos(omega[i]*t[j] + phase[i])
        cos_matrix = np.cos(np.outer(t, omega) + phases)  # (n_t, n_comp)
        signal = cos_matrix @ self.amplitudes  # (n_t,)
        
        signal += mean_value
        
        return t, signal
    
    def compute_psd(self, signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute PSD of generated signal using Welch's method."""
        n = len(signal)
        # Use FFT-based periodogram
        freqs_fft = np.fft.rfftfreq(n, d=dt)
        fft_vals = np.fft.rfft(signal)
        psd_fft = np.abs(fft_vals)**2 * dt / n  # Periodogram: (dt/N)*|FFT|^2
        
        # Only return positive frequencies in range
        mask = (freqs_fft >= self.psd_model.f_low) & (freqs_fft <= self.psd_model.f_high)
        return freqs_fft[mask], psd_fft[mask]


# =============================================================
# 4. NONLINEAR NEWMARK-BETA + NEWTON-RAPHSON SOLVER
# =============================================================

class NonlinearBacklashSpring:
    """Nonlinear spring element with backlash for assembly into system."""
    
    def __init__(self, backlash_model: BacklashModel, node_left: int, node_right: int):
        self.model = backlash_model
        self.node_left = node_left
        self.node_right = node_right
    
    def compute_torque_and_jacobian(self, theta: np.ndarray) -> Tuple[float, float, float]:
        """Compute element torque and tangent stiffness."""
        rel_theta = theta[self.node_right] - theta[self.node_left]
        T = self.model.torque(np.array([rel_theta]))[0]
        k_t = self.model.jacobian(np.array([rel_theta]))[0]
        return T, k_t, rel_theta


class NonlinearNewmarkSolver:
    """Newmark-beta solver with Newton-Raphson iteration for nonlinear systems.
    
    Solves: M*x'' + C*x' + F_int(x) = F_ext(t)
    where F_int(x) includes nonlinear backlash elements.
    
    Uses average acceleration method (gamma=0.5, beta=0.25).
    """
    
    def __init__(self, M: np.ndarray, C: np.ndarray, K_linear: np.ndarray,
                 backlash_springs: Optional[List[NonlinearBacklashSpring]] = None,
                 gamma: float = 0.5, beta: float = 0.25,
                 max_iter: int = 50, tol: float = 1e-8):
        self.M = M
        self.C = C
        self.K_linear = K_linear
        self.backlash_springs = backlash_springs or []
        self.gamma = gamma
        self.beta = beta
        self.max_iter = max_iter
        self.tol = tol
        self.n_dof = M.shape[0]
    
    def _compute_internal_force(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute internal force vector and tangent stiffness.
        
        F_int = K_linear * x + F_backlash(x)
        
        Returns
        -------
        F_int : ndarray
            Internal force vector
        K_t : ndarray
            Tangent stiffness matrix
        """
        F_int = self.K_linear @ x
        K_t = self.K_linear.copy()
        
        for spring in self.backlash_springs:
            T, k_t, rel_theta = spring.compute_torque_and_jacobian(x)
            i, j = spring.node_left, spring.node_right
            
            # Add backlash torque contribution to internal force
            # The spring torque acts on both nodes
            F_int[i] -= T  # Reaction on left node
            F_int[j] += T  # Reaction on right node
            
            # Remove linear stiffness contribution (already in K_linear)
            k_lin = spring.model.stiffness
            K_t[i, i] -= k_lin
            K_t[j, j] -= k_lin
            K_t[i, j] += k_lin
            K_t[j, i] += k_lin
            
            # Add nonlinear tangent stiffness
            K_t[i, i] += k_t
            K_t[j, j] += k_t
            K_t[i, j] -= k_t
            K_t[j, i] -= k_t
        
        return F_int, K_t
    
    def _compute_energy(self, x: np.ndarray, v: np.ndarray) -> Tuple[float, float, float]:
        """Compute kinetic, potential, and total energy."""
        KE = 0.5 * v @ self.M @ v
        
        # Potential energy from linear stiffness
        PE_lin = 0.5 * x @ self.K_linear @ x
        
        # Potential energy from backlash springs
        PE_nl = 0.0
        for spring in self.backlash_springs:
            rel = x[spring.node_right] - x[spring.node_left]
            delta = spring.model.backlash
            k = spring.model.stiffness
            if abs(rel) > delta:
                PE_nl += 0.5 * k * (abs(rel) - delta)**2
        
        PE = PE_lin + PE_nl
        TE = KE + PE
        
        return KE, PE, TE
    
    def solve(self, t: np.ndarray, F_ext: np.ndarray,
              x0: Optional[np.ndarray] = None,
              v0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Time integration with Newton-Raphson iteration.
        
        Equation: M*a + C*v + F_int(x) = F_ext
        """
        n_steps = len(t) - 1
        dt = t[1] - t[0]
        n = self.n_dof
        
        x = np.zeros((n_steps + 1, n))
        v = np.zeros((n_steps + 1, n))
        a = np.zeros((n_steps + 1, n))
        energy = np.zeros(n_steps + 1)
        iterations = np.zeros(n_steps + 1, dtype=int)
        
        if x0 is not None:
            x[0] = x0
        if v0 is not None:
            v[0] = v0
        
        # Initial acceleration: M*a = F_ext - C*v - F_int(x)
        F_int0, _ = self._compute_internal_force(x[0])
        a[0] = np.linalg.solve(self.M, F_ext[0] - self.C @ v[0] - F_int0)
        
        # Newmark constants
        g = self.gamma
        b = self.beta
        
        # Newmark update coefficients
        a1 = 1.0 / (b * dt**2)
        a2 = 1.0 / (b * dt)
        a3 = 1.0 / (2.0 * b) - 1.0
        a4 = g / (b * dt)
        a5 = 1.0 - g / b
        a6 = dt * (1.0 - g / (2.0 * b))
        
        for step in range(n_steps):
            # Predictor
            x_pred = x[step] + dt * v[step] + dt**2 * (0.5 - b) * a[step]
            v_pred = v[step] + dt * (1.0 - g) * a[step]
            
            # Newton-Raphson iteration
            x_new = x_pred.copy()
            converged = False
            
            for itr in range(self.max_iter):
                # Compute acceleration and velocity from displacement
                # a_{n+1} = (x_{n+1} - x_pred) / (beta * dt^2)
                a_new = a1 * (x_new - x_pred)
                v_new = v_pred + g * dt * a_new
                
                # Compute internal force and tangent stiffness
                F_int, K_t = self._compute_internal_force(x_new)
                
                # Residual: R = F_ext - M*a - C*v - F_int
                R = F_ext[step + 1] - self.M @ a_new - self.C @ v_new - F_int
                
                # Check convergence
                res_norm = np.linalg.norm(R)
                if itr == 0:
                    res_norm0 = max(res_norm, 1e-15)
                
                if res_norm / res_norm0 < self.tol:
                    converged = True
                    iterations[step + 1] = itr + 1
                    break
                
                # Effective tangent stiffness for Newton update
                # dR/dx = -K_t - (gamma/(beta*dt))*C - (1/(beta*dt^2))*M
                K_eff = K_t + a4 * self.C + a1 * self.M
                
                # Solve for correction
                dx = np.linalg.solve(K_eff, R)
                x_new += dx
            
            if not converged:
                iterations[step + 1] = self.max_iter
            
            x[step + 1] = x_new
            a[step + 1] = a1 * (x[step + 1] - x_pred)
            v[step + 1] = v_pred + g * dt * a[step + 1]
            
            # Energy
            _, _, energy[step + 1] = self._compute_energy(x[step + 1], v[step + 1])
        
        # Initial energy
        _, _, energy[0] = self._compute_energy(x[0], v[0])
        
        return {
            't': t,
            'x': x,
            'v': v,
            'a': a,
            'energy': energy,
            'iterations': iterations,
        }


# =============================================================
# 5. MONTE CARLO SIMULATOR
# =============================================================

class MonteCarloShaftSimulator:
    """Monte Carlo simulation for shaft torsional response under random ice loading."""
    
    def __init__(self, shaft: MultiSegmentShaft, 
                 ice_psd: IceLoadPSD,
                 backlash: Optional[BacklashModel] = None,
                 damping_ratio: float = 0.02,
                 fixed_node: int = 0,
                 load_node: Optional[int] = None):
        self.shaft = shaft
        self.ice_psd = ice_psd
        self.backlash = backlash
        self.damping_ratio = damping_ratio
        self.fixed_node = fixed_node
        self.load_node = load_node if load_node is not None else shaft.n_nodes - 1
    
    def _build_system(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        """Build M, C, K matrices and backlash springs."""
        M_full, K_full = self.shaft.build_matrices()
        
        # Remove fixed node
        free = [i for i in range(self.shaft.n_nodes) if i != self.fixed_node]
        M = np.diag(M_full)[free][:, free]
        K = K_full[np.ix_(free, free)]
        
        # Rayleigh damping
        M_inv_half = np.diag(1.0 / np.sqrt(np.diag(M)))
        A = M_inv_half @ K @ M_inv_half
        eig_vals, _ = np.linalg.eigh(A)
        omega1 = np.sqrt(max(eig_vals[0], 1e-10))
        omega2 = np.sqrt(max(eig_vals[1] if len(eig_vals) > 1 else eig_vals[0] * 4, 1e-10))
        
        zeta = self.damping_ratio
        a0 = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
        a1 = 2 * zeta / (omega1 + omega2)
        C = a0 * M + a1 * K
        
        # Build backlash springs
        backlash_springs = []
        if self.backlash is not None and not self.backlash.is_linear():
            load_local = free.index(self.load_node) if self.load_node in free else len(free) - 1
            if load_local > 0:
                spring = NonlinearBacklashSpring(self.backlash, load_local - 1, load_local)
                backlash_springs.append(spring)
        
        return M, C, K, backlash_springs
    
    def run_single(self, duration: float, dt: float, 
                   mean_torque: float = 50e3, seed: int = 0) -> Dict[str, Any]:
        """Run single realization."""
        # Generate random ice load
        sim = ShinozukaJanSimulator(self.ice_psd, n_components=64, seed=seed)
        t, ice_signal = sim.generate(duration, dt, mean_value=mean_torque)
        
        # Build system
        M, C, K, backlash_springs = self._build_system()
        n_dof = M.shape[0]
        
        # Force vector
        n_steps = len(t)
        F = np.zeros((n_steps, n_dof))
        free = [i for i in range(self.shaft.n_nodes) if i != self.fixed_node]
        load_local = free.index(self.load_node) if self.load_node in free else n_dof - 1
        F[:, load_local] = ice_signal
        
        # Solve
        solver = NonlinearNewmarkSolver(M, C, K, backlash_springs)
        result = solver.solve(t, F)
        
        # Extract response statistics
        x = result['x']
        max_response = np.max(np.abs(x[:, load_local]))
        rms_response = np.sqrt(np.mean(x[:, load_local]**2))
        
        return {
            't': t,
            'x': x,
            'max_response': max_response,
            'rms_response': rms_response,
            'energy': result['energy'],
            'load_signal': ice_signal,
        }
    
    def run_monte_carlo(self, n_realizations: int, duration: float, dt: float,
                        mean_torque: float = 50e3) -> Dict[str, Any]:
        """Run Monte Carlo simulation with N realizations."""
        max_responses = []
        rms_responses = []
        
        for i in range(n_realizations):
            result = self.run_single(duration, dt, mean_torque, seed=i)
            max_responses.append(result['max_response'])
            rms_responses.append(result['rms_response'])
        
        max_responses = np.array(max_responses)
        rms_responses = np.array(rms_responses)
        
        # Statistics
        mean_max = np.mean(max_responses)
        std_max = np.std(max_responses)
        std_err_max = std_max / np.sqrt(n_realizations)
        
        mean_rms = np.mean(rms_responses)
        std_rms = np.std(rms_responses)
        std_err_rms = std_rms / np.sqrt(n_realizations)
        
        return {
            'n_realizations': n_realizations,
            'max_responses': max_responses,
            'rms_responses': rms_responses,
            'mean_max': mean_max,
            'std_max': std_max,
            'std_err_max': std_err_max,
            'cv_max': std_max / mean_max if mean_max > 0 else 0,
            'mean_rms': mean_rms,
            'std_rms': std_rms,
            'std_err_rms': std_err_rms,
            'convergence_std_err_ratio': std_err_max / mean_max if mean_max > 0 else 0,
        }


# =============================================================
# 6. FATIGUE DAMAGE CALCULATOR (Miner's Rule, DNVGL S-N Curve)
# =============================================================

@dataclass
class DNVGLSnCurve:
    """DNVGL S-N curve parameters for fatigue analysis.
    
    S-N curve: log(N) = log(a) - m * log(S)
    """
    m: float = 3.0
    log_a: float = 11.764
    fatigue_limit: float = 0.0
    
    def cycles_to_failure(self, stress_range_mpa: float) -> float:
        """Compute cycles to failure for given stress range."""
        if stress_range_mpa <= self.fatigue_limit:
            return np.inf
        log_N = self.log_a - self.m * np.log10(stress_range_mpa)
        return 10.0 ** log_N
    
    def stress_from_cycles(self, N: float) -> float:
        """Compute stress range for given cycles."""
        log_S = (self.log_a - np.log10(N)) / self.m
        return 10.0 ** log_S


class FatigueDamageCalculator:
    """Miner's linear cumulative damage rule with rainflow counting."""
    
    def __init__(self, sn_curve: Optional[DNVGLSnCurve] = None):
        self.sn_curve = sn_curve or DNVGLSnCurve()
    
    def rainflow_count(self, signal: np.ndarray) -> List[Tuple[float, float]]:
        """Simplified rainflow counting algorithm."""
        # Extract turning points
        turning_points = []
        for i in range(1, len(signal) - 1):
            if (signal[i] - signal[i-1]) * (signal[i+1] - signal[i]) < 0:
                turning_points.append(signal[i])
        
        if len(turning_points) < 3:
            return []
        
        # 4-point rainflow method
        cycles = []
        points = list(turning_points)
        
        while len(points) >= 4:
            found = False
            for i in range(len(points) - 3):
                r1 = abs(points[i+1] - points[i])
                r2 = abs(points[i+2] - points[i+1])
                r3 = abs(points[i+3] - points[i+2])
                
                if r2 <= r1 and r2 <= r3:
                    range_val = r2
                    mean_val = (points[i+1] + points[i+2]) / 2
                    cycles.append((range_val, mean_val))
                    points.pop(i+1)
                    points.pop(i+1)
                    found = True
                    break
            
            if not found:
                break
        
        return cycles
    
    def compute_damage(self, signal: np.ndarray, dt: float) -> Dict[str, Any]:
        """Compute fatigue damage from time series."""
        cycles = self.rainflow_count(signal)
        
        total_damage = 0.0
        cycle_details = []
        
        for range_val, mean_val in cycles:
            stress_amp = range_val / 2.0
            stress_range_mpa = stress_amp / 1e6
            
            if stress_range_mpa > 0:
                N_f = self.sn_curve.cycles_to_failure(stress_range_mpa)
                if N_f < np.inf:
                    damage = 1.0 / N_f
                    total_damage += damage
                    cycle_details.append({
                        'range': range_val,
                        'mean': mean_val,
                        'stress_range_mpa': stress_range_mpa,
                        'N_f': N_f,
                        'damage': damage,
                    })
        
        return {
            'total_damage': total_damage,
            'n_cycles': len(cycles),
            'cycle_details': cycle_details,
        }
    
    def constant_amplitude_damage(self, stress_range_pa: float, 
                                  n_cycles: int, duration: float) -> float:
        """Compute damage for constant amplitude loading."""
        stress_range_mpa = stress_range_pa / 1e6
        N_f = self.sn_curve.cycles_to_failure(stress_range_mpa)
        if N_f == np.inf or N_f <= 0:
            return 0.0
        return n_cycles / N_f


# =============================================================
# 7. INTEGRATED NONLINEAR SHAFT ANALYZER
# =============================================================

class NonlinearShaftAnalyzer:
    """Top-level analyzer for nonlinear shaft torsional vibration."""
    
    def __init__(self, segments: List[ShaftSegment],
                 backlash: Optional[BacklashModel] = None,
                 ice_psd: Optional[IceLoadPSD] = None,
                 sn_curve: Optional[DNVGLSnCurve] = None,
                 elements_per_segment: int = 10,
                 damping_ratio: float = 0.02):
        self.shaft = MultiSegmentShaft(segments, elements_per_segment)
        self.backlash = backlash or BacklashModel()
        self.ice_psd = ice_psd or IceLoadPSD()
        self.sn_curve = sn_curve or DNVGLSnCurve()
        self.damping_ratio = damping_ratio
        
        self.simulator = MonteCarloShaftSimulator(
            self.shaft, self.ice_psd, self.backlash,
            self.damping_ratio
        )
        self.fatigue_calc = FatigueDamageCalculator(self.sn_curve)
    
    def modal_analysis(self, n_modes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute natural frequencies."""
        return self.shaft.solve_eigenvalue(n_modes=n_modes)
    
    def transient_analysis(self, duration: float = 2.0, dt: float = 1e-4,
                           mean_torque: float = 50e3, seed: int = 0) -> Dict[str, Any]:
        """Run single transient analysis."""
        return self.simulator.run_single(duration, dt, mean_torque, seed)
    
    def monte_carlo_analysis(self, n_realizations: int = 100,
                             duration: float = 2.0, dt: float = 1e-4,
                             mean_torque: float = 50e3) -> Dict[str, Any]:
        """Run Monte Carlo simulation."""
        return self.simulator.run_monte_carlo(n_realizations, duration, dt, mean_torque)
    
    def fatigue_analysis(self, signal: np.ndarray, dt: float) -> Dict[str, Any]:
        """Compute fatigue damage from stress/torque time series."""
        return self.fatigue_calc.compute_damage(signal, dt)
    
    def run_full_analysis(self, n_mc: int = 100, duration: float = 2.0,
                          dt: float = 1e-4, mean_torque: float = 50e3) -> Dict[str, Any]:
        """Run complete nonlinear shaft analysis."""
        result = {}
        
        # 1. Modal analysis
        freqs, modes = self.modal_analysis()
        result['natural_frequencies'] = freqs.tolist()
        result['mode_shapes'] = modes.tolist()
        
        # 2. Single transient
        transient = self.transient_analysis(duration, dt, mean_torque)
        result['transient'] = {
            'max_response': transient['max_response'],
            'rms_response': transient['rms_response'],
        }
        
        # 3. Monte Carlo
        mc = self.monte_carlo_analysis(n_mc, duration, dt, mean_torque)
        result['monte_carlo'] = {
            'n_realizations': mc['n_realizations'],
            'mean_max': mc['mean_max'],
            'std_max': mc['std_max'],
            'std_err_max': mc['std_err_max'],
            'convergence_ratio': mc['convergence_std_err_ratio'],
        }
        
        # 4. Fatigue
        fatigue = self.fatigue_analysis(transient['x'][:, -1], dt)
        result['fatigue'] = {
            'total_damage': fatigue['total_damage'],
            'n_cycles': fatigue['n_cycles'],
        }
        
        return result
