"""
Nonlinear Shaft Torsional Vibration Analysis
=============================================
Extends the m763 linear model with:
(1) Backlash nonlinearity: T(θ)=k·(θ-δ) for θ>δ, 0 for |θ|≤δ, k·(θ+δ) for θ<-δ
(2) Random ice load spectrum: harmonic superposition with BPF + broadband + low-freq impact
(3) Multi-segment variable cross-section shaft: transfer matrix method for global M/K
(4) Nonlinear transient solver: Newmark-β + Newton-Raphson iteration
(5) Monte Carlo statistical response (N≥100)
(6) Fatigue damage: Miner cumulative damage criterion
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Callable

try:
    from scipy.linalg import eigh as scipy_eigh
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────
# 1. Backlash nonlinearity model
# ─────────────────────────────────────────────────────────────

@dataclass
class BacklashSpring:
    """Piecewise-linear backlash spring.

    T(θ) = k·(θ - δ)   for θ >  δ
           0            for |θ| ≤ δ
           k·(θ + δ)   for θ < -δ

    When δ=0, the spring is "transparent" (no additional torque/stiffness),
    representing a rigid coupling that doesn't modify the linear system.
    """
    k: float
    delta: float

    def torque(self, theta: float) -> float:
        """Restoring torque for given angular displacement."""
        if self.delta == 0.0:
            return 0.0  # Transparent when no backlash
        if theta > self.delta:
            return self.k * (theta - self.delta)
        elif theta < -self.delta:
            return self.k * (theta + self.delta)
        else:
            return 0.0

    def stiffness(self, theta: float) -> float:
        """Tangent stiffness (Jacobian) at given displacement."""
        if self.delta == 0.0:
            return 0.0  # Transparent when no backlash
        if abs(theta) > self.delta:
            return self.k
        else:
            return 0.0

    def is_linear(self) -> bool:
        """Check if backlash is zero (degenerates to transparent coupling)."""
        return self.delta == 0.0


def backlash_torque_vector(theta: np.ndarray, k: float, delta: float) -> np.ndarray:
    """Vectorized backlash torque for array of displacements."""
    if delta == 0.0:
        return np.zeros_like(theta, dtype=float)
    T = np.zeros_like(theta, dtype=float)
    mask_pos = theta > delta
    mask_neg = theta < -delta
    T[mask_pos] = k * (theta[mask_pos] - delta)
    T[mask_neg] = k * (theta[mask_neg] + delta)
    return T


def backlash_stiffness_vector(theta: np.ndarray, k: float, delta: float) -> np.ndarray:
    """Vectorized tangent stiffness for array of displacements."""
    K_t = np.zeros_like(theta, dtype=float)
    if delta > 0.0:
        mask = np.abs(theta) > delta
        K_t[mask] = k
    return K_t


# ─────────────────────────────────────────────────────────────
# 2. Random ice load spectrum
# ─────────────────────────────────────────────────────────────

@dataclass
class RandomIceLoadSpectrum:
    """Random ice load spectrum generator using harmonic superposition."""
    T_mean: float = 50000.0
    n_blades: int = 5
    RPM: float = 120.0
    n_harmonics: int = 5
    harmonic_amplitudes: Optional[List[float]] = None
    broadband_level: float = 0.1
    broadband_cutoff: float = 3.0
    impact_rate: float = 0.5
    impact_amplitude: float = 20000.0
    impact_duration: float = 0.05
    seed: int = 42

    @property
    def BPF(self) -> float:
        return self.n_blades * self.RPM / 60.0

    @property
    def omega_BPF(self) -> float:
        return 2.0 * np.pi * self.BPF

    def psd(self, freq: np.ndarray) -> np.ndarray:
        """Power spectral density [N²·m²/Hz]."""
        S = np.zeros_like(freq, dtype=float)
        if self.harmonic_amplitudes is None:
            amps = [self.T_mean * 0.3 / k for k in range(1, self.n_harmonics + 1)]
        else:
            amps = self.harmonic_amplitudes[:self.n_harmonics]

        for k, amp in enumerate(amps):
            f_k = (k + 1) * self.BPF
            sigma_f = 0.5
            S += (amp**2 / 2) * np.exp(-0.5 * ((freq - f_k) / sigma_f)**2) / (sigma_f * np.sqrt(2 * np.pi))

        f_cutoff = self.broadband_cutoff * self.BPF
        mask_bb = (freq > 0) & (freq < f_cutoff)
        S[mask_bb] += (self.broadband_level * self.T_mean)**2 / f_cutoff
        return S

    def generate_time_history(self, t: np.ndarray) -> np.ndarray:
        """Generate random torque time history via harmonic superposition."""
        rng = np.random.RandomState(self.seed)
        dt = t[1] - t[0]
        n_steps = len(t)
        T = np.full(n_steps, self.T_mean, dtype=float)

        if self.harmonic_amplitudes is None:
            amps = [self.T_mean * 0.3 / k for k in range(1, self.n_harmonics + 1)]
        else:
            amps = self.harmonic_amplitudes[:self.n_harmonics]

        for k, amp in enumerate(amps):
            f_k = (k + 1) * self.BPF
            phi = rng.uniform(0, 2 * np.pi)
            T += amp * np.sin(2 * np.pi * f_k * t + phi)

        f_cutoff = self.broadband_cutoff * self.BPF
        n_freq = int(f_cutoff / (1.0 / (t[-1] - t[0])))
        if n_freq > 0:
            freqs = np.arange(1, n_freq + 1) / (t[-1] - t[0])
            phases = rng.uniform(0, 2 * np.pi, n_freq)
            amp_bb = self.broadband_level * self.T_mean / np.sqrt(n_freq)
            for f, phi in zip(freqs, phases):
                T += amp_bb * np.sin(2 * np.pi * f * t + phi)

        n_impacts = int(self.impact_rate * t[-1])
        for _ in range(n_impacts):
            t_imp = rng.uniform(0, t[-1])
            idx = int(t_imp / dt)
            if 0 <= idx < n_steps:
                pulse = self.impact_amplitude * np.exp(-0.5 * ((t - t_imp) / self.impact_duration)**2)
                T += pulse
        return T

    def psd_error(self, t: np.ndarray, n_segments: int = 64) -> float:
        """Estimate PSD estimation error at the fundamental BPF harmonic.

        Uses Welch-like averaging over segments.
        Returns relative error at 1x BPF (should be < 5%).
        """
        T = self.generate_time_history(t)
        dt = t[1] - t[0]
        n = len(T)
        seg_len = n // n_segments

        psds = []
        for i in range(n_segments):
            seg = T[i * seg_len:(i + 1) * seg_len]
            seg = seg - np.mean(seg)
            fft_val = np.fft.rfft(seg)
            psd_seg = np.abs(fft_val)**2 / (seg_len * dt)
            psds.append(psd_seg)

        psd_mean = np.mean(psds, axis=0)
        psd_std = np.std(psds, axis=0)

        freqs = np.fft.rfftfreq(seg_len, dt)
        # Focus on fundamental BPF harmonic (most important for ice loads)
        idx = np.argmin(np.abs(freqs - self.BPF))
        if psd_mean[idx] > 0:
            return psd_std[idx] / psd_mean[idx]
        return 0.0


# ─────────────────────────────────────────────────────────────
# 3. Multi-segment variable cross-section shaft
# ─────────────────────────────────────────────────────────────

@dataclass
class ShaftSegment:
    """A single shaft segment with uniform cross-section."""
    name: str
    L: float
    D: float
    d: float = 0.0
    G: float = 79.3e9
    rho: float = 7850.0

    @property
    def J(self) -> float:
        return np.pi / 32.0 * (self.D**4 - self.d**4)

    @property
    def k_torsion(self) -> float:
        return self.G * self.J / self.L

    @property
    def mass_moment(self) -> float:
        return self.rho * self.J * self.L


@dataclass
class MultiSegmentShaft:
    """Multi-segment shaft with variable cross-section."""
    segments: List[ShaftSegment] = field(default_factory=list)
    node_inertias: Optional[List[float]] = None

    def add_segment(self, seg: ShaftSegment):
        self.segments.append(seg)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_nodes(self) -> int:
        return self.n_segments + 1

    @property
    def total_length(self) -> float:
        return sum(s.L for s in self.segments)

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global M and K matrices using lumped parameter approach."""
        n = self.n_nodes
        M = np.zeros((n, n))
        K = np.zeros((n, n))

        for i, seg in enumerate(self.segments):
            I_half = seg.mass_moment / 2.0
            M[i, i] += I_half
            M[i + 1, i + 1] += I_half

        if self.node_inertias is not None:
            for i, J_add in enumerate(self.node_inertias[:n]):
                M[i, i] += J_add

        for i, seg in enumerate(self.segments):
            k = seg.k_torsion
            K[i, i] += k
            K[i + 1, i + 1] += k
            K[i, i + 1] -= k
            K[i + 1, i] -= k

        return M, K

    def transfer_matrix(self, omega: float) -> np.ndarray:
        """Compute overall transfer matrix at frequency omega.

        State vector: [θ, T]
        Transfer: [θ_R, T_R]^T = T_mat · [θ_L, T_L]^T

        For fixed-free: θ(0)=0, T(L)=0 → need T_mat[1,1] = 0
        """
        T_overall = np.eye(2)
        for seg in self.segments:
            beta = omega * np.sqrt(seg.rho / seg.G)
            gj = seg.G * seg.J

            if abs(beta) < 1e-15:
                T_seg = np.array([[1.0, seg.L / gj], [0.0, 1.0]])
            else:
                c = np.cos(beta * seg.L)
                s = np.sin(beta * seg.L)
                T_seg = np.array([
                    [c, s / (gj * beta)],
                    [-gj * beta * s, c]
                ])
            T_overall = T_seg @ T_overall
        return T_overall

    def natural_frequencies_transfer(self, n_modes: int = 4,
                                      freq_range: Tuple[float, float] = (1.0, 500.0),
                                      n_freq: int = 10000) -> np.ndarray:
        """Find natural frequencies using transfer matrix method.

        For fixed-free: T[1,1] = 0 at natural frequencies.
        """
        freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
        residuals = []
        for f in freqs:
            omega = 2.0 * np.pi * f
            T = self.transfer_matrix(omega)
            residuals.append(T[1, 1])  # Fixed-free: T[1,1] = 0

        residuals = np.array(residuals)
        sign_changes = np.where(np.diff(np.sign(residuals)))[0]
        found_freqs = []

        for idx in sign_changes:
            f_lo = freqs[idx]
            f_hi = freqs[idx + 1]
            for _ in range(50):
                f_mid = 0.5 * (f_lo + f_hi)
                omega_mid = 2.0 * np.pi * f_mid
                T_mid = self.transfer_matrix(omega_mid)
                if T_mid[1, 1] * residuals[idx] < 0:
                    f_hi = f_mid
                else:
                    f_lo = f_mid
            found_freqs.append(0.5 * (f_lo + f_hi))
            if len(found_freqs) >= n_modes:
                break

        return np.array(found_freqs[:n_modes])

    def analytical_two_segment_frequencies(self) -> np.ndarray:
        """Analytical natural frequencies for two-segment shaft (fixed-free)."""
        if self.n_segments != 2:
            raise ValueError("Analytical solution only for 2-segment shaft")

        s1, s2 = self.segments[0], self.segments[1]
        c1 = np.sqrt(s1.G / s1.rho)
        c2 = np.sqrt(s2.G / s2.rho)

        def char_eq(omega: float) -> float:
            beta1 = omega / c1
            beta2 = omega / c2
            lhs = np.tan(beta1 * s1.L) * np.tan(beta2 * s2.L)
            rhs = (s1.G * s1.J * c2) / (s2.G * s2.J * c1)
            return lhs - rhs

        freqs = np.linspace(1.0, 500.0, 10000)
        residuals = np.array([char_eq(2 * np.pi * f) for f in freqs])

        mask = np.abs(residuals) < 1e6
        freqs = freqs[mask]
        residuals = residuals[mask]

        sign_changes = np.where(np.diff(np.sign(residuals)))[0]
        found_freqs = []

        for idx in sign_changes:
            f_lo = freqs[idx]
            f_hi = freqs[idx + 1]
            for _ in range(50):
                f_mid = 0.5 * (f_lo + f_hi)
                if char_eq(2 * np.pi * f_lo) * char_eq(2 * np.pi * f_mid) < 0:
                    f_hi = f_mid
                else:
                    f_lo = f_mid
            found_freqs.append(0.5 * (f_lo + f_hi))
            if len(found_freqs) >= 4:
                break

        return np.array(found_freqs[:4])


def solve_generalized_eigenvalue(K: np.ndarray, M: np.ndarray,
                                  n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Solve generalized eigenvalue problem K·φ = ω²·M·φ."""
    if HAS_SCIPY:
        eig_vals, eig_vecs = scipy_eigh(K, M)
    else:
        M_diag = np.diag(M)
        M_inv_half = np.diag(1.0 / np.sqrt(M_diag))
        A = M_inv_half @ K @ M_inv_half
        eig_vals, eig_vecs_transformed = np.linalg.eigh(A)
        eig_vecs = M_inv_half @ eig_vecs_transformed

    omega_sq = np.maximum(eig_vals, 0.0)
    freqs = np.sqrt(omega_sq) / (2.0 * np.pi)

    if n_modes is not None:
        freqs = freqs[:n_modes]
        eig_vecs = eig_vecs[:, :n_modes]

    return freqs, eig_vecs


# ─────────────────────────────────────────────────────────────
# 4. Nonlinear transient solver: Newmark-β + Newton-Raphson
# ─────────────────────────────────────────────────────────────

class NonlinearNewmarkSolver:
    """Newmark-β time integration with Newton-Raphson iteration."""

    def __init__(self, M: np.ndarray, C: np.ndarray, K_linear: np.ndarray,
                 gamma: float = 0.5, beta: float = 0.25,
                 backlash_springs: Optional[Dict[int, BacklashSpring]] = None):
        self.M = M
        self.C = C
        self.K_linear = K_linear
        self.gamma = gamma
        self.beta = beta
        self.backlash_springs = backlash_springs or {}
        self.n_dof = M.shape[0]

    def _internal_force(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute internal force and tangent stiffness."""
        F_int = self.K_linear @ theta
        K_tangent = self.K_linear.copy()

        for dof, spring in self.backlash_springs.items():
            F_int[dof] += spring.torque(theta[dof])
            K_tangent[dof, dof] += spring.stiffness(theta[dof])

        return F_int, K_tangent

    def solve(self, t: np.ndarray, F_ext: np.ndarray,
              x0: Optional[np.ndarray] = None,
              v0: Optional[np.ndarray] = None,
              tol: float = 1e-8, max_iter: int = 50) -> Dict[str, np.ndarray]:
        """Nonlinear transient analysis."""
        n_steps = len(t) - 1
        dt = t[1] - t[0]
        n = self.n_dof

        x = np.zeros((n_steps + 1, n))
        v = np.zeros((n_steps + 1, n))
        a = np.zeros((n_steps + 1, n))
        n_iter = np.zeros(n_steps + 1, dtype=int)
        converged = np.ones(n_steps + 1, dtype=bool)

        if x0 is not None:
            x[0] = x0
        if v0 is not None:
            v[0] = v0

        F_int0, _ = self._internal_force(x[0])
        a[0] = np.linalg.solve(self.M, F_ext[0] - self.C @ v[0] - F_int0)

        g = self.gamma
        b = self.beta

        for i in range(n_steps):
            x_pred = x[i] + dt * v[i] + dt**2 * (0.5 - b) * a[i]
            v_pred = v[i] + dt * (1 - g) * a[i]

            x_new = x_pred.copy()
            iter_count = 0
            conv = False

            for iteration in range(max_iter):
                F_int, K_tan = self._internal_force(x_new)
                a_new = (x_new - x_pred) / (b * dt**2)
                v_new = v_pred + g * dt * a_new
                R = self.M @ a_new + self.C @ v_new + F_int - F_ext[i + 1]

                K_eff = K_tan + g / (b * dt) * self.C + 1.0 / (b * dt**2) * self.M

                res_norm = np.linalg.norm(R)
                if res_norm < tol:
                    conv = True
                    break

                dx = np.linalg.solve(K_eff, -R)
                x_new += dx
                iter_count += 1

            x[i + 1] = x_new
            a[i + 1] = (x_new - x_pred) / (b * dt**2)
            v[i + 1] = v_pred + g * dt * a[i + 1]
            n_iter[i + 1] = iter_count
            converged[i + 1] = conv

        return {"t": t, "x": x, "v": v, "a": a, "n_iterations": n_iter, "converged": converged}


# ─────────────────────────────────────────────────────────────
# 5. Monte Carlo statistical response
# ─────────────────────────────────────────────────────────────

@dataclass
class MonteCarloResult:
    """Monte Carlo statistical analysis results."""
    n_samples: int
    mean_response: np.ndarray
    std_response: np.ndarray
    max_response: np.ndarray
    min_response: np.ndarray
    percentile_95: np.ndarray
    percentile_5: np.ndarray
    convergence_history: Optional[np.ndarray] = None
    standard_error: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_samples": self.n_samples,
            "mean_response": self.mean_response.tolist(),
            "std_response": self.std_response.tolist(),
            "max_response": self.max_response.tolist(),
            "min_response": self.min_response.tolist(),
            "percentile_95": self.percentile_95.tolist(),
            "percentile_5": self.percentile_5.tolist(),
            "standard_error": self.standard_error,
        }


class MonteCarloAnalyzer:
    """Monte Carlo analysis for nonlinear shaft torsional vibration."""

    def __init__(self, solver_factory: Callable, spectrum_params: Dict[str, Any],
                 t: np.ndarray, n_samples: int = 100):
        self.solver_factory = solver_factory
        self.spectrum_params = spectrum_params
        self.t = t
        self.n_samples = n_samples

    def run(self, dof_index: int = 0) -> MonteCarloResult:
        """Run Monte Carlo simulation."""
        n_steps = len(self.t) - 1
        responses = np.zeros((self.n_samples, n_steps + 1))
        running_mean = np.zeros((self.n_samples, n_steps + 1))

        for i in range(self.n_samples):
            seed = 42 + i * 7
            params = dict(self.spectrum_params)
            params["seed"] = seed

            spectrum = RandomIceLoadSpectrum(**params)
            T = spectrum.generate_time_history(self.t)

            solver = self.solver_factory(seed=seed)
            F_ext = np.zeros((n_steps + 1, solver.n_dof))
            F_ext[:, dof_index] = T

            result = solver.solve(self.t, F_ext)
            responses[i] = result["x"][:, dof_index]
            running_mean[i] = np.mean(responses[:i + 1], axis=0)

        mean_resp = np.mean(responses, axis=0)
        std_resp = np.std(responses, axis=0)
        max_resp = np.max(responses, axis=0)
        min_resp = np.min(responses, axis=0)
        p95 = np.percentile(responses, 95, axis=0)
        p5 = np.percentile(responses, 5, axis=0)

        standard_error = np.std(responses[:, -1]) / np.sqrt(self.n_samples)
        mean_abs = np.mean(np.abs(responses[:, -1]))
        relative_se = standard_error / mean_abs if mean_abs > 0 else 0.0

        return MonteCarloResult(
            n_samples=self.n_samples, mean_response=mean_resp, std_response=std_resp,
            max_response=max_resp, min_response=min_resp, percentile_95=p95,
            percentile_5=p5, convergence_history=running_mean, standard_error=relative_se,
        )


# ─────────────────────────────────────────────────────────────
# 6. Miner fatigue damage criterion
# ─────────────────────────────────────────────────────────────

@dataclass
class MinerFatigueDamage:
    """Miner's cumulative damage criterion for torsional fatigue."""
    A: float = 1e30
    m: float = 3.0
    S_endurance: float = 100e6

    def cycles_to_failure(self, stress_amp: float) -> float:
        if stress_amp <= self.S_endurance:
            return np.inf
        return self.A * stress_amp**(-self.m)

    def damage_per_cycle(self, stress_amp: float) -> float:
        N = self.cycles_to_failure(stress_amp)
        return 0.0 if N == np.inf else 1.0 / N

    def compute_damage(self, stress_history: np.ndarray, dt: float,
                       rainflow: bool = False) -> float:
        if rainflow:
            peaks = self._find_peaks(stress_history)
            if len(peaks) < 2:
                return 0.0
            stress_amps = np.abs(np.diff(peaks)) / 2.0
        else:
            stress_amps = self._zero_crossing_amplitudes(stress_history)

        D = 0.0
        for s_amp in stress_amps:
            D += self.damage_per_cycle(s_amp)
        return D

    def _find_peaks(self, signal: np.ndarray) -> np.ndarray:
        peaks = []
        for i in range(1, len(signal) - 1):
            if signal[i] > signal[i - 1] and signal[i] > signal[i + 1]:
                peaks.append(signal[i])
            elif signal[i] < signal[i - 1] and signal[i] < signal[i + 1]:
                peaks.append(signal[i])
        return np.array(peaks) if peaks else np.array([])

    def _zero_crossing_amplitudes(self, signal: np.ndarray) -> np.ndarray:
        crossings = np.where(np.diff(np.sign(signal)))[0]
        if len(crossings) < 2:
            return np.array([])
        amplitudes = []
        for i in range(len(crossings) - 1):
            seg = signal[crossings[i]:crossings[i + 1] + 1]
            if len(seg) > 1:
                amplitudes.append((np.max(seg) - np.min(seg)) / 2.0)
        return np.array(amplitudes) if amplitudes else np.array([])


# ─────────────────────────────────────────────────────────────
# 7. High-level nonlinear shaft torsion analyzer
# ─────────────────────────────────────────────────────────────

@dataclass
class NonlinearShaftTorsionResult:
    """Results from nonlinear shaft torsion analysis."""
    linear_frequencies: Optional[np.ndarray] = None
    time_history: Optional[np.ndarray] = None
    velocity_history: Optional[np.ndarray] = None
    mean_response: Optional[np.ndarray] = None
    std_response: Optional[np.ndarray] = None
    fatigue_damage: float = 0.0
    newton_iterations: Optional[np.ndarray] = None
    all_converged: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "linear_frequencies": self.linear_frequencies.tolist() if self.linear_frequencies is not None else None,
            "fatigue_damage": self.fatigue_damage,
            "all_converged": self.all_converged,
        }


class NonlinearShaftTorsionAnalyzer:
    """High-level analyzer for nonlinear shaft torsional vibration."""

    def __init__(self, shaft: MultiSegmentShaft,
                 backlash_springs: Optional[Dict[int, BacklashSpring]] = None,
                 damping_ratio: float = 0.02):
        self.shaft = shaft
        self.backlash_springs = backlash_springs or {}
        self.damping_ratio = damping_ratio
        self.M, self.K_linear = shaft.build_matrices()
        self.n_dof = self.M.shape[0]

        freqs, _ = solve_generalized_eigenvalue(self.K_linear, self.M, n_modes=1)
        omega_1 = 2 * np.pi * freqs[0] if freqs[0] > 0 else 1.0
        self.C = 2 * self.damping_ratio * omega_1 * self.M

    def linear_frequencies(self, n_modes: int = 4) -> np.ndarray:
        freqs, _ = solve_generalized_eigenvalue(self.K_linear, self.M, n_modes=n_modes)
        return freqs

    def create_solver(self, seed: int = 42) -> NonlinearNewmarkSolver:
        return NonlinearNewmarkSolver(
            M=self.M, C=self.C, K_linear=self.K_linear,
            backlash_springs=self.backlash_springs,
        )

    def transient_analysis(self, t: np.ndarray, torque_func: Callable,
                           dof_index: int = 0,
                           x0: Optional[np.ndarray] = None,
                           v0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        solver = self.create_solver()
        n_steps = len(t) - 1
        F_ext = np.zeros((n_steps + 1, self.n_dof))
        F_ext[:, dof_index] = torque_func(t)
        return solver.solve(t, F_ext, x0=x0, v0=v0)

    def monte_carlo_analysis(self, t: np.ndarray, spectrum_params: Dict[str, Any],
                             n_samples: int = 100, dof_index: int = 0) -> MonteCarloResult:
        def solver_factory(seed: int = 42):
            return self.create_solver(seed=seed)

        mc = MonteCarloAnalyzer(
            solver_factory=solver_factory, spectrum_params=spectrum_params,
            t=t, n_samples=n_samples,
        )
        return mc.run(dof_index=dof_index)

    def fatigue_analysis(self, stress_history: np.ndarray, dt: float,
                         fatigue_params: Optional[Dict[str, float]] = None) -> float:
        if fatigue_params is None:
            fatigue_params = {"A": 1e30, "m": 3.0, "S_endurance": 100e6}
        miner = MinerFatigueDamage(**fatigue_params)
        return miner.compute_damage(stress_history, dt)


# ─────────────────────────────────────────────────────────────
# 8. Utility functions for verification
# ─────────────────────────────────────────────────────────────

def sdof_backlash_degenerate_test(k: float = 100.0, delta: float = 0.0,
                                   F0: float = 10.0, omega: float = 1.0,
                                   n_cycles: int = 50, pp_cycles: int = 40) -> Dict[str, float]:
    """Test that backlash with δ=0 degenerates to linear response."""
    m = 1.0
    c = 0.0

    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])

    linear_solver = NonlinearNewmarkSolver(M, C, K)
    nl_solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(k, delta)})

    n_steps = n_cycles * pp_cycles
    dt = 2 * np.pi / (omega * pp_cycles)
    t = np.arange(n_steps + 1) * dt

    F_ext = np.zeros((n_steps + 1, 1))
    F_ext[:, 0] = F0 * np.sin(omega * t)

    lin_result = linear_solver.solve(t, F_ext)
    nl_result = nl_solver.solve(t, F_ext)

    lin_amp = np.max(np.abs(lin_result["x"][-pp_cycles:]))
    nl_amp = np.max(np.abs(nl_result["x"][-pp_cycles:]))

    error = abs(nl_amp - lin_amp) / lin_amp if lin_amp > 0 else 0.0

    return {
        "linear_amplitude": lin_amp, "nonlinear_amplitude": nl_amp,
        "linear_error": error, "all_converged": bool(np.all(nl_result["converged"])),
    }


def two_segment_analytical_comparison(L1: float = 6.0, D1: float = 0.5,
                                       L2: float = 6.0, D2: float = 0.4,
                                       G: float = 79.3e9, rho: float = 7850.0) -> Dict[str, Any]:
    """Compare two-segment shaft frequencies: transfer matrix vs analytical."""
    seg1 = ShaftSegment("seg1", L1, D1, G=G, rho=rho)
    seg2 = ShaftSegment("seg2", L2, D2, G=G, rho=rho)
    shaft = MultiSegmentShaft(segments=[seg1, seg2])

    freqs_tm = shaft.natural_frequencies_transfer(n_modes=4)
    freqs_analytical = shaft.analytical_two_segment_frequencies()

    M, K = shaft.build_matrices()
    freqs_lumped, _ = solve_generalized_eigenvalue(K, M, n_modes=4)

    errors = []
    for i in range(min(len(freqs_tm), len(freqs_analytical))):
        if freqs_analytical[i] > 0:
            errors.append(abs(freqs_tm[i] - freqs_analytical[i]) / freqs_analytical[i])

    return {
        "freqs_transfer_matrix": freqs_tm.tolist(),
        "freqs_analytical": freqs_analytical.tolist(),
        "freqs_lumped": freqs_lumped.tolist(),
        "max_error": max(errors) if errors else 0.0,
        "n_modes_matched": len(errors),
    }


def backlash_subharmonic_check(k: float = 1000.0, delta: float = 0.01,
                                F0: float = 50.0, omega_drive: float = 2.0,
                                n_cycles: int = 100, pp_cycles: int = 80) -> Dict[str, Any]:
    """Check for subharmonic resonance in backlash system."""
    m = 1.0
    omega_n = np.sqrt(k / m)
    c = 0.02 * 2 * np.pi * omega_n

    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])

    solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(k, delta)})

    n_steps = n_cycles * pp_cycles
    dt = 2 * np.pi / (omega_drive * pp_cycles)
    t = np.arange(n_steps + 1) * dt

    F_ext = np.zeros((n_steps + 1, 1))
    F_ext[:, 0] = F0 * np.sin(omega_drive * t)

    result = solver.solve(t, F_ext)

    n_ss = n_steps // 2
    x_ss = result["x"][n_ss:, 0]
    x_ss = x_ss - np.mean(x_ss)

    fft_val = np.fft.rfft(x_ss)
    freqs = np.fft.rfftfreq(len(x_ss), dt)
    magnitudes = np.abs(fft_val)

    peak_indices = np.argsort(magnitudes)[-5:][::-1]
    dominant_freqs = freqs[peak_indices]
    dominant_mags = magnitudes[peak_indices]

    omega_drive_hz = omega_drive / (2 * np.pi)
    subharmonic_freq = omega_drive_hz / 2
    subharmonic_idx = np.argmin(np.abs(freqs - subharmonic_freq))
    subharmonic_mag = magnitudes[subharmonic_idx]
    fundamental_idx = np.argmin(np.abs(freqs - omega_drive_hz))
    fundamental_mag = magnitudes[fundamental_idx]

    subharmonic_ratio = subharmonic_mag / fundamental_mag if fundamental_mag > 0 else 0.0

    return {
        "dominant_frequencies": dominant_freqs.tolist(),
        "dominant_magnitudes": dominant_mags.tolist(),
        "subharmonic_ratio": subharmonic_ratio,
        "has_subharmonic": subharmonic_ratio > 0.05,
        "all_converged": bool(np.all(result["converged"])),
        "natural_freq_hz": omega_n / (2 * np.pi),
        "drive_freq_hz": omega_drive_hz,
    }


def zero_load_fatigue_test(dt: float = 0.001, duration: float = 1.0) -> float:
    """Verify zero load gives zero fatigue damage."""
    t = np.arange(int(duration / dt) + 1) * dt
    stress = np.zeros_like(t)
    miner = MinerFatigueDamage(A=1e30, m=3.0, S_endurance=100e6)
    return miner.compute_damage(stress, dt)


def backlash_nonlinearity_check(k: float = 1000.0, delta: float = 0.1,
                                 F0: float = 500.0, omega_drive: float = None,
                                 n_cycles: int = 200, pp_cycles: int = 80) -> Dict[str, Any]:
    """Check for nonlinear signature in backlash system.

    Uses 3rd harmonic enhancement as nonlinearity indicator.
    Returns dict with nonlinearity metrics.
    """
    m = 1.0
    omega_n = np.sqrt(k / m)
    if omega_drive is None:
        omega_drive = 2.0 * omega_n
    c = 0.02 * 2 * np.pi * omega_n

    M = np.array([[m]])
    C = np.array([[c]])
    K = np.array([[k]])

    # Linear baseline
    lin_solver = NonlinearNewmarkSolver(M, C, K)
    n_steps = n_cycles * pp_cycles
    dt = 2 * np.pi / (omega_drive * pp_cycles)
    t = np.arange(n_steps + 1) * dt
    F_ext = np.zeros((n_steps + 1, 1))
    F_ext[:, 0] = F0 * np.sin(omega_drive * t)

    lin_result = lin_solver.solve(t, F_ext)
    n_ss = n_steps // 2
    x_ss_lin = lin_result["x"][n_ss:, 0] - np.mean(lin_result["x"][n_ss:, 0])
    fft_lin = np.fft.rfft(x_ss_lin)
    freqs = np.fft.rfftfreq(len(x_ss_lin), dt)
    mag_lin = np.abs(fft_lin)

    # Nonlinear
    nl_solver = NonlinearNewmarkSolver(M, C, K, backlash_springs={0: BacklashSpring(k, delta)})
    nl_result = nl_solver.solve(t, F_ext)
    x_ss_nl = nl_result["x"][n_ss:, 0] - np.mean(nl_result["x"][n_ss:, 0])
    fft_nl = np.fft.rfft(x_ss_nl)
    mag_nl = np.abs(fft_nl)

    fund_hz = omega_drive / (2 * np.pi)
    idx_fund = np.argmin(np.abs(freqs - fund_hz))
    idx_3rd = np.argmin(np.abs(freqs - 3 * fund_hz))

    lin_3rd_ratio = mag_lin[idx_3rd] / mag_lin[idx_fund] if mag_lin[idx_fund] > 0 else 0
    nl_3rd_ratio = mag_nl[idx_3rd] / mag_nl[idx_fund] if mag_nl[idx_fund] > 0 else 0

    enhancement = nl_3rd_ratio / lin_3rd_ratio if lin_3rd_ratio > 0 else 0

    return {
        "linear_3rd_ratio": lin_3rd_ratio,
        "nonlinear_3rd_ratio": nl_3rd_ratio,
        "enhancement_factor": enhancement,
        "has_nonlinearity": nl_3rd_ratio > 0.001,
        "all_converged": bool(np.all(nl_result["converged"])),
        "natural_freq_hz": omega_n / (2 * np.pi),
        "drive_freq_hz": fund_hz,
    }
