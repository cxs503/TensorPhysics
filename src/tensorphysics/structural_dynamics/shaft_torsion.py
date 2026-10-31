"""
Propulsion Shaft Torsional Vibration Analysis
==============================================
Ice milling torque pulse → torsional vibration response & safety margin.

Analytical anchor
-----------------
Uniform shaft (fixed–free):  f_n = (2n-1)·√(G/ρ) / (4L),  n = 1,…

Lumped model
------------
N-element chain:  propeller – shaft segments – gear – motor
Matrix eigenvalue + Holzer table cross-check.

Transient
---------
Newmark-β (average acceleration, γ=½, β=¼) with ice-milling torque
(BPF + phase jitter).

Outputs
-------
TAF (Torque Amplification Factor), shaft segment shear stress amplitude,
comparison with IACS ice-class torsional vibration limits.
"""

from __future__ import annotations

import json
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Tuple, Dict, Any


# ─────────────────────────────────────────────────────────────
# 1. Analytical uniform-shaft modal solution
# ─────────────────────────────────────────────────────────────

@dataclass
class UniformShaftAnalytical:
    """Closed-form natural frequencies for a uniform shaft (fixed–free).

    f_n = (2n-1) * c / (4L),   c = √(G/ρ)
    """
    G: float       # Shear modulus [Pa]
    rho: float     # Density [kg/m³]
    L: float       # Shaft length [m]
    J: float = 0.006136  # Polar moment of area [m⁴] (default ~500 mm dia.)

    @property
    def c(self) -> float:
        """Torsional wave speed [m/s]."""
        return np.sqrt(self.G / self.rho)

    def frequency(self, n: int) -> float:
        """Natural frequency of mode n (1-indexed) [Hz]."""
        return (2 * n - 1) * self.c / (4.0 * self.L)

    def frequencies(self, n_modes: int) -> np.ndarray:
        """First *n_modes* natural frequencies [Hz]."""
        ns = np.arange(1, n_modes + 1)
        return (2 * ns - 1) * self.c / (4.0 * self.L)

    def omega(self, n: int) -> float:
        """Circular natural frequency [rad/s]."""
        return 2.0 * np.pi * self.frequency(n)


# ─────────────────────────────────────────────────────────────
# 2. Lumped inertia–torsional spring chain
# ─────────────────────────────────────────────────────────────

@dataclass
class LumpedInertia:
    """A single lumped inertia node."""
    name: str
    J: float       # Mass moment of inertia [kg·m²]
    location: float  # Axial position [m]


@dataclass
class TorsionalSpring:
    """A torsional spring (shaft segment or coupling)."""
    name: str
    k: float       # Torsional stiffness [N·m/rad]
    left_node: int
    right_node: int


class LumpedMassSpringModel:
    """Build M, K matrices for a lumped inertia–spring chain.

    Nodes are ordered 0 … n_nodes-1.
    Boundary: node 0 is typically the engine/gearbox (may be fixed or free).
    """

    def __init__(self):
        self.inertias: List[LumpedInertia] = []
        self.springs: List[TorsionalSpring] = []
        self.fixed_nodes: List[int] = []

    def add_inertia(self, name: str, J: float, location: float) -> int:
        idx = len(self.inertias)
        self.inertias.append(LumpedInertia(name, J, location))
        return idx

    def add_spring(self, name: str, k: float, left: int, right: int):
        self.springs.append(TorsionalSpring(name, k, left, right))

    def fix_node(self, node: int):
        self.fixed_nodes.append(node)

    @property
    def n_nodes(self) -> int:
        return len(self.inertias)

    def build_matrices(self) -> Tuple[np.ndarray, np.ndarray]:
        """Assemble global M (diagonal) and K (sparse → dense) matrices."""
        n = self.n_nodes
        M = np.zeros(n)
        K = np.zeros((n, n))

        for iner in self.inertias:
            idx = self.inertias.index(iner)
            M[idx] = iner.J

        for sp in self.springs:
            i, j = sp.left_node, sp.right_node
            K[i, i] += sp.k
            K[j, j] += sp.k
            K[i, j] -= sp.k
            K[j, i] -= sp.k

        return M, K

    def solve_eigenvalue(self, n_modes: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Solve generalized eigenvalue problem K·φ = ω²·M·φ.

        Returns (frequencies_Hz, mode_shapes) after removing fixed DOFs.
        """
        M, K = self.build_matrices()
        free = [i for i in range(self.n_nodes) if i not in self.fixed_nodes]
        M_f = np.diag(M)[free][:, free]
        K_f = K[np.ix_(free, free)]

        # Solve M⁻¹K eigenvalue problem (symmetric → use eigh)
        M_f_inv_half = np.diag(1.0 / np.sqrt(np.diag(M_f)))
        A = M_f_inv_half @ K_f @ M_f_inv_half
        eig_vals, eig_vecs = np.linalg.eigh(A)

        # Convert to physical mode shapes
        phi = M_f_inv_half @ eig_vecs

        # Frequencies
        omega_sq = np.maximum(eig_vals, 0.0)  # clip numerical noise
        freqs = np.sqrt(omega_sq) / (2.0 * np.pi)

        if n_modes is not None:
            freqs = freqs[:n_modes]
            phi = phi[:, :n_modes]

        return freqs, phi

    def holzer_table(self, freq_hz: float) -> List[Dict[str, float]]:
        """Holzer method for verifying modal frequencies.

        For fixed-free system: start from first free node with unit amplitude.
        At natural frequency, residual torque at free end ≈ 0.
        """
        omega = 2.0 * np.pi * freq_hz
        n = self.n_nodes

        # Build spring map
        spring_map: Dict[Tuple[int, int], float] = {}
        for sp in self.springs:
            spring_map[(sp.left_node, sp.right_node)] = sp.k
            spring_map[(sp.right_node, sp.left_node)] = sp.k

        # Find first free node
        free_nodes = [i for i in range(n) if i not in self.fixed_nodes]
        if not free_nodes:
            return []

        # Start Holzer sweep from first free node
        theta = np.zeros(n)
        start_node = free_nodes[0]
        theta[start_node] = 1.0

        residual_torque = 0.0
        table = []

        # Sweep from first free node to last
        for i in range(start_node, n):
            # Inertia torque at this node
            inertia_torque = -self.inertias[i].J * omega**2 * theta[i]
            residual_torque += inertia_torque

            row = {
                "node": i,
                "name": self.inertias[i].name,
                "J": self.inertias[i].J,
                "theta": theta[i],
                "inertia_torque": inertia_torque,
                "residual_torque": residual_torque,
            }
            table.append(row)

            # Transmit to next node via spring
            if i < n - 1 and (i, i + 1) in spring_map:
                k = spring_map[(i, i + 1)]
                if abs(k) > 1e-30:
                    theta[i + 1] = theta[i] + residual_torque / k

        return table


def build_uniform_shaft_model(
    shaft: UniformShaftAnalytical,
    N: int,
    propeller_J: float = 0.0,
    engine_J: float = 0.0,
) -> LumpedMassSpringModel:
    """Discretize a uniform shaft into N elements.

    Fixed–free: node 0 (engine end) fixed, node N (propeller end) free.
    """
    model = LumpedMassSpringModel()
    h = shaft.L / N  # element length
    k_elem = shaft.G * shaft.J / h  # element torsional stiffness
    I_elem = shaft.rho * shaft.J * h  # element mass moment of inertia

    # Node 0: engine inertia (or half-element if no engine)
    J0 = engine_J if engine_J > 0 else I_elem / 2.0
    model.add_inertia("engine", J0, 0.0)
    model.fix_node(0)

    # Interior nodes 1 … N-1
    for i in range(1, N):
        model.add_inertia(f"shaft_{i}", I_elem, i * h)

    # Node N: propeller
    JN = propeller_J + I_elem / 2.0
    model.add_inertia("propeller", JN, shaft.L)

    # Springs
    for i in range(N):
        model.add_spring(f"seg_{i}", k_elem, i, i + 1)

    return model


# ─────────────────────────────────────────────────────────────
# 3. Newmark-β transient solver
# ─────────────────────────────────────────────────────────────

class NewmarkBetaSolver:
    """Average-acceleration Newmark-β (γ=½, β=¼) for M·x'' + C·x' + K·x = F(t).

    Unconditionally stable for linear systems.
    """

    def __init__(self, M: np.ndarray, C: np.ndarray, K: np.ndarray,
                 gamma: float = 0.5, beta: float = 0.25):
        self.M = M
        self.C = C
        self.K = K
        self.gamma = gamma
        self.beta = beta
        self.n_dof = M.shape[0]

    def solve(self, t: np.ndarray, F: np.ndarray,
              x0: Optional[np.ndarray] = None,
              v0: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Time integration.

        Parameters
        ----------
        t : (n_steps+1,) time array
        F : (n_steps+1, n_dof) force array
        x0, v0 : initial conditions (default zero)

        Returns
        -------
        dict with 't', 'x', 'v', 'a', 'energy'
        """
        n_steps = len(t) - 1
        dt = t[1] - t[0]
        n = self.n_dof

        x = np.zeros((n_steps + 1, n))
        v = np.zeros((n_steps + 1, n))
        a = np.zeros((n_steps + 1, n))

        if x0 is not None:
            x[0] = x0
        if v0 is not None:
            v[0] = v0

        # Initial acceleration
        a[0] = np.linalg.solve(self.M, F[0] - self.C @ v[0] - self.K @ x[0])

        # Effective stiffness
        g = self.gamma
        b = self.beta
        K_eff = (self.K
                 + g / (b * dt) * self.C
                 + 1.0 / (b * dt**2) * self.M)
        K_eff_inv = np.linalg.inv(K_eff)

        for i in range(n_steps):
            # Effective force
            F_eff = (F[i + 1]
                     + self.M @ (x[i] / (b * dt**2) + v[i] / (b * dt) + (1 - 2 * b) / (2 * b) * a[i])
                     + self.C @ (g / (b * dt) * x[i] + (g / b - 1) * v[i]
                                 + dt / 2 * (g / b - 2) * a[i]))

            x[i + 1] = K_eff_inv @ F_eff
            a[i + 1] = ((x[i + 1] - x[i]) / (b * dt**2)
                        - v[i] / (b * dt)
                        - (1 - 2 * b) / (2 * b) * a[i])
            v[i + 1] = (v[i]
                        + dt * ((1 - g) * a[i] + g * a[i + 1]))

        # Energy computation
        energy = self._compute_energy(x, v, a, F, dt)

        return {"t": t, "x": x, "v": v, "a": a, "energy": energy}

    def _compute_energy(self, x, v, a, F, dt) -> np.ndarray:
        """Compute total energy at each time step.

        E = ½ vᵀMv + ½ xᵀKx
        """
        n_steps = len(v)
        energy = np.zeros(n_steps)
        for i in range(n_steps):
            KE = 0.5 * v[i] @ self.M @ v[i]
            SE = 0.5 * x[i] @ self.K @ x[i]
            energy[i] = KE + SE
        return energy


# ─────────────────────────────────────────────────────────────
# 4. Ice milling excitation model
# ─────────────────────────────────────────────────────────────

@dataclass
class IceMillingExcitation:
    """Ice milling torque pulse model.

    T(t) = T_mean + Σ A_k · sin(2π·k·BPF·t + φ_k)

    BPF = n_blades × RPM / 60
    Phase jitter φ_k adds stochastic content.
    """
    T_mean: float = 50000.0     # Mean torque [N·m]
    n_blades: int = 5           # Number of blades
    RPM: float = 120.0          # Shaft speed [rpm]
    n_harmonics: int = 3        # Number of BPF harmonics
    amplitudes: Optional[List[float]] = None  # Harmonic amplitudes [N·m]
    phase_jitter_std: float = 0.1  # Phase jitter std [rad]
    seed: int = 42

    @property
    def BPF(self) -> float:
        return self.n_blades * self.RPM / 60.0

    def torque(self, t: np.ndarray) -> np.ndarray:
        """Generate torque time history."""
        rng = np.random.RandomState(self.seed)
        T = np.full_like(t, self.T_mean, dtype=float)

        if self.amplitudes is None:
            amps = [self.T_mean * 0.3 / k for k in range(1, self.n_harmonics + 1)]
        else:
            amps = self.amplitudes[:self.n_harmonics]

        for k, amp in enumerate(amps):
            freq = (k + 1) * self.BPF
            phase = rng.uniform(-self.phase_jitter_std, self.phase_jitter_std)
            T += amp * np.sin(2 * np.pi * freq * t + phase)

        return T


# ─────────────────────────────────────────────────────────────
# 5. SDOF resonance test (for Q-factor verification)
# ─────────────────────────────────────────────────────────────

def sdof_resonance_test(zeta: float, freq_hz: float = 1.0,
                        F0: float = 1.0, n_cycles: int = 200,
                        pp_cycles: int = 100) -> Dict[str, float]:
    """SDOF harmonic resonance test.

    m·x'' + c·x' + k·x = F0·sin(ω·t),  ω = ω_n

    Steady-state amplification:  X / X_static = 1 / (2ζ)

    Returns dict with 'Q_analytical', 'Q_numerical', 'error'.
    """
    m = 1.0
    omega_n = 2 * np.pi * freq_hz
    k = m * omega_n**2
    c = 2 * zeta * m * omega_n

    M_mat = np.array([[m]])
    C_mat = np.array([[c]])
    K_mat = np.array([[k]])

    solver = NewmarkBetaSolver(M_mat, C_mat, K_mat)

    total_time = n_cycles / freq_hz
    n_steps = n_cycles * pp_cycles
    t = np.linspace(0, total_time, n_steps + 1)
    F = np.zeros((n_steps + 1, 1))
    F[:, 0] = F0 * np.sin(omega_n * t)

    result = solver.solve(t, F)

    # Extract steady-state amplitude (last 20% of signal)
    x = result["x"][:, 0]
    n_ss = len(x) // 5
    x_ss = x[-n_ss:]
    X_numerical = np.max(x_ss) - np.min(x_ss)
    X_numerical /= 2.0  # amplitude (half peak-to-peak)

    X_static = F0 / k
    Q_analytical = 1.0 / (2 * zeta)
    Q_numerical = X_numerical / X_static

    return {
        "zeta": zeta,
        "Q_analytical": Q_analytical,
        "Q_numerical": Q_numerical,
        "error": abs(Q_numerical - Q_analytical) / Q_analytical,
    }


# ─────────────────────────────────────────────────────────────
# 6. Harmonic response (frequency sweep)
# ─────────────────────────────────────────────────────────────

def sdof_frequency_response(
    zeta: float = 0.02,
    freq_n: float = 1.0,
    freq_range: Tuple[float, float] = (0.1, 3.0),
    n_freq: int = 200,
    F0: float = 1.0,
) -> Dict[str, np.ndarray]:
    """SDOF frequency response via steady-state harmonic analysis.

    Compare numerical (time-domain → amplitude) vs analytical transfer function.

    |H(ω)| = 1 / √((k - mω²)² + (cω)²)
    """
    m = 1.0
    omega_n = 2 * np.pi * freq_n
    k = m * omega_n**2
    c = 2 * zeta * m * omega_n

    freqs = np.linspace(freq_range[0], freq_range[1], n_freq)
    omegas = 2 * np.pi * freqs

    # Analytical transfer function
    H_analytical = 1.0 / np.sqrt((k - m * omegas**2)**2 + (c * omegas)**2)

    # Numerical: run time simulation for each frequency, extract amplitude
    H_numerical = np.zeros(n_freq)
    for i, freq in enumerate(freqs):
        omega = 2 * np.pi * freq
        n_cycles = max(50, int(80 * freq_n / freq))
        total_time = n_cycles / freq
        pp_cycle = 100  # Increased for better accuracy at high frequencies
        n_steps = n_cycles * pp_cycle
        t = np.linspace(0, total_time, n_steps + 1)
        F = np.zeros((n_steps + 1, 1))
        F[:, 0] = F0 * np.sin(omega * t)

        M_mat = np.array([[m]])
        C_mat = np.array([[c]])
        K_mat = np.array([[k]])
        solver = NewmarkBetaSolver(M_mat, C_mat, K_mat)
        result = solver.solve(t, F)

        # Steady-state amplitude (last 25% of signal)
        x = result["x"][:, 0]
        n_ss = len(x) // 4
        x_ss = x[-n_ss:]
        X = (np.max(x_ss) - np.min(x_ss)) / 2.0
        H_numerical[i] = X / F0

    return {
        "freqs": freqs,
        "H_analytical": H_analytical,
        "H_numerical": H_numerical,
        "error": np.abs(H_numerical - H_analytical) / np.maximum(H_analytical, 1e-30),
    }


# ─────────────────────────────────────────────────────────────
# 7. Main orchestrator
# ─────────────────────────────────────────────────────────────

@dataclass
class ShaftTorsionResult:
    """Container for all analysis results."""
    # Modal
    analytical_freqs: List[float] = field(default_factory=list)
    numerical_freqs: List[float] = field(default_factory=list)
    modal_errors: List[float] = field(default_factory=list)

    # Convergence
    convergence_N: List[int] = field(default_factory=list)
    convergence_errors: List[List[float]] = field(default_factory=list)

    # Transient
    TAF: float = 0.0
    max_shear_stress: float = 0.0
    energy_drift: float = 0.0

    # Resonance
    resonance_Q_results: List[Dict] = field(default_factory=list)

    # Frequency response
    freq_response_max_error: float = 0.0

    # IACS comparison
    iacs_margin: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ShaftTorsionAnalyzer:
    """Top-level analyzer orchestrating all sub-analyses."""

    def __init__(self, G: float = 79.3e9, rho: float = 7850.0,
                 L: float = 12.0, D: float = 0.5,
                 propeller_J: float = 5000.0,
                 engine_J: float = 10000.0):
        J = np.pi * D**4 / 32.0
        self.shaft = UniformShaftAnalytical(G=G, rho=rho, L=L, J=J)
        self.propeller_J = propeller_J
        self.engine_J = engine_J
        self.D = D

    def modal_analysis(self, N: int = 50, n_modes: int = 4) -> Tuple[np.ndarray, np.ndarray]:
        """Compute numerical natural frequencies and compare to analytical."""
        model = build_uniform_shaft_model(
            self.shaft, N,
            propeller_J=self.propeller_J,
            engine_J=self.engine_J,
        )
        freqs_num, _ = model.solve_eigenvalue(n_modes=n_modes)
        freqs_ana = self.shaft.frequencies(n_modes)
        return freqs_ana, freqs_num

    def convergence_study(self, N_values: List[int], n_modes: int = 4) -> Dict:
        """Mesh convergence: error vs N."""
        freqs_ana = self.shaft.frequencies(n_modes)
        results = {"N": [], "errors": []}
        for N in N_values:
            model = build_uniform_shaft_model(
                self.shaft, N,
                propeller_J=self.propeller_J,
                engine_J=self.engine_J,
            )
            freqs_num, _ = model.solve_eigenvalue(n_modes=n_modes)
            errs = np.abs(freqs_num - freqs_ana) / freqs_ana
            results["N"].append(N)
            results["errors"].append(errs.tolist())
        return results

    def transient_analysis(self, N: int = 30, duration: float = 2.0,
                           dt: float = 1e-4,
                           excitation: Optional[IceMillingExcitation] = None
                           ) -> Dict[str, Any]:
        """Newmark-β transient under ice milling torque."""
        if excitation is None:
            excitation = IceMillingExcitation()

        model = build_uniform_shaft_model(
            self.shaft, N,
            propeller_J=self.propeller_J,
            engine_J=self.engine_J,
        )
        M_full, K_full = model.build_matrices()
        free = [i for i in range(model.n_nodes) if i not in model.fixed_nodes]
        M = np.diag(M_full)[free][:, free]
        K = K_full[np.ix_(free, free)]
        n_dof = len(free)

        # Rayleigh damping: ζ = 0.02 for first two modes
        freqs_num, _ = model.solve_eigenvalue(n_modes=2)
        omega1 = 2 * np.pi * freqs_num[0]
        omega2 = 2 * np.pi * freqs_num[1] if len(freqs_num) > 1 else omega1 * 3
        zeta = 0.02
        a0 = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
        a1 = 2 * zeta / (omega1 + omega2)
        C = a0 * M + a1 * K

        # Time array
        t = np.arange(0, duration + dt, dt)
        n_steps = len(t)

        # Excitation
        T_t = excitation.torque(t)

        # Force vector: torque applied at propeller node (last free DOF)
        F = np.zeros((n_steps, n_dof))
        prop_idx = free[-1]  # propeller is last node
        local_prop = free.index(prop_idx) if prop_idx in free else n_dof - 1
        F[:, local_prop] = T_t

        # Solve
        solver = NewmarkBetaSolver(M, C, K)
        result = solver.solve(t, F)

        # Post-process
        x = result["x"]  # (n_steps, n_dof) angular displacements
        energy = result["energy"]

        # TAF: max dynamic torque / mean static torque
        # Compute segment torques: T_seg = k * (θ_{i+1} - θ_i)
        max_seg_torque = 0.0
        k_elem = self.shaft.G * self.shaft.J / (self.shaft.L / N)
        for i in range(n_dof - 1):
            seg_torque = k_elem * (x[:, i + 1] - x[:, i])
            max_seg_torque = max(max_seg_torque, np.max(np.abs(seg_torque)))

        TAF = max_seg_torque / excitation.T_mean if excitation.T_mean > 0 else 0.0

        # Max shear stress: τ = T·r/J = T·(D/2)/J_polar
        r = self.D / 2.0
        max_shear = max_seg_torque * r / self.shaft.J

        # Energy drift
        E0 = energy[1] if energy[1] > 0 else 1.0
        energy_drift = np.max(np.abs(energy[1:] - energy[0])) / max(abs(energy[0]), 1.0)

        return {
            "t": t,
            "x": x,
            "energy": energy,
            "TAF": TAF,
            "max_shear_stress": max_shear,
            "energy_drift": energy_drift,
            "max_seg_torque": max_seg_torque,
        }

    def run_full_analysis(self, N: int = 50, n_modes: int = 4) -> ShaftTorsionResult:
        """Run complete analysis and return structured result."""
        result = ShaftTorsionResult()

        # 1. Modal analysis
        freqs_ana, freqs_num = self.modal_analysis(N, n_modes)
        result.analytical_freqs = freqs_ana.tolist()
        result.numerical_freqs = freqs_num.tolist()
        result.modal_errors = (np.abs(freqs_num - freqs_ana) / freqs_ana).tolist()

        # 2. Convergence study
        N_values = [5, 10, 15, 20, 30, 40, 50, 80, 100]
        conv = self.convergence_study(N_values, n_modes)
        result.convergence_N = conv["N"]
        result.convergence_errors = conv["errors"]

        # 3. Transient
        trans = self.transient_analysis(N=N)
        result.TAF = trans["TAF"]
        result.max_shear_stress = trans["max_shear_stress"]
        result.energy_drift = trans["energy_drift"]

        # 4. SDOF resonance Q-factor
        for zeta in [0.02, 0.05, 0.1]:
            q_res = sdof_resonance_test(zeta)
            result.resonance_Q_results.append(q_res)

        # 5. Frequency response
        fr = sdof_frequency_response()
        result.freq_response_max_error = float(np.max(fr["error"]))

        # 6. IACS margin (simplified: allowable TAF ≤ 2.0 for ice class)
        iacs_allowable_TAF = 2.0
        result.iacs_margin = (iacs_allowable_TAF - result.TAF) / iacs_allowable_TAF

        return result
