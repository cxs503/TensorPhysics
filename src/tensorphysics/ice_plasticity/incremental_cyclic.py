"""Incremental cyclic elasto-plastic analysis with return mapping.

Implements:
- Return mapping algorithm for 1D plasticity with kinematic hardening
- Cycle-by-cycle tracking of plastic work and residual deformation
- Convergence monitoring

Constitutive model (1D):
  ε = εe + εp
  σ = E·εe
  f = |σ - β| - σy0 ≤ 0  (yield function)
  β = H·εp  (linear kinematic hardening)
  εp_dot = γ·sign(σ - β)  (flow rule)
  γ ≥ 0, f ≤ 0, γ·f = 0  (KKT)

Return mapping (backward Euler):
  Trial: σ_tr = E·(ε_n+1 - εp_n), β_tr = H·εp_n
  f_tr = |σ_tr - β_tr| - σy0
  If f_tr ≤ 0: elastic step
  If f_tr > 0: plastic correction
    Δγ = f_tr / (E + H)
    σ_n+1 = σ_tr - E·Δγ·sign(σ_tr - β_tr)
    β_n+1 = β_tr + H·Δγ·sign(σ_tr - β_tr)
    εp_n+1 = εp_n + Δγ·sign(σ_tr - β_tr)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class KinematicHardening1D:
    """1D kinematic hardening material state and update."""
    E: float        # Young's modulus [Pa]
    sigma_y0: float  # initial yield stress [Pa]
    H: float        # kinematic hardening modulus [Pa]

    # State variables
    sigma: float = 0.0
    beta: float = 0.0      # back stress
    eps_p: float = 0.0     # plastic strain
    eps: float = 0.0       # total strain

    def reset(self):
        self.sigma = 0.0
        self.beta = 0.0
        self.eps_p = 0.0
        self.eps = 0.0

    def trial_stress(self, eps_new: float) -> Tuple[float, float]:
        """Compute trial stress and back stress."""
        sigma_tr = self.E * (eps_new - self.eps_p)
        beta_tr = self.H * self.eps_p
        return sigma_tr, beta_tr

    def return_mapping(self, eps_new: float) -> dict:
        """Return mapping step.

        Parameters
        ----------
        eps_new : float
            New total strain

        Returns
        -------
        result : dict
            Updated state and plastic flag
        """
        sigma_tr, beta_tr = self.trial_stress(eps_new)
        xi_tr = sigma_tr - beta_tr  # relative stress
        f_tr = abs(xi_tr) - self.sigma_y0

        if f_tr <= 0:
            # Elastic step
            self.sigma = sigma_tr
            self.beta = beta_tr
            self.eps = eps_new
            return {"plastic": False, "delta_gamma": 0.0, "f_trial": f_tr}

        # Plastic step
        delta_gamma = f_tr / (self.E + self.H)
        sign_xi = np.sign(xi_tr)

        self.sigma = sigma_tr - self.E * delta_gamma * sign_xi
        self.beta = beta_tr + self.H * delta_gamma * sign_xi
        self.eps_p = self.eps_p + delta_gamma * sign_xi
        self.eps = eps_new

        return {"plastic": True, "delta_gamma": delta_gamma, "f_trial": f_tr}

    def stress(self, eps: float) -> float:
        """Compute stress for given strain (single step from zero)."""
        self.reset()
        result = self.return_mapping(eps)
        return self.sigma


@dataclass
class CyclicHistory:
    """Storage for cyclic analysis history."""
    cycle_number: List[int] = field(default_factory=list)
    eps_max: List[float] = field(default_factory=list)
    eps_min: List[float] = field(default_factory=list)
    sigma_max: List[float] = field(default_factory=list)
    sigma_min: List[float] = field(default_factory=list)
    eps_p_accum: List[float] = field(default_factory=list)
    plastic_work: List[float] = field(default_factory=list)
    residual_strain: List[float] = field(default_factory=list)


def run_cyclic_analysis(mat: KinematicHardening1D,
                        eps_amplitude: float,
                        n_cycles: int,
                        load_ratio: float = 0.0,
                        n_steps_per_cycle: int = 20) -> CyclicHistory:
    """Run cyclic strain-controlled analysis.

    Parameters
    ----------
    mat : KinematicHardening1D
        Material (will be modified in place)
    eps_amplitude : float
        Strain amplitude
    n_cycles : int
        Number of cycles
    load_ratio : float
        R = eps_min / eps_max (0 for pulsating, -1 for fully reversed)
    n_steps_per_cycle : int
        Time steps per cycle

    Returns
    -------
    history : CyclicHistory
    """
    mat.reset()
    history = CyclicHistory()

    eps_max = eps_amplitude
    eps_min = load_ratio * eps_amplitude

    total_work = 0.0

    for cycle in range(n_cycles):
        # Generate strain path for this cycle
        eps_path = np.linspace(eps_min, eps_max, n_steps_per_cycle // 2)
        eps_path = np.concatenate([eps_path, eps_path[::-1][1:]])

        cycle_work = 0.0
        sigma_prev = mat.sigma

        for eps in eps_path:
            result = mat.return_mapping(eps)
            if result["plastic"]:
                # Plastic work increment: σ·dεp
                d_eps_p = mat.eps_p - (mat.eps_p - result["delta_gamma"] * np.sign(mat.sigma - mat.beta))
                cycle_work += mat.sigma * abs(d_eps_p)

        total_work += cycle_work

        history.cycle_number.append(cycle + 1)
        history.eps_max.append(eps_max)
        history.eps_min.append(eps_min)
        history.sigma_max.append(mat.sigma if mat.eps == eps_max else mat.sigma)
        history.sigma_min.append(mat.sigma if mat.eps == eps_min else mat.sigma)
        history.eps_p_accum.append(abs(mat.eps_p))
        history.plastic_work.append(total_work)
        history.residual_strain.append(mat.eps_p)

    return history


def run_stress_cyclic_analysis(E: float, sigma_y0: float, H: float,
                                stress_amplitude: float,
                                n_cycles: int,
                                load_ratio: float = 0.0,
                                n_steps_per_cycle: int = 20) -> CyclicHistory:
    """Run cyclic stress-controlled analysis.

    For stress control, we need to find the strain that produces the target stress.
    Uses Newton iteration within each step.

    Parameters
    ----------
    E, sigma_y0, H : material parameters
    stress_amplitude : float
        Stress amplitude [Pa]
    n_cycles : int
    load_ratio : float
    n_steps_per_cycle : int

    Returns
    -------
    history : CyclicHistory
    """
    mat = KinematicHardening1D(E=E, sigma_y0=sigma_y0, H=H)
    history = CyclicHistory()

    sigma_max = stress_amplitude
    sigma_min = load_ratio * stress_amplitude

    total_work = 0.0

    for cycle in range(n_cycles):
        # Generate stress path
        sigma_path = np.linspace(sigma_min, sigma_max, n_steps_per_cycle // 2)
        sigma_path = np.concatenate([sigma_path, sigma_path[::-1][1:]])

        cycle_work = 0.0
        eps_p_start = mat.eps_p

        for sigma_target in sigma_path:
            # Find strain that produces sigma_target
            # For elastic: eps = sigma/E
            # For plastic: need iteration
            eps = _find_strain_for_stress(mat, sigma_target)

            if mat.eps_p != eps_p_start:
                d_eps_p = mat.eps_p - eps_p_start
                cycle_work += sigma_target * abs(d_eps_p)
                eps_p_start = mat.eps_p

        total_work += cycle_work

        history.cycle_number.append(cycle + 1)
        history.eps_max.append(mat.eps)
        history.eps_min.append(mat.eps)
        history.sigma_max.append(sigma_max)
        history.sigma_min.append(sigma_min)
        history.eps_p_accum.append(abs(mat.eps_p))
        history.plastic_work.append(total_work)
        history.residual_strain.append(mat.eps_p)

    return history


def _find_strain_for_stress(mat: KinematicHardening1D, sigma_target: float,
                             tol: float = 1e-12, max_iter: int = 50) -> float:
    """Find strain that produces target stress using Newton iteration."""
    eps = mat.eps + sigma_target / mat.E  # initial guess

    for _ in range(max_iter):
        # Forward: compute stress at this strain
        sigma_tr = mat.E * (eps - mat.eps_p)
        beta_tr = mat.H * mat.eps_p
        xi_tr = sigma_tr - beta_tr
        f_tr = abs(xi_tr) - mat.sigma_y0

        if f_tr <= 0:
            sigma = sigma_tr
        else:
            delta_gamma = f_tr / (mat.E + mat.H)
            sign_xi = np.sign(xi_tr)
            sigma = sigma_tr - mat.E * delta_gamma * sign_xi

        residual = sigma - sigma_target
        if abs(residual) < tol:
            break

        # Tangent modulus
        if f_tr <= 0:
            Et = mat.E
        else:
            Et = mat.E * mat.H / (mat.E + mat.H)

        deps = -residual / Et
        eps += deps

    # Update material state
    mat.return_mapping(eps)
    return eps


def classify_regime(history: CyclicHistory, tol: float = 1e-9) -> str:
    """Classify cyclic response regime.

    Returns one of:
    - "elastic_shakedown": no plastic deformation after initial cycles
    - "plastic_shakedown": plastic deformation stabilizes (ratcheting rate → 0)
    - "ratcheting": permanent strain accumulates linearly
    - "alternating_plasticity": plastic strain oscillates without accumulation
    """
    if len(history.residual_strain) < 3:
        return "unknown"

    eps_p = np.array(history.residual_strain)

    # Check if any plasticity occurred
    if np.max(np.abs(eps_p)) < tol:
        return "elastic_shakedown"

    # Check ratcheting rate (slope of eps_p vs cycle)
    n = len(eps_p)
    if n >= 10:
        # Use last 10 cycles
        x = np.arange(n - 10, n, dtype=float)
        y = eps_p[-10:]
        slope = np.polyfit(x, y, 1)[0]

        if abs(slope) < tol:
            # Check if plastic strain oscillates or is constant
            if np.std(y) < tol:
                return "plastic_shakedown"
            else:
                return "alternating_plasticity"
        else:
            return "ratcheting"
    else:
        # Not enough cycles
        if abs(eps_p[-1] - eps_p[0]) < tol:
            return "plastic_shakedown"
        else:
            return "ratcheting"


def monotonic_degradation_test(E: float, sigma_y0: float, H: float,
                                eps_amplitudes: np.ndarray) -> dict:
    """Test monotonic degradation: as amplitude → 0, permanent deformation → 0.

    Parameters
    ----------
    E, sigma_y0, H : material parameters
    eps_amplitudes : array
        Strain amplitudes to test

    Returns
    -------
    result : dict
    """
    results = []
    for eps_amp in eps_amplitudes:
        mat = KinematicHardening1D(E=E, sigma_y0=sigma_y0, H=H)
        history = run_cyclic_analysis(mat, eps_amp, n_cycles=10, load_ratio=0.0)
        results.append({
            "eps_amplitude": eps_amp,
            "residual_strain": history.residual_strain[-1],
            "plastic_work": history.plastic_work[-1],
        })

    # Check: as eps_amp → 0, residual → 0
    return {
        "results": results,
        "max_residual_at_small_amp": max(r["residual_strain"] for r in results if r["eps_amplitude"] < sigma_y0 / E),
    }
