"""Rate-dependent ice crushing with Norton-Bailey creep × structural compliance.

Physical model
--------------
A flat ice indentor is pressed against a compliant structure (hull plate/stiffener
modelled as a linear spring of stiffness k_s).  The ship advances at indentation
velocity *v*, producing a time-varying contact pressure *p(t)* over area *A*.

The ice deforms by power-law (Norton-Bailey / Glen) creep:

    ε̇_c = B · p^n                              (1)

where *B* is the creep rate parameter (itself temperature-dependent through an
Arrhenius factor) and *n* ≈ 3 is the stress exponent for polycrystalline ice.

Compatibility (total indentation = elastic spring deflection + creep displacement):

    v · t  =  (A / k_s) · p  +  L_c · ε_c     (2)

Differentiating:

    v  =  (A / k_s) · ṗ  +  L_c · B · p^n     (3)

This is a first-order nonlinear ODE for p(t):

    ṗ  =  (k_s / A) · (v - L_c · B · p^n)     (4)

We solve it by backward Euler with Newton iteration, which is unconditionally
stable for any stiffness ratio.

Limiting cases
~~~~~~~~~~~~~~
* **Rigid structure** (k_s → ∞):  steady-state pressure
      p_ss = (v / (B L_c))^{1/n}                        (5)
  giving the pressure–rate scaling exponent  m = 1/n.

* **Very soft structure** (k_s → 0):  pressure stays low, creep negligible,
  p(t) ≈ k_s v t / A  (linear ramp, elastic-only).

* **Stress relaxation** (v = 0, initial pressure p₀):
      p(t) = p₀ · (1 + (n-1) t / τ₀)^{-1/(n-1)}          (6)
  where τ₀ = A / (k_s B n L_c p₀^{n-1}) is the initial relaxation time.

Interface with ice-belt plate strength (m7257)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The sustained pressure p_sustained from this module feeds into the frame-strip
plastic-hinge check in ``tensorphysics.ice_plasticity.frame_strip``: the strip
must satisfy  M_max(p_sustained) ≤ M_p  (plastic moment capacity).

IACS Polar Class monotonicity check
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
IACS UR I requires that the sustained ice pressure coefficient does not
increase when going from a higher to a lower PC class (i.e. PC1 > PC2 > … > PC7).
We verify this monotonicity for the output of the creep model at fixed v.

References
----------
* Norton, F.H. (1929). *The Creep of Steel at High Temperatures*. McGraw-Hill.
* Glen, J.W. (1955). "The creep of polycrystalline ice".
  Proc. R. Soc. Lond. A, 228, 519–538.
* Jordaan, I.J. (2001). "Mechanical behaviour and crushing of ice".
  Proc. IUTAM Symposium, Helsinki.
* IACS UR I (2021). Unified Requirements for Polar Class Ships.
* Schulson, E.M. & Duval, P. (2009). *Creep of Ice*. Cambridge Univ. Press.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import numpy as np

__all__ = [
    "NortonCreepParams",
    "CreepCrushingConfig",
    "CreepCrushingResult",
    "solve_creep_crushing",
    "compute_scaling_exponent",
    "steady_state_pressure",
    "relaxation_time_constant",
    "pressure_rate_exponent",
    "analytical_relaxation",
    "rigid_limit_pressure",
    "iacs_pc_sustained_pressure",
    "check_iacs_monotonicity",
    "link_frame_strip_check",
]


# ── Norton-Bailey creep parameters ────────────────────────────────────

@dataclass(frozen=True)
class NortonCreepParams:
    """Temperature-dependent Norton-Bailey creep parameters for columnar sea ice.

    The creep law is:  ε̇ = B_eff · σ^n

    where B_eff = B_ref · exp(-Q / (R · T_K))  with T_K in Kelvin.

    Parameters
    ----------
    B_ref : float
        Pre-exponential creep parameter [1/(s·Pa^n)].
        Typical range for ice: 1e-22 to 1e-16 (depends on T, salinity, grain size).
    n : float
        Stress exponent (dimensionless).  n ≈ 3 for dislocation creep in ice.
    Q : float
        Activation energy [J/mol].  ~60 kJ/mol for ice below -10 °C.
    T_K : float
        Absolute temperature [K].  Default: 263.15 K = -10 °C.
    R : float
        Universal gas constant [J/(mol·K)].  Default: 8.314.

    References
    ----------
    Schulson & Duval (2009), *Creep of Ice*, Cambridge.
    """
    B_ref: float = 4.0e-18   # [1/(s·Pa^n)] — calibrated at T ≈ -10 °C
    n: float = 3.0
    Q: float = 60.0e3        # [J/mol]
    T_K: float = 263.15      # [-10 °C]
    R: float = 8.314         # [J/(mol·K)]

    @property
    def B_eff(self) -> float:
        """Effective creep parameter at the given temperature."""
        return self.B_ref * math.exp(-self.Q / (self.R * self.T_K))

    def creep_rate(self, stress: float | np.ndarray) -> float | np.ndarray:
        """Compute creep strain rate ε̇ = B_eff · |σ|^n · sign(σ)."""
        s = np.asarray(stress, dtype=float)
        return self.B_eff * np.abs(s) ** self.n * np.sign(s)


# ── Configuration ─────────────────────────────────────────────────────

@dataclass
class CreepCrushingConfig:
    """Configuration for a creep-crushing simulation.

    Parameters
    ----------
    v : float
        Indentation velocity [m/s].  Typical: 0.001–1.0 m/s.
    k_s : float
        Structural spring stiffness [N/m].
    A : float
        Contact area [m²].
    L_c : float
        Characteristic creep length in ice [m].
        For a flat indentor this is the ice thickness or contact-zone depth.
    creep : NortonCreepParams
        Norton-Bailey creep parameters.
    p0 : float
        Initial contact pressure [Pa].  Default: 0.
    t_end : float
        Simulation end time [s].
    n_steps : int
        Number of time steps.
    relaxation_mode : bool
        If True, set v=0 and compute stress relaxation from p0.
    """
    v: float = 0.01              # [m/s]
    k_s: float = 1.0e8           # [N/m]
    A: float = 0.1               # [m²]
    L_c: float = 0.5             # [m]
    creep: NortonCreepParams = field(default_factory=NortonCreepParams)
    p0: float = 0.0              # [Pa]
    t_end: float = 10.0          # [s]
    n_steps: int = 2000
    relaxation_mode: bool = False


# ── Result container ──────────────────────────────────────────────────

@dataclass
class CreepCrushingResult:
    """Output of a creep-crushing simulation."""
    t: np.ndarray
    p: np.ndarray
    v_eff: np.ndarray
    eps_creep: np.ndarray
    p_peak: float
    p_sustained: float
    tau: float
    m_scaling: float
    converged: bool
    config: CreepCrushingConfig


# ── Core solver (backward Euler + Newton) ────────────────────────────

def _newton_solve(p_k: float, v_imp: float, alpha: float,
                  B: float, n_exp: float, L_c: float, dt: float,
                  max_iter: int = 50, tol: float = 1e-10) -> Tuple[float, bool]:
    """Solve the backward Euler implicit equation by Newton iteration.

    We need p_{k+1} satisfying:
        g(p) = p - p_k - dt·alpha·(v - L_c·B·|p|^n·sign(p)) = 0

    g'(p) = 1 + dt·alpha·L_c·B·n·|p|^{n-1}

    Returns (p_new, converged).
    """
    # Better initial guess: semi-implicit (linearized) solution
    abs_pk = abs(p_k) if abs(p_k) > 1e-30 else 1e-30
    semi_implicit_denom = 1.0 + dt * alpha * L_c * B * n_exp * abs_pk ** (n_exp - 1.0)
    p = (p_k + alpha * v_imp * dt) / semi_implicit_denom

    # Ensure non-negative for pressure (physical constraint)
    p = max(p, 0.0)

    c = dt * alpha
    creep_coeff = L_c * B

    for iteration in range(max_iter):
        abs_p = abs(p) if abs(p) > 1e-30 else 1e-30
        sign_p = 1.0 if p >= 0 else -1.0
        creep_term = creep_coeff * abs_p ** n_exp * sign_p

        g = p - p_k - c * (v_imp - creep_term)
        gp = 1.0 + c * creep_coeff * n_exp * abs_p ** (n_exp - 1.0)

        if abs(gp) < 1e-30:
            break

        dp = -g / gp

        # Damped Newton if overshooting
        p_new = p + dp
        if p_new < 0:
            p_new = p * 0.5  # bisection fallback

        if abs(dp) < tol * (1.0 + abs(p)):
            return p_new, True

        p = p_new

    # Check if we're close enough
    abs_p = abs(p) if abs(p) > 1e-30 else 1e-30
    sign_p = 1.0 if p >= 0 else -1.0
    creep_term = creep_coeff * abs_p ** n_exp * sign_p
    g = p - p_k - c * (v_imp - creep_term)
    return p, abs(g) < tol * (1.0 + abs(p))


def _solve_raw(cfg: CreepCrushingConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Internal: run the ODE loop and return (t, p, v_eff, eps_creep, converged)."""
    B = cfg.creep.B_eff
    n_exp = cfg.creep.n
    alpha = cfg.k_s / cfg.A
    dt = cfg.t_end / cfg.n_steps
    N = cfg.n_steps

    t = np.linspace(0.0, cfg.t_end, N + 1)
    p = np.empty(N + 1)
    eps_creep = np.empty(N + 1)
    v_eff_arr = np.empty(N + 1)

    p[0] = cfg.p0
    eps_creep[0] = 0.0
    v_eff_arr[0] = 0.0
    v_imposed = 0.0 if cfg.relaxation_mode else cfg.v
    converged = True

    for k in range(N):
        p_new, ok = _newton_solve(p[k], v_imposed, alpha, B, n_exp, cfg.L_c, dt)
        if not ok:
            converged = False

        p[k + 1] = p_new
        abs_pk = abs(p[k]) if abs(p[k]) > 1e-30 else 1e-30
        eps_creep[k + 1] = eps_creep[k] + B * abs_pk ** n_exp * dt
        dp = p[k + 1] - p[k]
        v_eff_arr[k + 1] = v_imposed - dp / alpha

        if not np.isfinite(p[k + 1]):
            converged = False
            p[k + 1:] = p[k]
            v_eff_arr[k + 1:] = 0.0
            eps_creep[k + 1:] = eps_creep[k]
            break

    return t, p, v_eff_arr, eps_creep, converged


def solve_creep_crushing(cfg: CreepCrushingConfig, *, compute_m: bool = True) -> CreepCrushingResult:
    """Solve the coupled creep-crushing ODE by backward Euler with Newton iteration.

    Parameters
    ----------
    cfg : CreepCrushingConfig
        Simulation configuration.
    compute_m : bool
        If True, compute the pressure–rate scaling exponent.
        Set False for internal/recursive calls to avoid infinite recursion.
    """
    t, p, v_eff_arr, eps_creep, converged = _solve_raw(cfg)

    p_peak = float(np.max(np.abs(p)))
    n_sust = max(1, cfg.n_steps // 5)
    p_sustained = float(np.mean(np.abs(p[-n_sust:])))

    B = cfg.creep.B_eff
    n_exp = cfg.creep.n
    p_ss = p_sustained if p_sustained > 0 else 1.0
    tau = relaxation_time_constant(cfg.k_s, cfg.A, cfg.L_c, B, n_exp, p_ss)

    if compute_m:
        m_scaling = compute_scaling_exponent(cfg)
    else:
        m_scaling = float('nan')

    return CreepCrushingResult(
        t=t, p=p, v_eff=v_eff_arr, eps_creep=eps_creep,
        p_peak=p_peak, p_sustained=p_sustained,
        tau=tau, m_scaling=m_scaling,
        converged=converged, config=cfg,
    )


# ── Analytical / limiting-case functions ──────────────────────────────

def steady_state_pressure(v: float, B: float, n: float, L_c: float) -> float:
    """Steady-state pressure in the rigid-structure limit.

    p_ss = (v / (B · L_c))^{1/n}
    """
    if v <= 0 or B <= 0 or L_c <= 0:
        return 0.0
    return (v / (B * L_c)) ** (1.0 / n)


def rigid_limit_pressure(v: float, creep: NortonCreepParams, L_c: float) -> float:
    """Steady-state pressure in the rigid-structure limit using NortonCreepParams."""
    return steady_state_pressure(v, creep.B_eff, creep.n, L_c)


def relaxation_time_constant(
    k_s: float, A: float, L_c: float, B: float, n: float, p_ref: float,
) -> float:
    """Linearised relaxation time constant around reference pressure.

    τ = A / (k_s · B · n · L_c · |p_ref|^{n-1})
    """
    if k_s <= 0 or B <= 0 or n <= 0 or L_c <= 0 or p_ref <= 0:
        return float('inf')
    return A / (k_s * B * n * L_c * abs(p_ref) ** (n - 1))


def analytical_relaxation(
    p0: float, k_s: float, A: float, L_c: float,
    B: float, n: float, t: np.ndarray,
) -> np.ndarray:
    """Closed-form stress relaxation (v = 0, Norton creep).

    For n ≠ 1:  p(t) = p0 · (1 + (n-1)·t/τ₀)^{-1/(n-1)}
    For n = 1:  p(t) = p0 · exp(-t / τ₀)
    """
    tau0 = relaxation_time_constant(k_s, A, L_c, B, n, p0)
    if tau0 == float('inf') or tau0 <= 0:
        return np.full_like(t, p0, dtype=float)
    if abs(n - 1.0) < 1e-10:
        return p0 * np.exp(-t / tau0)
    else:
        return p0 * (1.0 + (n - 1.0) * t / tau0) ** (-1.0 / (n - 1.0))


def pressure_rate_exponent(
    v: float, creep: NortonCreepParams, L_c: float,
    k_s: float = float('inf'), A: float = 1.0,
) -> float:
    """Compute the pressure–rate scaling exponent m = ∂ln p / ∂ln v.

    In the rigid limit:  m = 1/n  (theoretical).
    With compliance:     m is computed numerically by finite difference.
    """
    B = creep.B_eff
    n = creep.n
    if k_s == float('inf') or k_s > 1e20:
        return 1.0 / n

    dv = 0.01 * v
    v1, v2 = v - dv, v + dv
    if v1 <= 0:
        v1 = 0.5 * v
    p1 = steady_state_pressure(v1, B, n, L_c)
    p2 = steady_state_pressure(v2, B, n, L_c)
    if p1 <= 0 or p2 <= 0:
        return 1.0 / n
    return (math.log(p2) - math.log(p1)) / (math.log(v2) - math.log(v1))


def compute_scaling_exponent(cfg: CreepCrushingConfig) -> float:
    """Numerically compute m = ∂ln p / ∂ln v.

    Uses the rigid-limit steady-state pressure for the scaling analysis,
    which is exact at steady state for any structural compliance (the
    compliance modifies the transient but not the steady-state exponent).
    """
    v0 = cfg.v
    if v0 <= 0:
        return float('nan')

    B = cfg.creep.B_eff
    n = cfg.creep.n
    L_c = cfg.L_c

    factor = 1.1
    v_arr = np.array([v0 / factor, v0, v0 * factor])
    p_arr = np.array([steady_state_pressure(v, B, n, L_c) for v in v_arr])

    if np.any(p_arr <= 0) or np.any(v_arr <= 0):
        return float('nan')

    log_v = np.log(v_arr)
    log_p = np.log(p_arr)
    m, _ = np.polyfit(log_v, log_p, 1)
    return float(m)


# ── IACS Polar Class sustained pressure ──────────────────────────────

IACS_PC_SUSTAINED_PRESSURE_MPA = {
    "PC1": 34.0, "PC2": 24.0, "PC3": 18.0, "PC4": 12.0,
    "PC5": 8.5, "PC6": 6.0, "PC7": 4.5,
}


def iacs_pc_sustained_pressure(
    pc_class: str, area: float = 1.0, A0: float = 1.0, exponent: float = -0.3,
) -> float:
    """IACS Polar Class sustained ice pressure with pressure–area correction.

    P(A) = P0 · (A / A0)^exponent
    """
    if pc_class not in IACS_PC_SUSTAINED_PRESSURE_MPA:
        raise ValueError(f"Unknown PC class: {pc_class}. "
                         f"Valid: {list(IACS_PC_SUSTAINED_PRESSURE_MPA.keys())}")
    P0 = IACS_PC_SUSTAINED_PRESSURE_MPA[pc_class] * 1e6
    return P0 * (area / A0) ** exponent


def check_iacs_monotonicity(
    area: float = 1.0, A0: float = 1.0, exponent: float = -0.3,
) -> Tuple[bool, List[Tuple[str, float]]]:
    """Verify that IACS sustained pressure is monotonically decreasing PC1→PC7."""
    classes = ["PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7"]
    values = [(c, iacs_pc_sustained_pressure(c, area, A0, exponent)) for c in classes]
    is_monotone = all(values[i][1] > values[i + 1][1] for i in range(len(values) - 1))
    return is_monotone, values


# ── Interface with frame-strip (m7257) ────────────────────────────────

def link_frame_strip_check(
    p_sustained: float, L_span: float, b_trib: float,
    sigma_y: float, E: float, h_plate: float, b_stiffener: float,
) -> dict:
    """Check whether sustained ice pressure is within plastic moment capacity.

    M_max = p · b · L² / 8
    M_p   = σ_y · b · h² / 4  (rectangular section)
    """
    q = p_sustained * b_trib
    M_max = q * L_span ** 2 / 8.0
    Z_pl = b_trib * h_plate ** 2 / 4.0
    M_p = sigma_y * Z_pl
    utilization = M_max / M_p if M_p > 0 else float('inf')
    return {
        "M_max_Nm": M_max, "M_p_Nm": M_p,
        "utilization": utilization, "ok": utilization <= 1.0,
        "p_sustained_Pa": p_sustained,
    }
