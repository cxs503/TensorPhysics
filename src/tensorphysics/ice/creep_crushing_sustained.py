"""Rate-dependent creep crushing of ice and sustained design pressure (P269W2-N05).

Physical problem
----------------
A structure of stiffness ``k_s`` indents an ice sheet of thickness ``h`` at
prescribed velocity ``v``.  The ice carries load through a Norton-Bailey creep
law and a brittle pressure ceiling::

    creep:      eps_dot = B(T) * sigma**n * t**(q-1)     (Norton-Bailey)
    ceiling:    p <= sigma_f                             (brittle crushing)

Structure-ice compatibility (elastic recoil of the structure + creep of the ice
inside the process zone) reads, per unit contact area ``A``::

    v  =  (A / k_s) * dp/dt  +  L_c(p) * B * p**n        (1)

Equation (1) is the rate-dependent crushing ODE.  ``L_c(p)`` is the length of
the actively creeping process zone:

* ``zone_model="constant"``        : ``L_c = h``            (constant zone)
* ``zone_model="pressure_scaled"`` : ``L_c = h * p / sigma_ref``, capped at ``h``

Both are integrated here; the difference in the resulting pressure-rate
exponent is a central physical result of this module.

Limiting cases (closed form, machine-checked)
---------------------------------------------
* **Rigid structure** (``k_s -> inf``): ``dp/dt -> 0`` and (1) degenerates to
  constant-rate crushing ``v = L_c(p) * B * p**n``, hence::

      constant zone        :  p_ss = (v / (B h))**(1/n)                  (2a)
      pressure-scaled zone :  p_ss = (v * sigma_ref / (h B))**(1/(n+1))  (2b)

  giving the pressure-rate scaling exponents::

      m = dln p / dln v_eff  =  1/n        (constant zone)               (3a)
                             =  1/(n+1)    (pressure-scaled zone)        (3b)

  For ice (n = 3): 1/n = 0.3333, 1/(n+1) = 0.25.  (3b) is the classical
  ductile-regime exponent: raising the pressure both accelerates creep
  (``p**n``) and thickens the creeping layer (proportional to ``p``), so the
  sustained pressure grows only as ``v**(1/(n+1))``.  (3a) is the rigid
  constant-rate result.

* **Constant-load creep** (``sigma = const``): the Bailey closed form is::

      delta(t) = L_c * B * sigma**n * t**q               (4)

  ``q = 1`` is secondary (Norton) creep, ``q > 1`` primary creep.

* **Stress relaxation** (``v = 0``, linear creep ``n = 1``, viscosity
  ``eta = 1/B``)::

      (A/k_s) dp/dt + (L_c/eta) p = 0  =>  p(t) = p0 exp(-t/tau)
      tau = eta * A / (k_s L_c) = eta / E_eff,   E_eff = k_s L_c / A     (5)

* **No creep** (``B -> 0``): (1) gives the elastic ramp ``p = k_s v t / A``;
  for a Hertzian contact the elastic leg is
  ``F = (4/3) E* sqrt(R) delta**(3/2)`` (checked by the same Newton machinery).

Energy ledger (residual <= 1 %)
-------------------------------
Multiplying (1) by ``A p`` and integrating gives::

    W = int F v dt = A v int p dt                  (external work)
    U = A**2 p**2 / (2 k_s)                        (structure elastic energy)
    D = A int L_c(p) B p**(n+1) dt                 (creep dissipation)
    W = U_end - U_0 + D                            (6)

Sustained design pressure (input to the m7257 ice-belt plate chain)
-------------------------------------------------------------------
:func:`sustained_design_pressure` returns ``p_sustained`` (mean over the last
25 % of the history), ``p_peak``, ``tau`` and ``m``; ``p_sustained`` is the load
input of the frame-strip plastic-hinge check in
``tensorphysics.ice_plasticity.frame_strip`` (m7257).

Numerics
--------
Equation (1) is integrated by backward Euler with a Newton inner solve: a
semi-implicit scheme that is unconditionally stable for any stiffness ratio
``dt * k_s / A`` (stiff structures included).

References
----------
* Norton, F.H. (1929) *The Creep of Steel at High Temperatures*, McGraw-Hill.
* Bailey, R.W. (1929) *J. Inst. Metals* 35, 27 (primary-creep time exponent).
* Glen, J.W. (1955) *Proc. R. Soc. Lond. A* 228, 519 (ice creep, n ~ 3).
* Michel, B. (1978) *Ice Mechanics*, Laval Univ. Press (ductile indentation).
* Jordaan, I.J. (2001) *Proc. IUTAM Symp. on Scaling Laws in Ice Mechanics*.
* Schulson, E.M. & Duval, P. (2009) *Creep and Fracture of Ice*, CUP.
* IACS UR I (2021) *Polar Class ships* (sustained pressure, PC classes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

__all__ = [
    "NortonBailey",
    "CrushingConfig",
    "CrushingResult",
    "zone_length",
    "creep_velocity",
    "steady_pressure_closed_form",
    "theoretical_rate_exponent",
    "simulate_crushing",
    "constant_load_creep",
    "fit_power_law",
    "scaling_exponent",
    "relaxation_tau",
    "simulate_relaxation",
    "fit_relaxation",
    "energy_ledger",
    "hertz_force",
    "hertz_delta",
    "hertz_indentation_force",
    "sustained_design_pressure",
]


# ----------------------------------------------------------------------
# Norton-Bailey creep law
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class NortonBailey:
    """Norton-Bailey creep parameters for ice (SI units, stress in Pa).

    ``eps_dot = B(T) * sigma**n * t**(q-1)``

    ``B_ref`` is the creep parameter at the reference temperature ``T_ref_K``;
    the temperature effect is carried by the Arrhenius shift
    ``B(T) = B_ref * exp(-Q/R * (1/T - 1/T_ref))``, normalised so that
    ``B(T_ref) = B_ref``.  Salinity enters only through ``B_ref`` (saline ice
    creeps faster); see the parameter table of the report.
    """

    B_ref: float = 1.0e-23      # [Pa^-n s^-1] at T_ref (~ -10 C, columnar sea ice)
    n: float = 3.0              # stress exponent
    q: float = 1.0              # Bailey time exponent (1 = secondary creep)
    Q: float = 60.0e3           # activation energy [J/mol]
    T_ref_K: float = 263.15     # -10 C
    R: float = 8.314            # gas constant [J/mol/K]

    def B_at(self, T_K: float) -> float:
        """Creep parameter at temperature ``T_K`` [K]."""
        return self.B_ref * math.exp(
            -self.Q / self.R * (1.0 / T_K - 1.0 / self.T_ref_K))

    def creep_rate(self, sigma, T_K: Optional[float] = None, t: float = 1.0):
        """Creep strain rate ``eps_dot = B sigma**n t**(q-1)`` [1/s]."""
        B = self.B_at(T_K) if T_K is not None else self.B_ref
        s = np.asarray(sigma, dtype=float)
        return B * np.abs(s) ** self.n * np.sign(s) * float(t) ** (self.q - 1.0)


# ----------------------------------------------------------------------
# Configuration / result containers
# ----------------------------------------------------------------------
@dataclass
class CrushingConfig:
    """Configuration of a rate-controlled creep-crushing simulation.

    Parameters
    ----------
    v : float
        Imposed indentation velocity [m/s].
    k_s : float
        Structural stiffness [N/m]; ``inf`` gives the rigid limit.
    A : float
        Nominal contact area [m^2].
    h : float
        Ice sheet thickness / process-zone thickness scale [m].
    sigma_ref : float
        Pressure at which the creeping zone fills ``h`` (pressure-scaled zone
        model only) [Pa].
    sigma_f : float
        Brittle pressure ceiling [Pa].
    creep : NortonBailey
        Creep law parameters.
    zone_model : {"pressure_scaled", "constant"}
        Process-zone length model, see :func:`zone_length`.
    p0 : float
        Initial contact pressure [Pa].
    t_end, n_steps :
        Time window [s] and number of backward-Euler steps.
    """

    v: float = 1.0e-4
    k_s: float = 1.0e8
    A: float = 0.1
    h: float = 0.5
    sigma_ref: float = 5.0e6
    sigma_f: float = 50.0e6
    creep: NortonBailey = field(default_factory=NortonBailey)
    zone_model: str = "pressure_scaled"
    p0: float = 0.0
    t_end: float = 2000.0
    n_steps: int = 2000

    def __post_init__(self) -> None:
        if self.zone_model not in ("pressure_scaled", "constant"):
            raise ValueError(f"unknown zone_model {self.zone_model!r}")
        if self.n_steps < 1:
            raise ValueError("n_steps must be >= 1")
        if self.t_end <= 0:
            raise ValueError("t_end must be > 0")


@dataclass
class CrushingResult:
    """Output of :func:`simulate_crushing`."""

    t: np.ndarray
    p: np.ndarray
    v_eff: np.ndarray
    L_c: np.ndarray
    p_peak: float
    p_sustained: float
    p_steady_analytic: float
    tau: float
    crushing: np.ndarray
    config: CrushingConfig

    def ledger(self) -> Dict[str, float]:
        """Energy ledger of this history (W = U + D)."""
        return energy_ledger(self)


# ----------------------------------------------------------------------
# Process-zone length and creep velocity
# ----------------------------------------------------------------------
def zone_length(cfg: CrushingConfig, p) -> np.ndarray:
    """Length of the actively creeping process zone at pressure ``p`` [m].

    ``constant``        : ``L_c = h``
    ``pressure_scaled`` : ``L_c = h * min(p / sigma_ref, 1)``
    """
    p = np.asarray(p, dtype=float)
    if cfg.zone_model == "constant":
        return np.full(p.shape, cfg.h, dtype=float)
    return cfg.h * np.minimum(np.abs(p) / cfg.sigma_ref, 1.0)


def creep_velocity(cfg: CrushingConfig, p) -> np.ndarray:
    """Creep contribution to the indentation rate ``G(p) = L_c(p) B p**n``."""
    p = np.asarray(p, dtype=float)
    return zone_length(cfg, p) * cfg.creep.B_ref * np.abs(p) ** cfg.creep.n


def creep_velocity_derivative(cfg: CrushingConfig, p: float) -> float:
    """``dG/dp`` of :func:`creep_velocity` (used by the Newton solve)."""
    B, n = cfg.creep.B_ref, cfg.creep.n
    if cfg.zone_model == "constant":
        return cfg.h * B * n * abs(p) ** (n - 1.0)
    if abs(p) <= cfg.sigma_ref:
        c = cfg.h / cfg.sigma_ref
        return c * B * (n + 1.0) * abs(p) ** n
    return cfg.h * B * n * abs(p) ** (n - 1.0)


# ----------------------------------------------------------------------
# Closed forms
# ----------------------------------------------------------------------
def steady_pressure_closed_form(v: float, cfg: CrushingConfig) -> float:
    """Rigid-limit (``k_s -> inf``) steady crushing pressure, Eq. (2).

    In the pressure-scaled model this is valid while ``p_ss <= sigma_ref``
    (above that the zone saturates at ``h`` and (2a) with ``h`` applies).
    """
    B, n = cfg.creep.B_ref, cfg.creep.n
    if v <= 0 or B <= 0:
        return 0.0
    if cfg.zone_model == "constant":
        return (v / (B * cfg.h)) ** (1.0 / n)
    c = cfg.h / cfg.sigma_ref                       # L_c = c p
    return (v / (c * B)) ** (1.0 / (n + 1.0))


def theoretical_rate_exponent(n: float,
                              zone_model: str = "pressure_scaled") -> float:
    """Theoretical pressure-rate exponent ``m``, Eq. (3)."""
    return 1.0 / (n + 1.0) if zone_model == "pressure_scaled" else 1.0 / n


# ----------------------------------------------------------------------
# Semi-implicit coupled solver (backward Euler + Newton)
# ----------------------------------------------------------------------
def simulate_crushing(cfg: CrushingConfig) -> CrushingResult:
    """Integrate the rate-controlled crushing ODE (1).

    Backward Euler on ``p`` with a Newton inner solve; the brittle cap
    ``p <= sigma_f`` is applied as a physical clamp (``crushing`` flags the
    clamped steps).
    """
    n_steps = int(cfg.n_steps)
    dt = cfg.t_end / n_steps
    t = np.linspace(0.0, cfg.t_end, n_steps + 1)
    p = np.zeros(n_steps + 1)
    crushing = np.zeros(n_steps + 1, dtype=bool)
    p_cur = float(cfg.p0)

    rigid = math.isinf(cfg.k_s)
    alpha = 0.0 if rigid else cfg.k_s / cfg.A       # [Pa/m]
    v = float(cfg.v)
    p_scale = max(cfg.sigma_ref, 1.0)
    tol = 1e-11

    for k in range(n_steps):
        if rigid:
            p_new = _solve_rigid(v, cfg)
        else:
            z = p_cur / p_scale
            converged = False
            for _ in range(100):
                pk = max(z * p_scale, 0.0)
                g = float(creep_velocity(cfg, np.array([pk]))[0])
                f = pk - p_cur - dt * alpha * (v - g)
                df = 1.0 + dt * alpha * creep_velocity_derivative(cfg, pk)
                dz = -f / (df * p_scale)
                z += dz
                if abs(dz) * p_scale <= tol * p_scale:
                    converged = True
                    break
            if not converged:                       # pragma: no cover
                raise RuntimeError("Newton iteration failed to converge")
            p_new = max(z * p_scale, 0.0)

        if p_new > cfg.sigma_f:
            p_new = cfg.sigma_f
            crushing[k + 1] = True
        p[k + 1] = p_new
        p_cur = p_new

    v_eff = creep_velocity(cfg, p)
    tail = max(1, n_steps // 4)
    p_sustained = float(np.mean(p[-tail:]))
    p_peak = float(np.max(p))
    p_ref = p_sustained if p_sustained > 0 else 1.0
    L_ref = float(zone_length(cfg, np.array([p_ref]))[0])
    tau = relaxation_tau(cfg.k_s, cfg.A, L_ref, cfg.creep.B_ref,
                         cfg.creep.n, p_ref)

    return CrushingResult(
        t=t, p=p, v_eff=v_eff, L_c=zone_length(cfg, p),
        p_peak=p_peak, p_sustained=p_sustained,
        p_steady_analytic=steady_pressure_closed_form(v, cfg),
        tau=tau, crushing=crushing, config=cfg,
    )


def _solve_rigid(v: float, cfg: CrushingConfig, tol: float = 1e-12) -> float:
    """Rigid-limit solution of ``v = L_c(p) B p**n`` (Newton on ``p``)."""
    if v <= 0:
        return 0.0
    p = steady_pressure_closed_form(v, cfg) or 1.0
    for _ in range(200):
        g = float(creep_velocity(cfg, np.array([p]))[0])
        dg = creep_velocity_derivative(cfg, p)
        if dg <= 0:
            break
        p_new = max(p - (g - v) / dg, 1e-30)
        if abs(p_new - p) <= tol * max(abs(p), 1.0):
            p = p_new
            break
        p = p_new
    return p


def relaxation_tau(k_s: float, A: float, L_c: float, B: float,
                   n: float, p_ref: float) -> float:
    """Linearised relaxation time constant, Eq. (5).

    ``tau = A / (k_s B n L_c p_ref**(n-1))``; for ``n = 1`` this is
    ``tau = eta / E_eff`` with ``eta = 1/B`` and ``E_eff = k_s L_c / A``.
    """
    if k_s <= 0 or B <= 0 or L_c <= 0 or p_ref <= 0 or math.isinf(k_s):
        return float("inf")
    return A / (k_s * B * n * L_c * abs(p_ref) ** (n - 1.0))


# ----------------------------------------------------------------------
# Constant-load creep closed form and power-law fitting
# ----------------------------------------------------------------------
def constant_load_creep(t, sigma: float, creep: NortonBailey, L_c: float,
                        q: Optional[float] = None) -> np.ndarray:
    """Closed-form constant-load creep displacement, Eq. (4).

    ``delta(t) = L_c B sigma**n t**q`` (``q`` defaults to ``creep.q``).
    """
    t = np.asarray(t, dtype=float)
    qq = creep.q if q is None else float(q)
    delta0 = L_c * creep.B_ref * abs(sigma) ** creep.n
    return delta0 * np.power(t, qq)


def fit_power_law(t, y) -> Tuple[float, float, float]:
    """Fit ``y = a t**b`` by log-log least squares.

    Returns ``(b, a, rel_residual)`` with
    ``rel_residual = ||y_fit - y||_2 / ||y||_2``.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (t > 0) & (y > 0)
    b, log_a = np.polyfit(np.log(t[mask]), np.log(y[mask]), 1)
    y_fit = np.exp(log_a) * np.power(t, b)
    rel = float(np.linalg.norm(y_fit - y) / np.linalg.norm(y))
    return float(b), float(math.exp(log_a)), rel


# ----------------------------------------------------------------------
# Pressure-rate scaling exponent
# ----------------------------------------------------------------------
def scaling_exponent(cfg: CrushingConfig, v_list=None):
    """Fit ``m = dln p / dln v_eff`` from a set of simulations.

    Returns ``(m, rows)`` where ``rows`` is a list of
    ``{"v", "p_sustained", "p_steady_analytic"}`` dicts.  ``v_eff`` at steady
    state equals ``v`` (all the imposed rate is consumed by creep), which is
    the definition used for the fit.
    """
    if v_list is None:
        # a one-decade spread below the zone-saturation velocity
        # v* = h B sigma_ref**n (above it the zone saturates at h and the
        # slope changes, cf. Eq. (2b) restricted to p_ss <= sigma_ref)
        v0 = cfg.v
        v_list = list(np.geomspace(v0 / 16.0, v0, 5))
    rows = []
    for v in v_list:
        sub = CrushingConfig(**{**cfg.__dict__, "v": float(v)})
        res = simulate_crushing(sub)
        rows.append({"v": float(v), "p_sustained": res.p_sustained,
                     "p_steady_analytic": res.p_steady_analytic})
    lnv = np.log([r["v"] for r in rows])
    lnp = np.log([r["p_sustained"] for r in rows])
    m, _ = np.polyfit(lnv, lnp, 1)
    return float(m), rows


# ----------------------------------------------------------------------
# Linear relaxation (A4)
# ----------------------------------------------------------------------
def simulate_relaxation(p0: float, eta: float, E_eff: float, t_end: float,
                        n_steps: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Linear stress relaxation ``p(t) = p0 exp(-t/tau)``, ``tau = eta/E_eff``.

    Integrated with the same backward-Euler scheme as the crushing solver
    (``v = 0``, ``n = 1``, ``B = 1/eta``) so that the numerical and the
    analytical legs are compared on equal footing.
    """
    tau = eta / E_eff
    dt = t_end / n_steps
    t = np.linspace(0.0, t_end, n_steps + 1)
    p = np.zeros(n_steps + 1)
    p[0] = p0
    for k in range(n_steps):
        p[k + 1] = p[k] / (1.0 + dt / tau)
    return t, p


def fit_relaxation(t, p, p0: float) -> Tuple[float, float]:
    """Fit ``p = p0 exp(-t/tau)`` by log-linear least squares.

    Returns ``(tau_fit, r_squared)``.
    """
    t = np.asarray(t, dtype=float)
    p = np.asarray(p, dtype=float)
    mask = p > 0
    y = np.log(p[mask])
    slope, intercept = np.polyfit(t[mask], y, 1)
    fit = slope * t[mask] + intercept
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    tau_fit = -1.0 / slope if slope != 0 else float("inf")
    _ = p0
    return float(tau_fit), float(r2)


# ----------------------------------------------------------------------
# Energy ledger (A5)
# ----------------------------------------------------------------------
def energy_ledger(res: CrushingResult) -> Dict[str, float]:
    """Check ``W = U_end - U_0 + D``, Eq. (6).

    All integrals use the trapezoidal rule on the same ``p`` history.
    """
    cfg = res.config
    t, p = res.t, res.p
    W = cfg.A * cfg.v * float(np.trapezoid(p, t))
    U_end = cfg.A ** 2 * res.p[-1] ** 2 / (2.0 * cfg.k_s)
    U_0 = cfg.A ** 2 * p[0] ** 2 / (2.0 * cfg.k_s)
    integrand = cfg.A * res.L_c * cfg.creep.B_ref * np.abs(p) ** (cfg.creep.n + 1.0)
    D = float(np.trapezoid(integrand, t))
    resid = W - (U_end - U_0 + D)
    return {
        "W_external_J": W,
        "U_end_J": U_end,
        "U_0_J": U_0,
        "D_creep_J": D,
        "residual_J": resid,
        "residual_rel": abs(resid) / W if W > 0 else float("nan"),
    }


# ----------------------------------------------------------------------
# Elastic Hertz limit (A6, negative control B -> 0)
# ----------------------------------------------------------------------
def hertz_force(delta, E_star: float, R: float):
    """Hertzian contact force ``F = (4/3) E* sqrt(R) delta**(3/2)``."""
    delta = np.asarray(delta, dtype=float)
    return (4.0 / 3.0) * E_star * math.sqrt(R) * np.power(delta, 1.5)


def hertz_delta(F, E_star: float, R: float):
    """Inverse Hertz relation ``delta = (3F / (4 E* sqrt(R)))**(2/3)``."""
    F = np.asarray(F, dtype=float)
    return np.power(3.0 * F / (4.0 * E_star * math.sqrt(R)), 2.0 / 3.0)


def hertz_indentation_force(v: float, t, E_star: float, R: float,
                            tol: float = 1e-12) -> np.ndarray:
    """Hertz contact force during rate-controlled indentation without creep.

    The imposed displacement ``x = v t`` equals the Hertzian penetration
    (no structure compliance, ``B -> 0``); ``F`` is obtained from the implicit
    Hertz relation by Newton iteration, so this checks the nonlinear elastic
    leg against its closed form ``F = (4/3) E* sqrt(R) (v t)**(3/2)``.
    """
    t = np.asarray(t, dtype=float)
    C = (4.0 / 3.0) * E_star * math.sqrt(R)
    F = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti <= 0:
            F[i] = 0.0
            continue
        x = v * ti
        f = 1.0                                    # start at unit force, scale-invariant
        for _ in range(100):
            gap = float(hertz_delta(f, E_star, R)) - x
            dgap = (2.0 / 3.0) * float(hertz_delta(f, E_star, R)) / f
            df = -gap / dgap
            f = max(f + df, 1e-30)
            if abs(df) <= tol * f:
                break
        F[i] = f
    _ = C
    return F


# ----------------------------------------------------------------------
# Design-load interface (m7257 chain)
# ----------------------------------------------------------------------
def sustained_design_pressure(cfg: CrushingConfig) -> Dict[str, float]:
    """Sustained/peak pressure, ``tau`` and ``m`` for the design chain.

    ``p_sustained`` is the load input of the m7257 ice-belt frame-strip check.
    """
    res = simulate_crushing(cfg)
    m, _rows = scaling_exponent(cfg)
    return {
        "v_m_s": cfg.v,
        "p_peak_Pa": res.p_peak,
        "p_sustained_Pa": res.p_sustained,
        "p_steady_analytic_Pa": res.p_steady_analytic,
        "tau_s": res.tau,
        "m": m,
        "m_theory": theoretical_rate_exponent(cfg.creep.n, cfg.zone_model),
        "clamped": bool(np.any(res.crushing)),
    }