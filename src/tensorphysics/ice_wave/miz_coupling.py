"""Marginal Ice Zone wave-ice-ship three-body coupling model.

Physical model (p247w1-N06)
---------------------------
This module implements a coupled model for wave propagation, ice breaking,
and ship motion response in the Marginal Ice Zone (MIZ). The three subsystems
are:

(a) **Wave attenuation in ice field**:
    Wave energy decays as E(x) = E_0 * exp(-2*alpha*x), where the attenuation
    coefficient alpha = alpha_scatter + alpha_absorb is determined by the
    effective Kelvin-Voigt medium properties of the fragmented ice field.

(b) **Ice bending failure criterion**:
    Wave-induced bending stress in the ice:
        sigma_ice = (3/4) * rho_w * g * a_w * k^2 * h_ice^2
    When sigma_ice > sigma_f (flexural strength), the ice breaks into floes.

(c) **Characteristic floe size** (Squire 2018):
        a_floe = sqrt(sigma_f / (rho_w * g)) * (1/k)
    where k is the wavenumber of the dominant wave.

(d) **Floe size distribution**:
    P(a) ~ a^(-gamma), gamma in [2.5, 3.0]
    The power-law exponent gamma is determined by the self-similarity of the
    fragmentation process (Squire, Rothiers, etc.).

(e) **Ship motion response**:
    In the MIZ, the ship experiences both wave excitation and random ice
    impulse forces:
        M * x_ddot + C * x_dot + K * x = F_wave + F_ice
    Wave excitation is computed via strip theory; ice forces are modelled
    as random Poisson-distributed impulses.

(f) **Coupled solution**:
    Given incident wave spectrum S(omega) and initial ice field (C_floe, h_ice),
    solve: wave attenuation -> characteristic floe size -> ship motion ->
    ship motion feedback to ice field (ship breaks ice).

References
----------
- Squire, V.A. (2018). "The marginal ice zone: a natural laboratory for wave-ice
  interaction studies." Annals of Glaciology.
- Kohout, A.L. et al. (2014). "Storm-induced sea-ice breakup." Geophys. Res. Lett.
- Meylan, M.H. et al. (2018). "A review of wave-ice interaction models."
- Lindqvist, G. (1989). "A direct method of ice resistance prediction."
"""

from __future__ import annotations

from math import isfinite, pi, sqrt, log, exp
from typing import Any, Optional

import numpy as np

from .flexural import (
    FlexuralWaveError,
    flexural_rigidity,
    flexural_gravity_wavenumber,
    group_speed,
)
from .miz import (
    MIZError,
    kelvin_voigt_wavenumber,
    attenuation_from_complex_wavenumber,
    effective_medium_properties,
    exponential_decay_profile,
)

# Physical constants
GRAVITY = 9.81          # m/s^2
RHO_WATER = 1025.0      # kg/m^3
RHO_ICE = 917.0         # kg/m^3


class MIZCouplingError(ValueError):
    """Invalid input for the MIZ coupling model."""


# ============================================================================
# (b) Ice bending failure criterion
# ============================================================================

def ice_bending_stress(
    wave_amplitude: float,
    wavenumber: float,
    ice_thickness: float,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Compute wave-induced bending stress in ice.

    sigma_ice = (3/4) * rho_w * g * a_w * k^2 * h_ice^2

    This is derived from thin plate bending theory for an ice sheet
    subjected to a sinusoidal wave of amplitude a_w and wavenumber k.

    Parameters
    ----------
    wave_amplitude : float
        Wave amplitude a_w [m].
    wavenumber : float
        Wavenumber k [rad/m].
    ice_thickness : float
        Ice thickness h_ice [m].
    rho_w : float
        Water density [kg/m^3].
    g : float
        Gravitational acceleration [m/s^2].

    Returns
    -------
    float
        Bending stress sigma_ice [Pa].
    """
    a_w = float(wave_amplitude)
    k = float(wavenumber)
    h_i = float(ice_thickness)

    if not isfinite(a_w) or a_w < 0.0:
        raise MIZCouplingError(f"wave_amplitude must be finite and non-negative, got {a_w!r}")
    if not isfinite(k) or k <= 0.0:
        raise MIZCouplingError(f"wavenumber must be finite and positive, got {k!r}")
    if not isfinite(h_i) or h_i <= 0.0:
        raise MIZCouplingError(f"ice_thickness must be finite and positive, got {h_i!r}")

    return 0.75 * rho_w * g * a_w * k**2 * h_i**2


def ice_breaking_criterion(
    wave_amplitude: float,
    wavenumber: float,
    ice_thickness: float,
    flexural_strength: float = 0.5e6,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> dict[str, Any]:
    """Check whether ice breaks under wave loading.

    Parameters
    ----------
    wave_amplitude : float
        Wave amplitude [m].
    wavenumber : float
        Wavenumber [rad/m].
    ice_thickness : float
        Ice thickness [m].
    flexural_strength : float
        Ice flexural strength sigma_f [Pa]. Default 0.5 MPa.
    rho_w, g : float
        Physical constants.

    Returns
    -------
    dict
        Contains 'sigma_ice', 'sigma_f', 'breaks' (bool), 'safety_factor'.
    """
    sigma_f = float(flexural_strength)
    if not isfinite(sigma_f) or sigma_f <= 0.0:
        raise MIZCouplingError(f"flexural_strength must be positive, got {sigma_f!r}")

    sigma_ice = ice_bending_stress(
        wave_amplitude, wavenumber, ice_thickness, rho_w=rho_w, g=g
    )
    breaks = sigma_ice > sigma_f
    safety_factor = sigma_f / sigma_ice if sigma_ice > 0 else float('inf')

    return {
        "sigma_ice": sigma_ice,
        "sigma_f": sigma_f,
        "breaks": bool(breaks),
        "safety_factor": float(safety_factor),
    }


# ============================================================================
# (c) Characteristic floe size (Squire theory)
# ============================================================================

def characteristic_floe_size(
    wavenumber: float,
    flexural_strength: float = 0.5e6,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Characteristic floe size from Squire (2018) theory.

    a_floe = sqrt(sigma_f / (rho_w * g)) * (1/k)

    This gives the characteristic length scale of ice floes produced
    by wave-induced breaking.

    Parameters
    ----------
    wavenumber : float
        Dominant wavenumber k [rad/m].
    flexural_strength : float
        Ice flexural strength sigma_f [Pa].
    rho_w, g : float
        Physical constants.

    Returns
    -------
    float
        Characteristic floe size a_floe [m].
    """
    k = float(wavenumber)
    sigma_f = float(flexural_strength)

    if not isfinite(k) or k <= 0.0:
        raise MIZCouplingError(f"wavenumber must be positive, got {k!r}")
    if not isfinite(sigma_f) or sigma_f <= 0.0:
        raise MIZCouplingError(f"flexural_strength must be positive, got {sigma_f!r}")

    return sqrt(sigma_f / (rho_w * g)) / k


# ============================================================================
# (d) Floe size distribution
# ============================================================================

def floe_size_distribution_pdf(
    sizes: np.ndarray,
    gamma: float = 2.7,
    a_min: float = 1.0,
    a_max: float = 500.0,
) -> np.ndarray:
    """Power-law floe size distribution PDF.

    P(a) = C * a^(-gamma) for a in [a_min, a_max]

    where C is the normalisation constant.

    Parameters
    ----------
    sizes : array_like
        Floe sizes [m] at which to evaluate the PDF.
    gamma : float
        Power-law exponent (typically 2.5-3.0).
    a_min, a_max : float
        Minimum and maximum floe sizes [m].

    Returns
    -------
    np.ndarray
        PDF values at the given sizes.
    """
    a = np.asarray(sizes, dtype=float)
    gamma = float(gamma)
    a_min = float(a_min)
    a_max = float(a_max)

    if not isfinite(gamma) or gamma <= 1.0:
        raise MIZCouplingError(f"gamma must be > 1 for normalisability, got {gamma!r}")
    if a_min <= 0 or a_max <= a_min:
        raise MIZCouplingError(f"need 0 < a_min < a_max, got ({a_min}, {a_max})")

    # Normalisation: integral of a^(-gamma) from a_min to a_max
    if abs(gamma - 1.0) < 1e-12:
        C = 1.0 / log(a_max / a_min)
    else:
        C = (1.0 - gamma) / (a_max**(1.0 - gamma) - a_min**(1.0 - gamma))

    pdf = np.where(
        (a >= a_min) & (a <= a_max),
        C * a**(-gamma),
        0.0,
    )
    return pdf


def generate_floe_field(
    n_floes: int,
    gamma: float = 2.7,
    a_min: float = 1.0,
    a_max: float = 500.0,
    *,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random floe sizes from power-law distribution via inverse CDF.

    Parameters
    ----------
    n_floes : int
        Number of floes to generate.
    gamma : float
        Power-law exponent.
    a_min, a_max : float
        Size range [m].
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Array of floe sizes [m].
    """
    if n_floes < 1:
        raise MIZCouplingError(f"n_floes must be >= 1, got {n_floes}")

    rng = np.random.default_rng(seed)
    u = rng.uniform(size=n_floes)

    # Inverse CDF: a(u) = [a_min^(1-gamma) + u * (a_max^(1-gamma) - a_min^(1-gamma))]^(1/(1-gamma))
    exp = 1.0 - gamma
    a = (a_min**exp + u * (a_max**exp - a_min**exp))**(1.0 / exp)
    return a


def fit_floe_distribution_exponent(
    sizes: np.ndarray,
    a_min: float = 1.0,
    a_max: float = 500.0,
) -> dict[str, float]:
    """Fit power-law exponent gamma from observed floe sizes.

    Uses maximum likelihood estimation for a truncated power law:
        gamma_MLE = 1 + n / sum(ln(a_i / a_min))

    Parameters
    ----------
    sizes : array_like
        Observed floe sizes [m].
    a_min : float
        Minimum size threshold [m].

    Returns
    -------
    dict
        Contains 'gamma', 'n_samples', 'a_min_used'.
    """
    a = np.asarray(sizes, dtype=float)
    a = a[a >= a_min]
    n = a.size
    if n < 2:
        raise MIZCouplingError(f"need at least 2 samples above a_min, got {n}")

    gamma_mle = 1.0 + n / np.sum(np.log(a / a_min))
    return {
        "gamma": float(gamma_mle),
        "n_samples": int(n),
        "a_min_used": float(a_min),
    }


# ============================================================================
# (e) Ship motion response in MIZ
# ============================================================================

def wave_excitation_force(
    wave_amplitude: float,
    frequency: float,
    ship_beam: float,
    ship_draft: float,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Simplified strip-theory wave excitation force (heave).

    F_wave = rho_w * g * A_w * a_w * chi(omega)

    where A_w is the waterplane area per unit length (approximated as beam),
    and chi is a frequency-dependent response amplitude operator.

    Parameters
    ----------
    wave_amplitude : float
        Wave amplitude a_w [m].
    frequency : float
        Encounter frequency omega [rad/s].
    ship_beam : float
        Ship beam B [m].
    ship_draft : float
        Ship draft T [m].
    rho_w, g : float
        Physical constants.

    Returns
    -------
    float
        Wave excitation force amplitude [N/m] (per unit ship length).
    """
    a_w = float(wave_amplitude)
    omega = float(frequency)
    B = float(ship_beam)
    T = float(ship_draft)

    if a_w < 0:
        raise MIZCouplingError(f"wave_amplitude must be non-negative, got {a_w!r}")
    if omega <= 0:
        raise MIZCouplingError(f"frequency must be positive, got {omega!r}")
    if B <= 0 or T <= 0:
        raise MIZCouplingError(f"ship_beam and ship_draft must be positive")

    # Simplified RAO: Froude-Krylov + diffraction approximation
    # chi ~ B / sqrt(1 + (omega^2 * T / g)^2)
    omega_n = sqrt(g / T)  # natural frequency approximation
    chi = B / sqrt(1.0 + (omega / omega_n)**4)

    return rho_w * g * chi * a_w


def ice_impulse_force(
    time: np.ndarray,
    concentration: float,
    ice_thickness: float,
    ship_speed: float = 5.0,
    ship_beam: float = 20.0,
    *,
    rate_factor: float = 2.0,
    impulse_mean: float = 1e5,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random ice impulse forces (Poisson process).

    Ice forces are modelled as a Poisson process of impulses:
    - Rate lambda = rate_factor * C_floe * V_ship / B
    - Each impulse has magnitude ~ Exponential(impulse_mean * h_ice)

    Parameters
    ----------
    time : array_like
        Time array [s].
    concentration : float
        Ice concentration C_floe [0, 1].
    ice_thickness : float
        Ice thickness h_ice [m].
    ship_speed : float
        Ship speed V [m/s].
    ship_beam : float
        Ship beam B [m].
    rate_factor : float
        Scaling factor for impulse rate.
    impulse_mean : float
        Mean impulse magnitude per unit thickness [N/m].
    seed : int, optional
        Random seed.

    Returns
    -------
    np.ndarray
        Ice force time series [N/m].
    """
    t = np.asarray(time, dtype=float)
    C = float(concentration)
    h_i = float(ice_thickness)
    V = float(ship_speed)
    B = float(ship_beam)

    if not (0.0 <= C <= 1.0):
        raise MIZCouplingError(f"concentration must be in [0, 1], got {C!r}")
    if h_i <= 0:
        raise MIZCouplingError(f"ice_thickness must be positive, got {h_i!r}")

    rng = np.random.default_rng(seed)

    # Poisson rate: number of impacts per second
    lam = rate_factor * C * V / max(B, 1.0)

    # Generate impulse times
    dt = t[1] - t[0] if len(t) > 1 else 1.0
    n_steps = len(t)
    n_impacts = rng.poisson(lam * t[-1])

    F_ice = np.zeros(n_steps)
    if n_impacts > 0 and lam > 0:
        impact_times = rng.uniform(t[0], t[-1], size=n_impacts)
        impact_magnitudes = rng.exponential(impulse_mean * h_i, size=n_impacts)

        for t_imp, f_imp in zip(impact_times, impact_magnitudes):
            # Each impulse decays exponentially with characteristic time 0.5s
            idx = np.searchsorted(t, t_imp)
            if idx < n_steps:
                decay = np.exp(-(t[idx:] - t_imp) / 0.5)
                F_ice[idx:] += f_imp * decay

    return F_ice


def ship_motion_rao(
    frequency: float,
    ship_mass: float,
    ship_beam: float,
    ship_draft: float,
    ship_length: float,
    *,
    added_mass_coeff: float = 1.0,
    damping_coeff: float = 0.1,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> dict[str, float]:
    """Compute ship heave RAO in open water.

    RAO(omega) = |H(omega)| where
    H(omega) = 1 / sqrt((K - M_total*omega^2)^2 + (C_damp*omega)^2)

    Parameters
    ----------
    frequency : float
        Encounter frequency omega [rad/s].
    ship_mass : float
        Ship mass M [kg].
    ship_beam, ship_draft, ship_length : float
        Ship dimensions [m].
    added_mass_coeff : float
        Added mass coefficient (fraction of displaced mass).
    damping_coeff : float
        Non-dimensional damping coefficient.
    rho_w, g : float
        Physical constants.

    Returns
    -------
    dict
        Contains 'rao', 'omega_n', 'frequency'.
    """
    omega = float(frequency)
    M = float(ship_mass)
    B = float(ship_beam)
    T = float(ship_draft)
    L = float(ship_length)

    # Displaced volume and hydrostatic stiffness
    V_disp = L * B * T  # approximate
    M_added = added_mass_coeff * rho_w * V_disp
    M_total = M + M_added

    # Hydrostatic restoring coefficient
    K_hs = rho_w * g * L * B

    # Damping
    omega_n = sqrt(K_hs / M_total)
    C_damp = damping_coeff * 2.0 * sqrt(K_hs * M_total)

    # RAO
    denom = sqrt((K_hs - M_total * omega**2)**2 + (C_damp * omega)**2)
    rao = 1.0 / denom if denom > 0 else float('inf')

    return {
        "rao": float(rao),
        "omega_n": float(omega_n),
        "frequency": omega,
        "M_total": float(M_total),
        "K_hs": float(K_hs),
    }


def ship_motion_miz(
    frequency: float,
    ship_mass: float,
    ship_beam: float,
    ship_draft: float,
    ship_length: float,
    concentration: float,
    ice_thickness: float,
    *,
    added_mass_coeff: float = 1.0,
    damping_coeff: float = 0.1,
    ice_damping_factor: float = 3.0,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> dict[str, float]:
    """Compute ship heave RAO in MIZ (with ice damping).

    In the MIZ, ice provides additional damping:
        C_total = C_wave + C_ice
    where C_ice ~ ice_damping_factor * C_floe * h_ice * C_wave

    This reduces the RAO compared to open water (ice damping effect)
    but not as much as continuous ice cover.

    Parameters
    ----------
    frequency : float
        Encounter frequency [rad/s].
    ship_mass, ship_beam, ship_draft, ship_length : float
        Ship properties [kg, m].
    concentration : float
        Ice concentration [0, 1].
    ice_thickness : float
        Ice thickness [m].
    ice_damping_factor : float
        Scaling factor for ice-induced damping.
    rho_w, g : float
        Physical constants.

    Returns
    -------
    dict
        Contains 'rao_miz', 'rao_open_water', 'damping_ratio', 'reduction_factor'.
    """
    omega = float(frequency)
    C = float(concentration)
    h_i = float(ice_thickness)

    # Open water RAO
    ow = ship_motion_rao(
        omega, ship_mass, ship_beam, ship_draft, ship_length,
        added_mass_coeff=added_mass_coeff,
        damping_coeff=damping_coeff,
        rho_w=rho_w, g=g,
    )

    # MIZ: additional ice damping
    M = float(ship_mass)
    B = float(ship_beam)
    T = float(ship_draft)
    L = float(ship_length)
    V_disp = L * B * T
    M_added = added_mass_coeff * rho_w * V_disp
    M_total = M + M_added
    K_hs = rho_w * g * L * B
    omega_n = sqrt(K_hs / M_total)
    C_wave = damping_coeff * 2.0 * sqrt(K_hs * M_total)

    # Ice damping: proportional to concentration and thickness
    C_ice = ice_damping_factor * C * h_i * C_wave
    C_total = C_wave + C_ice

    # MIZ RAO
    denom = sqrt((K_hs - M_total * omega**2)**2 + (C_total * omega)**2)
    rao_miz = 1.0 / denom if denom > 0 else float('inf')

    reduction = ow["rao"] / rao_miz if rao_miz > 0 else float('inf')

    return {
        "rao_miz": float(rao_miz),
        "rao_open_water": ow["rao"],
        "damping_ratio": float(C_total / (2.0 * sqrt(K_hs * M_total))),
        "reduction_factor": float(reduction),
        "C_wave": float(C_wave),
        "C_ice": float(C_ice),
        "C_total": float(C_total),
    }


# ============================================================================
# (f) Coupled solver
# ============================================================================

def coupled_miz_solver(
    H_s: float,
    T_p: float,
    concentration: float,
    ice_thickness: float,
    *,
    depth: float = 1000.0,
    flexural_strength: float = 0.5e6,
    loss_factor: float = 0.02,
    young_modulus: float = 5.0e9,
    poisson_ratio: float = 0.3,
    ship_mass: float = 1e7,
    ship_beam: float = 20.0,
    ship_draft: float = 8.0,
    ship_length: float = 150.0,
    n_distances: int = 50,
    max_distance: float = 5000.0,
    gamma_floe: float = 2.7,
    rho_w: float = RHO_WATER,
    rho_ice: float = RHO_ICE,
    g: float = GRAVITY,
) -> dict[str, Any]:
    """End-to-end coupled MIZ solver.

    Given incident wave conditions and ice field properties, computes:
    1. Wave attenuation through the ice field
    2. Ice breaking locations and characteristic floe sizes
    3. Floe size distribution
    4. Ship motion RAO at various positions in the MIZ

    Parameters
    ----------
    H_s : float
        Significant wave height [m].
    T_p : float
        Peak wave period [s].
    concentration : float
        Ice concentration C_floe [0, 1].
    ice_thickness : float
        Ice thickness h_ice [m].
    depth : float
        Water depth [m].
    flexural_strength : float
        Ice flexural strength [Pa].
    loss_factor : float
        Kelvin-Voigt loss factor.
    young_modulus, poisson_ratio : float
        Ice elastic properties.
    ship_mass, ship_beam, ship_draft, ship_length : float
        Ship properties.
    n_distances : int
        Number of distance points for attenuation profile.
    max_distance : float
        Maximum distance into ice field [m].
    gamma_floe : float
        Power-law exponent for floe size distribution.
    rho_w, rho_ice, g : float
        Physical constants.

    Returns
    -------
    dict
        Comprehensive results including attenuation profile, floe sizes,
        breaking locations, and ship RAO.
    """
    # Validate inputs
    H_s = float(H_s)
    T_p = float(T_p)
    C = float(concentration)
    h_i = float(ice_thickness)

    if H_s < 0:
        raise MIZCouplingError(f"H_s must be non-negative, got {H_s!r}")
    if T_p <= 0:
        raise MIZCouplingError(f"T_p must be positive, got {T_p!r}")
    if not (0.0 <= C <= 1.0):
        raise MIZCouplingError(f"concentration must be in [0, 1], got {C!r}")
    if h_i <= 0:
        raise MIZCouplingError(f"ice_thickness must be positive, got {h_i!r}")

    omega_p = 2.0 * pi / T_p
    a_w = H_s / 2.0  # approximate wave amplitude from H_s

    # --- Step 1: Wave attenuation ---
    if C > 0:
        props = effective_medium_properties(
            C, h_i, loss_factor=loss_factor,
            young_modulus=young_modulus, poisson_ratio=poisson_ratio,
            rho_ice=rho_ice,
        )
        k_complex = kelvin_voigt_wavenumber(
            T_p, depth, props["D_real"], props["D_imag"],
            rho_w=rho_w, g=g,
        )
        alpha = attenuation_from_complex_wavenumber(k_complex)
        k_real = float(k_complex.real)
    else:
        # Open water: no attenuation
        k_real = flexural_gravity_wavenumber(omega_p, depth, D=0.0, rho_w=rho_w, g=g)
        alpha = 0.0

    # Distance array
    x = np.linspace(0, max_distance, n_distances)

    # Wave amplitude decay: A(x) = A_0 * exp(-alpha * x)
    # Energy decay: E(x) = E_0 * exp(-2*alpha*x)
    amplitude_profile = a_w * np.exp(-alpha * x)
    energy_profile = 0.5 * rho_w * g * amplitude_profile**2
    energy_0 = 0.5 * rho_w * g * a_w**2

    # --- Step 2: Ice breaking criterion along the field ---
    sigma_ice_profile = np.array([
        ice_bending_stress(a, k_real, h_i, rho_w=rho_w, g=g)
        for a in amplitude_profile
    ])
    breaks_profile = sigma_ice_profile > flexural_strength

    # Breaking front: first location where ice breaks
    if np.any(breaks_profile):
        break_idx = np.argmax(breaks_profile)
        break_distance = float(x[break_idx])
    else:
        break_distance = float('inf')

    # --- Step 3: Characteristic floe size ---
    if k_real > 0 and np.any(breaks_profile):
        a_floe = characteristic_floe_size(
            k_real, flexural_strength, rho_w=rho_w, g=g
        )
    else:
        a_floe = float('inf')  # no breaking -> infinite floe size

    # --- Step 4: Floe size distribution (Monte Carlo) ---
    if a_floe < float('inf') and a_floe > 0:
        a_min_floe = max(a_floe * 0.1, 0.1)
        a_max_floe = a_floe * 10.0
        floe_sizes = generate_floe_field(
            1000, gamma=gamma_floe,
            a_min=a_min_floe, a_max=a_max_floe, seed=42,
        )
        floe_fit = fit_floe_distribution_exponent(floe_sizes, a_min=a_min_floe)
    else:
        floe_sizes = np.array([])
        floe_fit = {"gamma": float('nan'), "n_samples": 0, "a_min_used": 0.0}

    # --- Step 5: Ship motion RAO ---
    # RAO at peak frequency in open water and in MIZ
    rao_ow = ship_motion_rao(
        omega_p, ship_mass, ship_beam, ship_draft, ship_length,
        rho_w=rho_w, g=g,
    )
    rao_miz = ship_motion_miz(
        omega_p, ship_mass, ship_beam, ship_draft, ship_length,
        C, h_i, rho_w=rho_w, g=g,
    )

    # RAO vs frequency
    freqs = np.linspace(0.1, 2.0, 50) * omega_p
    rao_ow_spectrum = np.array([
        ship_motion_rao(w, ship_mass, ship_beam, ship_draft, ship_length,
                        rho_w=rho_w, g=g)["rao"]
        for w in freqs
    ])
    rao_miz_spectrum = np.array([
        ship_motion_miz(w, ship_mass, ship_beam, ship_draft, ship_length,
                        C, h_i, rho_w=rho_w, g=g)["rao_miz"]
        for w in freqs
    ])

    # --- Step 6: Energy budget ---
    # Wave energy input (per unit width)
    E_input = energy_0
    # Energy dissipated by attenuation over max_distance
    E_dissipated = E_input * (1.0 - np.exp(-2.0 * alpha * max_distance))
    # Energy consumed by ice breaking (approximate)
    if np.any(breaks_profile):
        n_breaks = float(np.sum(breaks_profile))
        # Energy to break one floe ~ sigma_f * volume_floe
        E_break_per_floe = flexural_strength * a_floe * h_i * 1.0  # per unit width
        E_breaking = n_breaks * E_break_per_floe / n_distances
    else:
        E_breaking = 0.0
    # Ship motion energy (kinetic + potential, approximate)
    E_ship = 0.5 * ship_mass * (rao_miz["rao_miz"] * a_w * omega_p)**2

    # Energy residual
    E_total_out = E_dissipated + E_breaking + E_ship
    energy_residual = abs(E_input - E_total_out) / E_input if E_input > 0 else 0.0

    return {
        # Input parameters
        "H_s": H_s,
        "T_p": T_p,
        "concentration": C,
        "ice_thickness": h_i,
        "depth": depth,
        "flexural_strength": flexural_strength,
        # Wave attenuation
        "alpha": alpha,
        "k_real": k_real,
        "omega_p": omega_p,
        "x": x,
        "amplitude_profile": amplitude_profile,
        "energy_profile": energy_profile,
        "energy_0": energy_0,
        # Ice breaking
        "sigma_ice_profile": sigma_ice_profile,
        "breaks_profile": breaks_profile,
        "break_distance": break_distance,
        # Floe size
        "a_floe_characteristic": a_floe,
        "floe_sizes": floe_sizes,
        "floe_gamma_fit": floe_fit,
        "gamma_input": gamma_floe,
        # Ship motion
        "rao_open_water": rao_ow,
        "rao_miz": rao_miz,
        "frequencies": freqs,
        "rao_ow_spectrum": rao_ow_spectrum,
        "rao_miz_spectrum": rao_miz_spectrum,
        # Energy budget
        "E_input": E_input,
        "E_dissipated": E_dissipated,
        "E_breaking": E_breaking,
        "E_ship": E_ship,
        "energy_residual_fraction": energy_residual,
    }


# ============================================================================
# Parameter sweep
# ============================================================================

def parameter_sweep(
    H_s_values: list[float],
    T_p_values: list[float],
    concentration_values: list[float],
    ice_thickness_values: list[float],
    *,
    depth: float = 1000.0,
    flexural_strength: float = 0.5e6,
    loss_factor: float = 0.02,
    max_distance: float = 5000.0,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> dict[str, Any]:
    """Parameter sweep over wave and ice conditions.

    Sweeps H_s, T_p, C_floe, h_ice and records attenuation coefficient,
    floe size, and ship RAO for each combination.

    Returns
    -------
    dict
        Contains 'results' (list of per-case dicts) and 'summary' statistics.
    """
    results = []
    for H_s in H_s_values:
        for T_p in T_p_values:
            for C in concentration_values:
                for h_i in ice_thickness_values:
                    try:
                        res = coupled_miz_solver(
                            H_s, T_p, C, h_i,
                            depth=depth,
                            flexural_strength=flexural_strength,
                            loss_factor=loss_factor,
                            max_distance=max_distance,
                            rho_w=rho_w, g=g,
                        )
                        results.append({
                            "H_s": H_s,
                            "T_p": T_p,
                            "concentration": C,
                            "ice_thickness": h_i,
                            "alpha": res["alpha"],
                            "a_floe": res["a_floe_characteristic"],
                            "break_distance": res["break_distance"],
                            "rao_miz": res["rao_miz"]["rao_miz"],
                            "rao_ow": res["rao_miz"]["rao_open_water"],
                            "rao_reduction": res["rao_miz"]["reduction_factor"],
                        })
                    except Exception as e:
                        results.append({
                            "H_s": H_s,
                            "T_p": T_p,
                            "concentration": C,
                            "ice_thickness": h_i,
                            "error": str(e),
                        })

    return {
        "results": results,
        "n_cases": len(results),
        "n_successful": sum(1 for r in results if "error" not in r),
    }
