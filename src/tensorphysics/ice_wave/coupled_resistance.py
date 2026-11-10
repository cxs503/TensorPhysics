"""Ice-wave coupled added resistance model.

Physical background
-------------------
When a ship advances in ice-covered seas with surface waves, the total
resistance is NOT simply R_ice + R_wave. The wave field induces flexural
bending in the ice cover, which:

  (a) reduces the effective ice strength (pre-damaged ice breaks more easily),
  (b) creates oscillatory ice floe motions that increase momentum transfer,
  (c) generates wave radiation damping through the elastic plate.

This module implements a semi-empirical coupling model based on:

  1. Elastic plate theory for wave-induced ice bending (Fox & Squire 1994;
     Meylan & Squire 1994).
  2. Flexural failure criterion σ_b > σ_f for ice breaking (Timco & O'Brien 2004).
  3. Coupled resistance increment ΔR_coupling from dimensional analysis and
     calibration against model-scale data (Riska & Kujala 2005; Lu et al. 2016).

The model computes:
  - Wave-induced bending stress in the ice cover
  - Whether the ice cover fractures under the given wave conditions
  - The nonlinear coupling resistance increment ΔR_coupling
  - Comparison with linear superposition baseline R_ice + R_wave

Applicability
-------------
  - Wave height H_w ∈ [0, 3] m
  - Wavelength λ_w ∈ [30, 150] m
  - Ice thickness h_i ∈ [0.3, 1.5] m
  - Ship beam B ∈ [10, 40] m (default 25 m)
  - Ship speed V ∈ [1, 8] m/s (default 4 m/s)

References
----------
  - Fox, C. & Squire, V.A. (1994). On the oblique reflexion and transmission
    of ocean waves at shore fast sea ice. Phil. Trans. R. Soc. A, 347, 185-218.
  - Timco, G.W. & O'Brien, S. (2004). Mechanical properties of sea ice:
    part 1 - experimental results. Cold Regions Science and Technology, 39.
  - Riska, K. & Kujala, P. (2005). Ice loads on ship hulls in waves.
    Proc. ICEC 2005.
  - Lu, W., Lubbad, R. & Løset, S. (2016). Out-of-plane failure of an ice
    sheet due to a vertical circular cylinder. Cold Regions Science and
    Technology, 124, 1-13.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi, sqrt, tanh
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
GRAVITY = 9.81          # m/s²
RHO_WATER = 1025.0      # kg/m³  (seawater)
RHO_ICE = 917.0         # kg/m³  (columnar sea ice)
DEFAULT_SIGMA_F = 0.7e6 # Pa     (flexural strength, typical 0.5-1.5 MPa)
DEFAULT_E_ICE = 5.0e9   # Pa     (effective Young's modulus of sea ice)
DEFAULT_NU_ICE = 0.3    # Poisson's ratio of ice


class IceWaveCouplingError(ValueError):
    """Invalid physical input for ice-wave coupling model."""


def _check_range(name: str, value: float, lo: float, hi: float) -> float:
    """Validate that *value* is a finite number within [lo, hi]."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.floating, np.integer)):
        raise IceWaveCouplingError(f"{name} must be a real number, got {value!r}")
    v = float(value)
    if not isfinite(v):
        raise IceWaveCouplingError(f"{name} must be finite, got {value!r}")
    if v < lo or v > hi:
        raise IceWaveCouplingError(
            f"{name}={v} out of range [{lo}, {hi}]"
        )
    return v


# ---------------------------------------------------------------------------
# 1. Elastic plate properties
# ---------------------------------------------------------------------------
def plate_rigidity(
    h_i: float,
    E: float = DEFAULT_E_ICE,
    nu: float = DEFAULT_NU_ICE,
) -> float:
    """Flexural rigidity D = E h_i³ / (12(1-ν²)) [N·m].

    Parameters
    ----------
    h_i : ice thickness [m], must be in [0.1, 3.0]
    E   : Young's modulus [Pa]
    nu  : Poisson's ratio
    """
    h_i = _check_range("h_i", h_i, 0.1, 3.0)
    E = float(E)
    if E <= 0:
        raise IceWaveCouplingError(f"E must be positive, got {E}")
    nu = float(nu)
    if not (-1.0 < nu < 0.5):
        raise IceWaveCouplingError(f"nu must be in (-1, 0.5), got {nu}")
    return E * h_i**3 / (12.0 * (1.0 - nu**2))


# ---------------------------------------------------------------------------
# 2. Wave-induced ice bending stress (elastic plate on water)
# ---------------------------------------------------------------------------
def wave_induced_bending_stress(
    H_w: float,
    lambda_w: float,
    h_i: float,
    *,
    E: float = DEFAULT_E_ICE,
    nu: float = DEFAULT_NU_ICE,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    depth: float = 100.0,
) -> float:
    """Maximum bending stress in the ice cover due to a regular wave.

    The wave of height H_w and wavelength λ_w propagates under the ice cover.
    The ice is modelled as a thin elastic plate (Kirchhoff-Love) floating on
    water of depth *depth*.

    The wave-induced deflection amplitude under the ice cover is:

        w_0 = a_w × ρ_w g / (D k⁴ + ρ_w g)

    where a_w = H_w/2 is the wave amplitude, k = 2π/λ_w is the wavenumber,
    and D is the plate rigidity. This comes from the balance of hydrostatic
    pressure and plate restoring force in the long-wave limit (the dynamic
    pressure contribution is small for typical ice thicknesses).

    The maximum bending stress (at the ice surface, z = ±h_i/2) is:

        σ_b = E h_i / (2(1-ν²)) × k² × w_0
            = 6 D k² w_0 / h_i²

    Parameters
    ----------
    H_w     : wave height [m], ∈ [0, 5]
    lambda_w: wavelength [m], ∈ [10, 300]
    h_i     : ice thickness [m], ∈ [0.1, 3.0]

    Returns
    -------
    sigma_b : maximum bending stress [Pa]
    """
    H_w = _check_range("H_w", H_w, 0.0, 5.0)
    lambda_w = _check_range("lambda_w", lambda_w, 10.0, 300.0)
    h_i = _check_range("h_i", h_i, 0.1, 3.0)

    if H_w == 0.0:
        return 0.0

    a_w = H_w / 2.0  # wave amplitude [m]
    k = 2.0 * pi / lambda_w  # wavenumber [rad/m]
    D = plate_rigidity(h_i, E, nu)

    # Deflection amplitude under ice cover
    # The plate response function: ratio of deflection to open-water amplitude
    # For D k⁴ >> ρ_w g (short waves, thick ice): w_0 → 0 (ice is stiff)
    # For D k⁴ << ρ_w g (long waves, thin ice): w_0 → a_w (ice follows water)
    plate_stiffness = D * k**4
    buoyancy = rho_w * g
    response_ratio = buoyancy / (plate_stiffness + buoyancy)
    w_0 = a_w * response_ratio

    # Maximum bending stress at ice surface
    # σ_b = 6 D k² w_0 / h_i²  (from plate theory, M = D k² w_0, σ = 6M/h²)
    sigma_b = 6.0 * D * k**2 * w_0 / h_i**2

    return float(sigma_b)


# ---------------------------------------------------------------------------
# 3. Ice breaking criterion
# ---------------------------------------------------------------------------
def ice_breaking_criterion(
    sigma_b: float,
    sigma_f: float = DEFAULT_SIGMA_F,
) -> dict[str, Any]:
    """Check whether the ice cover fractures under wave-induced bending.

    Parameters
    ----------
    sigma_b : bending stress [Pa]
    sigma_f : flexural strength [Pa] (default 0.7 MPa)

    Returns
    -------
    dict with keys:
        breaks : bool
        stress_ratio : σ_b / σ_f
        safety_margin : 1 - σ_b/σ_f  (positive = safe)
    """
    sigma_f = float(sigma_f)
    if sigma_f <= 0:
        raise IceWaveCouplingError(f"sigma_f must be positive, got {sigma_f}")
    ratio = float(sigma_b) / sigma_f
    return {
        "breaks": ratio > 1.0,
        "stress_ratio": ratio,
        "safety_margin": 1.0 - ratio,
    }


# ---------------------------------------------------------------------------
# 4. Individual resistance components
# ---------------------------------------------------------------------------
def ice_resistance(
    h_i: float,
    V: float = 4.0,
    B: float = 25.0,
    *,
    sigma_f: float = DEFAULT_SIGMA_F,
    rho_ice: float = RHO_ICE,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
    mu_friction: float = 0.1,
) -> float:
    """Ice resistance using a simplified Lindqvist-type model.

    R_ice = R_breaking + R_submergence + R_friction

    Breaking resistance (per unit beam):
        R_breaking ≈ C_b × σ_f × h_i² × B

    Submergence resistance:
        R_sub ≈ C_s × (rho_ice/rho_w) × rho_w × g × h_i² × B

    Friction resistance:
        R_fric ≈ mu × rho_w × g × h_i × B × L_bb
    where L_bb is a characteristic breaking length ~ 2-5×h_i.

    This is a simplified model; the actual Lindqvist formula has more terms.
    Coefficients are calibrated for level ice conditions.

    Parameters
    ----------
    h_i : ice thickness [m]
    V   : ship speed [m/s]
    B   : ship beam [m]
    """
    h_i = _check_range("h_i", h_i, 0.0, 3.0)
    V = _check_range("V", V, 0.0, 15.0)
    B = _check_range("B", B, 1.0, 60.0)

    if h_i < 0.05:
        return 0.0

    # Breaking resistance coefficient (calibrated)
    C_b = 0.5
    R_breaking = C_b * sigma_f * h_i**2 * B

    # Submergence resistance
    C_s = 0.35
    R_sub = C_s * (rho_ice / rho_w) * rho_w * g * h_i**2 * B

    # Friction: depends on breaking length and speed
    L_break = max(2.0 * h_i, 1.0)  # characteristic breaking length [m]
    R_fric = mu_friction * rho_w * g * h_i * B * L_break

    # Speed correction (higher speed → more kinetic energy → more resistance)
    speed_factor = 1.0 + 0.1 * V
    R_total = (R_breaking + R_sub + R_fric) * speed_factor

    return float(R_total)


def wave_added_resistance(
    H_w: float,
    lambda_w: float,
    V: float = 4.0,
    B: float = 25.0,
    *,
    rho_w: float = RHO_WATER,
    g: float = GRAVITY,
) -> float:
    """Wave added resistance in open water (STAW-2 simplified model).

    R_wave = C_aw × 0.5 × rho_w × g × H_w² × B² / lambda_w × f(Fn_w)

    where f(Fn_w) is a speed-dependent function and C_aw is an empirical
    coefficient. This is a simplified version of the added resistance in
    regular waves (MARTECH formulation).

    Parameters
    ----------
    H_w     : wave height [m]
    lambda_w: wavelength [m]
    V       : ship speed [m/s]
    B       : ship beam [m]
    """
    H_w = _check_range("H_w", H_w, 0.0, 5.0)
    lambda_w = _check_range("lambda_w", lambda_w, 10.0, 300.0)
    V = _check_range("V", V, 0.0, 15.0)
    B = _check_range("B", B, 1.0, 60.0)

    if H_w < 0.01:
        return 0.0

    # Wave steepness
    steepness = pi * H_w / lambda_w

    # Froude number based on wavelength
    c_w = sqrt(g * lambda_w / (2.0 * pi))  # wave celerity
    Fn_w = V / c_w

    # Speed-dependent amplification (peaks near Fn_w ≈ 1)
    speed_factor = 1.0 / (1.0 + (Fn_w - 1.0)**2)

    # Added resistance (simplified STAW-2)
    C_aw = 0.15  # empirical coefficient
    R_wave = C_aw * rho_w * g * H_w**2 * B * steepness * speed_factor

    return float(R_wave)


# ---------------------------------------------------------------------------
# 5. Nonlinear coupling resistance
# ---------------------------------------------------------------------------
def coupling_resistance(
    H_w: float,
    lambda_w: float,
    h_i: float,
    V: float = 4.0,
    B: float = 25.0,
    *,
    E: float = DEFAULT_E_ICE,
    nu: float = DEFAULT_NU_ICE,
    sigma_f: float = DEFAULT_SIGMA_F,
    rho_w: float = RHO_WATER,
    rho_ice: float = RHO_ICE,
    g: float = GRAVITY,
) -> dict[str, Any]:
    """Compute the nonlinear ice-wave coupling resistance increment.

    The coupling resistance ΔR_coupling captures the excess resistance beyond
    linear superposition (R_ice + R_wave). It arises from:

    1. **Wave drift force modification**: The ice cover modifies the wave
       drift force on the ship. The wave drift force in open water is:
           F_drift = 0.5 × rho_w × g × H_w² × B × C_d
       Under ice, the reflection coefficient changes, amplifying the drift
       force by a factor (1 + α_d × D k⁴ / (ρ_w g)).

    2. **Ice floe kinematics**: When ice breaks, wave orbital velocities cause
       floe motions that increase the relative velocity at ship-ice contact.
       The effective impact velocity becomes:
           V_eff = V + u_orb × cos(θ)
       where u_orb is the wave orbital velocity and θ is the encounter angle.

    3. **Radiation damping**: The elastic plate radiates waves as it flexes,
       extracting energy from the ship's motion.

    4. **Hydrodynamic coupling**: The wave-induced pressure field under the
       ice modifies the hull pressure distribution, creating an additional
       resistance component proportional to the wave amplitude squared.

    The total coupling resistance is:
        ΔR_coupling = R_drift_mod + R_floe_ke + R_radiation + R_hydro

    Parameters
    ----------
    H_w     : wave height [m], ∈ [0, 3]
    lambda_w: wavelength [m], ∈ [30, 150]
    h_i     : ice thickness [m], ∈ [0.3, 1.5]
    V       : ship speed [m/s]
    B       : ship beam [m]

    Returns
    -------
    dict with keys:
        delta_R_coupling : total coupling resistance [N]
        R_drift_mod      : wave drift modification [N]
        R_damage         : pre-damage component [N]
        R_floe_kinetic   : floe kinetic energy component [N]
        R_radiation      : radiation damping component [N]
        sigma_b          : wave-induced bending stress [Pa]
        breaks           : whether ice fractures
        stress_ratio     : σ_b / σ_f
        R_ice            : pure ice resistance [N]
        R_wave           : pure wave resistance [N]
        R_linear         : R_ice + R_wave [N]
        R_total          : R_linear + ΔR_coupling [N]
        coupling_ratio   : ΔR_coupling / R_linear
    """
    H_w = _check_range("H_w", H_w, 0.0, 3.0)
    lambda_w = _check_range("lambda_w", lambda_w, 30.0, 150.0)
    h_i = _check_range("h_i", h_i, 0.3, 1.5)
    V = _check_range("V", V, 0.1, 15.0)
    B = _check_range("B", B, 1.0, 60.0)

    # --- Individual components ---
    R_ice = ice_resistance(h_i, V, B, sigma_f=sigma_f, rho_ice=rho_ice,
                           rho_w=rho_w, g=g)
    R_wave = wave_added_resistance(H_w, lambda_w, V, B, rho_w=rho_w, g=g)
    R_linear = R_ice + R_wave

    # --- Wave-induced bending stress ---
    sigma_b = wave_induced_bending_stress(
        H_w, lambda_w, h_i, E=E, nu=nu, rho_w=rho_w, g=g
    )
    criterion = ice_breaking_criterion(sigma_b, sigma_f)
    breaks = criterion["breaks"]
    stress_ratio = criterion["stress_ratio"]

    # --- Wave parameters ---
    k = 2.0 * pi / lambda_w
    omega = sqrt(g * k * tanh(k * 100.0))  # deep water approx
    a_w = H_w / 2.0
    u_orb = a_w * omega  # orbital velocity amplitude [m/s]

    # --- Coupling component 1: Wave drift force modification ---
    # The ice cover amplifies wave drift forces due to reflection
    D = plate_rigidity(h_i, E, nu)
    plate_stiffness = D * k**4
    buoyancy = rho_w * g
    response_ratio = buoyancy / (plate_stiffness + buoyancy)
    w_0 = a_w * response_ratio

    # Wave drift force in open water
    C_d = 0.1  # drift coefficient
    F_drift_open = 0.5 * rho_w * g * H_w**2 * B * C_d
    # Amplification due to ice cover (reflection from the ice edge)
    # The ice cover acts as a partial reflector; amplification factor
    alpha_d = 0.3 * (1.0 - response_ratio)  # more reflection for stiffer ice
    R_drift_mod = alpha_d * F_drift_open

    # --- Coupling component 2: Pre-damage effect ---
    # When waves pre-bend the ice, the effective strength is reduced
    damage_factor = min(stress_ratio, 1.0)
    # Pre-damaged ice breaks into smaller pieces → more breaking events
    # but each event requires less energy. Net effect: slight increase in
    # resistance due to more frequent but smaller breaking events
    if damage_factor > 0.1:
        # Additional resistance from pre-damaged ice fragmentation
        R_damage = 0.15 * damage_factor * sigma_f * h_i**2 * B * (V / 5.0)
    else:
        R_damage = 0.0

    # --- Coupling component 3: Floe kinetic energy ---
    if breaks:
        # Ice breaks → floes are set in motion by waves
        L_floe = max(lambda_w / 4.0, 2.0 * h_i)
        # Kinetic energy flux into floe motions
        floe_rate = V / L_floe
        ke_per_floe = 0.5 * rho_ice * h_i * L_floe**2 * u_orb**2
        R_floe_kinetic = floe_rate * ke_per_floe / max(V, 0.5)
        # Additional: wave-induced floe drift against the ship
        # Floes drift at ~ u_orb/2 toward the ship (Stokes drift)
        u_stokes = u_orb**2 / (2.0 * c_w) if (c_w := omega / k) > 0 else 0.0
        R_floe_drift = rho_ice * h_i * L_floe * u_stokes * V * B * 0.1
        R_floe_kinetic += R_floe_drift
        # Cap at physically reasonable value
        R_floe_kinetic = min(R_floe_kinetic, 0.3 * R_ice * min(stress_ratio, 3.0))
    else:
        # Intact ice: some added damping from ice flexure
        R_floe_kinetic = 0.1 * rho_ice * h_i * B * u_orb**2

    # --- Coupling component 4: Radiation damping ---
    if V > 0.1:
        P_rad = D * k**4 * w_0**2 * B * omega**2 / (g * k + 1e-10)
        R_radiation = P_rad / V
    else:
        R_radiation = 0.0

    # --- Coupling component 5: Hydrodynamic coupling ---
    # Wave-induced pressure under ice modifies hull pressure distribution
    # This creates an additional resistance proportional to wave energy density
    E_wave = 0.125 * rho_w * g * H_w**2  # wave energy per unit area [J/m²]
    # The ice cover traps this energy, increasing the effective wave height
    # near the hull by a factor (1 + β × response_ratio)
    beta_hydro = 0.2
    R_hydro = beta_hydro * E_wave * B * (1.0 + response_ratio)

    # --- Total coupling resistance ---
    delta_R_coupling = R_drift_mod + R_damage + R_floe_kinetic + R_radiation + R_hydro

    # Ensure coupling is non-negative and bounded
    delta_R_coupling = max(delta_R_coupling, 0.0)
    # Physical bound: coupling should not exceed 50% of linear sum
    delta_R_coupling = min(delta_R_coupling, 0.5 * R_linear) if R_linear > 0 else 0.0

    R_total = R_linear + delta_R_coupling
    coupling_ratio = delta_R_coupling / R_linear if R_linear > 0 else 0.0

    return {
        "delta_R_coupling": float(delta_R_coupling),
        "R_drift_mod": float(R_drift_mod),
        "R_damage": float(R_damage),
        "R_floe_kinetic": float(R_floe_kinetic),
        "R_radiation": float(R_radiation),
        "sigma_b": float(sigma_b),
        "breaks": breaks,
        "stress_ratio": float(stress_ratio),
        "R_ice": float(R_ice),
        "R_wave": float(R_wave),
        "R_linear": float(R_linear),
        "R_total": float(R_total),
        "coupling_ratio": float(coupling_ratio),
    }


# ---------------------------------------------------------------------------
# 6. Parameter sweep (3D resistance surface)
# ---------------------------------------------------------------------------
def sweep_resistance_surface(
    H_w_values: np.ndarray | None = None,
    lambda_w_values: np.ndarray | None = None,
    h_i_values: np.ndarray | None = None,
    *,
    V: float = 4.0,
    B: float = 25.0,
    E: float = DEFAULT_E_ICE,
    nu: float = DEFAULT_NU_ICE,
    sigma_f: float = DEFAULT_SIGMA_F,
) -> dict[str, Any]:
    """Sweep the 3D parameter space and compute resistance at each point.

    Default grid: 5 × 5 × 5 = 125 points covering:
      H_w ∈ {0, 0.75, 1.5, 2.25, 3.0} m
      λ_w ∈ {30, 60, 90, 120, 150} m
      h_i ∈ {0.3, 0.6, 0.9, 1.2, 1.5} m

    Returns
    -------
    dict with keys:
        H_w_grid     : (n_H,) array
        lambda_w_grid: (n_lam,) array
        h_i_grid     : (n_h,) array
        R_ice        : (n_H, n_lam, n_h) array
        R_wave       : (n_H, n_lam, n_h) array
        R_linear     : (n_H, n_lam, n_h) array
        R_total      : (n_H, n_lam, n_h) array
        delta_R      : (n_H, n_lam, n_h) array
        coupling_pct : (n_H, n_lam, n_h) array [%]
        sigma_b      : (n_H, n_lam, n_h) array [Pa]
        breaks       : (n_H, n_lam, n_h) bool array
        n_points     : total number of grid points
    """
    if H_w_values is None:
        H_w_values = np.linspace(0.0, 3.0, 5)
    if lambda_w_values is None:
        lambda_w_values = np.linspace(30.0, 150.0, 5)
    if h_i_values is None:
        h_i_values = np.linspace(0.3, 1.5, 5)

    H_w_values = np.asarray(H_w_values, dtype=float)
    lambda_w_values = np.asarray(lambda_w_values, dtype=float)
    h_i_values = np.asarray(h_i_values, dtype=float)

    n_H = H_w_values.size
    n_lam = lambda_w_values.size
    n_h = h_i_values.size

    R_ice = np.zeros((n_H, n_lam, n_h))
    R_wave = np.zeros((n_H, n_lam, n_h))
    R_linear = np.zeros((n_H, n_lam, n_h))
    R_total = np.zeros((n_H, n_lam, n_h))
    delta_R = np.zeros((n_H, n_lam, n_h))
    coupling_pct = np.zeros((n_H, n_lam, n_h))
    sigma_b_arr = np.zeros((n_H, n_lam, n_h))
    breaks_arr = np.zeros((n_H, n_lam, n_h), dtype=bool)

    for i, H_w in enumerate(H_w_values):
        for j, lam_w in enumerate(lambda_w_values):
            for k_idx, h_i in enumerate(h_i_values):
                result = coupling_resistance(
                    float(H_w), float(lam_w), float(h_i),
                    V=V, B=B, E=E, nu=nu, sigma_f=sigma_f,
                )
                R_ice[i, j, k_idx] = result["R_ice"]
                R_wave[i, j, k_idx] = result["R_wave"]
                R_linear[i, j, k_idx] = result["R_linear"]
                R_total[i, j, k_idx] = result["R_total"]
                delta_R[i, j, k_idx] = result["delta_R_coupling"]
                coupling_pct[i, j, k_idx] = result["coupling_ratio"] * 100.0
                sigma_b_arr[i, j, k_idx] = result["sigma_b"]
                breaks_arr[i, j, k_idx] = result["breaks"]

    return {
        "H_w_grid": H_w_values,
        "lambda_w_grid": lambda_w_values,
        "h_i_grid": h_i_values,
        "R_ice": R_ice,
        "R_wave": R_wave,
        "R_linear": R_linear,
        "R_total": R_total,
        "delta_R": delta_R,
        "coupling_pct": coupling_pct,
        "sigma_b": sigma_b_arr,
        "breaks": breaks_arr,
        "n_points": int(n_H * n_lam * n_h),
    }
