"""Thermophysical properties of moist air and water used by the icing kernels.

All correlations are the textbook forms (Sutherland viscosity, Buck saturation
vapour pressure, Clausius-Clapeyron latent heats) so that the icing chain does
not depend on any tabulated data file.

References
----------
* White, *Viscous Fluid Flow*, 3rd ed., App. A (Sutherland law).
* Buck, A. L. (1981) *J. Appl. Meteorol.* 20, 1527-1532 (vapour pressure).
* Incropera & DeWitt, *Fundamentals of Heat and Mass Transfer* (props.).
"""

from __future__ import annotations

import numpy as np

# --- universal constants -------------------------------------------------
R_D = 287.05  # dry-air gas constant [J/(kg K)]
R_V = 461.51  # water-vapour gas constant [J/(kg K)]
P_ATM = 101325.0  # sea-level pressure [Pa]
L_F = 3.337e5  # latent heat of fusion [J/kg]
L_V = 2.501e6  # latent heat of vaporisation at 0 C [J/kg]
L_S = 2.834e6  # latent heat of sublimation [J/kg]
C_W = 4218.0  # specific heat of liquid water [J/(kg K)]
C_ICE = 2100.0  # specific heat of ice [J/(kg K)]
K_ICE = 2.22  # thermal conductivity of ice [W/(m K)]
RHO_ICE_GLAZE = 900.0  # glaze (wet-growth) ice density [kg/m3]
RHO_ICE_RIME = 850.0  # rime (dry-growth) ice density [kg/m3]
SIGMA_SB = 5.670374419e-8  # Stefan-Boltzmann [W/(m2 K4)]
EPS_ICE = 0.97  # long-wave emissivity of ice
K_WATER = 0.569  # thermal conductivity of liquid water [W/(m K)]


def tk(t_c: float) -> float:
    """Celsius -> Kelvin."""
    return t_c + 273.15


def rho_air(t_c: float, p: float = P_ATM) -> float:
    """Density of dry air [kg/m3] (ideal gas)."""
    return p / (R_D * tk(t_c))


def mu_air(t_c: float) -> float:
    """Dynamic viscosity of air [Pa s] (Sutherland law)."""
    t = tk(t_c)
    return 1.458e-6 * t**1.5 / (t + 110.4)


def nu_air(t_c: float) -> float:
    """Kinematic viscosity of air [m2/s]."""
    return mu_air(t_c) / rho_air(t_c)


def k_air(t_c: float) -> float:
    """Thermal conductivity of air [W/(m K)] (Incropera, linear fit)."""
    return 0.02424 * (tk(t_c) / 273.15) ** 0.86


def cp_air(t_c: float) -> float:
    """Specific heat of air [J/(kg K)]."""
    return 1005.0 + 0.02 * t_c


def pr_air(t_c: float) -> float:
    """Prandtl number of air [-]; essentially constant over -60..+40 C."""
    return cp_air(t_c) * mu_air(t_c) / k_air(t_c)


def sc_air(t_c: float) -> float:
    """Schmidt number of water vapour in air [-].

    From the standard mass-transfer analogy ``Sc ~= Pr`` (Lewis number ~ 1.0 for
    water vapour in air at icing temperatures); Le = Sc/Pr = 0.89 is used.
    """
    return 0.89 * pr_air(t_c)


def le_air(t_c: float) -> float:
    """Lewis number [-]."""
    return sc_air(t_c) / pr_air(t_c)


def p_sat_water(t_c: float) -> float:
    """Saturation vapour pressure over liquid water [Pa] (Buck 1981)."""
    return 611.21 * np.exp((18.678 - t_c / 234.5) * (t_c / (257.14 + t_c)))


def p_sat_ice(t_c: float) -> float:
    """Saturation vapour pressure over ice [Pa] (Buck 1981)."""
    return 611.15 * np.exp((23.036 - t_c / 333.7) * (t_c / (279.82 + t_c)))


def rho_v_sat(t_c: float) -> float:
    """Saturation vapour density [kg/m3]; over ice when t_c < 0 C."""
    e = np.where(t_c < 0.0, p_sat_ice(t_c), p_sat_water(t_c))
    return e / (R_V * tk(t_c))


def l_vap(t_c: float) -> float:
    """Latent heat of vaporisation/sublimation [J/kg] (sublimation below 0 C)."""
    return np.where(t_c < 0.0, L_S, L_V)


def rho_ice(regime: str) -> float:
    """Ice density for the given growth regime."""
    return RHO_ICE_RIME if regime == "rime" else RHO_ICE_GLAZE