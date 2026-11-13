"""Published correlations encoded for MPDB v1 (SI in, SI out).

Every function here is a faithful transcription of a published equation; the
``SOURCE_REGISTRY`` entry named in the docstring is what the generated rows
carry as provenance.  No constants are invented: each is quoted with its
origin, and :data:`tensorphysics.material_properties.sources.ANCHORS` holds the
independent check values used by ``build.py`` to prove the transcription.

Temperature arguments are **kelvin**; salinities are **g/kg** (numerically equal
to the older "ppt"); pressures are **pascal**.
"""

from __future__ import annotations

import math
from typing import Sequence

T0_K = 273.15
P_ATM = 101325.0
R_AIR = 287.058            # J/(kg K), ISO 2533 dry air
GAMMA_AIR = 1.4
CP_AIR = 1005.0            # J/(kg K), 250-350 K
MU_AIR_0 = 1.716e-5        # Pa s at 273.15 K
SUTHERLAND_S = 110.4       # K
CP_WATER_TABLE = (         # (T in degC, cp in J/(kg K)) CRC Handbook p. 6-4
    (0.0, 4217.6), (10.0, 4192.1), (20.0, 4181.8), (25.0, 4181.1),
    (30.0, 4178.4), (40.0, 4178.6), (50.0, 4181.1), (60.0, 4184.9),
    (70.0, 4189.9), (80.0, 4197.1), (90.0, 4206.3), (100.0, 4216.2),
)
LATENT_HEAT_FUSION = 333.55e3  # J/kg at 273.15 K (IAPWS R14-08 / CRC p. 6-4)


# --------------------------------------------------------------------------- #
# Fresh water
# --------------------------------------------------------------------------- #
def water_density(T_K: float, p_Pa: float = P_ATM) -> float:
    """Density of ordinary (fresh) water on the 0.1 MPa isobar, kg/m^3.

    Kell (1975) / IAPWS R7-97 eq. (2): a 5-term numerator over a linear
    denominator in t [degC]; valid 273.15 <= T <= 423.15 K.
    """
    t = T_K - T0_K
    if not (0.0 - 1e-9 <= t <= 150.0):
        raise ValueError(f"Kell density valid 0..150 degC, got {t:.2f} degC")
    num = (
        999.83952
        + 16.945176 * t
        - 7.9870401e-3 * t ** 2
        - 46.170461e-6 * t ** 3
        + 105.56302e-9 * t ** 4
        - 280.54253e-12 * t ** 5
    )
    return num / (1.0 + 16.87985e-3 * t)


def water_dynamic_viscosity(T_K: float) -> float:
    """Dynamic viscosity of fresh water, Pa*s (Vogel form of Kestin et al. 1978).

    mu = 2.414e-5 * 10^(247.8 / (T - 140))  with T in kelvin, valid 273-373 K
    to about 1% (3% at the 0 degC edge, see ANCHORS).
    """
    t = T_K - T0_K
    if not (-8.0 <= t <= 150.0):
        raise ValueError(f"Vogel viscosity valid -8..150 degC, got {t:.2f} degC")
    return 2.414e-5 * 10.0 ** (247.8 / (T_K - 140.0))


def water_specific_heat(T_K: float) -> float:
    """Specific heat capacity of fresh water, J/(kg*K), linear interp of CRC table."""
    t = T_K - T0_K
    if not (CP_WATER_TABLE[0][0] <= t <= CP_WATER_TABLE[-1][0]):
        raise ValueError(f"cp table valid 0..100 degC, got {t:.2f} degC")
    for (t1, c1), (t2, c2) in zip(CP_WATER_TABLE, CP_WATER_TABLE[1:]):
        if t1 <= t <= t2:
            w = (t - t1) / (t2 - t1)
            return c1 + w * (c2 - c1)
    return CP_WATER_TABLE[-1][1]


def water_thermal_conductivity(T_K: float) -> float:
    """Thermal conductivity of fresh water, W/(m*K).

    Cubic fit to the IAPWS R15-11 0.1 MPa isobar (273-373 K):
    k = 0.5610 + 1.876e-3 t - 7.5e-6 t^2  + 5e-9 t^3   (t in degC)
    reproduces the R15-11 isobar to <= 1.5% over 273-373 K.
    """
    t = T_K - T0_K
    if not (0.0 <= t <= 100.0):
        raise ValueError(f"conductivity fit valid 0..100 degC, got {t:.2f} degC")
    return 0.5610 + 1.876e-3 * t - 7.5e-6 * t ** 2 + 5.0e-9 * t ** 3


def water_sound_speed(T_K: float) -> float:
    """Speed of sound in fresh water, m/s (Bilaniuk & Wong 1993, eq. 2)."""
    t = T_K - T0_K
    if not (0.0 <= t <= 100.0):
        raise ValueError(f"Bilaniuk-Wong fit valid 0..100 degC, got {t:.2f} degC")
    return (
        1402.388
        + 5.03830 * t
        - 5.81090e-2 * t ** 2
        + 3.3432e-4 * t ** 3
        - 1.47797e-6 * t ** 4
        + 3.1419e-9 * t ** 5
    )


def water_freezing_temperature(p_Pa: float = P_ATM) -> float:
    """Freezing temperature of fresh water at atmospheric pressure, K."""
    if abs(p_Pa - P_ATM) > 1.0:
        raise ValueError("fresh-water freezing point only tabulated at 0.1 MPa")
    return T0_K


# --------------------------------------------------------------------------- #
# Seawater (Sharqawy, Lienhard & Zubair 2010)
# --------------------------------------------------------------------------- #
def _sw_salinity_check(S: float) -> float:
    if not (0.0 <= S <= 160.0):
        raise ValueError(f"Sharqawy correlations valid S = 0..160 g/kg, got {S}")
    return float(S)


def seawater_density(T_K: float, S: float, p_Pa: float = P_ATM) -> float:
    """Density of standard seawater, kg/m^3 (Sharqawy et al. 2010, eq. 2-5)."""
    _sw_salinity_check(S)
    t = T_K - T0_K
    if not (0.0 <= t <= 180.0):
        raise ValueError(f"seawater density valid 0..180 degC, got {t:.2f} degC")
    rho_w = (
        999.842594
        + 6.793952e-2 * t
        - 9.095290e-3 * t ** 2
        + 1.001685e-4 * t ** 3
        - 1.120083e-6 * t ** 4
        + 6.536332e-9 * t ** 5
    )
    A = (
        8.24493e-1
        - 4.0899e-3 * t
        + 7.6438e-5 * t ** 2
        - 8.2467e-7 * t ** 3
        + 5.3875e-9 * t ** 4
    )
    B = -5.72466e-3 + 1.0227e-4 * t - 1.6546e-6 * t ** 2
    C = 4.8314e-4
    return rho_w + A * S + B * S ** 1.5 + C * S ** 2


def seawater_dynamic_viscosity(T_K: float, S: float) -> float:
    """Dynamic viscosity of seawater, Pa*s (Sharqawy et al. 2010, eq. 22-24).

    Note on units: in the quoted equations the salinity enters as a **mass
    fraction** (kg/kg), unlike the density and cp correlations of the same paper
    which take S in g/kg.  The argument of this function is g/kg (the MPDB
    convention), so it is converted internally -- mixing the two conventions is
    the classic way to get a viscosity 200x too large (caught by the
    ITTC-15 degC anchor: nu(S=35) must be 1.1883e-6 m^2/s).
    """
    _sw_salinity_check(S)
    t = T_K - T0_K
    if not (0.0 <= t <= 180.0):
        raise ValueError(f"seawater viscosity valid 0..180 degC, got {t:.2f} degC")
    S_frac = S / 1000.0            # g/kg -> kg/kg for eq. (22)-(24)
    mu_w = 4.2844e-5 + 1.0 / (0.157 * (t + 64.993) ** 2 - 91.296)
    A = 1.541 + 1.998e-2 * t - 9.52e-5 * t ** 2
    B = 7.974 - 7.561e-2 * t + 4.724e-4 * t ** 2
    return mu_w * (1.0 + A * S_frac + B * S_frac ** 2)


def seawater_specific_heat(T_K: float, S: float) -> float:
    """Specific heat capacity of seawater, J/(kg*K) (Sharqawy et al. 2010, eq. 9)."""
    _sw_salinity_check(S)
    t = T_K - T0_K
    if not (0.0 <= t <= 180.0):
        raise ValueError(f"seawater cp valid 0..180 degC, got {t:.2f} degC")
    cp_w = (
        4217.4
        - 3.720283 * t
        + 0.1412855 * t ** 2
        - 2.654387e-3 * t ** 3
        + 2.093236e-5 * t ** 4
    )
    d1 = -7.64357 + 0.1072763 * t - 1.38385e-3 * t ** 2
    d2 = 0.1770383 - 4.07718e-3 * t + 5.148e-5 * t ** 2
    return cp_w + S * d1 + S ** 1.5 * d2


def seawater_freezing_temperature(S: float, p_Pa: float = P_ATM) -> float:
    """Freezing temperature of seawater at 1 atm, K (Millero & Leung 1976)."""
    _sw_salinity_check(S)
    if abs(p_Pa - P_ATM) > 1.0:
        raise ValueError("seawater freezing point tabulated at 1 atm only")
    t_f = -0.0575 * S + 1.710523e-3 * S ** 1.5 - 2.154996e-4 * S ** 2
    return T0_K + t_f


# --------------------------------------------------------------------------- #
# Air (dry)
# --------------------------------------------------------------------------- #
def air_density(T_K: float, p_Pa: float = P_ATM) -> float:
    """Ideal-gas density of dry air, kg/m^3 (ISO 2533 / R = 287.058 J/(kg K))."""
    if T_K <= 0.0 or p_Pa <= 0.0:
        raise ValueError("air density needs T > 0 K and p > 0 Pa")
    return p_Pa / (R_AIR * T_K)


def air_dynamic_viscosity(T_K: float) -> float:
    """Sutherland law for dry air, Pa*s (White 2006 Table 1-2 coefficients)."""
    if not (100.0 <= T_K <= 2000.0):
        raise ValueError(f"Sutherland law used 100..2000 K, got {T_K:.2f} K")
    return MU_AIR_0 * (T_K / T0_K) ** 1.5 * (T0_K + SUTHERLAND_S) / (T_K + SUTHERLAND_S)


def air_thermal_conductivity(T_K: float) -> float:
    """Thermal conductivity of dry air, W/(m*K) (CRC modified-Eucken fit)."""
    if not (200.0 <= T_K <= 1500.0):
        raise ValueError(f"air conductivity fit used 200..1500 K, got {T_K:.2f} K")
    return 2.646e-3 * T_K ** 1.5 / (T_K + 245.4 * 10.0 ** (-12.0 / T_K))


def air_sound_speed(T_K: float) -> float:
    """Speed of sound in dry air, m/s: a = sqrt(gamma R T)."""
    return math.sqrt(GAMMA_AIR * R_AIR * T_K)


def air_prandtl(T_K: float) -> float:
    """Prandtl number of dry air (mu cp / k), dimensionless."""
    return air_dynamic_viscosity(T_K) * CP_AIR / air_thermal_conductivity(T_K)


# --------------------------------------------------------------------------- #
# Sea ice
# --------------------------------------------------------------------------- #
def brine_volume_fraction(T_K: float, S: float) -> float:
    """Brine volume fraction of sea ice (Frankenstein & Garner 1967, eq. 1).

    v_b = S * (49.185 / |T_degC| + 0.532),  valid -0.5 >= T >= -22.9 degC,
    S in g/kg.
    """
    t_c = T_K - T0_K
    if not (-22.9 <= t_c <= -0.5):
        raise ValueError(f"brine volume fit valid -22.9..-0.5 degC, got {t_c:.2f} degC")
    return S * (49.185 / abs(t_c) + 0.532) / 1000.0


def sea_ice_flexural_strength(v_b: float) -> float:
    """Flexural strength of first-year sea ice, Pa (Timco & Weeks 2010, eq. 8).

    sigma_f = 1.76 exp(-5.88 sqrt(v_b)) MPa, valid 0 <= v_b <= 0.3.
    """
    if not (0.0 <= v_b <= 0.3):
        raise ValueError(f"flexural-strength fit valid v_b = 0..0.3, got {v_b:.3f}")
    return 1.76 * math.exp(-5.88 * math.sqrt(v_b)) * 1.0e6


def fresh_ice_density(T_K: float) -> float:
    """Density of bubble-free ice Ih, kg/m^3, linear thermal contraction from 0 degC.

    Hobbs (1974) Table 2.1 gives 916.8 kg/m^3 at 273.15 K; the volume thermal
    expansion coefficient of ice Ih near 0 degC is 1.53e-4 1/K, i.e.
    rho(T) = rho_0 / (1 + beta (T0 - T)).
    """
    t_c = T_K - T0_K
    if not (-60.0 <= t_c <= 0.0):
        raise ValueError(f"ice density parametrisation used -60..0 degC, got {t_c:.2f} degC")
    beta = 1.53e-4
    return 916.8 / (1.0 + beta * (0.0 - t_c))


def interpolate(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    """Piecewise-linear interpolation with endpoint clamping (no extrapolation)."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("interpolate needs equal-length non-empty tables")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for x1, y1, x2, y2 in zip(xs, xs[1:], ys, ys[1:]):
        if x1 <= x <= x2:
            w = (x - x1) / (x2 - x1)
            return y1 + w * (y2 - y1)
    return ys[-1]