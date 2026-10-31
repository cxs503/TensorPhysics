"""Adfreeze bond degradation mechanisms.

Three mechanisms destroy an existing ice/steel bond:

1. **Mechanical peel / debonding** (:func:`peel_softening`, :func:`cohesive_law`)
   A bilinear cohesive law with interface fracture energy ``G_c``.  The slip
   at failure follows from the energy balance ``G_c = 0.5*tau_peak*delta_c``,
   so ``delta_c = 2*G_c/tau_peak``.  This is the mechanism exploited by
   heeling/rocking the hull.
2. **Thermal melting** (:func:`thermal_degradation_factor`)
   The strength is controlled by the undercooling, so warming the joint to
   its freezing point destroys it.  The factor is the ratio of the strength
   model (:mod:`tensorphysics.ice.adfreeze`) at the new and the old
   interface temperature.
3. **Salinity / freshening** (:func:`salinity_degradation_factor`)
   Brine at the interface carries no shear stress.  The factor is the ratio
   of the brine penalties at the new and old salinity; freshening (river
   runoff, melt water) therefore *restores* strength, while brine drainage
   from a saline ice cover reduces it.

All factors are multiplicative and lie in ``[0, 1]`` (except a salinity
freshening factor, which can exceed 1), so the residual bond is

    tau_residual = tau_adfreeze * f_peel * f_thermal * f_salinity
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .adfreeze import (
    DEFAULT_ROUGHNESS_UM,
    adfreeze_strength,
    brine_volume_fraction,
    seawater_freezing_point,
)

#: Interface fracture energy of an ice/steel adfreeze joint [J/m^2].
#: Order of magnitude of brittle ice interface energies (1-10 J/m^2).
DEFAULT_INTERFACE_G_C = 5.0


def critical_slip(tau_peak_mpa: float, g_c_j_m2: float = DEFAULT_INTERFACE_G_C) -> float:
    """Slip at complete debonding [m]: ``delta_c = 2*G_c/tau_peak``."""
    if tau_peak_mpa <= 0.0:
        raise ValueError("tau_peak_mpa must be positive")
    if g_c_j_m2 <= 0.0:
        raise ValueError("g_c_j_m2 must be positive")
    return float(2.0 * g_c_j_m2 / (tau_peak_mpa * 1.0e6))


def peel_softening(
    tau_peak_mpa: float,
    slip_m: float,
    g_c_j_m2: float = DEFAULT_INTERFACE_G_C,
) -> float:
    """Residual strength fraction after a given slip (bilinear cohesive law).

    ``delta_c = 2*G_c/tau_peak``; the response is ``1 - slip/delta_c`` for
    ``slip < delta_c`` and ``0`` afterwards, so the dissipated energy per unit
    area equals ``G_c`` exactly (verified in the test suite).
    """
    if tau_peak_mpa <= 0.0:
        raise ValueError("tau_peak_mpa must be positive")
    if slip_m < 0.0:
        raise ValueError("slip must be non-negative")
    delta_c = critical_slip(tau_peak_mpa, g_c_j_m2)
    return float(max(0.0, 1.0 - slip_m / delta_c))


def cohesive_law(
    tau_peak_mpa: float,
    slips_m: Iterable[float],
    g_c_j_m2: float = DEFAULT_INTERFACE_G_C,
) -> np.ndarray:
    """Shear traction [MPa] along the bilinear cohesive law."""
    return np.array(
        [tau_peak_mpa * peel_softening(tau_peak_mpa, s, g_c_j_m2) for s in slips_m],
        dtype=float,
    )


def thermal_degradation_factor(
    temperature_old_c: float,
    temperature_new_c: float,
    salinity_gkg: float = 0.0,
    contact_time_s: float = 1.0e4,
    roughness_um: float = DEFAULT_ROUGHNESS_UM,
) -> float:
    """Ratio ``tau_new/tau_old`` when the joint temperature changes.

    Evaluated with the full strength model on both states, so the frozen-area
    fraction is re-evaluated at the new undercooling.  Returns ``0`` when the
    new state is at or above the liquidus (bond melted away).
    """
    tau_old = adfreeze_strength(
        temperature_old_c, contact_time_s, salinity_gkg, roughness_um
    )
    tau_new = adfreeze_strength(
        temperature_new_c, contact_time_s, salinity_gkg, roughness_um
    )
    if tau_old <= 0.0:
        raise ValueError("initial state carries no bond to degrade")
    return float(tau_new / tau_old)


def salinity_degradation_factor(
    temperature_c: float,
    salinity_old_gkg: float,
    salinity_new_gkg: float,
) -> float:
    """Ratio of brine penalties between two salinity states at the same T.

    Freshening (``S_new < S_old``) gives a factor ``> 1`` (strength recovery);
    brine enrichment gives ``< 1``.  Returns ``0`` if the new state cannot
    freeze at all.
    """
    if salinity_old_gkg < 0.0 or salinity_new_gkg < 0.0:
        raise ValueError("salinity must be non-negative")
    tf_old = seawater_freezing_point(salinity_old_gkg)
    tf_new = seawater_freezing_point(salinity_new_gkg)
    if float(temperature_c) >= tf_old:
        raise ValueError("initial state carries no bond to degrade")
    if float(temperature_c) >= tf_new:
        return 0.0
    vb_old = brine_volume_fraction(temperature_c, salinity_old_gkg)
    vb_new = brine_volume_fraction(temperature_c, salinity_new_gkg)
    from .adfreeze import SALINITY_EXPONENT

    return float(((1.0 - vb_new) / (1.0 - vb_old)) ** SALINITY_EXPONENT)


@dataclass(frozen=True)
class DegradationState:
    """Combined degradation of an adfreeze bond."""

    tau_peak_mpa: float
    f_peel: float
    f_thermal: float
    f_salinity: float

    @property
    def tau_residual_mpa(self) -> float:
        return self.tau_peak_mpa * self.f_peel * self.f_thermal * self.f_salinity

    @property
    def total_factor(self) -> float:
        return self.f_peel * self.f_thermal * self.f_salinity


def combined_degradation(
    tau_peak_mpa: float,
    slip_m: float = 0.0,
    g_c_j_m2: float = DEFAULT_INTERFACE_G_C,
    temperature_old_c: float | None = None,
    temperature_new_c: float | None = None,
    salinity_old_gkg: float | None = None,
    salinity_new_gkg: float | None = None,
    contact_time_s: float = 1.0e4,
    roughness_um: float = DEFAULT_ROUGHNESS_UM,
) -> DegradationState:
    """Multiplicative combination of the three degradation mechanisms.

    Any mechanism may be omitted (``None``) in which case its factor is 1.
    """
    f_peel = peel_softening(tau_peak_mpa, slip_m, g_c_j_m2) if slip_m > 0.0 else 1.0
    f_thermal = 1.0
    if temperature_old_c is not None and temperature_new_c is not None:
        f_thermal = thermal_degradation_factor(
            temperature_old_c,
            temperature_new_c,
            salinity_old_gkg or 0.0,
            contact_time_s,
            roughness_um,
        )
    f_sal = 1.0
    if (
        salinity_old_gkg is not None
        and salinity_new_gkg is not None
        and temperature_old_c is not None
    ):
        f_sal = salinity_degradation_factor(
            temperature_old_c, salinity_old_gkg, salinity_new_gkg
        )
    return DegradationState(tau_peak_mpa, f_peel, f_thermal, f_sal)


def peel_sweep(
    tau_peak_mpa: float,
    slip_max_m: float,
    g_c_j_m2: float = DEFAULT_INTERFACE_G_C,
    n: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slip sweep for plotting/energetics: ``(slip, traction, dissipated)``."""
    if slip_max_m <= 0.0:
        raise ValueError("slip_max_m must be positive")
    slips = np.linspace(0.0, slip_max_m, n)
    tractions = cohesive_law(tau_peak_mpa, slips, g_c_j_m2)
    delta_c = critical_slip(tau_peak_mpa, g_c_j_m2)
    # analytic dissipation per unit area up to each slip
    diss = np.where(
        slips < delta_c,
        tau_peak_mpa
        * 1.0e6
        * (slips - 0.5 * slips**2 / max(delta_c, 1e-30)),
        tau_peak_mpa * 1.0e6 * 0.5 * delta_c,
    )
    return slips, tractions, diss