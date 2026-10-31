"""Ultimate limit state (ULS) check of an ice-loaded plate panel.

The check follows the format used for Polar Class shell plating (IACS UR I
limit-state practice, consistent with DNV-RP-C201 plate capacity and the
partial-factor format of the Polar Code):

    utilisation  eta = gamma_F * S_k / R_k  <=  1.0

with

* ``S_k`` the characteristic (unfactored) load effect: the ice patch pressure
  plus the still-water pressure applied to the panel,
* ``R_k`` the characteristic resistance: the pressure that the panel carries at
  its ultimate (limit) point, obtained here from the arc-length nonlinear FE
  analysis rather than from an empirical capacity formula,
* ``gamma_F`` the load factor on the variable (ice) action.

Two levels of checking are reported:

1. **stress check** -- maximum von Mises stress under the factored load must not
   exceed ``sigma_y / gamma_M``;
2. **ultimate-capacity check** -- the factored demand must not exceed the
   numerically determined ultimate pressure, i.e. ``eta <= 1``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

__all__ = ["ULSCheckResult", "uls_check", "DEFAULT_FACTORS"]

#: Default partial factors (explicit and overridable).
DEFAULT_FACTORS = {
    "gamma_ice": 1.25,     # variable action factor on the ice load
    "gamma_water": 1.0,    # permanent action factor on the water pressure
    "gamma_m": 1.15,       # material factor on yield stress
    "target_sf": 1.5,      # required safety factor on the ultimate capacity
}


@dataclass
class ULSCheckResult:
    """Outcome of the ULS check (all quantities SI)."""

    sigma_y: float
    sigma_vm_max: float
    sigma_allow: float
    p_ice_design: float
    p_water_design: float
    p_ult: float
    utilisation: float
    safety_factor: float
    required_safety_factor: float
    stress_utilisation: float
    pass_stress: bool
    pass_ultimate: bool
    factors: Dict[str, float] = field(default_factory=dict)
    notes: str = ""

    @property
    def passed(self) -> bool:
        return bool(self.pass_stress and self.pass_ultimate)

    def as_dict(self) -> Dict[str, object]:
        d = {
            "sigma_y_MPa": self.sigma_y / 1e6,
            "sigma_vm_max_MPa": self.sigma_vm_max / 1e6,
            "sigma_allow_MPa": self.sigma_allow / 1e6,
            "p_ice_design_MPa": self.p_ice_design / 1e6,
            "p_water_design_MPa": self.p_water_design / 1e6,
            "p_ult_MPa": self.p_ult / 1e6,
            "utilisation": self.utilisation,
            "stress_utilisation": self.stress_utilisation,
            "safety_factor": self.safety_factor,
            "required_safety_factor": self.required_safety_factor,
            "pass_stress": self.pass_stress,
            "pass_ultimate": self.pass_ultimate,
            "passed": self.passed,
            "factors": self.factors,
            "notes": self.notes,
        }
        return d


def uls_check(sigma_y: float, sigma_vm_max: float, p_ice: float, p_water: float,
              p_ult: float, factors: Optional[Dict[str, float]] = None) -> ULSCheckResult:
    """Evaluate the ULS check of an ice-loaded plate panel.

    Parameters
    ----------
    sigma_y:
        Yield stress (Pa).
    sigma_vm_max:
        Maximum von Mises stress under the *factored* load (Pa).
    p_ice, p_water:
        Characteristic (unfactored) ice patch pressure and lateral water
        pressure (Pa).
    p_ult:
        Ultimate ice pressure the panel carries, defined as the pressure at the
        limit point of the pressure-deflection path with the water pressure
        held constant (Pa).
    """
    fac = dict(DEFAULT_FACTORS)
    if factors:
        fac.update(factors)

    sigma_allow = sigma_y / fac["gamma_m"]
    demand = fac["gamma_ice"] * p_ice + fac["gamma_water"] * p_water
    utilisation = demand / p_ult if p_ult > 0 else float("inf")
    # safety factor delivered by the panel: capacity over characteristic demand
    safety = p_ult / (p_ice + p_water) if (p_ice + p_water) > 0 else float("inf")
    stress_util = sigma_vm_max / sigma_allow

    return ULSCheckResult(
        sigma_y=sigma_y,
        sigma_vm_max=sigma_vm_max,
        sigma_allow=sigma_allow,
        p_ice_design=fac["gamma_ice"] * p_ice,
        p_water_design=fac["gamma_water"] * p_water,
        p_ult=p_ult,
        utilisation=utilisation,
        safety_factor=safety,
        required_safety_factor=fac["target_sf"],
        stress_utilisation=stress_util,
        pass_stress=bool(stress_util <= 1.0),
        pass_ultimate=bool(utilisation <= 1.0 and safety >= fac["target_sf"]),
        factors=fac,
        notes=("capacity from the nonlinear FE limit point; demand uses "
               "characteristic ice patch + still-water pressure"),
    )