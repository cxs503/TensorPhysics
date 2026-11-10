"""IACS Polar Class torsional-vibration screening and ice-milling fatigue life.

What is a rule and what is a model parameter
--------------------------------------------
* Rule *framework*: IACS Unified Requirement I (Polar Class) grades PC1..PC7 and
  requires ice-class shafting to be checked for torsional vibration and fatigue.
  The framework (which checks, which polar classes) is taken from the rule.
* The numeric allowable alternating shear stresses below are **model parameters**
  assembled from common classification practice for ice-class shafting.  They are
  NOT transcribed rule text and must be replaced by the applicable class society's
  values before any certification use.  Every returned dict carries the flag
  ``"parameter_status": "model_parameter"`` for that reason.

Fatigue
-------
S-N curve in shear with a single slope and a fatigue knee (steel shafting)::

    N(tau) = N_ref * (tau_ref / tau) ** m,      m = 3,   tau > tau_fl
    N(tau) = infinity,                                   tau <= tau_fl
    tau_fl = tau_ref * (N_ref / N_knee) ** (1 / m)

With the defaults (``tau_ref = 45 MPa`` at ``N_ref = 2e6``, ``N_knee = 1e7``)
the knee sits at ``tau_fl = 26.3 MPa``: amplitudes below it do not consume
damage, which is what the standard design curves assume for shaft steel.  Damage
is accumulated with Miner's rule over the discrete stress spectrum
``D = sum_i n_i / N_i``; the predicted life is ``1 / D`` exposure hours.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping, Optional, Sequence

__all__ = [
    "POLAR_CLASS_TORSION_LIMITS",
    "allowable_torsional_stress",
    "fatigue_limit",
    "sn_life",
    "miner_damage",
    "exposure_life",
    "iacs_torsion_check",
    "PARAMETER_STATUS",
]

PARAMETER_STATUS = "model_parameter"

#: Allowable alternating (zero-to-peak) shear stress in MPa for continuous
#: ice-milling operation.  Model parameters, see module docstring.
POLAR_CLASS_TORSION_LIMITS: Dict[str, float] = {
    "PC1": 30.0e6,
    "PC2": 32.0e6,
    "PC3": 34.0e6,
    "PC4": 36.0e6,
    "PC5": 40.0e6,
    "PC6": 44.0e6,
    "PC7": 48.0e6,
}

#: Transient (short-term, ice-milling peak) allowance factor on the above.
TRANSIENT_ALLOWANCE_FACTOR = 1.5


def allowable_torsional_stress(polar_class: str, transient: bool = False) -> float:
    """Allowable alternating shear stress [Pa] for a Polar Class."""
    key = polar_class.upper()
    if key not in POLAR_CLASS_TORSION_LIMITS:
        raise ValueError(f"unknown polar class {polar_class!r}; "
                         f"expected one of {sorted(POLAR_CLASS_TORSION_LIMITS)}")
    value = POLAR_CLASS_TORSION_LIMITS[key]
    return value * (TRANSIENT_ALLOWANCE_FACTOR if transient else 1.0)


def fatigue_limit(tau_ref: float = 45.0e6, n_ref: float = 2.0e6, slope: float = 3.0,
                  knee_cycles: float = 1.0e7) -> float:
    """Stress amplitude at the S-N knee (endurance limit) [Pa]."""
    if not (tau_ref > 0.0 and n_ref > 0.0 and slope > 0.0 and knee_cycles > n_ref):
        raise ValueError("require positive S-N parameters and knee_cycles > n_ref")
    return tau_ref * (n_ref / knee_cycles) ** (1.0 / slope)


def sn_life(stress_amplitude: float, tau_ref: float = 45.0e6,
            n_ref: float = 2.0e6, slope: float = 3.0,
            knee_cycles: float = 1.0e7) -> float:
    """Cycles to failure ``N = N_ref (tau_ref / tau) ** slope`` above the knee."""
    if stress_amplitude <= 0.0:
        return math.inf
    if not (tau_ref > 0.0 and n_ref > 0.0 and slope > 0.0):
        raise ValueError("S-N parameters must be positive")
    if stress_amplitude <= fatigue_limit(tau_ref, n_ref, slope, knee_cycles):
        return math.inf
    return n_ref * (tau_ref / stress_amplitude) ** slope


def miner_damage(amplitudes: Sequence[float], cycles: Sequence[float],
                 tau_ref: float = 45.0e6, n_ref: float = 2.0e6,
                 slope: float = 3.0, knee_cycles: float = 1.0e7) -> float:
    """Miner's rule damage ``D = sum n_i / N_i`` (dimensionless)."""
    if len(amplitudes) != len(cycles):
        raise ValueError("amplitudes and cycles must have the same length")
    damage = 0.0
    for amp, n in zip(amplitudes, cycles):
        if amp <= 0.0 or n <= 0.0:
            continue
        damage += n / sn_life(amp, tau_ref, n_ref, slope, knee_cycles)
    return damage


def exposure_life(cycles_per_hour: float, stress_amplitude: float,
                  tau_ref: float = 45.0e6, n_ref: float = 2.0e6,
                  slope: float = 3.0, knee_cycles: float = 1.0e7) -> Dict[str, float]:
    """Predicted life in hours and years for a single dominant stress amplitude.

    ``hours = 1 / (cycles_per_hour / N)``; years use ``hours_per_year`` = 6000
    (ice-operating exposure, a model assumption stated in the result).
    """
    if cycles_per_hour <= 0.0:
        raise ValueError("cycles_per_hour must be positive")
    n_fail = sn_life(stress_amplitude, tau_ref, n_ref, slope, knee_cycles)
    hours = math.inf if n_fail == math.inf else n_fail / cycles_per_hour
    years = hours / 6000.0
    return {"cycles_to_failure": n_fail, "life_hours": hours, "life_years": years,
            "hours_per_year": 6000.0,
            "fatigue_limit": fatigue_limit(tau_ref, n_ref, slope, knee_cycles)}


def iacs_torsion_check(polar_class: str, stress_amplitude: float,
                       cycles_per_hour: float, design_life_years: float = 20.0,
                       transient: bool = False, tau_ref: float = 45.0e6,
                       n_ref: float = 2.0e6, slope: float = 3.0,
                       knee_cycles: float = 1.0e7) -> Dict[str, object]:
    """Combined stress + fatigue screening for one Polar Class."""
    allowable = allowable_torsional_stress(polar_class, transient)
    life = exposure_life(cycles_per_hour, stress_amplitude, tau_ref, n_ref, slope,
                         knee_cycles)
    return {
        "polar_class": polar_class.upper(),
        "stress_amplitude": stress_amplitude,
        "allowable_stress": allowable,
        "stress_utilization": stress_amplitude / allowable,
        "stress_ok": stress_amplitude <= allowable,
        "design_life_years": design_life_years,
        "life_years": life["life_years"],
        "life_ok": life["life_years"] >= design_life_years,
        "parameter_status": PARAMETER_STATUS,
        "sn_ref": {"tau_ref": tau_ref, "n_ref": n_ref, "slope": slope,
                   "knee_cycles": knee_cycles, "fatigue_limit": life["fatigue_limit"]},
        "hours_per_year": life["hours_per_year"],
    }