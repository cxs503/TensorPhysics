"""Propeller blade ice-hydrodynamic coupled fatigue life (W7-N07).

Physics chain
-------------
1. ice load spectrum        : ice milling torque envelope -> blade-level impact
                              pulse train (phase locked to blade passage)
2. hydrodynamic alternating : unsteady blade element (Kuessner/Sears corrected)
                              driven by the circumferential wake harmonics
3. stress superposition     : mean (steady thrust/torque/centrifugal) + hydro
                              alternating + ice pulses
4. cycle counting           : ASTM E1049 rainflow on the combined history
5. Miner linear damage      : D = sum(n_i / N_i), N_i from Basquin S-N with
                              Goodman mean-stress correction
6. crack-initiation life    : exposure hours at D = 1, safety factor
                              SF = N_allowable / N_actual = 1 / D

Every number produced by this package is computed from a closed-form or
explicitly integrated physical model; no hard-coded result values are used.
Reference levels are stated per acceptance criterion in the report.
"""

from .materials_sn import MATERIALS, SNReference, BasquinSN, goodman_amplitude
from .beam_impact import Beam, AnalyticImpulse, finite_difference_beam, numerical_impulse
from .hydro_unsteady import (
    WakeField,
    blade_element_hydro,
    kussner_gust_response,
    sears_gust_function,
    sears_root_moment,
    theodorsen_function,
)
from .ice_spectrum import ICE_CASES, IceEnvelope, blade_impact_train, ice_milling_torque, impact_spectrum
from .blade_stress import BladeGeometry, root_stress_components
from .rainflow_miner import (
    initiation_life,
    miner_damage,
    rainflow_count,
    safety_factor,
    turning_points,
)

__all__ = [
    "MATERIALS",
    "SNReference",
    "BasquinSN",
    "goodman_amplitude",
    "Beam",
    "AnalyticImpulse",
    "finite_difference_beam",
    "numerical_impulse",
    "WakeField",
    "blade_element_hydro",
    "sears_gust_function",
    "sears_root_moment",
    "theodorsen_function",
    "kussner_gust_response",
    "ICE_CASES",
    "IceEnvelope",
    "ice_milling_torque",
    "impact_spectrum",
    "blade_impact_train",
    "BladeGeometry",
    "root_stress_components",
    "rainflow_count",
    "miner_damage",
    "initiation_life",
    "safety_factor",
    "turning_points",
]