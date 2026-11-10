"""Ice + wave combined fatigue: combination law, spectral closures, whipping share.

Public API::

    ice_anchor  -- m5528-calibrated ice load + hull-girder modal stress anchor
    sn          -- DNV-RP-C203 F3 (CP) S-N + Goodman + thickness + Miner
    processes   -- controlled stationary stress processes (spectra -> histories)
    combination -- the combination-law engine (ratio scan, spectral closures,
                   whipping accounting, rainflow commutativity)
"""

from .sn import (
    BASELINE_SN,
    CP_LIFE_FACTOR,
    DNV_F3_LOG_A1_AIR,
    DNV_F3_LOG_A1_CP,
    SNParams,
    SNCurveF3,
    goodman_effective_range,
    miner_damage,
)
from .ice_anchor import (
    ANCHOR_CASE,
    ANCHOR_REFERENCE_RESULT,
    ICE_PARAMS_CALIBRATED,
    FreeFreeBeamModes,
    IceContactParams,
    generate_ice_force,
    load_statistics,
    modal_stress_history,
    run_anchor_case,
    slow_varying_ice_params,
    whipping_share,
)

__all__ = [
    "BASELINE_SN", "CP_LIFE_FACTOR", "DNV_F3_LOG_A1_AIR", "DNV_F3_LOG_A1_CP",
    "SNParams", "SNCurveF3", "goodman_effective_range", "miner_damage",
    "ANCHOR_CASE", "ANCHOR_REFERENCE_RESULT", "ICE_PARAMS_CALIBRATED",
    "FreeFreeBeamModes", "IceContactParams", "generate_ice_force",
    "load_statistics", "modal_stress_history", "run_anchor_case",
    "slow_varying_ice_params", "whipping_share",
]
