"""Hull structural-detail fatigue maturity package (node p246w2-n09).

Capability chain
----------------
1. ``plane_stress_fem``         -- real T6 plane-stress FEM (patch-test verified),
                                   with the Kirsch hole, butt-weld toe notch and
                                   T-joint model problems;
2. ``hotspot_extrapolation``    -- 0.4t/0.9t/1.4t (and 0.5t/1t/1.5t) toe
                                   extrapolation with exact Lagrange coefficients;
3. ``sn_iiw``                   -- IIW FAT / DNV-RP-C203 S-N curves and log-log
                                   fitting (R^2 in log-log space);
4. ``damage``                   -- Palmgren-Miner accumulation, two-level block
                                   closed form, rainflow-based variable amplitude,
                                   and the explicit Corten-Dolan gap declaration;
5. ``acceptance``               -- C1..C7 acceptance suite producing the
                                   machine-readable evidence artifact.

The pre-existing ``tensorphysics.fatigue.detail_fatigue`` module is retained for
backward compatibility, but its ``fem_stress_field_*`` functions are analytic
surrogates -- this package is what provides a genuine, independent FEM path.
"""

from .sn_iiw import (  # noqa: F401
    IIWCurve,
    IIW_FAT_CLASSES,
    REFERENCE_CYCLES,
    SNLogLogFit,
    fatigue_curve,
    fat_from_log_a1,
    log_a1_from_fat,
    loglog_fit,
    reference_fit,
    simulate_test_series,
    standard_cross_check,
    design_to_mean_factor,
)
from .damage import (  # noqa: F401
    CORTEN_DOLAN_STATUS,
    KNOWN_LIMITATIONS,
    MinerDamage,
    block_history,
    blocks_to_failure,
    miner_damage,
    power_law_two_block_life,
    rainflow_block_damage,
    sequence_effect_probe,
    two_block_blocks_to_failure,
)
from .hotspot_extrapolation import (  # noqa: F401
    HotSpotResult,
    POINTS_IIW_LINEAR,
    POINTS_IIW_QUADRATIC,
    POINTS_TASK_QUADRATIC,
    extrapolate_from_profile,
    lagrange_extrapolate,
    standard_extrapolations,
)
from .acceptance import (  # noqa: F401
    run_acceptance,
    write_acceptance_artifact,
)

__all__ = [
    "IIWCurve", "IIW_FAT_CLASSES", "REFERENCE_CYCLES", "SNLogLogFit",
    "fatigue_curve", "fat_from_log_a1", "log_a1_from_fat", "loglog_fit",
    "reference_fit", "simulate_test_series", "standard_cross_check",
    "design_to_mean_factor",
    "CORTEN_DOLAN_STATUS", "KNOWN_LIMITATIONS", "MinerDamage", "block_history",
    "blocks_to_failure", "miner_damage", "power_law_two_block_life",
    "rainflow_block_damage", "sequence_effect_probe", "two_block_blocks_to_failure",
    "HotSpotResult", "POINTS_IIW_LINEAR", "POINTS_IIW_QUADRATIC",
    "POINTS_TASK_QUADRATIC", "extrapolate_from_profile", "lagrange_extrapolate",
    "standard_extrapolations",
    "run_acceptance", "write_acceptance_artifact",
]
