"""Ice + wave combined fatigue damage: combination law, spectral methods, whipping.

Node ``P82W4-N06`` (mission ``m7361``) of the TensorPhysics ice-and-wave
campaign.  The package answers one design question -- *can the ice-induced and
wave-induced fatigue damages simply be added?* -- and ships the machinery needed
to answer it reproducibly:

``dem``               reduced-order DEM ice-crushing load (port of the m5528
                      IN32-N07 calibration, identical series for identical seeds)
``beam``              free-free Euler-Bernoulli hull-girder modal superposition,
                      stress recovery, wave stress transfer function
``sn``                DNV-RP-C203 bilinear S-N curves (m5528 parameter family),
                      Goodman correction, Miner sum
``rainflow_stream``   ASTM E1049-85 four-point rainflow with a streaming residue
                      stack (block-wise == whole-record, bit for bit)
``waves``             JONSWAP / bimodal elevation spectra, strip-theory wave load,
                      hull-girder wave stress PSD, spectral width measures
``combination``       the superposition experiments: band-separation scan, the
                      nonlinear combination law, spectral-method benchmark,
                      whipping bookkeeping, rainflow commutability
``pipeline``          the node driver: runs every machine criterion and returns a
                      JSON-serialisable results dictionary
``bimodal``           (P82W5-N09) band-split bimodal spectral closures: the
                      two-band riding damage ``D = D_rider + kappa D_carrier``,
                      the calibrated ``kappa(r,N)`` closure, the decidable
                      sum-rule envelope, and the spectral-method benchmark
``whipping_regime``   (P82W5-N09) whipping-share calibers over a declared
                      loading plane, with the regime boundary of the m5528
                      ``< 8 %`` claim and the force-jump diagnostic
``w5n09``             (P82W5-N09) the node driver: criteria (a)-(e) in one JSON

Everything numerical that involves randomness is seeded; ``pipeline.run_study``
records seeds, record lengths, sampling rates and the git commit so that the
numbers in ``artifacts/m7361/results.json`` can be regenerated.
"""

from .beam import (
    BeamModes,
    FREE_FREE_BETA_L,
    free_free_beta_roots,
    modal_stress_history,
    quasi_static_moment_free_free,
    whipping_share,
)
from .combination import (
    HullGirderCase,
    SeaState,
    band_separation_scan,
    calibrate_combination_law,
    combination_case,
    combination_law,
    evaluate_combination_law,
    harmonic_ice_stress,
    ice_stress_history,
    kappa_enlargement,
    m5528_reproduction,
    mechanism_diagnostic,
    overlap_weight,
    physical_combination,
    physical_wave_component,
    riding_amplification,
    segmentation_invariance,
    spectral_method_benchmark,
    stretched_ice_stress,
    upcrossing_rate,
    WIDTH_LEVELS,
    whipping_accounting,
)
from .dem import (
    CALIBRATED_M5528,
    IceContactParams,
    generate_ice_force,
    load_statistics,
    scaled_drift_rate,
)
from .bimodal import (
    N_SEPARATED,
    BandSplit,
    KappaClosure,
    band_case,
    band_split,
    bimodal_benchmark,
    equal_damage_amplitude_ratio,
    fit_kappa_closure,
    kappa_calibration_grid,
    riding_kappa_convolution,
    sum_rule_envelope,
    two_band_riding_damage,
)
from .pipeline import run_study, summarise_criteria
from .w5n09 import (
    CRITERION_A_RATIOS,
    SUM_RULE_BAND,
    commutation_check,
    run_w5n09_study,
    sum_rule_scan,
)
from .whipping_regime import (
    JUMP_FRACTION_MAX,
    SHARE_THRESHOLD,
    THETA_MIN,
    regime_consistency,
    regime_plane,
    whipping_regime_scan,
)
from .rainflow_stream import (
    StreamingRainflow,
    damage_from_cycles,
    equivalent_stress_range,
    rainflow_cycles,
    turning_points,
)
from .waves import (
    bimodal_f,
    jonswap_f,
    jonswap_omega,
    normalised_psd,
    spectral_moments,
    spectral_width_measures,
    wave_load_shape,
    wave_stress_psd,
    waterline_breadth,
)

__all__ = [
    "BeamModes", "FREE_FREE_BETA_L", "free_free_beta_roots", "modal_stress_history",
    "quasi_static_moment_free_free", "whipping_share",
    "HullGirderCase", "SeaState", "band_separation_scan", "calibrate_combination_law",
    "combination_case", "combination_law", "evaluate_combination_law",
    "harmonic_ice_stress", "ice_stress_history", "kappa_enlargement",
    "m5528_reproduction", "mechanism_diagnostic", "overlap_weight",
    "physical_combination", "physical_wave_component", "riding_amplification",
    "segmentation_invariance",
    "spectral_method_benchmark", "stretched_ice_stress", "upcrossing_rate",
    "WIDTH_LEVELS", "whipping_accounting",
    "CALIBRATED_M5528", "IceContactParams", "generate_ice_force", "load_statistics",
    "scaled_drift_rate",
    "run_study", "summarise_criteria",
    "N_SEPARATED", "BandSplit", "KappaClosure", "band_case", "band_split",
    "bimodal_benchmark", "equal_damage_amplitude_ratio", "fit_kappa_closure",
    "kappa_calibration_grid", "riding_kappa_convolution", "sum_rule_envelope",
    "two_band_riding_damage",
    "CRITERION_A_RATIOS", "SUM_RULE_BAND", "commutation_check", "run_w5n09_study",
    "sum_rule_scan",
    "JUMP_FRACTION_MAX", "SHARE_THRESHOLD", "THETA_MIN", "regime_consistency",
    "regime_plane", "whipping_regime_scan",
    "StreamingRainflow", "damage_from_cycles", "equivalent_stress_range",
    "rainflow_cycles", "turning_points",
    "bimodal_f", "jonswap_f", "jonswap_omega", "normalised_psd", "spectral_moments",
    "spectral_width_measures", "wave_load_shape", "wave_stress_psd", "waterline_breadth",
]