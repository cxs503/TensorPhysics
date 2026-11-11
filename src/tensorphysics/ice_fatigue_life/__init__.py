"""Ice-induced fatigue life assessment chain.

Complete assessment pipeline for ice zone structural fatigue:

1. S-N curves (DNV-RP-C203): seawater environment welds with low-temperature correction
2. Hot-spot stress method: Peterson K_t at plate-beam weld toes
3. Rainflow counting: ASTM E1049-85 cycle extraction from ice load spectra
4. Miner cumulative damage: D = Σ(n_i / N_i), failure at D ≥ 1.0
5. Validation against published ice fatigue examples

Key features:
- DNV-RP-C203 D curve (seawater, cathodic protection)
- Low-temperature correction: +15% fatigue strength at -20°C
- Peterson stress concentration factor for weld toes
- FEM sub-model mesh convergence check (K_t variation ≤ 3%)
- Comprehensive pytest suite with negative controls

Campaign node: p351w1_n04 — Ice-induced fatigue life assessment chain

References
----------
- DNV-RP-C203 (2021): Fatigue Design of Offshore Steel Structures
- ISO 19906 (2019): Petroleum and natural gas industries — Arctic structures
- ASTM E1049-85 (2017): Cycle counting in fatigue analysis
- Peterson, R.E. (1953): Stress Concentration Factors
- Su, B. et al. (2010): Ice-induced fatigue of offshore structures
"""

from .sn_dnv_rp_c203 import (
    SNCurveRP203,
    DNV_RP_C203_CURVES,
    low_temperature_correction,
    seawater_environment_factor,
    thickness_correction,
)
from .hotspot_stress import (
    HotSpotStress,
    PetersonKt,
    FEMSubmodelConvergence,
    plate_beam_kt_peterson,
    hot_spot_stress_from_nominal,
    fem_submodel_convergence_study,
    peterson_material_constant,
)
from .miner_accumulation import (
    MinerResult,
    miner_cumulative_damage,
    miner_exact_verification,
    fatigue_life_from_damage,
    miner_two_level_loading,
)
from .ice_load_spectrum import (
    IceLoadSpectrum,
    RainflowResult,
    generate_ice_load_time_series,
    rainflow_counting,
    stress_range_histogram,
    spectral_moments,
)
from .fatigue_assessment import (
    IceFatigueAssessment,
    IceFatigueResult,
    run_ice_fatigue_assessment,
    validate_against_reference,
)
from .whipping_strain_fatigue import (
    DEFAULT_STEEL_YOUNGS_MODULUS_PA,
    DetailFatigueResult,
    StrainBayesPosterior,
    VoyageFatigueLife,
    WhippingMomentHistory,
    bayesian_posterior_grid,
    bayesian_response_scale_update,
    damage_vs_scale_samples,
    half_sine_force_signal,
    ice_exposure_blocks_per_year,
    moment_ranges_to_stress_ranges_mpa,
    rainflow_miner_damage_from_moment,
    voyage_fatigue_life,
    whipping_moment_history,
)

__all__ = [
    # S-N curves
    "SNCurveRP203",
    "DNV_RP_C203_CURVES",
    "low_temperature_correction",
    "seawater_environment_factor",
    "thickness_correction",
    # Hot-spot stress
    "HotSpotStress",
    "PetersonKt",
    "FEMSubmodelConvergence",
    "plate_beam_kt_peterson",
    "hot_spot_stress_from_nominal",
    "fem_submodel_convergence_study",
    "peterson_material_constant",
    # Miner damage
    "MinerResult",
    "miner_cumulative_damage",
    "miner_exact_verification",
    "fatigue_life_from_damage",
    "miner_two_level_loading",
    # Ice load spectrum
    "IceLoadSpectrum",
    "RainflowResult",
    "generate_ice_load_time_series",
    "rainflow_counting",
    "stress_range_histogram",
    "spectral_moments",
    # Assessment
    "IceFatigueAssessment",
    "IceFatigueResult",
    "run_ice_fatigue_assessment",
    "validate_against_reference",
    # whipping + measured-strain Bayesian chain (P226W3-N02)
    "DEFAULT_STEEL_YOUNGS_MODULUS_PA",
    "DetailFatigueResult",
    "StrainBayesPosterior",
    "VoyageFatigueLife",
    "WhippingMomentHistory",
    "bayesian_posterior_grid",
    "bayesian_response_scale_update",
    "damage_vs_scale_samples",
    "half_sine_force_signal",
    "ice_exposure_blocks_per_year",
    "moment_ranges_to_stress_ranges_mpa",
    "rainflow_miner_damage_from_moment",
    "voyage_fatigue_life",
    "whipping_moment_history",
]
