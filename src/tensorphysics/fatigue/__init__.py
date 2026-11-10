"""Hull Fatigue Damage Chain — ice-loaded ship structural fatigue assessment.

Implements:
- ASTM E1049-85 four-point rainflow counting (list reference + O(N) stack)
- Spectral (frequency-domain) fatigue: m0..m4 -> Rayleigh / Dirlik /
  Tovo-Benasciutti range models, bimodal combination limit, PSD -> Gaussian
  history -> rainflow time-domain triangulation
- DNV-RP-C203 D/F S-N curves (seawater, free corrosion / CP protected)
- Palmgren-Miner linear cumulative damage + bilinear correction
- Weibull/Rayleigh load fitting + Gumbel extreme value extrapolation
- Hot-spot stress via Kt factor
- Detail fatigue: weld toe geometry → Kt → hot-spot extrapolation → S-N life
"""

from .rainflow import rainflow_counting, RainflowResult
from .sn_curve import DNCurve, SNCurveResult
from .miner import miner_damage, MinerResult
from .extrapolation import weibull_fit, rayleigh_fit, gumbel_extreme
from .hotspot import hotspot_stress
from .chain import FatigueDamageChain, ChainResult
from .spectral import (
    SpectralMoments, GaussianSpectrum, PiersonMoskowitz, JONSWAP, BimodalSpectrum,
    moments_from_dict, spectral_moments, spectral_moments_callable,
    rayleigh_damage_rate, rayleigh_expected_range_power, dirlik_parameters,
    DirlikParameters, dirlik_range_pdf, dirlik_expected_range_power,
    dirlik_damage_rate, tovo_benasciutti_weight, benasciutti_tovo_damage_rate,
    bimodal_analytic_damage_rate, discrete_psd_grid, synthesize_gaussian_history,
    time_domain_damage_rate, monte_carlo_damage_rate, sensitivity_table,
    spectral_fatigue_assessment,
)
from .detail_fatigue import (
    WeldToeGeometry,
    kt_nii_t_joint,
    kt_maddox_butt_joint,
    kt_fillet_joint_approx,
    linear_extrapolation,
    quadratic_extrapolation,
    HotSpotExtrapolationResult,
    fem_stress_field_t_joint,
    fem_stress_field_butt_joint,
    fem_stress_field_fillet_joint,
    mesh_convergence_analysis,
    MeshConvergenceResult,
    detail_fatigue_assessment,
    DetailFatigueResult,
)

__all__ = [
    "rainflow_counting", "RainflowResult",
    "DNCurve", "SNCurveResult",
    "miner_damage", "MinerResult",
    "weibull_fit", "rayleigh_fit", "gumbel_extreme",
    "hotspot_stress",
    "FatigueDamageChain", "ChainResult",
    "SpectralMoments", "GaussianSpectrum", "PiersonMoskowitz", "JONSWAP",
    "BimodalSpectrum", "moments_from_dict", "spectral_moments",
    "spectral_moments_callable", "rayleigh_damage_rate",
    "rayleigh_expected_range_power", "dirlik_parameters", "DirlikParameters",
    "dirlik_range_pdf", "dirlik_expected_range_power", "dirlik_damage_rate",
    "tovo_benasciutti_weight", "benasciutti_tovo_damage_rate",
    "bimodal_analytic_damage_rate", "discrete_psd_grid",
    "synthesize_gaussian_history", "time_domain_damage_rate",
    "monte_carlo_damage_rate", "sensitivity_table",
    "spectral_fatigue_assessment",
    # Detail fatigue
    "WeldToeGeometry",
    "kt_nii_t_joint",
    "kt_maddox_butt_joint",
    "kt_fillet_joint_approx",
    "linear_extrapolation",
    "quadratic_extrapolation",
    "HotSpotExtrapolationResult",
    "fem_stress_field_t_joint",
    "fem_stress_field_butt_joint",
    "fem_stress_field_fillet_joint",
    "mesh_convergence_analysis",
    "MeshConvergenceResult",
    "detail_fatigue_assessment",
    "DetailFatigueResult",
]

from .strain_life import (
    StrainLifeMaterial, n_prime_from_coefficients, k_prime_from_coefficients,
    elastic_strain_amplitude, plastic_strain_amplitude, total_strain_amplitude,
    strain_life_curve, reversals_for_strain_amplitude, cycles_to_failure,
    transition_reversals, transition_life, cyclic_strain_amplitude,
    stress_amplitude_from_strain, NeuberNotchResult, neuber_notch_strain,
    notch_factor_neuber, notch_factor_peterson, notch_factor,
    DamageBlock, StrainLifeDamageResult, strain_life_damage,
    endurance_limit_strain_amplitude,
)
from .ice_pressure_lowcycle import (
    RAYLEIGH_COV, cycles_from_exposure, nominal_stress_range_MPa,
    weibull_shape_from_cov, pulse_amplitude_blocks,
    IcePressureLowCycleResult, ice_pressure_low_cycle_assessment,
)
