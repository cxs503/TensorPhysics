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
]
