"""Hull Fatigue Damage Chain — ice-loaded ship structural fatigue assessment.

Implements:
- ASTM E1049-85 four-point rainflow counting
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

__all__ = [
    "rainflow_counting", "RainflowResult",
    "DNCurve", "SNCurveResult",
    "miner_damage", "MinerResult",
    "weibull_fit", "rayleigh_fit", "gumbel_extreme",
    "hotspot_stress",
    "FatigueDamageChain", "ChainResult",
]
