"""Multi-Year Ice (MYI) Thermodynamic-Constitutive Model.

Implements the coupled thermodynamic-constitutive framework for multi-year ice,
including:
  - Stefan problem extension with salinity-dependent phase change temperature
  - Effective thermal conductivity as a function of porosity
  - Temperature-porosity dependent elastic modulus (Timco & Frederking, 2007)
  - Compressive strength model (Schulson, 1999)
  - Fracture toughness with Hall-Petch grain-size relation
  - Weibull strength distribution integration

References:
  - Timco, G.W. & Frederking, R.M.W. (2007). Cold Reg. Sci. Technol.
  - Schulson, E.M. (1999). J. Glaciol.
  - Dempsey, J.P. et al. (1999). Eng. Fract. Mech.
  - Frankenstein, G. & Garner, R. (1967). J. Glaciol.
"""

from .thermodynamics import (
    StefanSolver,
    melting_temperature,
    effective_thermal_conductivity,
    brine_volume_fraction,
    neumann_interface_position,
    neumann_temperature_profile,
)
from .constitutive import (
    elastic_modulus,
    compressive_strength,
    fracture_toughness,
    hall_petch_slope,
    WeibullStrengthDistribution,
)
from .myi_model import MYIModel

__all__ = [
    "StefanSolver",
    "melting_temperature",
    "effective_thermal_conductivity",
    "brine_volume_fraction",
    "neumann_interface_position",
    "neumann_temperature_profile",
    "elastic_modulus",
    "compressive_strength",
    "fracture_toughness",
    "hall_petch_slope",
    "WeibullStrengthDistribution",
    "MYIModel",
]
