"""Ice-structure contact phase change and melt-film evolution.

A self-contained verification-grade model of the three-way thermodynamic /
tribological coupling at a sliding ice/structure interface:

* :mod:`.pressure_melting` -- Clausius-Clapeyron pressure melting;
* :mod:`.stefan` -- Stefan moving-boundary reference solutions (Neumann similarity
  solution plus an independent 1-D enthalpy solver);
* :mod:`.film` -- melt-film thickness ODE from the mass and energy balance,
  including the refreezing (Stefan) sinks;
* :mod:`.adhesion` -- temperature dependence of ice adhesion strength through the
  interfacial quasi-liquid layer;
* :mod:`.tribology` -- Stribeck friction curve extended with the film parameter
  and a load-sharing mixture rule;
* :mod:`.competition` -- mechanism decomposition, melting onset speed and the
  parameter-space scan that demonstrates which mechanism dominates;
* :mod:`.verification` -- the acceptance-criteria harness that produces the
  machine-checkable JSON artefacts.

The package is a *contract-level* physics model: it is an auditable closure for
the orchestration layer, not a substitute for a specialist contact/CFD solver.
Every quantitative claim it makes is tied to an analytic reference solution or to
a documented empirical correlation -- see the module docstrings for the reference
of each relation.
"""

from .adhesion import IceAdhesionModel, qll_thickness, vdw_bridging_stress
from .competition import (
    direction_gates,
    dominant_mechanism,
    mechanism_decomposition,
    melting_onset_velocity,
    melting_onset_velocity_analytic,
    scan_mechanism_map,
)
from .film import ContactState, FilmModel
from .pressure_melting import (
    clapeyron_slope,
    integrate_melt_temperature,
    melt_point_depression,
    melt_temperature,
    melt_temperature_linear,
    pressure_melting_threshold,
    superheat,
)
from .properties import IceWaterProperties, StructureProperties, water_viscosity
from .stefan import (
    neumann_front,
    neumann_lambda,
    neumann_lambda_small_stefan,
    quasi_steady_front,
    solidification_time,
    solve_stefan_1d,
)
from .tribology import (
    StribeckParams,
    asperity_load_fraction,
    film_parameter,
    friction_coefficient,
    lubrication_regime,
    stribeck_curve,
    stribeck_curve_minimum,
    viscous_friction_coefficient,
)

__all__ = [
    "IceAdhesionModel",
    "qll_thickness",
    "vdw_bridging_stress",
    "direction_gates",
    "dominant_mechanism",
    "mechanism_decomposition",
    "melting_onset_velocity",
    "melting_onset_velocity_analytic",
    "scan_mechanism_map",
    "ContactState",
    "FilmModel",
    "clapeyron_slope",
    "integrate_melt_temperature",
    "melt_point_depression",
    "melt_temperature",
    "melt_temperature_linear",
    "pressure_melting_threshold",
    "superheat",
    "IceWaterProperties",
    "StructureProperties",
    "water_viscosity",
    "neumann_front",
    "neumann_lambda",
    "neumann_lambda_small_stefan",
    "quasi_steady_front",
    "solidification_time",
    "solve_stefan_1d",
    "StribeckParams",
    "asperity_load_fraction",
    "film_parameter",
    "friction_coefficient",
    "lubrication_regime",
    "stribeck_curve",
    "stribeck_curve_minimum",
    "viscous_friction_coefficient",
]