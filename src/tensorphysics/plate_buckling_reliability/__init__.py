"""Unified plate-buckling + structural-reliability framework (W4-N03).

The package couples the two independent capabilities delivered by the previous
campaign nodes into one assessment chain:

* :mod:`tensorphysics.plate_buckling` (W3-N03) -- arc-length nonlinear
  finite-element buckling, initial-defect sensitivity, ULS check;
* :mod:`tensorphysics.structural` (W3-N04) -- Nataf/FORM/SORM/Monte-Carlo
  structural reliability with ISO 19906 style ice-belt limit states.

Public interface
----------------
``UnifiedBucklingReliability``
    One panel in, ``sigma_ultimate`` + ``beta`` + ``P_f`` (+ inverse design
    thickness) out.
``PanelSpec``
    Geometry/material/mesh/load specification with the pre-analysis geometry
    gate.
``UltimateStrengthSurrogate``
    Response surface ``sigma_u/sigma_y = c0 + c1/beta + c2/beta^2`` fitted to
    the deterministic FE limit points, with its residual reported.
``PanelLimitStates``
    The three limit states of the panel (ultimate, yield, deflection).
``ProbabilisticAssessment``
    FORM/SORM/Ditlevsen/Monte-Carlo/sensitivity/design-inverse driver.
``handle_calculation``
    Framework-neutral handler of ``POST /calculations/plate-buckling-reliability``.

Example
-------
>>> from tensorphysics.plate_buckling_reliability import (   # doctest: +SKIP
...     PanelSpec, UnifiedBucklingReliability)
>>> model = UnifiedBucklingReliability(PanelSpec(t=19.49e-3))
>>> result = model.assess(target_beta=4.0)
>>> round(result["beta"], 2) > 0 and "target_thickness" in result
True
"""

from .api import ENDPOINT, handle_calculation, handle_request, make_router, make_server, serve
from .deterministic import DeterministicAssessment
from .limit_states import (
    DEFAULT_MODES,
    MODE_DEFLECTION,
    MODE_ELASTIC,
    MODE_ULTIMATE,
    MODE_YIELD,
    VAR_NAMES,
    PanelLimitStates,
)
from .panel import DEFAULT_E, DEFAULT_NU, DEFAULT_SIGMA_Y, GeometryGateError, PanelSpec
from .reliability import (
    GenericLimitStates,
    ProbabilisticAssessment,
    VariableModel,
    beta_from_pf,
)
from .surrogate import UltimateStrengthSurrogate, faulkner_reference, load_default_surrogate
from .unified import DEFAULT_CALIBRATION_BETAS, UnifiedBucklingReliability

__all__ = [
    "UnifiedBucklingReliability",
    "PanelSpec",
    "GeometryGateError",
    "DeterministicAssessment",
    "UltimateStrengthSurrogate",
    "faulkner_reference",
    "load_default_surrogate",
    "PanelLimitStates",
    "ProbabilisticAssessment",
    "GenericLimitStates",
    "VariableModel",
    "beta_from_pf",
    "MODE_ULTIMATE",
    "MODE_YIELD",
    "MODE_ELASTIC",
    "MODE_DEFLECTION",
    "DEFAULT_MODES",
    "VAR_NAMES",
    "DEFAULT_CALIBRATION_BETAS",
    "DEFAULT_E",
    "DEFAULT_NU",
    "DEFAULT_SIGMA_Y",
    "ENDPOINT",
    "handle_calculation",
    "handle_request",
    "make_router",
    "make_server",
    "serve",
]