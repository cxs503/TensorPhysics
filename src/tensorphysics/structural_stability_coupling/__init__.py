"""Structural–stability coupling verification chain (P167W2-N07).

Integrates FEM elastoplastic plate-panel ultimate strength (N05) with
inclining-test 3D inversion (N06) to form a closed structural–stability
verification loop.

Modules
-------
gz_curve
    Righting-arm (GZ) computation with optional structural degradation
    correction.  Wall-sided and large-angle (waterplane integration) methods.
error_propagation
    Propagation of plate-thickness uncertainty δ_iso through to GZ curve
    via weight/centre-of-gravity sensitivity.
stiffness_degradation
    Elastoplastic yield → effective section modulus reduction → BM/GM
    degradation model.  Implements Faulkner plate-strip ultimate strength
    as the negative-control asymptote.
cross_validation
    3D waterline cutting → wet-surface pressure → plate-panel load vs.
    ultimate capacity consistency check.
negative_control
    Stiffness → 0 limiting-case gate: GM must decrease monotonically and
    converge to the Faulkner strip-limit value.
"""

from .gz_curve import GZCurveResult, compute_gz_curve, wall_sided_gz, HullFormData  # noqa: F401
from .error_propagation import (  # noqa: F401
    propagate_delta_iso,
    DeltaPropagationResult,
    PlatePanelContribution,
)
from .stiffness_degradation import (  # noqa: F401
    StiffnessDegradationResult,
    compute_degradation,
    degraded_gm,
    faulkner_effective_width,
    PlatePanelStructural,
)
from .cross_validation import (  # noqa: F401
    CrossValidationResult,
    cross_validate,
    WaterlinePanelLoad,
)
from .negative_control import NegativeControlResult, negative_control_gate  # noqa: F401

__all__ = [
    "GZCurveResult",
    "compute_gz_curve",
    "wall_sided_gz",
    "HullFormData",
    "DeltaPropagationResult",
    "propagate_delta_iso",
    "PlatePanelContribution",
    "StiffnessDegradationResult",
    "compute_degradation",
    "degraded_gm",
    "faulkner_effective_width",
    "PlatePanelStructural",
    "CrossValidationResult",
    "cross_validate",
    "WaterlinePanelLoad",
    "NegativeControlResult",
    "negative_control_gate",
]
