"""LBM <-> DEM two-way coupling: momentum *and* energy conservation audit.

Campaign node ``p367seed-n07`` extends the existing energy contract of
:mod:`tensorphysics.coupling_energy` (``P40W3B-N09``) to the **momentum**
contract of an ice/water/fragmented-ice field, and states the momentum error
bound with a grid-refinement order.

Layers
------
``lattice``
    D2Q9 constants, Maxwellian, non-equilibrium stress -> rate of strain (the
    independent viscous-dissipation estimate used to close the energy budget).
``fluid``
    Closed-box BGK solver whose bounce-back exchange is *adjoint by
    construction*: the impulse handed to a wall/particle is exactly the impulse
    the fluid lost, and node creation/destruction is booked explicitly.
``dem``
    Rigid circular floes with linear spring contacts (Newton's third law is
    enforced by construction: one force per pair), plus the floe momentum and
    the contact elastic energy.
``coupling``
    The step loop, the momentum/energy ledgers and the three audit cases
    (single floe drift decay, fragmented-ice shear field, ship-ice push).
``gridscan``
    Fixed-Reynolds / fixed-lattice-Mach refinement map, three-level Richardson
    scan of the integrated interface impulse and the residual order.
``acceptance``
    Frozen, hashed criteria -> PASS/FAIL/WITHHELD verdict plus the JSON record.

Honest framing of the two residuals (they are different statements)
------------------------------------------------------------------
* ``momentum_residual`` is **structural**: because the exchange is adjoint, the
  interface cannot create momentum and the residual sits at round-off
  (``~1e-15``) independent of the grid.  It tests the *implementation*, and it
  is the quantity quoted against the ``1e-10`` criterion.
* ``momentum_order`` is a **discretisation** statement: the *integrated*
  interface impulse (drag / ship impulse, i.e. what a user reads off the
  simulation) still converges with ``dx``; its Richardson order ``p`` bounds the
  momentum error of a non-adjoint measurement.  ``p >= 1`` is the criterion.
"""
from __future__ import annotations

from .acceptance import (
    CRITERIA,
    CRITERIA_DIGEST,
    AcceptanceReport,
    Criterion,
    Outcome,
    evaluate,
    verdict_of,
)
from .cases import (
    all_cases,
    floe_drift_decay,
    ice_field_shear,
    negative_control_lbm_only,
    ship_ice_push,
)
from .coupling import AuditCase, CaseResult, run_case
from .dem import DEMConfig, DEMField, Floe
from .fluid import LBMFluid2D, StepDiagnostics
from .gridscan import RefinementLevel, refinement_levels, richardson_order

__all__ = [
    "CRITERIA",
    "CRITERIA_DIGEST",
    "AuditCase",
    "CaseResult",
    "Criterion",
    "DEMConfig",
    "DEMField",
    "Floe",
    "all_cases",
    "floe_drift_decay",
    "ice_field_shear",
    "negative_control_lbm_only",
    "ship_ice_push",
    "LBMFluid2D",
    "RefinementLevel",
    "StepDiagnostics",
    "AcceptanceReport",
    "Outcome",
    "evaluate",
    "refinement_levels",
    "richardson_order",
    "run_case",
    "verdict_of",
]