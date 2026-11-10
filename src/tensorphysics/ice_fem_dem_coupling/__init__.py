"""TensorFEM-DEM ice-structure coupling dynamics verification face.

This package implements the p299w4-n04 verification campaign node, covering
four physics stages of the ice-structure interaction chain:

1. **Plate bending (FEM)**: Winkler elastic foundation plate under concentrated
   load — bending stress vs Timoshenko & Woinowsky-Krieger (1959) analytical
   solution.  Criterion C1: error < 5%.

2. **Fracture handoff (FEM→DEM)**: When the FEM ice plate reaches its flexural
   strength, failure faces are exported as a ``failure_faces`` v3 packet and
   consumed by DEM as initial ice floes.  Criterion C2: mass conservation
   error < 1%.

3. **Floe drift (DEM)**: The broken ice floes drift in an ocean current field,
   validated against MT Uikku full-scale ice management trials.  Criterion C3:
   channel clearing area vs experiment error < 20%.

4. **Ice load time history**: The structural response to ice loading is
   characterised by peak force and Weibull distribution parameters, compared
   against model-scale and full-scale measurements.  Criterion C4: peak force
   error < 20%, Weibull shape parameter k error < 15%.

5. **Energy audit**: Global energy conservation across the FEM→DEM transition.
   Criterion C5: error < 10%.

References
----------
* Timoshenko, S. & Woinowsky-Krieger, S. (1959). *Theory of Plates and
  Shells*, 2nd ed., McGraw-Hill, Art. 61.
* Hetényi, M. (1946). *Beams on Elastic Foundation*, Univ. Michigan Press.
* Kerr, A.D. (1975). CRREL Report 75-1.
* Lu, W., Riska, K., & Ehlers, S. (2015). "Ice-induced loads on a ship hull",
  Proc. ICE 2015.
* Keinonen, A. et al. (1998). "MT Uikku full-scale ice trials",
  KVTT Report.
"""

from __future__ import annotations

from .plate_bending import (
    TimoshenkoPlateSolution,
    timoshenko_plate_point_load_stress,
    solve_plate_bending_verification,
    PlateBendingResult,
)
from .fracture_handoff import (
    FractureHandoffResult,
    run_fracture_handoff,
    mass_conservation_error,
)
from .floe_drift import (
    FloeProperties,
    DriftEnvironment,
    FloeDriftResult,
    simulate_floe_drift,
    channel_clearing_area,
    MT_UIKKU_REFERENCE,
)
from .ice_load_history import (
    IceLoadTimeHistory,
    fit_weibull,
    weibull_shape_parameter,
    IceLoadResult,
    run_ice_load_verification,
)
from .energy_audit import (
    EnergyAuditResult,
    run_energy_audit,
)
from .verification import (
    run_full_verification,
    VerificationReport,
)

__all__ = [
    # plate bending
    "TimoshenkoPlateSolution",
    "timoshenko_plate_point_load_stress",
    "solve_plate_bending_verification",
    "PlateBendingResult",
    # fracture handoff
    "FractureHandoffResult",
    "run_fracture_handoff",
    "mass_conservation_error",
    # floe drift
    "FloeProperties",
    "DriftEnvironment",
    "FloeDriftResult",
    "simulate_floe_drift",
    "channel_clearing_area",
    "MT_UIKKU_REFERENCE",
    # ice load
    "IceLoadTimeHistory",
    "fit_weibull",
    "weibull_shape_parameter",
    "IceLoadResult",
    "run_ice_load_verification",
    # energy audit
    "EnergyAuditResult",
    "run_energy_audit",
    # full verification
    "run_full_verification",
    "VerificationReport",
]
