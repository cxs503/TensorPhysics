"""DEM-FEM ice-structure coupling cross-validation face (node p380w2-n06).

Modules
-------
``lindqvist_ref``
    Adapter over the canonical Lindqvist implementation
    (:func:`tensorphysics.ice.resistance.level_ice_resistance`) providing the
    failure-instant and cycle-mean **line force** references together with the
    documented reading convention.
``dem_ice``
    Bonded-particle (lattice discrete element) ice sheet in a 2-D vertical
    slice: Euler-Bernoulli beam bonds, bending fracture at ``sigma_f``,
    Winkler buoyancy, and a bow plating whose contact is saturated by the ice
    crushing strength ``sigma_c``.  Produces the ice-resistance spectrum
    ``F_x(t)`` [N/m] with emergent sawtooth peaks.
``fem_plate``
    Elasto-plastic Mindlin plate (Q4, selective reduced integration) with
    layered J2 flow theory and plane-stress condensation, incremental Newton
    along the transferred load history, plus the independent Kirchhoff Navier
    reference solution.
``coupling``
    Impulse-conserving DEM -> FEM load transfer (exact piecewise-linear
    sub-cycling), synchronisation error metrics and the interface energy
    ledger.
``verification``
    Machine-checkable acceptance harness for criteria (a)-(f) of the campaign
    node, emitting ``artifacts/ice_structure_dem_fem/verification.json`` and
    the report figures.

Evidence grading of this package
--------------------------------
* DEM / FEM implementation : REAL (numerical, grid-converged, energy audited)
* plate elastic limit      : REAL (verified against the Kirchhoff Navier series)
* Lindqvist reference      : LITERATURE closed form (semi-empirical coefficients)
* absolute ice-load accuracy vs full scale : WITHHELD (no full-scale record)
* HSVA panel comparison    : WITHHELD (no digitised panel stress record)
"""

from .coupling import (  # noqa: F401
    CoupledResult,
    PatchLoad,
    iacs_patch_height,
    patch_load_from_spectrum,
    run_coupled_case,
    synchronisation_errors,
    transfer_impulse_consistent,
)
from .dem_ice import (  # noqa: F401
    IceSliceConfig,
    IceSliceResult,
    floating_beam_break_force,
    simulate_ice_slice,
)
from .fem_plate import (  # noqa: F401
    PlateConfig,
    PlateMaterial,
    PlateResponse,
    navier_reference_max_stress,
    solve_plate_history,
)
from .lindqvist_ref import PC6_SHIP, LindqvistReference, reference_for  # noqa: F401

__all__ = [
    "IceSliceConfig",
    "IceSliceResult",
    "PlateConfig",
    "PlateMaterial",
    "PlateResponse",
    "CoupledResult",
    "PatchLoad",
    "LindqvistReference",
    "PC6_SHIP",
    "simulate_ice_slice",
    "floating_beam_break_force",
    "solve_plate_history",
    "navier_reference_max_stress",
    "reference_for",
    "run_coupled_case",
    "patch_load_from_spectrum",
    "transfer_impulse_consistent",
    "synchronisation_errors",
    "iacs_patch_height",
]
