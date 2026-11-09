"""IN66-N08: cross-domain integration gate for the ice-navigation loop.

Public surface::

    from tensorphysics.ice_nav_integration import CaseConfig, run_gate
    verdict = run_gate(CaseConfig(), outdir="artifacts/in66_n08")

Modules
-------
* :mod:`~tensorphysics.ice_nav_integration.contracts` -- the three versioned
  interface contracts (DEM->RANS volume fraction, RANS->FSI traction,
  FSI->DEM structural feedback) with pure-function validators.
* :mod:`~tensorphysics.ice_nav_integration.dem_field` -- floe statistics,
  mass-exact Eulerian coverage field, conservative upwind advection.
* :mod:`~tensorphysics.ice_nav_integration.rans_1d`  -- reduced-order streamwise
  Reynolds-averaged flow through the ice lane, ITTC-57 wall shear, porous drag.
* :mod:`~tensorphysics.ice_nav_integration.fsi_coupling` -- partitioned two-way
  panel coupling with Aitken relaxation and the added-mass stability domain.
* :mod:`~tensorphysics.ice_nav_integration.energy_ledger` -- five-channel
  resistance/energy ledger with explicit residual attribution.
* :mod:`~tensorphysics.ice_nav_integration.gate` -- the loop, the C1..C5
  verdict, figures, provenance and the report driver.
"""

from .contracts import (
    SCHEMA_HULL_TRACTION,
    SCHEMA_ICE_FIELD,
    SCHEMA_STRUCT_FEEDBACK,
    Check,
    ContractError,
    all_passed,
    build_ice_field_payload,
    build_struct_feedback_payload,
    build_traction_payload,
    contract_manifest,
    grid_hash,
    same_dimension,
    summarize,
    validate_hull_traction,
    validate_ice_field,
    validate_struct_feedback,
)
from .dem_field import (
    ChannelGrid,
    IceFieldState,
    advect_upwind,
    build_ice_field_state,
    coverage_field,
    field_ice_volume,
    floe_statistics,
)
from .energy_ledger import CHANNELS, accumulate, brash_channel_split, build_ledger
from .fsi_coupling import (
    CouplingResult,
    StructuralSetup,
    analytic_stability_limit,
    linear_gain,
    packing_pressure,
    solve_two_way,
    stability_scan,
)
from .gate import (
    CPU_BUDGET_S,
    GATE_ID,
    GATE_SCHEMA,
    CaseConfig,
    evaluate_criteria,
    geometry_audit,
    ice_channels,
    make_figures,
    run_gate,
    run_loop,
)
from .rans_1d import (
    FlowSetup,
    distribution_weights,
    hull_profile,
    ittc57_cf,
    open_water_resistance,
    porous_drag_density,
    solve_streamwise,
)
from .report import write_report

__all__ = [
    "CPU_BUDGET_S",
    "CHANNELS",
    "CaseConfig",
    "ChannelGrid",
    "Check",
    "ContractError",
    "CouplingResult",
    "FlowSetup",
    "GATE_ID",
    "GATE_SCHEMA",
    "IceFieldState",
    "SCHEMA_HULL_TRACTION",
    "SCHEMA_ICE_FIELD",
    "SCHEMA_STRUCT_FEEDBACK",
    "StructuralSetup",
    "accumulate",
    "advect_upwind",
    "all_passed",
    "analytic_stability_limit",
    "brash_channel_split",
    "build_ice_field_payload",
    "build_ice_field_state",
    "build_ledger",
    "build_struct_feedback_payload",
    "build_traction_payload",
    "contract_manifest",
    "coverage_field",
    "distribution_weights",
    "evaluate_criteria",
    "field_ice_volume",
    "floe_statistics",
    "geometry_audit",
    "grid_hash",
    "hull_profile",
    "ice_channels",
    "ittc57_cf",
    "linear_gain",
    "make_figures",
    "open_water_resistance",
    "packing_pressure",
    "porous_drag_density",
    "run_gate",
    "run_loop",
    "same_dimension",
    "solve_streamwise",
    "solve_two_way",
    "stability_scan",
    "summarize",
    "validate_hull_traction",
    "validate_ice_field",
    "validate_struct_feedback",
    "write_report",
]