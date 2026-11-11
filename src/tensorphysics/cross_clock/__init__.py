"""Cross-clock multi-physics coupling: FEM / LBM / DEM with heterogeneous time steps.

The package answers one question in a machine-checkable form: **when three
solvers advance on different clocks, what exactly is conserved, what is
interpolated, and how large is the interpolation error?**

Modules
-------
``contract``
    the coupling-interface contract: solver clocks, integer clock ratios,
    interval-complete conserved-quantity samples and the per-window ledger
    (produced / applied / reference) with its drift definitions.
``remap``
    conservative time-rebinning of interval aggregates (donor-cell and
    slope-limited PLM), total-conserving by construction.
``interpolate``
    causal in-window reconstruction of a partner rate plus two *computable*
    error bounds (fine-sample and aggregate-only estimators).
``clock``
    the integer-ratio adaptive controller: refines on tolerance violation,
    coarsens only inside a hysteresis band, never churns.
``solvers``
    the three instruments: exact-step linear structure, D1Q3 lattice-BGK column
    with stage-wise momentum booking, exact-step drag particle, plus the
    analytic acoustic-structure eigenfrequency used as the independent anchor.
``scenarios``
    the two coupled cases (FEM-LBM FSI, DEM-LBM particle flow) driven through a
    clock-mismatched exchange, with reference attachment.
``acceptance``
    evaluates every node criterion from computed numbers and writes the
    verification record with its SHA256SUMS.
"""

from .contract import (
    QUANTITIES,
    ConservationPort,
    ContractError,
    CouplingContract,
    IntervalSample,
    SolverClock,
    WindowLedger,
    audit_record,
    check_tiling,
    drift_report,
)
from .remap import (
    RebinResult,
    conservative_regrid,
    first_moment,
    overlap_fractions,
    plm_slopes,
    regrid_report,
    van_leer_limiter,
)
from .interpolate import (
    HoldRate,
    LinearRate,
    bound_report,
    fine_rate_curvature,
    fine_rate_variation,
    interpolation_bounds,
    measured_error,
    reconstruct,
)
from .clock import AdaptiveClock, AdaptiveRun, ratio_scan
from .solvers import (
    DemParticle,
    FemStructure,
    LbmFluid1D,
    acoustic_structure_frequency,
    lattice_sound_speed,
)
from .scenarios import (
    DEM_CONTRACT,
    FEM_CONTRACT,
    FemLbmCase,
    attach_reference,
    default_dem_params,
    default_fem_params,
    dem_lbm_ode_reference,
    momentum_ledger,
    relative_bound,
    run_dem_lbm,
    run_fem_lbm,
)

__all__ = [
    "QUANTITIES",
    "ConservationPort",
    "ContractError",
    "CouplingContract",
    "IntervalSample",
    "SolverClock",
    "WindowLedger",
    "audit_record",
    "check_tiling",
    "drift_report",
    "RebinResult",
    "conservative_regrid",
    "first_moment",
    "overlap_fractions",
    "plm_slopes",
    "regrid_report",
    "van_leer_limiter",
    "HoldRate",
    "LinearRate",
    "bound_report",
    "fine_rate_curvature",
    "fine_rate_variation",
    "interpolation_bounds",
    "measured_error",
    "reconstruct",
    "AdaptiveClock",
    "AdaptiveRun",
    "ratio_scan",
    "DemParticle",
    "FemStructure",
    "LbmFluid1D",
    "acoustic_structure_frequency",
    "lattice_sound_speed",
    "DEM_CONTRACT",
    "FEM_CONTRACT",
    "FemLbmCase",
    "attach_reference",
    "default_dem_params",
    "default_fem_params",
    "dem_lbm_ode_reference",
    "momentum_ledger",
    "relative_bound",
    "run_dem_lbm",
    "run_fem_lbm",
]