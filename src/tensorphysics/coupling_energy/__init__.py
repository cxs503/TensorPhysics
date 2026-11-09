"""Cross-solver energy / momentum conservation contracts for partitioned coupling.

The package is split into three layers that are deliberately independent so
that each claim can be checked on its own:

``ledger``
    double-entry energy and momentum books, plus order estimation helpers.
``exchange``
    interface-history reconstruction (zero / first / second order) and the
    exchange-quadrature error that this reconstruction implies.
``solvers``
    minimal but real reference discretisations (D2Q9 thermal LBM, Q4 thermal
    FEM, Q4 plane-stress FEM, 2D DEM) used to produce genuine interface fluxes
    and forces.

``scenarios``
    the three coupled experiments of campaign node ``P40W3B-N09``:

    * ``run_thermal_lbm_fem``      — LBM <-> FEM conjugate heat transfer
    * ``run_mechanical_fem_dem``   — FEM <-> DEM contact (Newton's third law)
    * ``run_three_field``          — LBM <-> FEM <-> DEM energy chain

Every scenario returns the per-step ledger state, so the conservation residual
is reported as a time series and not as a single final number.
"""
from __future__ import annotations

from .exchange import (
    SCHEME_ORDER,
    ExchangeSample,
    MultiRateSchedule,
    TemporalScheme,
    quadrature_error,
    reconstruction_order_scan,
    reconstruction_poly,
)
from .ledger import (
    CouplingLedgerError,
    EnergyLedger,
    EnergySample,
    MomentumLedger,
    OrderEstimate,
    observed_order,
)
from .scenarios import (
    ScenarioResult,
    run_mechanical_fem_dem,
    run_suite,
    run_thermal_lbm_fem,
    run_three_field,
)

__all__ = [
    "SCHEME_ORDER",
    "ExchangeSample",
    "MultiRateSchedule",
    "TemporalScheme",
    "quadrature_error",
    "reconstruction_order_scan",
    "reconstruction_poly",
    "CouplingLedgerError",
    "EnergyLedger",
    "EnergySample",
    "MomentumLedger",
    "OrderEstimate",
    "observed_order",
    "ScenarioResult",
    "run_thermal_lbm_fem",
    "run_mechanical_fem_dem",
    "run_three_field",
    "run_suite",
]