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

``acceptance``
    frozen, hashed acceptance criteria (:class:`Criterion`), three-valued
    outcomes (PASS / FAIL / WITHHELD) and the verdict of a suite, including
    re-gating a stored ``verification.json`` record.

``exchange_operators``
    the exchange-operator algebra (``lag`` / ``centered`` /
    ``predictor_corrector``, one interface) with the stamp-convention rule and
    the interval-complete attribution ledger (campaign node ``p223w3-n06``).
``exchange_mms``
    manufactured-solution quadrature ladder: the temporal order of each operator
    measured on an analytic flux, with no solver in the loop.
``time_exchange``
    the three exchange chains (``viv`` / ``weld`` / ``settling``), each driven
    through a real receiver integrated exactly for a piecewise-constant input,
    with ``solve_ivp`` as oracle, plus the exchange-cadence ladder driver.
``exchange_gate``
    the acceptance gate of node ``p223w3-n06``: study + criteria + verdict.

``scenarios``
    the three coupled experiments of campaign node ``P40W3B-N09``:

    * ``run_thermal_lbm_fem``      — LBM <-> FEM conjugate heat transfer
    * ``run_mechanical_fem_dem``   — FEM <-> DEM contact (Newton's third law)
    * ``run_three_field``          — LBM <-> FEM <-> DEM energy chain

Every scenario returns the per-step ledger state, so the conservation residual
is reported as a time series and not as a single final number.
"""
from __future__ import annotations

from .acceptance import (
    CRITERIA,
    CRITERIA_DIGEST,
    AcceptanceError,
    AcceptanceReport,
    Criterion,
    CriterionOutcome,
    assert_accepted,
    audit_drift,
    criteria_digest,
    evaluate,
    evaluate_record,
    format_table,
    measurements,
)
from .exchange_gate import (
    ARMS,
    RATIOS,
    Criterion as ExchangeCriterion,
    gate_study,
    run_exchange_study,
)
from .exchange_mms import MMSLadder, MMSLevel, mms_order_ladder, mms_quadrature_level
from .exchange_operators import (
    SCHEME_TABLE as EXCHANGE_SCHEME_TABLE,
    ExchangeOperator,
    ExchangeScheme,
    IntervalAttribution,
    IntervalExchangeLedger,
    IntervalTransfer,
    StampConvention,
    StampRule,
    operator_for,
)
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
from .time_exchange import (
    CHAIN_SPECS,
    ChainResult,
    CouplingMMS,
    ExchangeChain,
    LadderResult,
    build_chain,
    run_cadence_ladder,
    run_ladder,
)
from .scenarios import (
    ScenarioResult,
    run_mechanical_fem_dem,
    run_suite,
    run_thermal_lbm_fem,
    run_three_field,
)

__all__ = [
    "ARMS",
    "EXCHANGE_SCHEME_TABLE",
    "RATIOS",
    "CHAIN_SPECS",
    "ChainResult",
    "CouplingMMS",
    "ExchangeChain",
    "ExchangeCriterion",
    "ExchangeOperator",
    "ExchangeScheme",
    "IntervalAttribution",
    "IntervalExchangeLedger",
    "IntervalTransfer",
    "LadderResult",
    "MMSLadder",
    "MMSLevel",
    "StampConvention",
    "StampRule",
    "build_chain",
    "gate_study",
    "mms_order_ladder",
    "mms_quadrature_level",
    "operator_for",
    "run_cadence_ladder",
    "run_exchange_study",
    "run_ladder",
    "CRITERIA",
    "CRITERIA_DIGEST",
    "AcceptanceError",
    "AcceptanceReport",
    "Criterion",
    "CriterionOutcome",
    "assert_accepted",
    "audit_drift",
    "criteria_digest",
    "evaluate",
    "evaluate_record",
    "format_table",
    "measurements",
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