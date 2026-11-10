"""Three-field (LBM / FEM / DEM) coupling energy contract verification.

Campaign node **P40W4-N09**: runtime energy ledger of the closed LBM->FEM->
DEM->LBM loop, multi-rate sub-cycling with a degree-``p`` interface
reconstruction, and the conservation residual as a function of ``p``.

Modules
-------
``contracts``
    the three declared interface crossings (heat flux, contact force, drag
    power), their units/sign conventions, the two bookings each requires, and
    the machine-checkable reciprocity clause.
``energy``
    the four-term runtime book ``E = E_kin + E_pot + E_int + E_diss`` filled
    from solver states, with relative and absolute residuals.
``subcycling``
    the three-rate schedule and the causal degree-``p`` Lagrange reconstruction
    (``p = 1, 2, 3``) with its interval integral.
``loop``
    the closed loop itself, the uncoupled negative control, and the order scan.
``acceptance``
    frozen, hashed criteria and three-valued (PASS / FAIL / WITHHELD) verdicts.
"""
from __future__ import annotations

from .acceptance import (
    CRITERIA,
    CRITERIA_DIGEST,
    AcceptanceReport,
    CriterionOutcome,
    audit_record,
    criteria_digest,
    evaluate,
    format_table,
)
from .contracts import (
    CONTRACTS,
    CONTRACT_INDEX,
    ContractError,
    InterfaceContract,
    Quantity,
    assert_reciprocity,
    contract,
    coupling_graph,
    validate_contracts,
)
from .energy import COMPONENTS, Component, EnergyBook, EnergyBookError, EnergySample
from .loop import (
    LoopConfig,
    LoopResult,
    run_negative_control,
    reconstruction_operator_study,
    run_three_field_loop,
    scan_reconstruction_order,
)
from .subcycling import (
    History,
    RateSchedule,
    SubcyclingError,
    integrate_history,
    lagrange_value,
    observed_order,
)

__all__ = [
    "CRITERIA",
    "CRITERIA_DIGEST",
    "AcceptanceReport",
    "CriterionOutcome",
    "audit_record",
    "criteria_digest",
    "evaluate",
    "format_table",
    "CONTRACTS",
    "CONTRACT_INDEX",
    "ContractError",
    "InterfaceContract",
    "Quantity",
    "assert_reciprocity",
    "contract",
    "coupling_graph",
    "validate_contracts",
    "COMPONENTS",
    "Component",
    "EnergyBook",
    "EnergyBookError",
    "EnergySample",
    "LoopConfig",
    "LoopResult",
    "run_negative_control",
    "run_three_field_loop",
    "scan_reconstruction_order",
    "reconstruction_operator_study",
    "History",
    "RateSchedule",
    "SubcyclingError",
    "integrate_history",
    "lagrange_value",
    "observed_order",
]