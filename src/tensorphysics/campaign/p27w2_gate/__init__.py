"""P27-W2 campaign integration gate (node N10, mission 6665).

Public surface::

    from tensorphysics.campaign.p27w2_gate import (
        run_gate, write_outputs,
        CampaignDAG, CycleError, build_dag_from_plan, inject_cycle,
        verify_convergence,
        validate_rao_handoff, validate_granular_flow_handoff,
        validate_gci_uq_handoff, validate_all,
        build_evidence_ledger, EvidenceEntry,
        build_maturity_report, MaturityReport,
    )

Sub-modules:

* :mod:`~tensorphysics.campaign.p27w2_gate.dag` -- DAG construction,
  topological sort, cycle detection, convergence verification.
* :mod:`~tensorphysics.campaign.p27w2_gate.contracts` -- cross-repo
  interface contracts (BEM->ship RAO, DEM->TensorPhysics granular flow,
  LBM->platform GCI/UQ).
* :mod:`~tensorphysics.campaign.p27w2_gate.evidence` -- per-node evidence
  ledger (result.json + artifacts + commit).
* :mod:`~tensorphysics.campaign.p27w2_gate.maturity` -- maturation-node
  (N04 spectral fatigue, N09 AMR) aggregation.
* :mod:`~tensorphysics.campaign.p27w2_gate.gate` -- six-criterion
  aggregation, verdict JSON, output writing.
* :mod:`~tensorphysics.campaign.p27w2_gate.live` -- thin I/O against the
  TensorPlatform API (login, fetch, live gate run).
"""
from .dag import (
    CampaignDAG,
    ConvergenceResult,
    CycleError,
    DAGNode,
    TERMINAL_STATES,
    build_dag_from_plan,
    inject_cycle,
    is_terminal,
    normalize_status,
    verify_convergence,
)
from .contracts import (
    ContractCheck,
    ContractResult,
    GCI_UQ_SCHEMA,
    GRANULAR_SCHEMA,
    RAO_SCHEMA,
    validate_all,
    validate_gci_uq_handoff,
    validate_granular_flow_handoff,
    validate_rao_handoff,
)
from .evidence import (
    EvidenceEntry,
    EvidenceLedger,
    build_evidence_ledger,
    entry_from_mission,
    ledger_from_bundle,
)
from .maturity import (
    MaturityReport,
    NodeMaturity,
    P27W2_MATURITY_NODES,
    build_maturity_report,
    maturity_coverage,
    parse_maturity_artifact,
)
from .gate import (
    GATE_ID,
    GATE_SCHEMA,
    GateCriterion,
    GateVerdict,
    criterion_capability,
    criterion_contracts,
    criterion_dag,
    criterion_evidence,
    criterion_maturity,
    criterion_pytest,
    run_gate,
    write_outputs,
)
from .live import (
    INTEGRATION_NODE_ID,
    P27W2_PLAN_ID,
    build_ledger_from_live,
    build_maturity_from_live,
    collect_live_state,
    fetch_mission,
    fetch_plan_nodes,
    login,
    run_live_gate,
    write_all_outputs,
)

__all__ = [
    "CampaignDAG", "ConvergenceResult", "CycleError", "DAGNode",
    "TERMINAL_STATES", "build_dag_from_plan", "inject_cycle",
    "is_terminal", "normalize_status", "verify_convergence",
    "ContractCheck", "ContractResult", "GCI_UQ_SCHEMA", "GRANULAR_SCHEMA",
    "RAO_SCHEMA", "validate_all", "validate_gci_uq_handoff",
    "validate_granular_flow_handoff", "validate_rao_handoff",
    "EvidenceEntry", "EvidenceLedger", "build_evidence_ledger",
    "entry_from_mission", "ledger_from_bundle",
    "MaturityReport", "NodeMaturity", "P27W2_MATURITY_NODES",
    "build_maturity_report", "maturity_coverage", "parse_maturity_artifact",
    "GATE_ID", "GATE_SCHEMA", "GateCriterion", "GateVerdict",
    "criterion_capability", "criterion_contracts", "criterion_dag",
    "criterion_evidence", "criterion_maturity", "criterion_pytest",
    "run_gate", "write_outputs",
    "INTEGRATION_NODE_ID", "P27W2_PLAN_ID", "build_ledger_from_live",
    "build_maturity_from_live", "collect_live_state", "fetch_mission",
    "fetch_plan_nodes", "login", "run_live_gate", "write_all_outputs",
]
