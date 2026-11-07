"""P20 campaign integration gate.

Public surface::

    from tensorphysics.campaign.p20_gate import (
        run_gate, write_outputs, render_markdown,
        CampaignDAG, CycleError, build_dag_from_plan, inject_cycle,
        validate_fsi_handoff, validate_hydrostatic_handoff,
        validate_seakeeping_handoff,
        collect_maturity, build_maturity_from_nodes,
    )

Sub-modules:

* :mod:`~tensorphysics.campaign.p20_gate.dag` -- DAG construction,
  topological sort, cycle detection, convergence verification.
* :mod:`~tensorphysics.campaign.p20_gate.contracts` -- cross-repo interface
  contracts (FSI LBM→FEM, hydrostatic CAD→ship, seakeeping BEM→ship).
* :mod:`~tensorphysics.campaign.p20_gate.maturity` -- maturity report
  collection and campaign-level summary.
* :mod:`~tensorphysics.campaign.p20_gate.gate` -- aggregation, verdict
  JSON, and markdown report generation.
"""
from .dag import (
    CampaignDAG,
    CycleError,
    ConvergenceResult,
    DAGNode,
    build_dag_from_plan,
    inject_cycle,
)
from .contracts import (
    ContractCheck,
    ContractResult,
    FSI_SCHEMA,
    HYDROSTATIC_SCHEMA,
    SEAKEEPING_SCHEMA,
    validate_fsi_handoff,
    validate_hydrostatic_handoff,
    validate_seakeeping_handoff,
)
from .maturity import (
    MATURITY_SCHEMA,
    P20_MATURITY_NODES,
    MaturitySummary,
    NodeMaturity,
    build_maturity_from_nodes,
    collect_maturity,
    parse_maturity_artifact,
)
from .gate import (
    GATE_ID,
    GATE_SCHEMA,
    REPORT_TITLE,
    load_evidence,
    render_markdown,
    run_gate,
    sha256_file,
    write_outputs,
)

__all__ = [
    # DAG
    "CampaignDAG",
    "CycleError",
    "ConvergenceResult",
    "DAGNode",
    "build_dag_from_plan",
    "inject_cycle",
    # Contracts
    "ContractCheck",
    "ContractResult",
    "FSI_SCHEMA",
    "HYDROSTATIC_SCHEMA",
    "SEAKEEPING_SCHEMA",
    "validate_fsi_handoff",
    "validate_hydrostatic_handoff",
    "validate_seakeeping_handoff",
    # Maturity
    "MATURITY_SCHEMA",
    "P20_MATURITY_NODES",
    "MaturitySummary",
    "NodeMaturity",
    "build_maturity_from_nodes",
    "collect_maturity",
    "parse_maturity_artifact",
    # Gate
    "GATE_ID",
    "GATE_SCHEMA",
    "REPORT_TITLE",
    "load_evidence",
    "render_markdown",
    "run_gate",
    "sha256_file",
    "write_outputs",
]
