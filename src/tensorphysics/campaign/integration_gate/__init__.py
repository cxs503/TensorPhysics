"""P16-G1 campaign integration gate.

Public surface::

    from tensorphysics.campaign.integration_gate import run_gate, write_outputs, render_markdown

    verdict = run_gate("artifacts/p16_gate")
    write_outputs(verdict, "artifacts/p16_gate/gate_verdict.json",
                  "docs/P16_CAMPAIGN_INTEGRATION_GATE.md")

Sub-modules:

* :mod:`~tensorphysics.campaign.integration_gate.contract` -- the two
  cross-domain interface contracts (LBM wind table -> ship-platform;
  FEM wet modes + added mass -> BEM Helmholtz) and the consumer-endpoint
  compatibility checks.
* :mod:`~tensorphysics.campaign.integration_gate.nodes` -- acceptance
  evaluation of the eight P16 maturity / new-domain nodes with
  PASS / FAIL / WITHHELD semantics.
* :mod:`~tensorphysics.campaign.integration_gate.regression` -- P15 -> P16
  regression baseline for the eleven delivered P15 nodes (``|Δ| < 1%``).
* :mod:`~tensorphysics.campaign.integration_gate.gate` -- aggregation,
  provenance, verdict JSON and the campaign-level report.
"""
from .contract import (
    ContractResult,
    Check,
    WET_MODAL_SCHEMA,
    WIND_SCHEMA,
    validate_endpoint_compatibility,
    validate_wet_modal_handoff,
    validate_wind_coefficient_table,
)
from .gate import (
    GATE_ID,
    GATE_SCHEMA,
    load_evidence,
    main,
    render_markdown,
    run_gate,
    sha256_file,
    write_outputs,
)
from .nodes import (
    NODE_ARTIFACT_SCHEMA,
    CriterionVerdict,
    NodeVerdict,
    evaluate_node,
    evaluate_nodes,
    parse_criteria,
)
from .regression import (
    REGRESSION_THRESHOLD,
    CaseVerdict,
    evaluate_case,
    evaluate_regression,
    parse_pytest_counts,
    run_probe,
    run_probes,
)

__all__ = [
    "Check",
    "ContractResult",
    "CriterionVerdict",
    "NodeVerdict",
    "CaseVerdict",
    "GATE_ID",
    "GATE_SCHEMA",
    "NODE_ARTIFACT_SCHEMA",
    "WET_MODAL_SCHEMA",
    "WIND_SCHEMA",
    "REGRESSION_THRESHOLD",
    "evaluate_case",
    "evaluate_node",
    "evaluate_nodes",
    "evaluate_regression",
    "load_evidence",
    "main",
    "parse_criteria",
    "parse_pytest_counts",
    "render_markdown",
    "run_gate",
    "run_probe",
    "run_probes",
    "sha256_file",
    "validate_endpoint_compatibility",
    "validate_wet_modal_handoff",
    "validate_wind_coefficient_table",
    "write_outputs",
]