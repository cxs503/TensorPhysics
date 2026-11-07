"""P34-G1 campaign integration gate (TensorPhysics).

Public API::

    from tensorphysics.campaign.p34_gate import (
        NODES, collect_campaign, collect_node, run_gate, evaluate_regression,
    )

The package is intentionally workflow-only: it evaluates delivered evidence
(test runs, artifacts, reports) and contains no domain kernel of its own.
"""
from __future__ import annotations

from .collect import (COLLECTION_SCHEMA, EVIDENCE_SCHEMA, PytestRecord,
                      collect_campaign, collect_node, check_json, check_png,
                      run_node_tests, sha256_file)
from .criteria import (FAIL, PASS, WITHHELD, Verdict, evaluate_c1, evaluate_c2,
                       evaluate_c3, evaluate_c4, evaluate_c5, scan_imports,
                       scan_lexicon, scan_placeholders)
from .gate import GATE_SCHEMA, run_gate, render_markdown
from .spec import (CAMPAIGN_ID, CAMPAIGN_TITLE, CRITERIA, DOMAIN_LEXICON,
                   GATE_ID, NODES, NODE_BY_ID, PLACEHOLDER_TOKENS,
                   REQUIRED_REGRESSION_FAMILIES, NodeSpec)

__all__ = [
    "CAMPAIGN_ID", "CAMPAIGN_TITLE", "CRITERIA", "GATE_ID", "NODES", "NODE_BY_ID",
    "REQUIRED_REGRESSION_FAMILIES", "NodeSpec", "DOMAIN_LEXICON",
    "PLACEHOLDER_TOKENS", "COLLECTION_SCHEMA", "EVIDENCE_SCHEMA", "GATE_SCHEMA",
    "PytestRecord", "Verdict", "collect_campaign", "collect_node", "check_json",
    "check_png", "run_node_tests", "run_gate", "render_markdown", "sha256_file",
    "evaluate_c1", "evaluate_c2", "evaluate_c3", "evaluate_c4", "evaluate_c5",
    "scan_imports", "scan_lexicon", "scan_placeholders", "PASS", "FAIL", "WITHHELD",
]