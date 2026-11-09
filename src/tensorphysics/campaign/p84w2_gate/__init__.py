"""P84-W2 campaign integration gate.

Campaign: P84W2-G12|N12 — Cross-domain integration & campaign gate.
Aggregates 11 nodes (N01-N11), performs cross-solver interface checks,
code-alignment verification for mature nodes, and lineage inheritance ruling.

Public surface::

    from tensorphysics.campaign.p84w2_gate import run_gate, write_outputs, render_markdown

    verdict = run_gate("artifacts/p84_w2")
    write_outputs(verdict, "artifacts/p84_w2/report.json",
                  "docs/P84_W2_CAMPAIGN_INTEGRATION_GATE.md")
"""
from .plan import CAMPAIGN_PLAN, PLAN_84_GAPS
from .contracts import (
    CrossDomainCheck,
    run_cross_domain_checks,
    LBM_TURBULENCE_CONTRACT,
    MULTIPHASE_PARTICLE_CONTRACT,
)
from .code_alignment import (
    CodeAlignmentCheck,
    run_code_alignment_checks,
    REGULATIONS,
)
from .lineage import run_lineage_ruling, LineageRuling
from .gate import run_gate, write_outputs, render_markdown, GATE_SCHEMA

__all__ = [
    "CAMPAIGN_PLAN", "PLAN_84_GAPS",
    "CrossDomainCheck", "run_cross_domain_checks",
    "LBM_TURBULENCE_CONTRACT", "MULTIPHASE_PARTICLE_CONTRACT",
    "CodeAlignmentCheck", "run_code_alignment_checks", "REGULATIONS",
    "LineageRuling", "run_lineage_ruling",
    "run_gate", "write_outputs", "render_markdown", "GATE_SCHEMA",
]
