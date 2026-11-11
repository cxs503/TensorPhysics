"""Residual-driven topic compiler: automatically generates candidate debt proposals
from campaign node limitations and audit capability gaps.

This module implements the p474w1-n03 residual compiler:
1. Scans done campaign nodes for limitations/untested/honest_fail fields
2. Scans audit capabilities for WITHHELD/NOT_SUPPORTED entries
3. Generates candidate proposal drafts with physical goals and criteria
4. Performs triple-retrieval deduplication (node names, proposal pool, concept grep)
5. Outputs prioritized candidate debt list

The compiler is fully automated with zero human intervention required.
"""
from .compiler import ResidualCompiler
from .schema import ProposalSchema, PROPOSAL_REQUIRED_FIELDS
from .scanner import CampaignNodeScanner, AuditCapabilityScanner
from .generator import ProposalGenerator
from .dedup import TripleRetrievalDedup

__all__ = [
    "ResidualCompiler",
    "ProposalSchema",
    "PROPOSAL_REQUIRED_FIELDS",
    "CampaignNodeScanner",
    "AuditCapabilityScanner",
    "ProposalGenerator",
    "TripleRetrievalDedup",
]
