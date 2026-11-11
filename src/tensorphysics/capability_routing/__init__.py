"""Capability-aware solver routing with fail-closed evidence gating.

Platform mechanism only: this package contains no domain vocabulary.  The
evidence grades, the solver/capability names and the forbidden-word lexicon are
all injected as data, so the same code serves every solver family.

Public surface::

    from tensorphysics.capability_routing import (
        AdmissibilityPolicy, CapabilityLedger, EvidenceGate, RoutingRejected,
        CapabilityAwareRouter, DecisionJournal, build_simulation,
    )

    ledger = CapabilityLedger.from_path("ledger.json")
    router = CapabilityAwareRouter(ledger, AdmissibilityPolicy(tier="staged"))
    decision = router.route("capability_x", preferred="solver_a")   # raises on refuse

See ``docs/CAPABILITY_AWARE_ROUTING_P285W2B_N04.md`` for the semantics and the
measured acceptance numbers.
"""
from __future__ import annotations

from .evidence import (
    AdmissibilityPolicy,
    EvidenceLevel,
    PolicyError,
    TIER_ADMISSIBLE,
    UnknownEvidenceLevel,
)
from .gate import REASON_CODES, Admission, EvidenceGate, RoutingRejected
from .integration import RoutedSimulation, RoutingIntegrationError, build_simulation, route_solver
from .ledger import CapabilityLedger, LedgerError, SolverRecord
from .router import (
    GATE_LATENCY_BUDGET_MS,
    CapabilityAwareRouter,
    DecisionJournal,
    RoutingDecision,
    latency_stats,
)

__all__ = [
    "AdmissibilityPolicy",
    "Admission",
    "CapabilityAwareRouter",
    "CapabilityLedger",
    "DecisionJournal",
    "EvidenceGate",
    "EvidenceLevel",
    "GATE_LATENCY_BUDGET_MS",
    "LedgerError",
    "PolicyError",
    "REASON_CODES",
    "RoutedSimulation",
    "RoutingDecision",
    "RoutingIntegrationError",
    "RoutingRejected",
    "SolverRecord",
    "TIER_ADMISSIBLE",
    "UnknownEvidenceLevel",
    "build_simulation",
    "latency_stats",
    "route_solver",
]

__version__ = "1.0.0"