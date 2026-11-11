"""Tests for capability-aware routing with fail-closed evidence gating (p285w2b-n04).

The suite is mechanism-level: capability ids, solver ids and the forbidden-word
lexicon are data, so no engineering domain is encoded here.  Each test pins one
acceptance criterion of the mission node:

* C1 refusal coverage   -> NOT_SUPPORTED / ungraded / placeholder refusals
* C2 fallback ordering  -> evidence-ordered fallback vs brute-force oracle
* C3 gate latency       -> <= 50 ms per route
* C4 zero domain words  -> injected lexicon scan of the mechanism surface
* C5 evidence trail     -> every decision records grade + reason code
* C6 no placeholder     -> no MOCK/STUB selection outside explicit consent
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.campaign.wave3_gate import Grade
from tensorphysics.capability_routing import (
    AdmissibilityPolicy,
    CapabilityAwareRouter,
    CapabilityLedger,
    DecisionJournal,
    EvidenceGate,
    EvidenceLevel,
    GATE_LATENCY_BUDGET_MS,
    LedgerError,
    PolicyError,
    REASON_CODES,
    RoutingDecision,
    RoutingIntegrationError,
    RoutingRejected,
    UnknownEvidenceLevel,
    build_simulation,
)
from tensorphysics.capability_routing.acceptance import (
    fallback_oracle_check,
    latency_benchmark,
    refusal_sweep,
    synthetic_ledger,
)
from tensorphysics.capability_routing.lexicon import default_lexicon_path, load_terms, scan_files
from tensorphysics.config import SimulationConfig
from tensorphysics.protocols import SolverState


# --------------------------------------------------------------------- helpers
class FakeSolver:
    """Minimal protocol-conforming solver used to prove the scheduler gate."""

    def __init__(self, name: str) -> None:
        self._name = name
        self.steps = 0
        self.events: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    def initialize(self, context) -> None:
        self.events.append("initialize")

    def advance(self, context) -> None:
        self.steps += 1

    def finalize(self, context) -> None:
        self.events.append("finalize")

    def snapshot(self, context) -> SolverState:
        return SolverState(self._name, context.step, context.time, {"steps": self.steps})


def _ledger(rows) -> CapabilityLedger:
    return CapabilityLedger.from_config({"schema_version": 1, "rows": list(rows)})


def _row(capability, solver, grade, available=True, priority=0):
    return {"capability_id": capability, "solver_id": solver, "evidence": grade,
            "available": available, "priority": priority, "note": ""}


# ------------------------------------------------------------------- taxonomy
def test_evidence_rank_is_total_order_and_parse_is_fail_closed():
    ordered = sorted(EvidenceLevel, key=lambda g: -g.rank)
    assert [g.value for g in ordered] == ["REAL", "WITHHELD", "MOCK", "STUB", "NOT_SUPPORTED"]
    assert len({g.rank for g in EvidenceLevel}) == len(list(EvidenceLevel))
    assert EvidenceLevel.parse("real") is EvidenceLevel.REAL
    assert EvidenceLevel.parse(EvidenceLevel.MOCK) is EvidenceLevel.MOCK
    with pytest.raises(UnknownEvidenceLevel):
        EvidenceLevel.parse("PROBABLY_REAL")
    with pytest.raises(UnknownEvidenceLevel):
        EvidenceLevel.parse(None)


def test_vocabulary_matches_platform_grading_taxonomy():
    """Regression guard: the router cannot drift from the platform vocabulary."""
    assert {g.value for g in EvidenceLevel} == {g.value for g in Grade}


def test_policy_rejects_unknown_tier_and_illegal_mock_consent():
    with pytest.raises(PolicyError):
        AdmissibilityPolicy(tier="yolo")
    with pytest.raises(PolicyError):
        AdmissibilityPolicy(tier="production", allow_mock=True)
    with pytest.raises(PolicyError):
        AdmissibilityPolicy(preference="trust_the_name")
    with pytest.raises(PolicyError):
        AdmissibilityPolicy(min_rank=9)
    assert not AdmissibilityPolicy(tier="production").admits(EvidenceLevel.WITHHELD)
    assert AdmissibilityPolicy(tier="staged").admits(EvidenceLevel.WITHHELD)
    assert not AdmissibilityPolicy(tier="development").admits(EvidenceLevel.MOCK)
    assert AdmissibilityPolicy(tier="development", allow_mock=True).admits(EvidenceLevel.MOCK)
    for policy in (AdmissibilityPolicy(), AdmissibilityPolicy(tier="staged"),
                   AdmissibilityPolicy(tier="development", allow_mock=True)):
        assert not policy.admits(EvidenceLevel.NOT_SUPPORTED)
        assert not policy.admits("NOT_A_GRADE")


# --------------------------------------------------------------------- ledger
def test_ledger_rejects_unknown_keys_duplicates_and_bad_grades():
    with pytest.raises(LedgerError):
        _ledger([{**_row("cap_a", "solver_a", "REAL"), "evidance": "REAL"}])
    with pytest.raises(LedgerError):
        _ledger([_row("cap_a", "solver_a", "REAL"), _row("cap_a", "solver_a", "MOCK")])
    with pytest.raises(LedgerError):
        _ledger([_row("cap_a", "solver_a", "REALISH")])
    with pytest.raises(LedgerError):
        _ledger([_row("", "solver_a", "REAL")])
    with pytest.raises(LedgerError):
        CapabilityLedger.from_config({"schema_version": 2, "rows": []})
    with pytest.raises(LedgerError):
        CapabilityLedger.from_config({"schema_version": 1, "rows": []})


def test_ledger_orders_candidates_by_evidence_then_priority():
    ledger = _ledger([
        _row("cap_a", "solver_mock", "MOCK", priority=0),
        _row("cap_a", "solver_real_b", "REAL", priority=5),
        _row("cap_a", "solver_real_a", "REAL", priority=1),
        _row("cap_a", "solver_withheld", "WITHHELD", priority=0),
    ])
    assert [r.solver_id for r in ledger.candidates("cap_a")] == [
        "solver_real_a", "solver_real_b", "solver_withheld", "solver_mock"
    ]
    assert ledger.levels("cap_a")[0] == ("solver_real_a", "REAL")
    assert ledger.record("cap_a", "solver_absent") is None
    assert ledger.ranked("cap_a", unavailable=["solver_real_a"])[0].solver_id == "solver_real_b"


# ----------------------------------------------------------------------- gate
def test_gate_refuses_not_supported_and_ungraded_in_every_tier():
    ledger = _ledger([
        _row("cap_a", "solver_missing", "NOT_SUPPORTED"),
        _row("cap_b", "solver_real", "REAL"),
    ])
    for tier, allow_mock in (("production", False), ("staged", False), ("development", True)):
        gate = EvidenceGate(ledger, AdmissibilityPolicy(tier=tier, allow_mock=allow_mock))
        assert gate.admit("cap_a", "solver_missing").code == "EVIDENCE_NOT_SUPPORTED"
        # ungraded solver and unknown capability are refusals, never default-allow
        assert gate.admit("cap_b", "solver_unlisted").code == "SOLVER_UNKNOWN"
        assert gate.admit("cap_zzz", "solver_real").code == "CAPABILITY_UNKNOWN"
        with pytest.raises(RoutingRejected) as excinfo:
            gate.require("cap_a", "solver_missing")
        assert excinfo.value.code == "EVIDENCE_NOT_SUPPORTED"
        assert "refused" in str(excinfo.value)
        assert gate.explain("cap_b")["candidates"][0]["admitted"] is True


def test_gate_admits_mock_only_with_explicit_development_consent():
    ledger = _ledger([_row("cap_a", "solver_mock", "MOCK"), _row("cap_a", "solver_stub", "STUB")])
    production = EvidenceGate(ledger, AdmissibilityPolicy())
    assert production.admit("cap_a", "solver_mock").code == "EVIDENCE_FABRICATED_NO_CONSENT"
    staged = EvidenceGate(ledger, AdmissibilityPolicy(tier="staged"))
    assert staged.admit("cap_a", "solver_stub").code == "EVIDENCE_FABRICATED_NO_CONSENT"
    development = EvidenceGate(ledger, AdmissibilityPolicy(tier="development", allow_mock=True))
    admission = development.admit("cap_a", "solver_mock")
    assert admission.admitted and admission.mock_consumed and admission.provisional
    assert admission.as_dict()["level"] == "MOCK"


def test_refusal_sweep_hits_not_supported_100_percent():
    result = refusal_sweep(synthetic_ledger())
    assert result["not_supported_routes"] > 0
    assert result["not_supported_refusal_rate"] == 1.0
    assert result["placeholder_leaks"] == 0
    assert result["pass"] is True


# --------------------------------------------------------------------- router
def test_router_falls_back_to_best_graded_available_solver():
    ledger = _ledger([
        _row("cap_a", "solver_real", "REAL", priority=0),
        _row("cap_a", "solver_withheld", "WITHHELD", priority=0),
        _row("cap_a", "solver_mock", "MOCK", priority=0),
    ])
    router = CapabilityAwareRouter(ledger, AdmissibilityPolicy(tier="staged"))
    # a MOCK preferred solver must not win over better-graded evidence
    decision = router.route("cap_a", preferred="solver_mock")
    assert decision.selected == "solver_real"
    assert decision.level is EvidenceLevel.REAL
    assert [e[0] for e in decision.examined] == ["solver_real"]
    # an outage of the best solver degrades to the next grade, never to MOCK
    decision = router.route("cap_a", unavailable=["solver_real"])
    assert decision.selected == "solver_withheld"
    assert decision.provisional is True and decision.mock_consumed is False
    with pytest.raises(RoutingRejected) as excinfo:
        router.route("cap_a", unavailable=["solver_real", "solver_withheld"])
    assert excinfo.value.code == "NO_ADMISSIBLE_SOLVER"


def test_requested_not_supported_solver_is_hard_refused_in_every_tier():
    """Naming an unimplemented solver never degrades to a lower-graded sibling."""
    ledger = _ledger([
        _row("cap_a", "solver_missing", "NOT_SUPPORTED"),
        _row("cap_a", "solver_mock", "MOCK"),
        _row("cap_a", "solver_real", "REAL"),
    ])
    for policy in (AdmissibilityPolicy(), AdmissibilityPolicy(tier="staged"),
                   AdmissibilityPolicy(tier="development", allow_mock=True)):
        router = CapabilityAwareRouter(ledger, policy)
        with pytest.raises(RoutingRejected) as excinfo:
            router.route("cap_a", preferred="solver_missing")
        assert excinfo.value.code == "EVIDENCE_NOT_SUPPORTED"
        assert excinfo.value.solver_id == "solver_missing"
        decision = router.journal.decisions[-1]
        assert decision.selected is None and decision.mock_consumed is False
        assert decision.examined == (("solver_missing", "NOT_SUPPORTED",
                                      "EVIDENCE_NOT_SUPPORTED"),)
    # an outage of an admissible solver is a different matter: it may degrade
    consented = CapabilityAwareRouter(ledger,
                                      AdmissibilityPolicy(tier="development", allow_mock=True))
    degraded = consented.route("cap_a", unavailable=["solver_real"])
    assert degraded.selected == "solver_mock" and degraded.mock_consumed is True


def test_router_refuses_when_no_admissible_candidate_is_left():
    ledger = _ledger([
        _row("cap_a", "solver_real", "REAL"),
        _row("cap_a", "solver_mock", "MOCK"),
        _row("cap_b", "solver_missing", "NOT_SUPPORTED"),
    ])
    router = CapabilityAwareRouter(ledger, AdmissibilityPolicy())
    with pytest.raises(RoutingRejected) as excinfo:
        router.route("cap_a", unavailable=["solver_real"])
    assert excinfo.value.code == "NO_ADMISSIBLE_SOLVER"
    assert excinfo.value.capability_id == "cap_a"
    with pytest.raises(RoutingRejected) as excinfo:
        router.route("cap_b", preferred="solver_missing")
    assert excinfo.value.code == "EVIDENCE_NOT_SUPPORTED"
    with pytest.raises(RoutingRejected) as excinfo:
        router.route("cap_b")  # no preferred name: still nothing admissible
    assert excinfo.value.code == "NO_ADMISSIBLE_SOLVER"
    with pytest.raises(RoutingRejected) as excinfo:
        router.route("cap_unknown")
    assert excinfo.value.code == "CAPABILITY_UNKNOWN"
    # refusals are journalled too: an audit sees the attempt, not silence
    assert len(router.journal) == 4
    assert all(not d.admitted for d in router.journal.decisions)
    assert [d.code for d in router.journal.decisions] == [
        "NO_ADMISSIBLE_SOLVER", "EVIDENCE_NOT_SUPPORTED", "NO_ADMISSIBLE_SOLVER",
        "CAPABILITY_UNKNOWN",
    ]


def test_router_fallback_matches_bruteforce_oracle():
    result = fallback_oracle_check(synthetic_ledger(), cases=180, seed=31337)
    assert result["mismatches"] == []
    assert result["agreement_rate"] == 1.0
    assert result["refusal_cases_observed"] == result["refusal_cases_expected"]
    assert result["pass"] is True


def test_router_preferred_first_mode_still_respects_evidence_floor():
    ledger = _ledger([
        _row("cap_a", "solver_real", "REAL"),
        _row("cap_a", "solver_withheld", "WITHHELD"),
    ])
    policy = AdmissibilityPolicy(tier="staged", preference="preferred_first")
    router = CapabilityAwareRouter(ledger, policy)
    assert router.route("cap_a", preferred="solver_withheld").selected == "solver_withheld"
    production = CapabilityAwareRouter(ledger, AdmissibilityPolicy(preference="preferred_first"))
    assert production.route("cap_a", preferred="solver_withheld").selected == "solver_real"


def test_every_decision_records_evidence_level_and_reason_code(tmp_path):
    journal = DecisionJournal(tmp_path / "decisions.jsonl")
    router = CapabilityAwareRouter(synthetic_ledger(), AdmissibilityPolicy(tier="staged"),
                                   journal=journal)
    for record in router.ledger.records():
        try:
            router.route(record.capability_id, preferred=record.solver_id)
        except RoutingRejected:
            continue
    decisions = journal.decisions
    assert decisions and all(isinstance(d, RoutingDecision) for d in decisions)
    assert all(d.code in REASON_CODES for d in decisions)
    assert all(d.level is not None for d in decisions if d.admitted)
    lines = (tmp_path / "decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(decisions)
    payloads = [json.loads(line) for line in lines]
    assert all("level" in p and "code" in p and "tier" in p for p in payloads)
    levels = {p["level"] for p in payloads if p["level"]}
    assert levels <= {g.value for g in EvidenceLevel}
    assert any(p["code"] == "NO_ADMISSIBLE_SOLVER" for p in payloads)


def test_gate_latency_within_budget():
    result = latency_benchmark(synthetic_ledger(), routes=600, seed=11)
    assert result["n"] == 600
    # wall-clock distribution of the gate itself
    assert result["gate"]["mean_ms"] < GATE_LATENCY_BUDGET_MS
    assert result["gate"]["p99_ms"] < GATE_LATENCY_BUDGET_MS
    assert result["gate"]["p99_ms"] < 1.0
    assert result["gate"]["max_ms"] < GATE_LATENCY_BUDGET_MS
    # CPU-time readings are only claimed when the instrument is self-consistent
    if result["cpu_clock_usable"]:
        assert result["route_cpu"]["max_ms"] < GATE_LATENCY_BUDGET_MS
    assert result["null_baseline_max_ms"] >= 0.0
    assert result["pass"] is True


# ------------------------------------------------------------------- scheduler
def test_build_simulation_runs_the_routed_solver():
    ledger = _ledger([
        _row("cap_a", "solver_real", "REAL", priority=1),
        _row("cap_a", "solver_mock", "MOCK", priority=0),
    ])
    built: list[str] = []

    def factory(solver_id):
        def make():
            built.append(solver_id)
            return FakeSolver(solver_id)
        return make

    routed = build_simulation(
        SimulationConfig(dt=0.1, end_time=0.5),
        "cap_a",
        {solver_id: factory(solver_id) for solver_id in ("solver_real", "solver_mock")},
        ledger=ledger,
        preferred="solver_mock",  # requested placeholder must not be instantiated
    )
    assert built == ["solver_real"]
    assert routed.decision.selected == "solver_real"
    assert routed.manifest()["evidence_level"] == "REAL"
    routed.simulation.run()
    assert routed.solver.steps == 5
    assert routed.simulation.state.value == "finished"


def test_build_simulation_refuses_instead_of_degrading_to_mock():
    ledger = _ledger([_row("cap_a", "solver_missing", "NOT_SUPPORTED")])
    instantiated: list[str] = []

    def make():
        instantiated.append("solver_missing")
        return FakeSolver("solver_missing")

    with pytest.raises(RoutingRejected) as excinfo:
        build_simulation(SimulationConfig(dt=0.1, end_time=0.1), "cap_a",
                         {"solver_missing": make}, ledger=ledger)
    assert excinfo.value.code == "NO_ADMISSIBLE_SOLVER"
    assert instantiated == []  # fail-closed happens before any solver is built


def test_build_simulation_rejects_ungraded_factory_and_name_drift():
    ledger = _ledger([_row("cap_a", "solver_real", "REAL")])
    with pytest.raises(RoutingIntegrationError):
        build_simulation(SimulationConfig(dt=0.1, end_time=0.1), "cap_a",
                         {"solver_real": lambda: FakeSolver("solver_real"),
                          "solver_ghost": lambda: FakeSolver("solver_ghost")},
                         ledger=ledger)
    with pytest.raises(RoutingIntegrationError):
        build_simulation(SimulationConfig(dt=0.1, end_time=0.1), "cap_a",
                         {"solver_real": lambda: FakeSolver("solver_other")},
                         ledger=ledger)


# ------------------------------------------------------- C4 zero domain words
def test_mechanism_surface_has_no_domain_vocabulary():
    root = pathlib.Path(__file__).resolve().parents[1]
    lexicon = default_lexicon_path(root)
    terms = load_terms(lexicon)
    assert terms, "lexicon must not be empty"
    files = sorted((root / "src" / "tensorphysics" / "capability_routing").glob("*.py"))
    files += sorted((root / "tests").glob("test_capability_routing*.py"))
    assert len(files) >= 8
    result = scan_files(files, terms)
    assert result.files_scanned == len(files)
    assert result.hits == (), f"domain vocabulary leaked into mechanism code: {result.hits}"


def test_lexicon_scanner_is_not_vacuous():
    """A planted lexicon token is reported, and an empty scan is not a pass."""
    from tensorphysics.capability_routing.lexicon import LexiconError, scan_text

    terms = load_terms(default_lexicon_path(pathlib.Path(__file__).resolve().parents[1]))
    offender = sorted(terms)[0]
    planted = "def route_" + offender + "_solver(capability_id):\n    return capability_id\n"
    hits = scan_text(planted, terms)
    assert hits and hits[0]["term"] == offender
    with pytest.raises(LexiconError):
        scan_files([], terms)
