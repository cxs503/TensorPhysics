"""Blocker graph + scheduler tests: no false triggers, one-cycle latency, idempotency."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.fail_revival import acceptance, scheduler as sched, synth
from tensorphysics.fail_revival.blockers import (Blocker, BlockerState, build_graph,
                                                 blocker_done, load_declarations)
from tensorphysics.fail_revival.extract import extract_artifact, load_layout
from tensorphysics.fail_revival.model import sha256_text, task_key

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "tensor_reports" / "campaign_nodes"
LAYOUT = load_layout()


def _record(mission: int, line_no: int, error: str, allowed: str = "<3%",
            markdown_extra: str = "") -> object:
    md = "\n".join([
        "| quantity | reference | computed | error | <3%? |",
        "|---|---|---|---|---|",
        "| q_%d | 1.0 | 2.0 | %s | 否 |%s" % (line_no, error, markdown_extra),
    ])
    art = {
        "schema": "campaign_node_artifact.v1",
        "mission_id": mission,
        "plan_id": 1,
        "goal": synth.criterion_goal(("q error <= 3%",)),
        "report": {"id": mission, "markdown": md, "sha256": sha256_text(md)},
    }
    return extract_artifact(art, layout=LAYOUT).records[0]


def _fixture():
    """Three FAILs blocked by the three blocker kinds."""
    r1 = _record(9501, 3, "+50.0%")
    r2 = _record(9502, 3, "+40.0%")
    r3 = _record(9503, 3, "+30.0%")
    blockers = {
        "capability:alpha": Blocker("capability:alpha", "capability_slug", (r1.fail_id,)),
        "mission:8001": Blocker("mission:8001", "upstream_mission", (r2.fail_id,)),
        "caliber:gamma": Blocker("caliber:gamma", "numerical_caliber_change", (r3.fail_id,)),
    }
    graph = build_graph([r1, r2, r3], blockers, mission_deps={9503: [8001]})
    states = [
        BlockerState(),
        BlockerState(capabilities_done=frozenset({"alpha"})),
        BlockerState(capabilities_done=frozenset({"alpha"}), missions_done=frozenset({8001}),
                     caliber_revisions=frozenset({"caliber:gamma"})),
    ]
    return graph, states, (r1, r2, r3)


def test_graph_is_acyclic_and_carries_the_campaign_dag():
    graph, states, _ = _fixture()
    order = graph.topological_order()
    assert order is not None
    kinds = {k for _, _, k in graph.edges}
    assert "blocked_by" in kinds and "upstream_of" in kinds


def test_nothing_landed_triggers_nothing():
    graph, states, _ = _fixture()
    store = sched.EmittedTasks()
    report = sched.tick(graph, states[0], store, cycle=1)
    assert report.emitted == ()
    assert sched.audit_triggers(graph, [states[0]], [report])["false_trigger_count"] == 0


def test_one_cycle_latency_and_correct_emissions():
    graph, states, (r1, r2, r3) = _fixture()
    reports = sched.run_cycles(graph, states)
    per_cycle = [[t.fail_key for t in rep.emitted] for rep in reports]
    assert per_cycle[0] == []
    assert per_cycle[1] == [r1.fail_id]
    assert sorted(per_cycle[2]) == sorted([r2.fail_id, r3.fail_id])
    audit = sched.audit_triggers(graph, states, reports)
    assert audit["false_trigger_count"] == 0
    assert audit["missed_count"] == 0


def test_idempotent_repeat_emits_no_duplicate_tasks():
    graph, states, _ = _fixture()
    store = sched.EmittedTasks()
    first = sched.tick(graph, states[-1], store, cycle=1)
    second = sched.tick(graph, states[-1], store, cycle=2)
    assert len(first.emitted) == 3
    assert second.emitted == ()
    assert store.duplicates_blocked == 3
    assert len({t.task_key for t in store.ordered()}) == 3
    assert {t.task_key for t in store.ordered()} == {
        task_key(r.criterion_sha256, sched.baseline_input_digest(r)) for r in (r for r in graph.records)
    }


def test_task_body_carries_digests_only():
    graph, states, _ = _fixture()
    reports = sched.run_cycles(graph, states)
    body = reports[-1].emitted[0].to_dict()
    assert set(body) == {"task_key", "fail_key", "criterion_sha256", "input_sha256",
                         "blocker_id", "blocker_kind", "mission_id", "cycle", "schema"}
    assert all(body[k] for k in ("criterion_sha256", "input_sha256", "task_key"))


def test_undeclared_blocker_never_auto_lifts():
    rec = _record(9601, 3, "+90.0%")
    graph = build_graph([rec], {})
    assert graph.dangling() == (rec.fail_id,)
    store = sched.EmittedTasks()
    state = BlockerState(capabilities_done=frozenset({"anything"}),
                         missions_done=frozenset({9601}), caliber_revisions=frozenset({"x"}))
    assert sched.tick(graph, state, store, cycle=1).emitted == ()


def test_blocker_done_semantics_per_kind():
    state = BlockerState(capabilities_done=frozenset({"alpha"}),
                         missions_done=frozenset({8001}),
                         caliber_revisions=frozenset({"caliber:gamma"}))
    assert blocker_done(Blocker("capability:alpha", "capability_slug", ()), state)
    assert blocker_done(Blocker("mission:8001", "upstream_mission", ()), state)
    assert blocker_done(Blocker("caliber:gamma", "numerical_caliber_change", ()), state)
    assert not blocker_done(Blocker("capability:beta", "capability_slug", ()), state)


@pytest.mark.skipif(not CORPUS.exists(), reason="real campaign corpus not synced")
def test_real_fixture_trigger_contract():
    subset, graph, _ = acceptance.real_fixture(CORPUS)
    assert {r.mission_id for r in subset} == {7813, 7815, 7827}
    result = acceptance.criterion_c3(graph, subset)
    assert result["pass"], json.dumps(result, ensure_ascii=False)[:800]
    assert result["false_triggers"] == 0
    assert result["cycle1_emissions_when_nothing_landed"] == 0
    assert result["max_latency"] == 1


@pytest.mark.skipif(not CORPUS.exists(), reason="real campaign corpus not synced")
def test_real_declarations_cover_every_fail():
    subset, graph, blockers = acceptance.real_fixture(CORPUS)
    declared = {f for b in blockers.values() for f in b.fail_keys}
    for rec in subset:
        assert rec.fail_id in declared