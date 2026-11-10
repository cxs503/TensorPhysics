"""Machine acceptance of the revival registry: C1..C6, with numbers and paths.

The acceptance run is a pure function of committed inputs:

* the real corpus ``tensor_reports/campaign_nodes/*.json`` (verbatim platform
  snapshots, written by ``tools/sync_campaign_nodes.py``);
* the blocker declarations ``tensor_reports/campaign_nodes/revival_blockers.json``
  (written by ``tools/make_revival_blockers.py``, provenance per declaration);
* the synthetic fixtures of :mod:`tensorphysics.fail_revival.synth` (abstract
  vocabulary only).

It writes ``artifacts/fail_revival/acceptance.json`` (numbers + paths for every
criterion) and ``artifacts/fail_revival/lineage.sqlite3`` (the lineage/audit
ledger produced by the demo adjudications).  Nothing is copied by hand.
"""
from __future__ import annotations

from pathlib import Path
import json
from typing import Any, Mapping, Sequence

from . import adjudicate as adj
from . import audit as audit_mod
from . import gate
from . import lineage as lin
from . import scan as scan_mod
from . import scheduler as sched
from . import synth
from .blockers import BlockerState, build_graph, load_declarations
from .extract import cross_check, extract_artifact, extract_directory, load_layout
from .model import Adjudication, FailRecord, RevivalRequest, sha256_text

REAL_FIXTURE_MISSIONS = (7813, 7815, 7827)
FIXED_AT = "2026-11-10T00:00:00+00:00"


# --------------------------------------------------------------------------- #
# corpus helpers
# --------------------------------------------------------------------------- #
def load_registry(corpus: Path | str) -> tuple[list[FailRecord], dict[str, FailRecord], dict, Any]:
    extraction = extract_directory(corpus)
    records = list(extraction.records)
    registry = {r.fail_id: r for r in records}
    blockers, _ = load_declarations(Path(corpus) / "revival_blockers.json")
    return records, registry, blockers, extraction


def mission_deps(corpus: Path | str) -> dict[int, tuple[int, ...]]:
    out: dict[int, tuple[int, ...]] = {}
    for path in sorted(Path(corpus).glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema") != synth.SYNTH_ARTIFACT_SCHEMA:
            continue
        deps = tuple(int(d) for d in (data.get("depends_on_missions") or ()))
        if deps:
            out[int(data["mission_id"])] = deps
    return out


# --------------------------------------------------------------------------- #
# C1 -- extraction completeness + digest consistency
# --------------------------------------------------------------------------- #
def criterion_c1(corpus: Path | str, records: Sequence[FailRecord], extraction,
                 *, sample: int = 20) -> dict[str, Any]:
    marks = sum(e.fail_marks for e in extraction.extractions)
    extracted = len(records)
    checked = cross_check(extraction, corpus, sample=sample)
    coverage = (extracted / marks) if marks else 1.0
    passed = (
        coverage >= 1.0
        and checked["sample_checked"] >= sample
        and checked["criterion_match_rate"] >= 1.0
        and checked["report_digest_match_rate"] >= 1.0
    )
    return {
        "id": "C1",
        "title": "extraction completeness + criterion/report digest consistency",
        "fail_marks_in_corpus": marks,
        "records_extracted": extracted,
        "extraction_rate": coverage,
        "withheld_rows_out_of_scope": sum(e.withheld_marks for e in extraction.extractions),
        "artifacts": len(extraction.extractions),
        "artifacts_with_report": sum(1 for e in extraction.extractions if e.status != "no_report"),
        "artifacts_tampered": [e.path for e in extraction.tampered],
        "cross_check_sample": checked["sample_checked"],
        "criterion_digest_match_rate": checked["criterion_match_rate"],
        "report_digest_match_rate": checked["report_digest_match_rate"],
        "cross_check_detail": checked["checked"],
        "pass": passed,
    }


# --------------------------------------------------------------------------- #
# C2 -- zero-rewrite gate
# --------------------------------------------------------------------------- #
def criterion_c2(records: Sequence[FailRecord], registry: Mapping[str, FailRecord]) -> dict[str, Any]:
    base = records[0]
    input_digest = sched.baseline_input_digest(base)
    cases = synth.tamper_cases(base.fail_id, base.criterion_sha256, input_digest)
    control = synth.admissible_case(base.fail_id, base.criterion_sha256, input_digest)
    outcomes = []
    for case in cases:
        req = gate.request_from_payload(case.payload)
        decision = gate.zero_rewrite_gate(req, registry)
        outcomes.append({
            "case": case.name,
            "expected": case.expected_reason,
            "actual": decision.reason,
            "accepted": decision.accepted,
            "reason_matches": decision.reason == case.expected_reason,
        })
    ctrl_req = gate.request_from_payload(control.payload)
    ctrl = gate.zero_rewrite_gate(ctrl_req, registry)
    rejected = sum(1 for o in outcomes if not o["accepted"])
    correct = sum(1 for o in outcomes if o["reason_matches"] and not o["accepted"])
    return {
        "id": "C2",
        "title": "zero-rewrite door: tampered/relaxed revival requests are refused",
        "cases": outcomes,
        "cases_total": len(outcomes),
        "rejected": rejected,
        "rejected_with_correct_reason": correct,
        "rejection_rate": rejected / len(outcomes) if outcomes else 0.0,
        "control_accepted": ctrl.accepted,
        "control_reason": ctrl.reason,
        "pass": rejected == len(outcomes) == 10 and correct == len(outcomes) and ctrl.accepted,
    }


# --------------------------------------------------------------------------- #
# C3 / C4 -- trigger correctness and idempotency
# --------------------------------------------------------------------------- #
def real_fixture(corpus: Path | str) -> tuple[list[FailRecord], Any, dict]:
    records, registry, blockers, extraction = load_registry(corpus)
    subset = [r for r in records if r.mission_id in REAL_FIXTURE_MISSIONS]
    graph = build_graph(subset, blockers, mission_deps=mission_deps(corpus),
                        provenance={"source": "plan 337 + m7827 real FAILs",
                                    "corpus": str(corpus)})
    return subset, graph, blockers


def _active_blockers(graph) -> list:
    """Blockers that actually cover a FAIL of the fixture (sorted by id)."""
    from .blockers import blocker_of

    ids = {b.blocker_id for r in graph.records for b in [blocker_of(r.fail_id, graph.blockers)] if b}
    return [graph.blockers[i] for i in sorted(ids)]


def _blockers_by_kind(graph, kind: str) -> list:
    return [b for b in _active_blockers(graph) if b.kind == kind]


def scenario_states(graph) -> list[BlockerState]:
    """Three cycles: nothing landed -> one capability landed -> everything landed.

    Blocker names are read from the graph (data), never hard-coded here, so the
    scenario cannot smuggle a domain term into the code.
    """
    capabilities = _blockers_by_kind(graph, "capability_slug")
    missions = _blockers_by_kind(graph, "upstream_mission")
    calibers = _blockers_by_kind(graph, "numerical_caliber_change")
    first_cap = capabilities[0]
    cap_slugs = {b.blocker_id.split(":", 1)[-1] for b in capabilities}
    mission_ids = {int(b.blocker_id.split(":", 1)[-1]) for b in missions}
    return [
        BlockerState(),
        BlockerState(capabilities_done=frozenset({first_cap.blocker_id.split(":", 1)[-1]})),
        BlockerState(
            capabilities_done=frozenset(cap_slugs),
            missions_done=frozenset(mission_ids),
            caliber_revisions=frozenset(b.blocker_id for b in calibers),
        ),
    ]


def criterion_c3(graph, records: Sequence[FailRecord]) -> dict[str, Any]:
    from .blockers import blocker_done, blocker_of

    states = scenario_states(graph)
    reports = sched.run_cycles(graph, states)
    emitted_per_cycle = [[t.fail_key for t in rep.emitted] for rep in reports]
    audit = sched.audit_triggers(graph, states, reports)

    # The landing event happens *after* the tick that still did not see it, so a
    # blocker first visible in the state of cycle k landed at the end of cycle
    # k-1.  Latency is then (emitted cycle - landing cycle): 1 means "the next
    # scheduler cycle", 0 would mean "the same cycle".
    landed: dict[str, int] = {}
    for rec in records:
        blocker = blocker_of(rec.fail_id, graph.blockers)
        if blocker is None or blocker.kind == "undeclared":
            continue
        for idx, state in enumerate(states, start=1):
            if blocker_done(blocker, state):
                landed[rec.fail_id] = max(idx - 1, 1)
                break
    emitted_at = {t.fail_key: rep.cycle for rep in reports for t in rep.emitted}
    latencies = {f: emitted_at[f] - landed[f] for f in emitted_at if f in landed}

    first_cap = _blockers_by_kind(graph, "capability_slug")[0]
    expected_first_cycle = sorted(first_cap.fail_keys)
    return {
        "id": "C3",
        "title": "trigger correctness on the real plan-337 fixture (m7813/m7815/m7827)",
        "fixture_fails": len(records),
        "cycles": len(states),
        "emitted_per_cycle": emitted_per_cycle,
        "tasks_total": audit["tasks"],
        "false_triggers": audit["false_trigger_count"],
        "missed_triggers": audit["missed_count"],
        "landed_cycle_after": landed,
        "latency_cycles": latencies,
        "max_latency": max(latencies.values()) if latencies else 0,
        "cycle1_emissions_when_nothing_landed": len(emitted_per_cycle[0]),
        "first_cycle_after_capability_landing": emitted_per_cycle[1],
        "expected_after_capability_landing": expected_first_cycle,
        "pass": (
            len(emitted_per_cycle[0]) == 0
            and audit["false_trigger_count"] == 0
            and audit["missed_count"] == 0
            and max(latencies.values() or [0]) <= 1
            and min(latencies.values() or [1]) >= 1
            and sorted(emitted_per_cycle[1]) == expected_first_cycle
            and len(landed) == len(set(emitted_at))
        ),
    }


def criterion_c4(graph) -> dict[str, Any]:
    states = scenario_states(graph)
    run_a = [rep.to_dict() for rep in sched.run_cycles(graph, states)]
    run_b = [rep.to_dict() for rep in sched.run_cycles(graph, states)]
    same = json.dumps(run_a, sort_keys=True, ensure_ascii=False) == json.dumps(
        run_b, sort_keys=True, ensure_ascii=False)

    store = sched.EmittedTasks()
    first = sched.tick(graph, states[-1], store, cycle=1)
    second = sched.tick(graph, states[-1], store, cycle=2)
    keys_first = {t.task_key for t in first.emitted}
    keys_second = {t.task_key for t in second.emitted}
    task_bodies = [t.to_dict() for t in store.ordered()]
    editable = [
        body for body in task_bodies
        if set(body) - {"task_key", "fail_key", "criterion_sha256", "input_sha256",
                        "blocker_id", "blocker_kind", "mission_id", "cycle", "schema"}
    ]
    unique_keys = len({t.task_key for t in store.ordered()}) == len(store.ordered())
    return {
        "id": "C4",
        "title": "idempotency / determinism of the scheduler",
        "identical_repeat_runs": same,
        "run_a_tasks": sum(len(r["emitted"]) for r in run_a),
        "run_b_tasks": sum(len(r["emitted"]) for r in run_b),
        "first_tick_tasks": len(keys_first),
        "second_tick_new_tasks": len(keys_second - keys_first),
        "duplicates_blocked": store.duplicates_blocked,
        "task_keys_unique": unique_keys,
        "task_body_fields_outside_core": editable,
        "pass": same and not keys_second and unique_keys and store.duplicates_blocked >= 1
                and not editable and len(keys_first) > 0,
    }


# --------------------------------------------------------------------------- #
# C5 -- anti-whitewash audit
# --------------------------------------------------------------------------- #
def criterion_c5(graph, records: Sequence[FailRecord], db_path: Path) -> dict[str, Any]:
    """Adjudicate the real revival tasks, prove NOT_COMPARABLE retention, and
    refuse a forged PASS.

    Real tasks: a caliber-revision revival adopts a *revised* caliber, so its
    verdict cannot be compared with the old one (NOT_COMPARABLE, change list
    mandatory, stays in the FAIL family); a capability/upstream-mission revival
    runs the registered criterion on a new code version but reproduces the
    baseline first (reproduction control), so it is comparable and -- with the
    old deviation re-observed -- STILL_FAIL.

    Two controls close the loop: a synthetic record with a registered tolerance
    and a within-tolerance measurement must reach PASS (the mechanism is not
    vacuously all-fail), and a forged PASS carrying an unregistered criterion
    digest must be refused and leave an OVERCLAIM row.
    """
    if db_path.exists():
        db_path.unlink()
    conn = lin.connect(db_path)
    store = sched.EmittedTasks()
    states = scenario_states(graph)
    for rep in sched.run_cycles(graph, states):
        for task in rep.emitted:
            store.add(task)
    tasks = store.ordered()
    by_fail = {r.fail_id: r for r in records}

    demo: list[dict[str, Any]] = []
    verdicts: list[Adjudication] = []
    for task in tasks:
        rec = by_fail[task.fail_key]
        revised = task.blocker_kind == "numerical_caliber_change"
        measured = None
        # a deviation is only used as a percentage measurement when the archived
        # cell actually carries a percent sign ("5x over" is not "5%")
        if not revised and rec.allowed_source == "header_threshold" and "%" in (rec.error or ""):
            measured = adj.parse_number(rec.error)
        outcome = adj.RevivalOutcome(
            inputs_sha256=task.input_sha256,
            caliber_id=task.blocker_id + (":revised" if revised else ""),
            code_version="revival-2026-11",
            measured=measured,
            reproduced_baseline=True,
        )
        verdict = adj.adjudicate(task, rec, outcome,
                                 baseline_inputs_sha256=task.input_sha256,
                                 baseline_caliber=task.blocker_id,
                                 baseline_code_version="node-run")
        verdicts.append(verdict)
        audit_mod.audited_adjudicate(conn, verdict, rec, task, at=FIXED_AT)
        demo.append({
            "fail_key": task.fail_key,
            "mission_id": rec.mission_id,
            "blocker_kind": task.blocker_kind,
            "verdict": verdict.verdict,
            "tolerance": rec.allowed,
            "measured": measured,
            "changes": [dict(c) for c in verdict.changes],
            "family_member": verdict.family_member,
            "reasons": list(verdict.reasons),
        })

    # ---- PASS control on a synthetic record (registered tolerance, in band)
    layout = load_layout()
    synth_art = synth.corpus()[0]
    synth_ex = extract_artifact(synth_art, path="<synth>", layout=layout)
    synth_rec = synth_ex.records[0]
    synth_task = sched.RevivalTask(
        task_key=sha256_text("synth|" + synth_rec.criterion_sha256),
        fail_key=synth_rec.fail_id, criterion_sha256=synth_rec.criterion_sha256,
        input_sha256=sha256_text("synth-input"), blocker_id="capability:synth-control",
        blocker_kind="capability_slug", mission_id=synth_rec.mission_id, cycle=1,
    )
    synth_outcome = adj.RevivalOutcome(inputs_sha256=synth_task.input_sha256,
                                       caliber_id="caliber:synth-control",
                                       code_version="revival-2026-11", measured=1.2,
                                       reproduced_baseline=True)
    synth_verdict = adj.adjudicate(synth_task, synth_rec, synth_outcome,
                                   baseline_inputs_sha256=synth_task.input_sha256,
                                   baseline_caliber="caliber:synth-control",
                                   baseline_code_version="revival-2026-11")
    audit_mod.audited_adjudicate(conn, synth_verdict, synth_rec, synth_task, at=FIXED_AT)

    # ---- forged PASS: unregistered criterion digest
    victim_task = tasks[0]
    victim_rec = by_fail[victim_task.fail_key]
    forged = Adjudication(task_key=victim_task.task_key, fail_key=victim_task.fail_key,
                          criterion_sha256="0" * 64, verdict="PASS", changes=(),
                          family_member=False, reasons=("forged_digest",))
    refused = False
    refusal_reason = ""
    try:
        audit_mod.audited_adjudicate(conn, forged, victim_rec, victim_task, at=FIXED_AT)
    except audit_mod.WhitewashRefused as exc:
        refused = True
        refusal_reason = exc.reason
    overclaims = audit_mod.overclaim_count(conn)
    forged_edges = [e for e in lin.edge_of_task(conn, victim_task.task_key)
                    if e["new_verdict"] == "PASS"]

    counts = {v: sum(1 for a in verdicts if a.verdict == v) for v in ("PASS", "STILL_FAIL", "NOT_COMPARABLE")}
    not_comparable_changes_ok = all(bool(a.changes) for a in verdicts if a.verdict == "NOT_COMPARABLE")
    family_ok = all(a.family_member for a in verdicts if a.verdict != "PASS")
    return {
        "id": "C5",
        "title": "three-way adjudication, family retention and anti-whitewash refusal",
        "adjudications": len(verdicts),
        "verdict_counts": counts,
        "not_comparable_with_change_list": not_comparable_changes_ok,
        "not_comparable_retained_in_family": family_ok,
        "pass_path_control_verdict": synth_verdict.verdict,
        "pass_path_control_reasons": list(synth_verdict.reasons),
        "forged_pass_refused": refused,
        "refusal_reason": refusal_reason,
        "overclaim_rows": overclaims,
        "forged_pass_edges_written": len(forged_edges),
        "lineage_edges": lin.edge_count(conn),
        "adjudication_demo": demo,
        "lineage_db": str(db_path),
        "pass": (
            synth_verdict.verdict == "PASS"
            and refused and overclaims == 1 and not forged_edges
            and counts["NOT_COMPARABLE"] >= 1 and counts["STILL_FAIL"] >= 1
            and not_comparable_changes_ok and family_ok
            and lin.edge_count(conn) == len(verdicts) + 1
        ),
    }


# --------------------------------------------------------------------------- #
# C6 -- zero domain semantics
# --------------------------------------------------------------------------- #
def criterion_c6(package: Path, tests: Sequence[Path]) -> dict[str, Any]:
    result = scan_mod.scan_paths([package, *tests])
    return {
        "id": "C6",
        "title": "zero domain semantics in code and test fixtures",
        "files_scanned": result["files_scanned"],
        "vocabulary_size": result["vocabulary_size"],
        "vocabulary_loaded": result["vocabulary_loaded"],
        "hits": result["hits"],
        "hit_count": result["hit_count"],
        "paths": result["paths"],
        "pass": bool(result["clean"]),
    }


# --------------------------------------------------------------------------- #
# whole run
# --------------------------------------------------------------------------- #
def _relativise(obj: Any, root: Path) -> Any:
    """Strip the workspace prefix from every recorded path (portable artifacts)."""
    prefix = str(root) + "/"
    if isinstance(obj, str):
        return obj.replace(prefix, "")
    if isinstance(obj, list):
        return [_relativise(v, root) for v in obj]
    if isinstance(obj, dict):
        return {k: _relativise(v, root) for k, v in obj.items()}
    return obj


def run(*, root: Path | None = None, out_dir: Path | None = None,
        sample: int = 20) -> dict[str, Any]:
    root = root or Path(__file__).resolve().parents[3]
    corpus = root / "tensor_reports" / "campaign_nodes"
    out_dir = out_dir or root / "artifacts" / "fail_revival"
    out_dir.mkdir(parents=True, exist_ok=True)

    records, registry, blockers, extraction = load_registry(corpus)
    subset, graph, _ = real_fixture(corpus)

    c1 = criterion_c1(corpus, records, extraction, sample=sample)
    c2 = criterion_c2(records, registry)
    c3 = criterion_c3(graph, subset)
    c4 = criterion_c4(graph)
    c5 = criterion_c5(graph, subset, out_dir / "lineage.sqlite3")
    tests = sorted((root / "tests").glob("test_fail_revival*.py"))
    c6 = criterion_c6(root / "src" / "tensorphysics" / "fail_revival", tests)

    graph_dict = graph.to_dict()
    graph_dict["resolvability_final_state"] = graph.resolvability(scenario_states(graph)[-1])
    graph_dict["mission_dependencies"] = {str(k): list(v) for k, v in mission_deps(corpus).items()}

    result = {
        "schema": "fail_revival_acceptance.v1",
        "generated_from": {
            "corpus": str(corpus),
            "artifacts": len(extraction.extractions),
            "declarations": str(corpus / "revival_blockers.json"),
            "records": len(records),
            "blockers": len(blockers),
        },
        "criteria": [c1, c2, c3, c4, c5, c6],
        "summary": {
            "C1_extraction_rate": c1["extraction_rate"],
            "C1_criterion_match_rate": c1["criterion_digest_match_rate"],
            "C2_rejected": "%d/%d" % (c2["rejected"], c2["cases_total"]),
            "C3_false_triggers": c3["false_triggers"],
            "C3_max_latency": c3["max_latency"],
            "C4_identical": c4["identical_repeat_runs"],
            "C5_overclaim_rows": c5["overclaim_rows"],
            "C6_hits": c6["hit_count"],
            "all_pass": all(c["pass"] for c in (c1, c2, c3, c4, c5, c6)),
        },
        "blocker_graph": graph_dict,
        "corpus_summary": extraction.to_dict(),
    }
    result = _relativise(result, root)
    out_path = out_dir / "acceptance.json"
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, sort_keys=True) + "\n",
                        encoding="utf-8")
    result["acceptance_path"] = str(out_path)
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--sample", type=int, default=20)
    args = ap.parse_args(argv)
    res = run(root=args.root, out_dir=args.out_dir, sample=args.sample)
    print(json.dumps(res["summary"], ensure_ascii=False, sort_keys=True))
    return 0 if res["summary"]["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())