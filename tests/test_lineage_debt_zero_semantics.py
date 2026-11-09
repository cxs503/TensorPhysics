"""Zero-domain-semantics proof (f) plus a fully synthetic end-to-end run.

(f) the domain vocabulary must be injected: a deny-list scan over the package's
``.py`` logic has to come back empty, and a pipeline fed an *invented*
vocabulary has to work end to end.  The second half is the real proof -- if any
domain word were hard-coded, the synthetic vocabulary could not drive extraction,
criteria parsing, coverage and the gate.
"""
from __future__ import annotations

import json

from lineage_debt_synth import SYNTH_VOCAB, table, report

from tensorphysics.lineage_debt import load_config
from tensorphysics.lineage_debt.query import query
from tensorphysics.lineage_debt.replay import build_ledger, load_bundle
from tensorphysics.lineage_debt.selfcheck import package_sources, scan_sources, selfcheck


def test_f_deny_list_scan_is_clean():
    """(f) no domain word in the package logic."""
    cfg = load_config()
    assert cfg.deny_list, "the shipped vocabulary must carry a deny list"
    result = selfcheck(cfg)
    assert result["verdict"] == "pass", result["hits"][:5]
    assert result["files_scanned"] >= 8
    assert result["hits"] == []
    # the scan really can fire (otherwise 'pass' would be meaningless)
    planted = package_sources()
    hits = scan_sources(planted, ["debt"], root=planted[0].parents[1])
    assert hits, "the deny-list scan must detect a word that is present"


def test_f_synthetic_vocabulary_drives_the_whole_pipeline():
    """An invented vocabulary produces a complete ledger (proof of injection)."""
    cfg = load_config(SYNTH_VOCAB)
    bundle = {
        "schema": "lineage_debt_input.v1",
        "fetched_utc": "2026-02-02T00:00:00Z",
        "source": {"api_base": "synthetic"},
        "plans": {
            "1": {"id": 1, "name": "alpha wave", "metadata": {"autonomy": {"new_plan": 2}}},
            "2": {"id": 2, "name": "beta wave", "metadata": {"parent_plan": 1}},
        },
        "nodes": {
            "1": [
                {"id": 11, "plan_id": 1, "mission_id": 501, "name": "alpha surface", "status": "done",
                 "metadata": {"acceptance_criteria": ["q1 error<10%", "q2 error<10%"]}},
                {"id": 12, "plan_id": 1, "mission_id": 502, "name": "second surface", "status": "blocked",
                 "metadata": {"goal": "unlabelled work"}, "block_reason": "no evidence"},
            ],
            "2": [
                {"id": 21, "plan_id": 2, "mission_id": 503, "name": "beta surface", "status": "planned",
                 "metadata": {"acceptance_criteria": ["q1 error<8%", "beta surface work"]}},
            ],
        },
        "reports": {
            "9001": report(9001, table(["metric", "case", "error"],
                                       [["q1", "case_A", "+40.0%"], ["q2", "case_A", "+2.0%"]]),
                           mission_id=501),
        },
        "downstream_plan": 2,
    }
    ledger = build_ledger(load_bundle(bundle), 1, cfg)
    assert ledger["schema"] == "lineage_debt_ledger.v1"
    assert ledger["coverage_target_plan"] == 2
    keys = {d["key"] for d in ledger["debts"]}
    assert "alpha|q1|CASE_A" in keys
    assert any(k.endswith("|node_terminal|node:12") for k in keys)
    covered = {r["key"] for r in query(ledger, coverage="covered")}
    assert covered == {"alpha|q1|CASE_A"}              # node 21 re-measures q1
    # node 12's gap names no domain anyone can match -> fail-closed
    assert ledger["seed_gate"]["verdict"] == "blocked"
    assert [m["node_id"] for m in ledger["seed_gate"]["missing"]] == [12]
    assert ledger["plan_metadata_after"]["autonomy"]["debt_unpaid"]
    assert json.dumps(ledger, sort_keys=True)            # serialisable as-is


def test_cli_selfcheck_and_query_are_usable():
    from tensorphysics.lineage_debt.__main__ import main

    assert main(["selfcheck"]) == 0