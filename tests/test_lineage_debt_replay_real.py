"""Real replay: plan 173 -> debt ledger (acceptance criteria (d) and (e)).

(d) the ledger built from the *published* plan-173 snapshot must mark mission
    m6753's ``w`` and ``t`` as ``debt_open`` -- 2/2 agreement with the hand
    label (both are the honest FAILs declared in report 8117);
(e) the same input replays to byte-identical output.

The snapshot under ``artifacts/lineage_debt/inputs`` is a read-only byte image of
the campaign API (plans 26/173/388, 18 reports), so these assertions are about
real campaign evidence, not fixtures.
"""
from __future__ import annotations

import json

import pytest
from lineage_debt_synth import REAL_BUNDLE

from tensorphysics.lineage_debt import load_bundle, load_config
from tensorphysics.lineage_debt.model import canonical_json
from tensorphysics.lineage_debt.query import query
from tensorphysics.lineage_debt.replay import build_ledger, write_ledger

pytestmark = pytest.mark.skipif(not REAL_BUNDLE.exists(), reason="campaign snapshot not present")


def _ledger(plan_id: int = 173, **kwargs) -> dict:
    cfg = load_config()
    bundle = load_bundle(REAL_BUNDLE)
    return build_ledger(bundle, plan_id, cfg, **kwargs)


def test_d_real_replay_marks_m6753_w_t_debt_open():
    """(d) mission 6753: exactly the two hand-labelled items, both debt_open."""
    ledger = _ledger(173)
    rows = query(ledger, mission_id=6753, state="debt_open")
    got = {(r["quantity"], r["condition"], r["report_id"]) for r in rows}
    gold = {("w", "KCS", 8117), ("t", "KCS", 8117)}
    assert got == gold, f"got={sorted(got)}"
    assert len(rows) == 2                                  # 2/2 against the hand label
    by_q = {r["quantity"]: r for r in rows}
    assert by_q["w"]["abs_err_pct"] == pytest.approx(25.2, abs=0.05)
    assert by_q["t"]["abs_err_pct"] == pytest.approx(15.1, abs=0.05)
    assert by_q["w"]["domain"] == "propulsion" and by_q["t"]["domain"] == "propulsion"
    # the debt must carry its evidence chain: plan 26 -> plan 173 -> node -> report
    chain = by_q["w"]["lineage"]
    assert [s["plan_id"] for s in chain if s["kind"] == "plan"] == [26, 173]
    assert [s["report_id"] for s in chain if s["kind"] == "report"] == [8117]
    assert chain[-1]["report_id"] == 8117
    # and the evidence itself is pinned to a report hash + line number
    ev = by_q["w"]["evidence"][0]
    assert ev["report_sha256"] and ev["line_no"] > 0


def test_e_same_input_replays_byte_identical(tmp_path):
    """(e) two replays of the same snapshot are byte-identical."""
    first = _ledger(173)
    second = _ledger(173)
    assert canonical_json(first) == canonical_json(second)
    p1 = write_ledger(first, tmp_path / "a.json")
    p2 = write_ledger(second, tmp_path / "b.json")
    assert p1.read_bytes() == p2.read_bytes()
    # the determinism is not an artefact of a single plan
    assert canonical_json(_ledger(26)) == canonical_json(_ledger(26))


def test_real_lineage_and_gate_are_consistent():
    """The real wave inherits 26's debts and blocks on the ones nobody re-measures."""
    ledger = _ledger(173)
    assert ledger["lineage"]["chain"] == [26, 173]
    assert ledger["lineage"]["inherited_debts"] >= 1
    assert ledger["summary"]["total"] == len(ledger["debts"])
    assert ledger["coverage_target_plan"] == 388                  # metadata.autonomy.new_plan
    verdict = ledger["seed_gate"]
    missing_keys = {m["debt_key"] for m in verdict["missing"]}
    assert verdict["verdict"] == "blocked"
    assert {"propulsion|w|KCS", "propulsion|t|KCS"} <= missing_keys
    assert ledger["plan_metadata_after"]["autonomy"]["debt_unpaid"] == sorted(missing_keys)
    # every blocking item carries the reason it is expected to supply
    for item in verdict["missing"]:
        assert item["required"] and item["debt_key"]


def test_plan26_ledger_is_inheritance_root():
    ledger = _ledger(26)
    assert ledger["lineage"]["chain"] == [26]
    assert ledger["lineage"]["inherited_debts"] == 0
    kinds = {d["source_kind"] for d in ledger["debts"]}
    assert kinds == {"error_row", "node_terminal"}
    # the blocked node of plan 26 is a debt with its block reason preserved
    terminal = [d for d in ledger["debts"] if d["source_kind"] == "node_terminal"]
    assert len(terminal) == 1 and terminal[0]["node_status"] == "blocked"
    assert terminal[0]["evidence"][0]["block_reason"]


def test_ledger_json_is_serialisable_and_queryable():
    ledger = _ledger(173)
    payload = json.loads(canonical_json(ledger))
    assert payload["schema"] == "lineage_debt_ledger.v1"
    assert payload["provenance"]["bundle_sha256"]
    rows = query(payload, coverage="dropped")
    assert rows and all(r["coverage"] == "dropped" for r in rows)
    assert [r["key"] for r in rows] == sorted(r["key"] for r in rows)