"""Child-node criteria parsing and the coverage matrix.

Acceptance criterion (b) lives here: three synthetic child nodes must produce
exactly ``covered`` / ``covered_by_domain`` / ``dropped``, matching the gold
labels.  The surrounding tests pin the resolution order (structured metadata >
regex > goal prose) and the rule priority (explicit reference > quantity >
domain > deferral > dropped).
"""
from __future__ import annotations

from lineage_debt_synth import SYNTH_VOCAB

from tensorphysics.lineage_debt import load_config
from tensorphysics.lineage_debt.coverage import build_matrix, match_debt
from tensorphysics.lineage_debt.criteria import parse_node_criteria
from tensorphysics.lineage_debt.model import DebtItem


def _cfg():
    return load_config(SYNTH_VOCAB)


def _node(nid: int, *, criteria=None, goal=None, name="synthetic node") -> dict:
    meta: dict = {}
    if criteria is not None:
        meta["acceptance_criteria"] = criteria
    if goal is not None:
        meta["goal"] = goal
    return {"id": nid, "plan_id": 7, "mission_id": 700 + nid, "name": name, "metadata": meta, "status": "planned"}


def _debt(key: str, domain: str, quantity: str | None, *, report_id: int | None = None,
          node_id: int | None = None, coverage: str | None = None) -> DebtItem:
    return DebtItem(key=key, domain=domain, quantity=quantity, condition="ALL",
                    source_kind="error_row", plan_id=1, node_id=node_id, mission_id=None,
                    report_id=report_id, err_pct=25.0, threshold_pct=15.0, coverage=coverage)


def test_structured_metadata_criteria_are_authoritative():
    node = _node(21, criteria=[
        {"id": "C1", "quantity": "q1", "threshold_pct": 10.0,
         "refs": {"report": [9001]}, "text": "q1 within 10%"},
    ])
    crits = parse_node_criteria(node, _cfg())
    assert len(crits) == 1
    c = crits[0]
    assert c.source == "metadata" and c.domain == "alpha" and c.quantity == "q1"
    assert c.threshold_pct == 10.0 and c.refs["report"] == (9001,)


def test_string_criteria_regex_fallback_canonicalises_vocabulary():
    node = _node(22, criteria=["q1 error<8% on case_A", "q9误差<3%"])
    crits = {c.quantity: c for c in parse_node_criteria(node, _cfg())}
    assert crits["q1"].domain == "alpha" and crits["q1"].threshold_pct == 8.0
    assert crits["q9"].domain == "beta" and crits["q9"].threshold_pct == 3.0
    assert all(c.source == "metadata_string" for c in crits.values())


def test_goal_prose_is_a_last_resort_criterion_source():
    node = _node(23, goal="do work; q2 must stay within <12% of the anchor; and be nice")
    crits = parse_node_criteria(node, _cfg())
    assert [c.source for c in crits] == ["goal_regex"]
    assert crits[0].quantity == "q2" and crits[0].domain == "alpha"


def test_b_coverage_matrix_gold_covered_by_domain_dropped():
    """(b) three synthetic child nodes -> covered / covered_by_domain / dropped."""
    cfg = _cfg()
    nodes = [
        _node(31, criteria=["q1 error<10%"]),        # says the quantity
        _node(32, criteria=["alpha surface review"]),  # names only the domain
        _node(33, criteria=["unrelated work"]),        # names neither
    ]
    criteria = [c for n in nodes for c in parse_node_criteria(n, cfg)]
    debts = [_debt("alpha|q1|ALL", "alpha", "q1"),
             _debt("alpha|q2|ALL", "alpha", "q2"),
             _debt("beta|q9|ALL", "beta", "q9")]
    matrix = build_matrix(debts, criteria, cfg)
    gold = {"alpha|q1|ALL": "covered",
            "alpha|q2|ALL": "covered_by_domain",
            "beta|q9|ALL": "dropped"}
    got = {row["debt_key"]: row["coverage"] for row in matrix["matrix"]}
    assert got == gold
    assert matrix["counts"] == {"covered": 1, "covered_by_domain": 1, "dropped": 1}
    # the matched criterion is recorded for every non-dropped debt (auditability)
    covered = [e for e in matrix["entries"] if e["coverage"] == "covered"][0]
    assert covered["matched"][0]["criterion_id"].startswith("n31:")
    assert [d.coverage for d in debts] == ["covered", "covered_by_domain", "dropped"]


def test_explicit_reference_outranks_quantity_match():
    cfg = _cfg()
    # a criterion that cites the report id but names no quantity of that domain
    node = _node(41, criteria=[{"id": "R1", "text": "re-measured per report 9001", "refs": {"report": [9001]}}])
    criteria = parse_node_criteria(node, cfg)
    debt = _debt("alpha|q1|ALL", "alpha", "q1", report_id=9001)
    entry = match_debt(debt, criteria, cfg)
    assert entry.coverage == "covered" and entry.rule == "explicit_ref"
    assert entry.matched[0]["kind"] == "ref" and entry.matched[0]["ids"] == [9001]


def test_deferral_requires_a_written_reason():
    cfg = _cfg()
    debt = _debt("beta|q9|ALL", "beta", "q9", node_id=77)
    no_reason = match_debt(debt, [], cfg, deferrals={"node:77": "   "})
    assert no_reason.coverage == "dropped" and no_reason.requires_reason
    with_reason = match_debt(debt, [], cfg, deferrals={"node:77": "carried to wave 9 by arbiter"})
    assert with_reason.coverage == "deferred"
    assert with_reason.deferred_reason == "carried to wave 9 by arbiter"


def test_unparsed_node_still_yields_an_identity():
    """A node with no criteria text at all keeps its title (never silently absent)."""
    crits = parse_node_criteria(_node(51, name="alpha surface"), _cfg())
    assert len(crits) == 1
    assert crits[0].source == "metadata_unparsed" and crits[0].domain == "alpha"