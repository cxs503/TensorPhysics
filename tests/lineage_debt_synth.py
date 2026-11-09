"""Synthetic fixtures for the lineage-debt tests (deliberately domain-free).

Every string in this module is invented ("alpha"/"beta"/"q1"/"q2"/"q9",
"plat_A"/"case_B").  If the ledger logic ever hard-coded a real engineering
vocabulary the tests here would keep passing, which is exactly why they are
kept synthetic: they prove the logic is *configuration-driven*.
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_BUNDLE = REPO_ROOT / "artifacts" / "lineage_debt" / "inputs" / "lineage_debt_bundle.json"

SYNTH_VOCAB: dict = {
    "schema": "lineage_debt_vocab.v1",
    "thresholds": {"default_pct": 15.0, "by_domain": {}, "by_quantity": {}},
    "table_contract": {
        "quantity_headers": ["量", "metric", "quantity"],
        "condition_headers": ["工况", "case", "state"],
        "error_header_patterns": ["误差", "error", "偏差", "err"],
        "verdict_markers": {"fail": ["FAIL", "honest FAIL"], "pass": ["PASS"]},
        "empty_markers": ["", "—", "-", "n/a", "NaN"],
        "fraction_interpretation": True,
    },
    "criteria_contract": {
        "threshold_patterns": [
            "(?P<qty>[^,;，；|]+?)\\s*[<≤]=?\\s*(?P<val>\\d+(?:\\.\\d+)?)\\s*%",
        ],
        "ref_patterns": {
            "mission": ["\\bm(?P<v>\\d{3,6})\\b"],
            "node": ["\\bnode\\s*#?(?P<v>\\d{2,6})\\b"],
            "report": ["\\breport\\s*#?(?P<v>\\d{3,6})\\b"],
            "plan": ["\\bplan\\s*#?(?P<v>\\d{2,6})\\b"],
        },
    },
    "domains": {
        "alpha": {
            "aliases": ["alpha"],
            "quantities": {
                "q1": {"aliases": ["q1"], "unit": "-"},
                "q2": {"aliases": ["q2"], "unit": "-"},
            },
        },
        "beta": {
            "aliases": ["beta"],
            "quantities": {
                "q3": {"aliases": ["q3"], "unit": "-"},
                "q9": {"aliases": ["q9"], "unit": "-"},
            },
        },
    },
    "states": {"open": "debt_open", "closed": "debt_closed", "unclassified_domain": "unclassified",
               "wildcard_condition": "ALL"},
    "coverage": {"covered": "covered", "covered_by_domain": "covered_by_domain",
                 "deferred": "deferred", "dropped": "dropped",
                 "rule_explicit_ref": "explicit_ref", "rule_quantity": "quantity",
                 "rule_domain": "domain", "rule_deferral": "deferral"},
    "deny_list": [],
}


def table(header: list[str], rows: list[list[str]]) -> str:
    """Render a markdown pipe table."""
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def report(rid: int, markdown: str, *, mission_id: int = 1001, created: str = "2026-01-01T00:00:00Z") -> dict:
    return {"id": rid, "markdown": markdown, "mission_id": mission_id, "created_at": created,
            "title": "synthetic report", "summary": "", "tags": [], "scenario_key": ""}


#: 6 synthetic reports; exactly 3 rows carry ``|err| > 15%`` (the gold debt set).
CONTRACT_REPORTS: list[dict] = [
    report(9001, "# r1\n\n" + table(["metric", "case", "error"], [
        ["q1", "case_A", "+22.0%"],
        ["q1", "case_B", "+5.0%"],
        ["q2", "case_A", "-3.0%"],
    ])),
    report(9002, "# r2\n\n" + table(["metric", "case", "error"], [
        ["q3", "case_A", "-31.5%"],
        ["q3", "case_B", "+12.0%"],
    ])),
    report(9003, "# r3\n\n" + table(["quantity", "state", "rel error"], [
        ["q9", "case_C", "16.0%"],
        ["q9", "case_D", "1.0%"],
    ])),
    report(9004, "# r4\n\n" + table(["metric", "case", "error"], [
        ["q2", "case_A", "14.9%"],
        ["q3", "case_B", "-0.5%"],
    ])),
    report(9005, "# r5\n\n" + table(["metric", "case", "error"], [
        ["q1", "case_C", "—"],
        ["q2", "case_C", "8.0%"],
    ])),
    report(9006, "# r6 (no error column)\n\n" + table(["metric", "case", "value"], [
        ["q1", "case_A", "1.0"],
        ["q2", "case_A", "2.0"],
    ])),
]

#: gold debt keys of :data:`CONTRACT_REPORTS` (hand-labelled).
CONTRACT_GOLD = {"alpha|q1|CASE_A", "beta|q3|CASE_A", "beta|q9|CASE_C"}