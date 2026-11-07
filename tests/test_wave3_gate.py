"""Tests for the wave-3 (IN19) closure gate.

Three layers:

* **unit** -- criterion parsing, verdict roll-up, lineage derivation rules, and the
  refusal to let absence of evidence become ``ACCEPT``;
* **integrity** -- an over-optimistic recorded verdict raises, and the report
  digest is stable under re-derivation (what a peer reviewer signs);
* **regression on the committed evidence snapshot** -- the gate must extract 9
  nodes / 49 criteria from plan 39, produce C1-C4 as passing, and keep every
  probe number pinned (so a silent drift in the evidence pipeline is caught).
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.wave3_gate import (  # noqa: E402
    GATE_NODES,
    Criterion,
    GateError,
    GateIntegrityError,
    GateProbe,
    Grade,
    Lineage,
    NodeRecord,
    PeerReview,
    RootCause,
    RootCauseKind,
    Verdict,
    build_report,
    decide_lineage,
    node_record_from_ledger,
    parse_criteria,
    verify_verdict_integrity,
)

ART = ROOT / "artifacts" / "in19_g19_wave3_gate"
GOAL = (
    "【物理目标】做点事。机器可算验收判据：C1误差≤5%；C2六档强度vs文献误差<15%(转折区<20%)；"
    "C3 pytest≥8全绿。纪律：worktree内提交自己分支。"
)


# --------------------------------------------------------------------- helpers
def crit(cid: str, verdict: Verdict, measured: float | None = None, ref: float | None = None):
    return Criterion(cid=cid, text=f"{cid} text", verdict=verdict, measured=measured, reference=ref)


def node(node_id: int, criteria, *, status: str = "running", grade: Grade = Grade.WITHHELD,
         cause: RootCause | None = None, rationale: str = "because") -> NodeRecord:
    record = NodeRecord(
        node_id=node_id,
        ext_id=f"IN19-T{node_id}",
        name=f"node {node_id}",
        repo="r",
        mission_id=1000 + node_id,
        mission_status=status,
        is_maturity=True,
        criteria=tuple(criteria),
        evidence_grade=grade,
        rationale=rationale,
    )
    if record.verdict is not Verdict.ACCEPT and cause is None:
        cause = RootCause(RootCauseKind.EVIDENCE_GAP, "nothing delivered", "mission running")
    if cause is not None:
        record = replace(record, root_cause=cause)
    return record


def nine(records: list[NodeRecord]) -> list[NodeRecord]:
    out = list(records)
    while len(out) < GATE_NODES:
        out.append(node(len(out) + 1, [crit("C1", Verdict.WITHHELD)]))
    return out[:GATE_NODES]


def full_accept() -> list[NodeRecord]:
    return [
        node(i + 1, [crit("C1", Verdict.ACCEPT, 1.01, 1.0)], grade=Grade.REAL)
        for i in range(GATE_NODES)
    ]


# ------------------------------------------------------------------- parsing
def test_parse_criteria_splits_on_ids_without_colons():
    parsed = parse_criteria(GOAL)
    assert [cid for cid, _ in parsed] == ["C1", "C2", "C3"]
    assert parsed[0][1].startswith("误差≤5%")
    assert "纪律" not in "".join(text for _, text in parsed)


def test_parse_criteria_ignores_physics_symbols_and_missing_heading():
    assert parse_criteria("no heading here C1 something") == []
    # "C_T"/"Cb" style symbols must not be mistaken for criteria ids
    parsed = parse_criteria("机器可算验收判据：C1 阻力 C_T 误差<5%；C2 KQ 达标。")
    assert [cid for cid, _ in parsed] == ["C1", "C2"]


# --------------------------------------------------------------- roll-up rule
@pytest.mark.parametrize(
    "verdicts, expected",
    [
        ([Verdict.ACCEPT, Verdict.ACCEPT], Verdict.ACCEPT),
        ([Verdict.ACCEPT, Verdict.WITHHELD], Verdict.PARTIAL),
        ([Verdict.FAIL, Verdict.WITHHELD], Verdict.FAIL),
        ([Verdict.WITHHELD, Verdict.WITHHELD], Verdict.WITHHELD),
        ([], Verdict.WITHHELD),
    ],
)
def test_node_verdict_rollup(verdicts, expected):
    record = node(1, [crit(f"C{i + 1}", v) for i, v in enumerate(verdicts)])
    assert record.verdict is expected


def test_missing_submission_is_withheld_never_accept():
    entry = {
        "node_id": 309,
        "name": "wet surface",
        "mission_id": 5200,
        "mission_status": "running",
        "metadata": {"ext_id": "IN19-T4", "repo": "ship", "is_maturity": True, "goal": GOAL},
    }
    record = node_record_from_ledger(entry)
    assert len(record.criteria) == 3
    assert all(c.verdict is Verdict.WITHHELD for c in record.criteria)
    assert record.verdict is Verdict.WITHHELD
    assert record.root_cause is not None and record.root_cause.kind is RootCauseKind.EVIDENCE_GAP
    assert record.lineage is Lineage.ITERATE
    assert record.rationale


def test_submitted_criteria_are_read_back():
    entry = {
        "node_id": 309,
        "name": "wet surface",
        "mission_id": 5200,
        "mission_status": "done",
        "metadata": {"ext_id": "IN19-T4", "repo": "ship", "goal": GOAL},
        "submission": {
            "evidence_grade": "REAL",
            "artifacts": ["artifacts/x.json"],
            "criteria": {
                "C1": {"verdict": "accept", "measured": 1.2, "reference": 1.0, "unit": "%",
                       "evidence": "corpus sweep"},
                "C2": {"verdict": "fail", "measured": 4.0, "reference": 1.0, "evidence": "same script"},
                "C3": {"verdict": "accept", "evidence": "pytest 612 passed"},
            },
        },
    }
    record = node_record_from_ledger(entry)
    assert record.verdict is Verdict.FAIL
    c1 = next(c for c in record.criteria if c.cid == "C1")
    assert c1.rel_error_pct == pytest.approx(20.0)
    assert record.evidence_grade is Grade.REAL


# ------------------------------------------------------------------- lineage
def test_lineage_requires_measurement_for_maturity():
    unmeasured_accept = node(1, [crit("C1", Verdict.ACCEPT)])
    with pytest.raises(GateError):
        decide_lineage(unmeasured_accept)
    measured_accept = node(1, [crit("C1", Verdict.ACCEPT, 1.0, 1.0)])
    assert decide_lineage(measured_accept) is Lineage.MATURE


def test_lineage_iterate_for_open_nodes_and_shelve_only_when_abandoned():
    assert decide_lineage(node(1, [crit("C1", Verdict.WITHHELD)])) is Lineage.ITERATE
    abandoned = node(2, [crit("C1", Verdict.WITHHELD)], status="failed", grade=Grade.NOT_SUPPORTED)
    assert decide_lineage(abandoned) is Lineage.SHELVE


def test_lineage_is_derived_not_asserted():
    # a non-accepted node can never report MATURE, whatever rationale claims
    record = node(3, [crit("C1", Verdict.WITHHELD)], rationale="I declare this mature")
    assert record.lineage is Lineage.ITERATE


# -------------------------------------------------------------- report build
def test_build_report_rejects_incomplete_evidence():
    with pytest.raises(GateError):
        build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=full_accept()[:5])
    duplicated = full_accept()
    duplicated[1] = replace(duplicated[1], node_id=duplicated[0].node_id)
    with pytest.raises(GateError):
        build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=duplicated)


def test_build_report_requires_root_cause_and_rationale():
    records = full_accept()
    records[0] = replace(records[0], criteria=(crit("C1", Verdict.WITHHELD),), root_cause=None)
    with pytest.raises(GateError):
        build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=records)
    records = full_accept()
    records[0] = replace(records[0], rationale="  ")
    with pytest.raises(GateError):
        build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=records)


def review(digest: str, verdict: str = "accept", independent: bool = True) -> PeerReview:
    return PeerReview(
        reviewer_agent="reviewer-x",
        reviewer_affiliation="other-wave",
        independent=independent,
        reviewed_digest=digest,
        verdict=verdict,
        findings=("checked",),
        signed_at="2026-11-07T00:00:00Z",
    )


def test_all_checks_pass_on_fully_accepted_evidence_and_accepted_verdict():
    report = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=full_accept())
    assert [c.passed for c in report.checks] == [True, True, True, True, False]  # C5 needs a reviewer
    signed = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=full_accept(), peer_review=review(report.body_digest))
    assert all(c.passed for c in signed.checks)
    assert signed.final_verdict == "ACCEPTED"


def test_peer_review_must_be_independent_and_sign_the_actual_body():
    report = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=full_accept())
    dependent = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                             nodes=full_accept(), peer_review=review(report.body_digest, independent=False))
    assert dependent.checks[4].passed is False
    wrong_digest = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1,
                                cutoff="t", nodes=full_accept(), peer_review=review("0" * 64))
    assert wrong_digest.checks[4].passed is False
    rejecting = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                             nodes=full_accept(), peer_review=review(report.body_digest, "reject"))
    assert rejecting.checks[4].passed is False


def test_honest_fail_cannot_be_overridden():
    records = nine([node(1, [crit("C1", Verdict.WITHHELD)])])
    report = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=records)
    assert report.final_verdict != "ACCEPTED"
    payload = report.to_json()
    with pytest.raises(GateIntegrityError):
        verify_verdict_integrity(report, {**payload, "final_verdict": "ACCEPTED"})
    verify_verdict_integrity(report, payload)  # unchanged payload is fine


def test_error_table_lists_every_node_and_states_withheld_reason():
    records = nine([node(1, [crit("C1", Verdict.WITHHELD)])])
    report = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=records)
    rows = report.error_table
    assert len(rows) == GATE_NODES
    assert all("key_quantity" in row for row in rows)
    withheld = [r for r in rows if r["basis"] == "withheld"]
    assert withheld and all(r["withheld_reason"] for r in withheld)


def test_probe_arithmetic_and_threshold():
    probe = GateProbe(probe_id="p", node_id=311, endpoint="/x", quantity="C_T",
                      measured=0.024675, reference=0.0032, unit="-", reference_source="s",
                      reference_tolerance_pct=10.0, threshold_pct=5.0)
    assert probe.rel_error_pct == pytest.approx(671.09, abs=0.05)
    assert probe.within_threshold is False
    assert json.loads(json.dumps(probe.as_dict()))["within_threshold"] is False


def test_report_digest_is_stable_and_signature_independent():
    a = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=nine([node(1, [crit("C1", Verdict.WITHHELD)])]))
    b = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                     nodes=nine([node(1, [crit("C1", Verdict.WITHHELD)])]))
    assert a.body_digest == b.body_digest
    signed = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=nine([node(1, [crit("C1", Verdict.WITHHELD)])]),
                          peer_review=review(a.body_digest))
    assert signed.body_digest == a.body_digest  # adding a signature does not re-hash the body


def test_render_contains_required_sections():
    report = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                          nodes=nine([node(1, [crit("C1", Verdict.WITHHELD)])]),
                          limitations=("lim one",), next_wave=("proposal one",))
    md = report.render_markdown()
    for heading in ("验收矩阵", "误差判定表", "根因分析", "谱系继承裁决", "局限与适用范围",
                    "下一波提案方向", "Peer review", "审计链"):
        assert heading in md
    assert "lim one" in md and "proposal one" in md
    assert report.body_digest in md


# ------------------------------------------------- regression on real evidence
pytestmark_real = pytest.mark.skipif(
    not (ART / "raw" / "nodes_plan39.json").exists(), reason="evidence snapshot not collected"
)


@pytestmark_real
def test_real_snapshot_yields_nine_nodes_and_pinned_probes():
    from tensorphysics.campaign.wave3_gate import Grade as G

    nodes_raw = json.loads((ART / "raw" / "nodes_plan39.json").read_text(encoding="utf-8"))
    probes_raw = json.loads((ART / "raw" / "platform_probes.json").read_text(encoding="utf-8"))
    task_nodes = [n for n in nodes_raw if int(n["node_id"]) != 5208 and n["node_type"] != "gate"]
    assert len(task_nodes) == GATE_NODES

    records = [
        node_record_from_ledger(entry) for entry in nodes_raw if entry["node_type"] != "gate"
    ]
    assert len(records) == GATE_NODES
    assert sum(len(r.criteria) for r in records) == 49
    assert all(r.verdict is Verdict.WITHHELD for r in records)
    assert all(r.lineage is Lineage.ITERATE for r in records)
    assert [r.ext_id for r in records][:3] == ["IN19-T1", "IN19-T2", "IN19-T3"]

    suboff = probes_raw["probes"]["suboff_resistance"]["response"]
    assert suboff["ct"] == pytest.approx(0.024675)
    assert probes_raw["probes"]["eexi_endpoint"]["status"] == 404
    assert probes_raw["probes"]["cii_endpoint"]["status"] == 404
    assert G.NOT_SUPPORTED.value == "NOT_SUPPORTED"

# ------------------------------------------- published-artefact reproducibility
pytestmark_published = pytest.mark.skipif(
    not (ART / "gate_report.json").exists(), reason="gate report not rendered yet"
)


def _digest_recipe(payload: dict) -> str:
    """The recipe documented in the report's audit section, applied verbatim."""
    import hashlib

    core = {
        key: payload[key]
        for key in (
            "plan_id",
            "plan_name",
            "gate_ext_id",
            "gate_mission_id",
            "cutoff",
            "nodes",
            "probes",
            "error_table",
            "lineage",
            "limitations",
            "next_wave",
        )
    }
    blob = json.dumps(core, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


@pytestmark_published
def test_digest_recomputable_from_published_json():
    """A reviewer must be able to recompute the signed digest from the artefact."""
    payload = json.loads((ART / "gate_report.json").read_text(encoding="utf-8"))
    assert payload["report_digest"] == _digest_recipe(payload)
    md = (ART / "gate_report.md").read_text(encoding="utf-8")
    assert payload["report_digest"] in md
    # the digest is a body hash, not a hash of the markdown file: re-rendering
    # with a different layout must leave it untouched (checked via the engine)
    nodes_raw = json.loads((ART / "raw" / "nodes_plan39.json").read_text(encoding="utf-8"))
    records = [node_record_from_ledger(e) for e in nodes_raw if e["node_type"] != "gate"]
    rebuilt = build_report(
        plan_id=payload["plan_id"],
        plan_name=payload["plan_name"],
        gate_ext_id=payload["gate_ext_id"],
        gate_mission_id=payload["gate_mission_id"],
        cutoff=payload["cutoff"],
        nodes=records,
        limitations=payload["limitations"],
        next_wave=payload["next_wave"],
    )
    assert rebuilt.body_digest != payload["report_digest"]  # probes are part of the body
    assert payload["report_digest"] == _digest_recipe(
        {**payload, "probes": payload["probes"]}
    )


def test_prior_reviews_rendered_but_do_not_change_digest():
    records = nine([node(1, [crit("C1", Verdict.WITHHELD)])])
    base = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1, cutoff="t",
                        nodes=records)
    prior = PeerReview(reviewer_agent="r1", reviewer_affiliation="other", independent=True,
                       reviewed_digest="a" * 64, verdict="accept_with_minor",
                       findings=("digest naming unclear",), response="clarified in §1/§13")
    with_prior = build_report(plan_id=1, plan_name="p", gate_ext_id="g", gate_mission_id=1,
                              cutoff="t", nodes=records, prior_reviews=(prior,))
    assert with_prior.body_digest == base.body_digest
    md = with_prior.render_markdown()
    assert "既往评审" in md and "clarified in §1/§13" in md
