"""P84-W2 campaign integration gate: tests.

Tests verify:
1. Gate can reach PASS with fully evidenced synthetic campaign
2. Gate fails on missing artifacts (honest FAIL)
3. Cross-domain interface checks work correctly
4. Code alignment checks verify regulation traceability
5. Lineage ruling correctly identifies gap resolution
6. End-to-end JSON output contains all required sections
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tensorphysics.campaign.p84w2_gate import (
    CAMPAIGN_PLAN,
    PLAN_84_GAPS,
    run_gate,
    write_outputs,
    render_markdown,
    run_cross_domain_checks,
    run_code_alignment_checks,
    run_lineage_ruling,
    LBM_TURBULENCE_CONTRACT,
    MULTIPHASE_PARTICLE_CONTRACT,
)
from tensorphysics.campaign.p84w2_gate.gate import evaluate_node, evaluate_all_nodes


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def full_artifacts():
    """Complete artifacts for all 11 nodes."""
    artifacts = {}
    for node in CAMPAIGN_PLAN["nodes"]:
        if node.get("artifact"):
            artifacts[node["ext_id"]] = node["artifact"]
    return artifacts


@pytest.fixture
def partial_artifacts():
    """Artifacts with N06-N11 missing (only N01-N05)."""
    artifacts = {}
    for node in CAMPAIGN_PLAN["nodes"][:4]:  # Only N01-N04 (N05 missing)
        if node.get("artifact"):
            artifacts[node["ext_id"]] = node["artifact"]
    return artifacts


# ---------------------------------------------------------------------------
# C1: Node verdict tests
# ---------------------------------------------------------------------------
class TestNodeVerdicts:
    """Tests for C1: all 11 nodes status=done and reproduction_verdict=pass."""

    def test_all_nodes_done(self):
        """All nodes in plan have status=done."""
        for node in CAMPAIGN_PLAN["nodes"]:
            assert node["status"] == "done", f"{node['ext_id']} not done"

    def test_all_nodes_repro_pass(self):
        """All nodes have reproduction_verdict=pass."""
        for node in CAMPAIGN_PLAN["nodes"]:
            assert node["reproduction_verdict"] == "pass", \
                f"{node['ext_id']} repro not pass"

    def test_evaluate_node_pass(self):
        """Node with all criteria passing gets PASS verdict."""
        node = CAMPAIGN_PLAN["nodes"][0]  # N01
        verdict = evaluate_node(node)
        assert verdict.verdict == "PASS"
        assert verdict.n_pass == verdict.n_criteria

    def test_evaluate_all_nodes(self):
        """All 11 nodes evaluate to PASS."""
        verdicts = evaluate_all_nodes()
        assert len(verdicts) == 11
        assert all(v.verdict == "PASS" for v in verdicts)

    def test_node_without_artifact_withheld(self):
        """Node without artifact gets WITHHELD verdict."""
        node = {"ext_id": "P84-N99", "status": "done", "reproduction_verdict": "pass"}
        verdict = evaluate_node(node)
        assert verdict.verdict == "WITHHELD"

    def test_node_with_failing_criterion(self):
        """Node with failing criterion gets FAIL verdict."""
        node = {
            "ext_id": "P84-N99",
            "status": "done",
            "reproduction_verdict": "pass",
            "artifact": {
                "criteria": {
                    "C1": {"value": 5.0, "limit": 3.0, "pass": False},
                }
            }
        }
        verdict = evaluate_node(node)
        assert verdict.verdict == "FAIL"


# ---------------------------------------------------------------------------
# C2: Cross-domain interface tests
# ---------------------------------------------------------------------------
class TestCrossDomainChecks:
    """Tests for C2: cross-domain interface checks ≥3 PASS."""

    def test_turbulence_contract_pass(self, full_artifacts):
        """LBM turbulence contract passes with valid artifacts."""
        result = run_cross_domain_checks(full_artifacts)
        turb = next(c for c in result["contracts"]
                    if c["contract_id"] == LBM_TURBULENCE_CONTRACT)
        assert turb["verdict"] == "PASS"
        assert turb["n_pass"] >= 3

    def test_multiphase_contract_pass(self, full_artifacts):
        """Multiphase-particle contract passes with valid artifacts."""
        result = run_cross_domain_checks(full_artifacts)
        multi = next(c for c in result["contracts"]
                     if c["contract_id"] == MULTIPHASE_PARTICLE_CONTRACT)
        assert multi["verdict"] == "PASS"
        assert multi["n_pass"] >= 3

    def test_total_checks_ge_3(self, full_artifacts):
        """Total cross-domain checks ≥3 PASS."""
        result = run_cross_domain_checks(full_artifacts)
        assert result["summary"]["n_pass"] >= 3

    def test_missing_artifact_withheld(self):
        """Missing artifacts result in WITHHELD checks."""
        result = run_cross_domain_checks({})
        assert result["summary"]["overall"] == "WITHHELD"

    def test_partial_artifacts_n05_missing(self, partial_artifacts):
        """Missing N05 results in multiphase checks partially withheld."""
        result = run_cross_domain_checks(partial_artifacts)
        # N04 present, N05 missing -> multiphase checks for N05 withheld
        multi = next(c for c in result["contracts"]
                     if c["contract_id"] == MULTIPHASE_PARTICLE_CONTRACT)
        # N04 checks (M1-M3) should pass, N05 checks (M4-M5) should be withheld
        assert multi["n_pass"] >= 3  # M1, M2, M3 from N04
        assert multi["n_withheld"] >= 2  # M4, M5 from N05


# ---------------------------------------------------------------------------
# C3: Code alignment tests
# ---------------------------------------------------------------------------
class TestCodeAlignment:
    """Tests for C3: mature nodes regulation clause alignment 100%."""

    def test_alignment_rate_100(self, full_artifacts):
        """All regulation clauses aligned with 100% rate."""
        result = run_code_alignment_checks(full_artifacts)
        assert result["summary"]["alignment_rate"] >= 1.0, \
            f"Alignment rate {result['summary']['alignment_rate']} < 1.0"

    def test_all_regulations_covered(self, full_artifacts):
        """All expected regulations are covered."""
        result = run_code_alignment_checks(full_artifacts)
        regs = result["regulations_covered"]
        assert "IMO_IS_CODE_2008" in regs
        assert "IACS_UR_S28" in regs
        assert "IMO_MEPC_335_76" in regs

    def test_missing_artifact_fails_alignment(self):
        """Missing artifacts fail alignment checks."""
        result = run_code_alignment_checks({})
        assert result["summary"]["alignment_rate"] == 0.0
        assert result["summary"]["verdict"] == "WITHHELD"


# ---------------------------------------------------------------------------
# C4: Lineage ruling tests
# ---------------------------------------------------------------------------
class TestLineageRuling:
    """Tests for C4: plan 84 gaps逐条回应."""

    def test_all_gaps_addressed(self, full_artifacts):
        """All plan 84 gaps are addressed."""
        ruling = run_lineage_ruling(full_artifacts)
        assert ruling.overall_status == "PASS"
        assert ruling.addressed == ruling.total_gaps

    def test_gap_count_matches(self):
        """Gap count matches PLAN_84_GAPS."""
        ruling = run_lineage_ruling({})
        assert ruling.total_gaps == len(PLAN_84_GAPS)

    def test_each_gap_has_ruling(self, full_artifacts):
        """Each gap has a corresponding ruling."""
        ruling = run_lineage_ruling(full_artifacts)
        gap_ids = {g["gap_id"] for g in PLAN_84_GAPS}
        ruling_ids = {r.gap_id for r in ruling.rulings}
        assert gap_ids == ruling_ids

    def test_missing_artifacts_partial(self, partial_artifacts):
        """Missing artifacts result in PARTIAL or NOT_ADDRESSED."""
        ruling = run_lineage_ruling(partial_artifacts)
        # Some gaps should not be fully addressed (N06-N11 missing)
        assert ruling.not_addressed > 0 or ruling.partial > 0


# ---------------------------------------------------------------------------
# C5: End-to-end JSON output tests
# ---------------------------------------------------------------------------
class TestEndToEndOutput:
    """Tests for C5: end-to-end JSON output with all required sections."""

    def test_gate_runs(self):
        """Gate runs without error."""
        verdict = run_gate()
        assert verdict is not None

    def test_gate_passes(self):
        """Gate reaches PASS with full evidence."""
        verdict = run_gate()
        assert verdict["overall_status"] == "PASS", \
            f"Gate failed: blocking={verdict.get('blocking')}"

    def test_json_has_required_sections(self):
        """JSON output contains all required sections."""
        verdict = run_gate()
        assert "node_verdicts" in verdict
        assert "cross_domain_checks" in verdict
        assert "code_alignment" in verdict
        assert "lineage_ruling" in verdict
        assert "campaign_report" in verdict

    def test_json_schema(self):
        """JSON output has correct schema tag."""
        verdict = run_gate()
        assert verdict["schema"] == "p84w2_campaign_gate_verdict.v1"

    def test_write_outputs(self, tmp_path):
        """write_outputs creates JSON and markdown files."""
        verdict = run_gate()
        out_json = tmp_path / "report.json"
        out_md = tmp_path / "report.md"
        paths = write_outputs(verdict, out_json, out_md)
        assert out_json.exists()
        assert out_md.exists()
        assert "json_sha256" in paths

    def test_render_markdown(self):
        """render_markdown produces valid markdown."""
        verdict = run_gate()
        md = render_markdown(verdict)
        assert "# P84-W2" in md
        assert "## 1. 门裁决总览" in md
        assert "## 2. 判据 C1" in md
        assert "## 3. 判据 C2" in md
        assert "## 4. 判据 C3" in md
        assert "## 5. 判据 C4" in md
        assert "## 6. 战役级汇总" in md


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------
class TestIntegration:
    """Integration tests for the full gate."""

    def test_full_gate_pass(self):
        """Full gate passes with all criteria met."""
        verdict = run_gate()
        assert verdict["overall_status"] == "PASS"
        assert len(verdict["blocking"]) == 0

    def test_campaign_report_summary(self):
        """Campaign report summary is consistent."""
        verdict = run_gate()
        cr = verdict["campaign_report"]
        assert cr["total_nodes"] == 11
        assert cr["nodes_pass"] == 11
        assert cr["overall_verdict"] == "PASS"

    def test_criteria_count(self):
        """Gate has exactly 5 criteria (C1-C5)."""
        verdict = run_gate()
        assert len(verdict["criteria"]) == 5
        ids = [c["criterion_id"] for c in verdict["criteria"]]
        assert ids == ["C1", "C2", "C3", "C4", "C5"]
