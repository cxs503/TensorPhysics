"""Comprehensive tests for P20 campaign integration gate.

Covers:
- C1: DAG acyclic verification (topological sort succeeds)
- C2: Cross-repo interface contract validation (FSI/hydrostatic/seakeeping)
- C3: Maturity report summary (≥2 maturity reports from N07/N08)
- C4: Campaign summary report six-element check
- C5: Negative control — intentional cycle → DAG check must fail
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from tensorphysics.campaign.p20_gate import (
    CampaignDAG,
    CycleError,
    DAGNode,
    build_dag_from_plan,
    collect_maturity,
    inject_cycle,
    load_evidence,
    render_markdown,
    run_gate,
    validate_fsi_handoff,
    validate_hydrostatic_handoff,
    validate_seakeeping_handoff,
)
from tensorphysics.campaign.p20_gate.contracts import (
    ContractCheck,
    ContractResult,
)
from tensorphysics.campaign.p20_gate.dag import ConvergenceResult
from tensorphysics.campaign.p20_gate.gate import GateCriterion, write_outputs
from tensorphysics.campaign.p20_gate.maturity import (
    MaturitySummary,
    NodeMaturity,
    build_maturity_from_nodes,
    parse_maturity_artifact,
)


# ===================================================================
# C1: DAG acyclic verification
# ===================================================================

class TestDAGAcyclic:
    """C1: Topological sort succeeds on acyclic graph."""

    def test_simple_linear_dag(self):
        nodes = [
            DAGNode("N1", "Node 1", "done"),
            DAGNode("N2", "Node 2", "done", depends_on=("N1",)),
            DAGNode("N3", "Node 3", "done", depends_on=("N2",)),
        ]
        dag = CampaignDAG(nodes)
        order = dag.topological_sort()
        assert order.index("N1") < order.index("N2") < order.index("N3")

    def test_diamond_dag(self):
        nodes = [
            DAGNode("A", "A", "done"),
            DAGNode("B", "B", "done", depends_on=("A",)),
            DAGNode("C", "C", "done", depends_on=("A",)),
            DAGNode("D", "D", "done", depends_on=("B", "C")),
        ]
        dag = CampaignDAG(nodes)
        order = dag.topological_sort()
        assert order.index("A") < order.index("B")
        assert order.index("A") < order.index("C")
        assert order.index("B") < order.index("D")
        assert order.index("C") < order.index("D")

    def test_empty_dag(self):
        dag = CampaignDAG([])
        assert dag.topological_sort() == []

    def test_single_node(self):
        dag = CampaignDAG([DAGNode("X", "X", "done")])
        assert dag.topological_sort() == ["X"]

    def test_convergence_all_done(self):
        nodes = [
            DAGNode("N1", "Node 1", "done"),
            DAGNode("N2", "Node 2", "done", depends_on=("N1",)),
            DAGNode("N3", "Node 3", "done", depends_on=("N1", "N2")),
        ]
        dag = CampaignDAG(nodes)
        result = dag.verify_convergence()
        assert result.acyclic is True
        assert result.all_deps_done is True
        assert result.converged is True
        assert result.done_count == 3
        assert result.total_count == 3

    def test_convergence_blocked(self):
        nodes = [
            DAGNode("N1", "Node 1", "done"),
            DAGNode("N2", "Node 2", "running", depends_on=("N1",)),
            DAGNode("N3", "Node 3", "queued", depends_on=("N2",)),
        ]
        dag = CampaignDAG(nodes)
        result = dag.verify_convergence()
        assert result.acyclic is True
        assert result.all_deps_done is False
        assert result.converged is False
        assert "N3" in result.blocked_nodes

    def test_build_from_plan(self):
        plan = {
            "plan_id": 27,
            "nodes": [
                {"ext_id": "P20-N01", "name": "Resistance", "status": "done",
                 "depends_on": []},
                {"ext_id": "P20-N02", "name": "Propulsion", "status": "done",
                 "depends_on": ["P20-N01"]},
                {"ext_id": "P20-N03", "name": "Seakeeping", "status": "done",
                 "depends_on": ["P20-N01"]},
            ],
        }
        dag = build_dag_from_plan(plan)
        order = dag.topological_sort()
        assert order.index("P20-N01") < order.index("P20-N02")
        assert order.index("P20-N01") < order.index("P20-N03")

    def test_duplicate_node_id_rejected(self):
        with pytest.raises(ValueError, match="duplicate"):
            CampaignDAG([
                DAGNode("X", "X", "done"),
                DAGNode("X", "X2", "done"),
            ])

    def test_unknown_dependency_rejected(self):
        with pytest.raises(ValueError, match="unknown node"):
            CampaignDAG([
                DAGNode("A", "A", "done", depends_on=("Z",)),
            ])

    def test_self_loop_rejected(self):
        with pytest.raises(ValueError, match="self-loop"):
            CampaignDAG([
                DAGNode("A", "A", "done", depends_on=("A",)),
            ])


# ===================================================================
# C5: Negative control — cycle detection
# ===================================================================

class TestCycleDetection:
    """C5: Intentional cycle → topological sort must raise CycleError."""

    def test_simple_cycle(self):
        nodes = [
            DAGNode("A", "A", "done", depends_on=("B",)),
            DAGNode("B", "B", "done", depends_on=("A",)),
        ]
        dag = CampaignDAG(nodes)
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_three_node_cycle(self):
        nodes = [
            DAGNode("A", "A", "done", depends_on=("C",)),
            DAGNode("B", "B", "done", depends_on=("A",)),
            DAGNode("C", "C", "done", depends_on=("B",)),
        ]
        dag = CampaignDAG(nodes)
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_cycle_convergence_fails(self):
        nodes = [
            DAGNode("A", "A", "done", depends_on=("B",)),
            DAGNode("B", "B", "done", depends_on=("A",)),
            DAGNode("C", "C", "done"),
        ]
        dag = CampaignDAG(nodes)
        result = dag.verify_convergence()
        assert result.acyclic is False
        assert result.converged is False
        assert result.sorted_ids is None

    def test_inject_cycle_creates_cycle(self):
        nodes = [
            DAGNode("A", "A", "done"),
            DAGNode("B", "B", "done"),
            DAGNode("C", "C", "done"),
        ]
        # Original is acyclic
        dag = CampaignDAG(nodes)
        dag.topological_sort()  # should not raise

        # Inject cycle
        mutated = inject_cycle(nodes, cycle_chain=["A", "B", "C"])
        dag2 = CampaignDAG(mutated)
        with pytest.raises(CycleError):
            dag2.topological_sort()

    def test_inject_cycle_default(self):
        nodes = [
            DAGNode("A", "A", "done"),
            DAGNode("B", "B", "done"),
        ]
        mutated = inject_cycle(nodes)  # uses first two nodes
        dag = CampaignDAG(mutated)
        with pytest.raises(CycleError):
            dag.topological_sort()

    def test_inject_cycle_unknown_node(self):
        nodes = [DAGNode("A", "A", "done")]
        with pytest.raises(ValueError, match="unknown node"):
            inject_cycle(nodes, cycle_chain=["A", "Z"])

    def test_inject_cycle_single_node(self):
        nodes = [DAGNode("A", "A", "done")]
        with pytest.raises(ValueError, match="at least 2"):
            inject_cycle(nodes)


# ===================================================================
# C2: Cross-repo interface contracts
# ===================================================================

class TestFSIHandoff:
    """FSI coupling: LBM surface forces → FEM boundary conditions."""

    def _valid_payload(self) -> dict:
        return {
            "schema": "fsi_surface_force_handoff.v1",
            "frame": {"x": "streamwise", "y": "transverse", "z": "vertical_up"},
            "timestamp_s": 0.05,
            "interface_nodes": [
                {"node_id": 1, "x": 0.0, "y": 0.1, "z": -0.05,
                 "Fx_N": 12.3, "Fy_N": 0.5, "Fz_N": -1.2},
                {"node_id": 2, "x": 0.1, "y": 0.2, "z": -0.04,
                 "Fx_N": 11.8, "Fy_N": 0.6, "Fz_N": -1.1},
            ],
            "force_components": ["Fx_N", "Fy_N", "Fz_N"],
            "total_force_N": [24.1, 1.1, -2.3],
            "mesh_id": "hull_surface_v2",
        }

    def test_valid_payload_passes(self):
        result = validate_fsi_handoff(self._valid_payload())
        assert result.passed is True
        assert result.contract_id == "fsi_handoff_v1"

    def test_missing_schema_fails(self):
        payload = self._valid_payload()
        payload["schema"] = "wrong_schema"
        result = validate_fsi_handoff(payload)
        assert result.passed is False
        assert result.failed_checks[0].check_id == "F1_schema"

    def test_wrong_frame_fails(self):
        payload = self._valid_payload()
        payload["frame"]["x"] = "wrong"
        result = validate_fsi_handoff(payload)
        assert result.passed is False

    def test_negative_timestamp_fails(self):
        payload = self._valid_payload()
        payload["timestamp_s"] = -1.0
        result = validate_fsi_handoff(payload)
        assert result.passed is False

    def test_empty_nodes_fails(self):
        payload = self._valid_payload()
        payload["interface_nodes"] = []
        result = validate_fsi_handoff(payload)
        assert result.passed is False

    def test_force_components_wrong_fails(self):
        payload = self._valid_payload()
        payload["force_components"] = ["Fx", "Fy", "Fz"]
        result = validate_fsi_handoff(payload)
        assert result.passed is False

    def test_total_force_inconsistent_fails(self):
        payload = self._valid_payload()
        payload["total_force_N"] = [999.0, 999.0, 999.0]
        result = validate_fsi_handoff(payload)
        assert result.passed is False


class TestHydrostaticHandoff:
    """Hydrostatic coupling: CAD geometry → ship platform."""

    def _valid_payload(self) -> dict:
        return {
            "schema": "cad_hydrostatic_handoff.v1",
            "geometry_id": "kcs_hull_v3",
            "principal_dimensions": {
                "Lpp_m": 230.0, "B_m": 32.2, "T_m": 10.0,
                "Cb": 0.65, "Cm": 0.98, "Cp": 0.66,
            },
            "displacement_m3": 48139.0,
            "wetted_area_m2": 8900.0,
        }

    def test_valid_payload_passes(self):
        result = validate_hydrostatic_handoff(self._valid_payload())
        assert result.passed is True

    def test_missing_schema_fails(self):
        payload = self._valid_payload()
        payload["schema"] = "wrong"
        result = validate_hydrostatic_handoff(payload)
        assert result.passed is False

    def test_dimensional_order_violated(self):
        payload = self._valid_payload()
        payload["principal_dimensions"]["B_m"] = 300.0  # B > Lpp
        result = validate_hydrostatic_handoff(payload)
        assert result.passed is False

    def test_cb_out_of_range(self):
        payload = self._valid_payload()
        payload["principal_dimensions"]["Cb"] = 0.15
        result = validate_hydrostatic_handoff(payload)
        assert result.passed is False

    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False
    def test_displacement_closure(self):
        payload = self._valid_payload()
        # Override displacement to create a mismatch
        payload["displacement_m3"] = 51234.0
        # Cb * Lpp * B * T = 0.65 * 230 * 32.2 * 10 = 48139
        # displacement = 51234 → rel_err = |48139 - 51234| / 51234 ≈ 0.06
        # This should fail the closure check (tolerance 1e-12)
        result = validate_hydrostatic_handoff(payload)
        # The closure check should fail because the numbers do not match
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is False

    def test_displacement_closure_exact(self):
        lpp, b, t, cb = 230.0, 32.2, 10.0, 0.65
        disp = cb * lpp * b * t  # exact
        payload = {
            "schema": "cad_hydrostatic_handoff.v1",
            "geometry_id": "test_hull",
            "principal_dimensions": {
                "Lpp_m": lpp, "B_m": b, "T_m": t, "Cb": cb,
            },
            "displacement_m3": disp,
        }
        result = validate_hydrostatic_handoff(payload)
        closure_check = next(c for c in result.checks if c.check_id == "H6_displacement_closure")
        assert closure_check.ok is True


class TestSeakeepingHandoff:
    """Seakeeping coupling: BEM potential flow → ship platform."""

    def _valid_payload(self) -> dict:
        return {
            "schema": "bem_seakeeping_handoff.v1",
            "frame": {"x": "surge", "y": "sway", "z": "heave",
                      "rx": "roll", "ry": "pitch", "rz": "yaw"},
            "frequency_rad_s": [0.5, 1.0, 1.5, 2.0],
            "wave_heading_deg": 180.0,
            "force_amplitude": [
                {"freq_rad_s": 0.5, "Fx_N": 1e5, "Fy_N": 0, "Fz_N": 2e5,
                 "Mx_Nm": 0, "My_Nm": 5e6, "Mz_Nm": 0},
                {"freq_rad_s": 1.0, "Fx_N": 1.2e5, "Fy_N": 0, "Fz_N": 2.5e5,
                 "Mx_Nm": 0, "My_Nm": 6e6, "Mz_Nm": 0},
                {"freq_rad_s": 1.5, "Fx_N": 0.8e5, "Fy_N": 0, "Fz_N": 1.8e5,
                 "Mx_Nm": 0, "My_Nm": 4e6, "Mz_Nm": 0},
                {"freq_rad_s": 2.0, "Fx_N": 0.5e5, "Fy_N": 0, "Fz_N": 1.2e5,
                 "Mx_Nm": 0, "My_Nm": 3e6, "Mz_Nm": 0},
            ],
            "encounter_convention": "positive_head_seas",
        }

    def test_valid_payload_passes(self):
        result = validate_seakeeping_handoff(self._valid_payload())
        assert result.passed is True

    def test_missing_schema_fails(self):
        payload = self._valid_payload()
        payload["schema"] = "wrong"
        result = validate_seakeeping_handoff(payload)
        assert result.passed is False

    def test_unsorted_frequencies_fails(self):
        payload = self._valid_payload()
        payload["frequency_rad_s"] = [2.0, 0.5, 1.0, 1.5]
        result = validate_seakeeping_handoff(payload)
        assert result.passed is False

    def test_negative_frequency_fails(self):
        payload = self._valid_payload()
        payload["frequency_rad_s"] = [-0.5, 1.0, 1.5, 2.0]
        result = validate_seakeeping_handoff(payload)
        assert result.passed is False

    def test_missing_encounter_convention_fails(self):
        payload = self._valid_payload()
        del payload["encounter_convention"]
        result = validate_seakeeping_handoff(payload)
        assert result.passed is False

    def test_force_count_mismatch_fails(self):
        payload = self._valid_payload()
        payload["force_amplitude"] = payload["force_amplitude"][:2]
        result = validate_seakeeping_handoff(payload)
        assert result.passed is False


# ===================================================================
# C3: Maturity report summary
# ===================================================================

class TestMaturity:
    """C3: ≥2 maturity reports (N07/N08) complete."""

    def _make_node(self, nid: str, trl: int = 7, delivered: bool = True,
                   validation: str = "passed") -> NodeMaturity:
        return NodeMaturity(
            node_id=nid, name=nid, trl=trl, delivered=delivered,
            artifact_count=3, tests_passed=12, tests_total=12,
            validation_status=validation,
        )

    def test_parse_maturity_artifact(self):
        art = {
            "schema": "p20_node_maturity.v1",
            "node_id": "P20-N07",
            "name": "Wind resistance",
            "trl": 7,
            "delivered": True,
            "artifacts": ["a.json", "b.json"],
            "tests": {"passed": 10, "total": 10},
            "validation_status": "passed",
            "notes": "Validated against wind tunnel data",
        }
        nm = parse_maturity_artifact(art, "P20-N07")
        assert nm.node_id == "P20-N07"
        assert nm.trl == 7
        assert nm.delivered is True
        assert nm.test_pass_rate == 1.0

    def test_build_maturity_from_nodes_all_pass(self):
        campaign_nodes = [
            {"ext_id": "P20-N07", "name": "Wind resistance", "status": "done"},
            {"ext_id": "P20-N08", "name": "Wet modes", "status": "done"},
        ]
        artifacts = {
            "P20-N07": {
                "schema": "p20_node_maturity.v1",
                "node_id": "P20-N07", "name": "Wind resistance",
                "trl": 7, "delivered": True,
                "artifacts": ["a.json"],
                "tests": {"passed": 10, "total": 10},
                "validation_status": "passed",
            },
            "P20-N08": {
                "schema": "p20_node_maturity.v1",
                "node_id": "P20-N08", "name": "Wet modes",
                "trl": 6, "delivered": True,
                "artifacts": ["b.json"],
                "tests": {"passed": 8, "total": 8},
                "validation_status": "passed",
            },
        }
        summary = build_maturity_from_nodes(campaign_nodes, artifacts)
        assert summary.delivered_count >= 2
        assert summary.validation_passed >= 2
        assert summary.overall_trl > 0

    def test_maturity_missing_nodes(self):
        summary = build_maturity_from_nodes([], {})
        assert summary.delivered_count == 0
        assert summary.validation_passed == 0
        assert summary.overall_trl == 0.0

    def test_node_test_pass_rate(self):
        nm = NodeMaturity(
            node_id="X", name="X", trl=5, delivered=True,
            artifact_count=1, tests_passed=8, tests_total=10,
            validation_status="passed",
        )
        assert nm.test_pass_rate == 0.8


# ===================================================================
# C4: Campaign summary report six-element check
# ===================================================================

class TestGateReport:
    """C4: Campaign summary report with six elements."""

    def test_render_markdown(self):
        verdict = {
            "schema": "p20_campaign_gate_verdict.v1",
            "gate_id": "P20-G1",
            "title": "P20 全战役集成验收门",
            "generated_utc": "2026-11-07T00:00:00+00:00",
            "campaign": {"plan_id": 27, "title": "P20", "status": "done", "n_nodes": 5},
            "overall_status": "PASS",
            "blocking": [],
            "criteria": [
                {"criterion_id": "a", "label": "DAG", "status": "PASS",
                 "detail": "all done", "evidence": {"acyclic": True, "all_deps_done": True,
                 "converged": True, "sorted_ids": ["N1", "N2"], "blocked_nodes": [],
                 "done_count": 2, "total_count": 2}},
                {"criterion_id": "b", "label": "Contracts", "status": "PASS",
                 "detail": "all pass", "evidence": {"contracts": [], "missing_payloads": []}},
                {"criterion_id": "c", "label": "Maturity", "status": "PASS",
                 "detail": "2/2", "evidence": {"delivered_count": 2, "total_count": 5,
                 "validation_passed": 2, "overall_trl": 6.5, "nodes": []}},
                {"criterion_id": "d", "label": "Report", "status": "PASS",
                 "detail": "6/6", "evidence": {"present": ["case_introduction", "methodology",
                 "results", "figures", "notes", "error_analysis"], "missing": []}},
            ],
            "provenance": {},
            "limitations": ["test limitation"],
        }
        md = render_markdown(verdict)
        assert "P20" in md
        assert "PASS" in md
        assert "DAG" in md
        assert "Maturity" in md or "成熟化" in md

    def test_gate_criterion_as_dict(self):
        gc = GateCriterion("a", "test", "PASS", "detail", {"key": "val"})
        d = gc.as_dict()
        assert d["criterion_id"] == "a"
        assert d["status"] == "PASS"
        assert d["evidence"] == {"key": "val"}


# ===================================================================
# Integration: full gate run with synthetic evidence
# ===================================================================

class TestFullGateRun:
    """End-to-end gate run with synthetic evidence directory."""

    def _make_evidence_dir(self, tmp_path: Path) -> Path:
        d = tmp_path / "p20_evidence"
        d.mkdir()

        # campaign_nodes.json
        campaign = {
            "plan_id": 27,
            "title": "P20 Campaign",
            "status": "done",
            "nodes": [
                {"ext_id": "P20-N01", "name": "Resistance", "status": "done",
                 "depends_on": []},
                {"ext_id": "P20-N02", "name": "Propulsion", "status": "done",
                 "depends_on": ["P20-N01"]},
                {"ext_id": "P20-N03", "name": "Seakeeping", "status": "done",
                 "depends_on": ["P20-N01"]},
                {"ext_id": "P20-N07", "name": "Wind resistance", "status": "done",
                 "depends_on": ["P20-N01"]},
                {"ext_id": "P20-N08", "name": "Wet modes", "status": "done",
                 "depends_on": ["P20-N01"]},
            ],
        }
        (d / "campaign_nodes.json").write_text(json.dumps(campaign))

        # contracts
        contracts_dir = d / "contracts"
        contracts_dir.mkdir()

        fsi = {
            "schema": "fsi_surface_force_handoff.v1",
            "frame": {"x": "streamwise", "y": "transverse", "z": "vertical_up"},
            "timestamp_s": 0.05,
            "interface_nodes": [
                {"node_id": 1, "x": 0.0, "y": 0.1, "z": -0.05,
                 "Fx_N": 12.3, "Fy_N": 0.5, "Fz_N": -1.2},
            ],
            "force_components": ["Fx_N", "Fy_N", "Fz_N"],
            "total_force_N": [12.3, 0.5, -1.2],
        }
        (contracts_dir / "fsi_handoff.json").write_text(json.dumps(fsi))

        hydro = {
            "schema": "cad_hydrostatic_handoff.v1",
            "geometry_id": "test_hull",
            "principal_dimensions": {
                "Lpp_m": 100.0, "B_m": 20.0, "T_m": 5.0, "Cb": 0.6,
            },
            "displacement_m3": 0.6 * 100 * 20 * 5,  # exact closure
        }
        (contracts_dir / "hydrostatic_handoff.json").write_text(json.dumps(hydro))

        seakeep = {
            "schema": "bem_seakeeping_handoff.v1",
            "frame": {"x": "surge", "y": "sway", "z": "heave",
                      "rx": "roll", "ry": "pitch", "rz": "yaw"},
            "frequency_rad_s": [0.5, 1.0, 1.5],
            "wave_heading_deg": 180.0,
            "force_amplitude": [
                {"freq_rad_s": 0.5, "Fx_N": 1e5, "Fy_N": 0, "Fz_N": 2e5,
                 "Mx_Nm": 0, "My_Nm": 5e6, "Mz_Nm": 0},
                {"freq_rad_s": 1.0, "Fx_N": 1.2e5, "Fy_N": 0, "Fz_N": 2.5e5,
                 "Mx_Nm": 0, "My_Nm": 6e6, "Mz_Nm": 0},
                {"freq_rad_s": 1.5, "Fx_N": 0.8e5, "Fy_N": 0, "Fz_N": 1.8e5,
                 "Mx_Nm": 0, "My_Nm": 4e6, "Mz_Nm": 0},
            ],
            "encounter_convention": "positive_head_seas",
        }
        (contracts_dir / "seakeeping_handoff.json").write_text(json.dumps(seakeep))

        # maturity
        maturity_dir = d / "maturity"
        maturity_dir.mkdir()
        for nid, trl in [("p20_n07", 7), ("p20_n08", 6)]:
            art = {
                "schema": "p20_node_maturity.v1",
                "node_id": nid.replace("_", "-").upper(),
                "name": nid,
                "trl": trl,
                "delivered": True,
                "artifacts": [f"{nid}.json"],
                "tests": {"passed": 10, "total": 10},
                "validation_status": "passed",
            }
            (maturity_dir / f"{nid}.json").write_text(json.dumps(art))

        # campaign_report.json with six elements
        report = {
            "case_introduction": "P20 campaign integration test",
            "methodology": "DAG + contract + maturity verification",
            "results": {"overall": "PASS"},
            "figures": ["dag.png", "contracts.png"],
            "notes": "Synthetic evidence for testing",
            "error_analysis": "All checks within tolerance",
        }
        (d / "campaign_report.json").write_text(json.dumps(report))

        return d

    def test_full_gate_pass(self, tmp_path):
        ev_dir = self._make_evidence_dir(tmp_path)
        verdict = run_gate(ev_dir)
        assert verdict["schema"] == "p20_campaign_gate_verdict.v1"
        assert verdict["gate_id"] == "P20-G1"
        assert verdict["overall_status"] == "PASS"
        assert verdict["blocking"] == []

        # Verify all criteria pass
        for c in verdict["criteria"]:
            assert c["status"] == "PASS", f"Criterion {c['criterion_id']} failed: {c['detail']}"

    def test_full_gate_with_cycle(self, tmp_path):
        ev_dir = self._make_evidence_dir(tmp_path)
        # Inject a cycle into the campaign nodes
        campaign_path = ev_dir / "campaign_nodes.json"
        campaign = json.loads(campaign_path.read_text())
        # Add circular dependency: N01 depends on N02, N02 depends on N01
        campaign["nodes"][0]["depends_on"] = ["P20-N02"]
        campaign["nodes"][1]["depends_on"] = ["P20-N01"]
        campaign_path.write_text(json.dumps(campaign))

        verdict = run_gate(ev_dir)
        assert verdict["overall_status"] == "FAIL"
        crit_a = next(c for c in verdict["criteria"] if c["criterion_id"] == "a")
        assert crit_a["status"] == "FAIL"
        assert "cycle" in crit_a["detail"].lower() or "blocked" in crit_a["detail"].lower()

    def test_full_gate_missing_maturity(self, tmp_path):
        ev_dir = self._make_evidence_dir(tmp_path)
        # Remove maturity directory
        import shutil
        shutil.rmtree(ev_dir / "maturity")

        verdict = run_gate(ev_dir)
        crit_c = next(c for c in verdict["criteria"] if c["criterion_id"] == "c")
        assert crit_c["status"] == "FAIL"

    def test_write_outputs(self, tmp_path):
        ev_dir = self._make_evidence_dir(tmp_path)
        verdict = run_gate(ev_dir)

        out_dir = tmp_path / "outputs"
        verdict_path = out_dir / "gate_verdict.json"
        report_path = out_dir / "P20_GATE_REPORT.md"

        write_outputs(verdict, verdict_path, report_path)

        assert verdict_path.exists()
        assert report_path.exists()
        assert verdict_path.stat().st_size > 100
        assert report_path.stat().st_size > 100

        # Verify JSON is valid
        loaded = json.loads(verdict_path.read_text())
        assert loaded["gate_id"] == "P20-G1"

        # Verify markdown contains key sections
        md = report_path.read_text()
        assert "P20" in md
        assert "DAG" in md or "dag" in md.lower()

    def test_load_evidence_missing_files(self, tmp_path):
        d = tmp_path / "empty_evidence"
        d.mkdir()
        ev = load_evidence(d)
        assert ev["campaign"] is None
        assert ev["fsi_payload"] is None
        assert ev["hydrostatic_payload"] is None
        assert ev["seakeeping_payload"] is None
        assert ev["report"] is None


# ===================================================================
# Contract result tests
# ===================================================================

class TestContractResult:
    def test_passed_property(self):
        cr = ContractResult(
            contract_id="test", title="Test", producer="A", consumer="B",
            checks=[ContractCheck("C1", True, "ok")],
        )
        assert cr.passed is True

    def test_failed_property(self):
        cr = ContractResult(
            contract_id="test", title="Test", producer="A", consumer="B",
            checks=[ContractCheck("C1", False, "fail")],
        )
        assert cr.passed is False
        assert len(cr.failed_checks) == 1

    def test_empty_checks_not_passed(self):
        cr = ContractResult(
            contract_id="test", title="Test", producer="A", consumer="B",
            checks=[],
        )
        assert cr.passed is False

    def test_as_dict(self):
        cr = ContractResult(
            contract_id="test", title="Test", producer="A", consumer="B",
            checks=[ContractCheck("C1", True, "ok", value=0.5, limit=1.0)],
        )
        d = cr.as_dict()
        assert d["verdict"] == "PASS"
        assert d["n_checks"] == 1
        assert d["n_failed"] == 0


# ===================================================================
# ConvergenceResult tests
# ===================================================================

class TestConvergenceResult:
    def test_converged_property(self):
        cr = ConvergenceResult(
            acyclic=True, all_deps_done=True, sorted_ids=["A", "B"],
            blocked_nodes=[], done_count=2, total_count=2, detail="ok",
        )
        assert cr.converged is True

    def test_not_converged_cyclic(self):
        cr = ConvergenceResult(
            acyclic=False, all_deps_done=False, sorted_ids=None,
            blocked_nodes=["A", "B"], done_count=0, total_count=2,
            detail="cycle",
        )
        assert cr.converged is False

    def test_not_converged_blocked(self):
        cr = ConvergenceResult(
            acyclic=True, all_deps_done=False, sorted_ids=["A", "B"],
            blocked_nodes=["B"], done_count=1, total_count=2,
            detail="blocked",
        )
        assert cr.converged is False

    def test_as_dict(self):
        cr = ConvergenceResult(
            acyclic=True, all_deps_done=True, sorted_ids=["A"],
            blocked_nodes=[], done_count=1, total_count=1, detail="ok",
        )
        d = cr.as_dict()
        assert d["converged"] is True
        assert d["acyclic"] is True
