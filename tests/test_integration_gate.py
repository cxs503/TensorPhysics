"""Contract tests for the P16-G1 campaign integration gate.

The gate is only useful if it can say *no*.  Every positive fixture here is
paired with a negative control that must flip the verdict, and the end-to-end
test builds a complete synthetic evidence tree so the gate's PASS path is
exercised as well (the real campaign tree in ``artifacts/p16_gate`` is
upstream-dependent and is not asserted on).
"""
from __future__ import annotations

import json

import pytest

from tensorphysics.campaign.integration_gate import (
    GATE_SCHEMA,
    NODE_ARTIFACT_SCHEMA,
    WET_MODAL_SCHEMA,
    WIND_SCHEMA,
    evaluate_case,
    evaluate_node,
    parse_criteria,
    parse_pytest_counts,
    render_markdown,
    run_gate,
    sha256_file,
    validate_endpoint_compatibility,
    validate_wet_modal_handoff,
    validate_wind_coefficient_table,
    write_outputs,
)

MATH = 2.0 * 3.141592653589793


# --------------------------------------------------------------------------- fixtures
def wind_payload(**over):
    payload = {
        "schema": WIND_SCHEMA,
        "units": {"beta_deg": "deg", "C_X": "1", "C_Y": "1", "C_N": "1",
                  "U_ref_m_s": "m/s", "rho_air_kg_m3": "kg/m^3",
                  "A_front_m2": "m^2", "A_lateral_m2": "m^2", "z_ref_m": "m"},
        "frame": {"x": "forward", "y": "port", "z": "up", "heading_zero": "bow-on"},
        "reference": {"A_front_m2": 312.0, "A_lateral_m2": 1580.0, "U_ref_m_s": 10.0,
                      "rho_air_kg_m3": 1.225, "z_ref_m": 10.0},
        "samples": [
            {"beta_deg": 0.0, "C_X": 0.82, "C_Y": 0.06, "C_N": 0.05},
            {"beta_deg": 30.0, "C_X": 0.78, "C_Y": 0.20, "C_N": 0.12},
            {"beta_deg": 60.0, "C_X": 0.55, "C_Y": 0.55, "C_N": 0.20},
            {"beta_deg": 90.0, "C_X": 0.12, "C_Y": 0.86, "C_N": 0.02},
            {"beta_deg": 120.0, "C_X": -0.30, "C_Y": 0.70, "C_N": -0.18},
            {"beta_deg": 150.0, "C_X": -0.52, "C_Y": 0.35, "C_N": -0.22},
            {"beta_deg": 180.0, "C_X": -0.62, "C_Y": 0.02, "C_N": -0.03},
            {"beta_deg": 210.0, "C_X": -0.55, "C_Y": -0.28, "C_N": 0.20},
            {"beta_deg": 240.0, "C_X": -0.34, "C_Y": -0.62, "C_N": 0.16},
            {"beta_deg": 270.0, "C_X": -0.10, "C_Y": -0.80, "C_N": 0.04},
            {"beta_deg": 300.0, "C_X": 0.36, "C_Y": -0.58, "C_N": -0.12},
            {"beta_deg": 330.0, "C_X": 0.70, "C_Y": -0.24, "C_N": -0.10},
            {"beta_deg": 360.0, "C_X": 0.82, "C_Y": 0.06, "C_N": 0.05},
        ],
    }
    payload.update(over)
    return payload


def wet_payload(**over):
    payload = {
        "schema": WET_MODAL_SCHEMA,
        "units": {"f_hz": "Hz", "omega_rad_s": "rad/s", "c_m_s": "m/s"},
        "fluid": {"rho_kg_m3": 1025.0, "c_m_s": 1500.0},
        "S_ref_m2": 3100.0,
        "modes": [
            {"mode_id": 1, "f_hz": 4.31, "omega_rad_s": MATH * 4.31, "dry_f_hz": 6.10,
             "added_mass_kg": 1.2e6, "modal_mass_kg": 2.0e6, "sigma": 0.42},
            {"mode_id": 2, "f_hz": 6.72, "omega_rad_s": MATH * 6.72, "dry_f_hz": 9.05,
             "added_mass_kg": 0.9e6, "modal_mass_kg": 1.7e6, "sigma": 0.51},
        ],
        "wavenumber_k": [MATH * 4.31 / 1500.0, MATH * 6.72 / 1500.0],
        "added_mass_matrix_kg": [[1.2e6, 0.0], [0.0, 0.9e6]],
    }
    payload.update(over)
    return payload


def node(cid="P16-N1", mission=5184, status="in-flight", criteria=None):
    return {
        "ext_id": cid,
        "name": f"{cid} synthetic node",
        "repo": "TensorSYN",
        "mission_id": mission,
        "status": status,
        "acceptance_criteria": criteria or ["C1: metric err <= 0.3%",
                                            "C2: negative control FAIL >= 8%"],
    }


def artifact(verdicts, cid="P16-N1", mission=5184, schema=NODE_ARTIFACT_SCHEMA):
    return {"schema": schema, "ext_id": cid, "mission_id": mission,
            "criteria": verdicts}


PASS_CRIT = {"C1": {"value": 0.0021, "limit": 0.003, "op": "le", "pass": True},
             "C2": {"value": 0.083, "limit": 0.08, "op": "ge", "pass": True}}


# --------------------------------------------------------------------- (b) wind table
def test_wind_table_valid_payload_passes():
    res = validate_wind_coefficient_table(wind_payload())
    assert res.passed, [c.as_dict() for c in res.failed_checks]
    assert {c.check_id for c in res.checks} >= {"W1_schema_tag", "W2_units", "W3_reference_quantities",
                                               "W4_frame_convention", "W5_samples_present",
                                               "W6_coefficient_magnitude", "W7_anchor_angles",
                                               "W8_angular_resolution"}


@pytest.mark.parametrize(
    "mutate, bad_check",
    [
        (lambda p: p.pop("units"), "W2_units"),
        (lambda p: p["units"].update({"U_ref_m_s": "kn"}), "W2_units"),
        (lambda p: p["reference"].update({"A_front_m2": 0.0}), "W3_reference_quantities"),
        (lambda p: p["frame"].update({"heading_zero": "stern-on"}), "W4_frame_convention"),
        (lambda p: p["samples"].pop(3), "W7_anchor_angles"),
        (lambda p: p["samples"].__setitem__(0, {"beta_deg": 0.0, "C_X": 8.2, "C_Y": 0.06, "C_N": 0.05}),
         "W6_coefficient_magnitude"),
        (lambda p: p["samples"].__setitem__(1, {"beta_deg": 30.0, "C_X": float("nan"), "C_Y": 0.2,
                                                "C_N": 0.1}), "W5_samples_present"),
        (lambda p: p.update({"schema": "wind_v0"}), "W1_schema_tag"),
    ],
)
def test_wind_table_negative_controls(mutate, bad_check):
    payload = wind_payload()
    mutate(payload)
    res = validate_wind_coefficient_table(payload)
    assert not res.passed
    assert bad_check in {c.check_id for c in res.failed_checks}


def test_wind_table_coarse_angles_are_rejected():
    """A 3-sample table (0/90/180) leaves 90 deg gaps -> cannot interpolate."""
    payload = wind_payload()
    payload["samples"] = [s for s in payload["samples"] if s["beta_deg"] in (0.0, 90.0, 180.0)]
    res = validate_wind_coefficient_table(payload)
    assert "W8_angular_resolution" in {c.check_id for c in res.failed_checks}


# ------------------------------------------------------------- (b) wet-modal handoff
def test_wet_modal_valid_payload_passes():
    res = validate_wet_modal_handoff(wet_payload())
    assert res.passed, [c.as_dict() for c in res.failed_checks]
    assert res.evidence["k_checked"] is True and res.evidence["added_mass_matrix_checked"] is True


def test_wet_modal_hz_rad_s_mismatch_is_caught():
    payload = wet_payload()
    payload["modes"][0]["omega_rad_s"] = 4.31          # Hz pasted into the rad/s slot
    res = validate_wet_modal_handoff(payload)
    assert "M6_hz_rad_s_identity" in {c.check_id for c in res.failed_checks}


def test_wet_modal_added_mass_must_soften_frequency():
    payload = wet_payload()
    payload["modes"][1]["dry_f_hz"] = 3.0              # wet > dry is unphysical
    res = validate_wet_modal_handoff(payload)
    assert "M7_added_mass_softening" in {c.check_id for c in res.failed_checks}


def test_wet_modal_wavenumber_mismatch_is_caught():
    payload = wet_payload()
    payload["wavenumber_k"] = [1.0, 2.0]               # arbitrary, not omega/c
    res = validate_wet_modal_handoff(payload)
    assert "M10_wavenumber_consistency" in {c.check_id for c in res.failed_checks}


def test_wet_modal_asymmetric_added_mass_matrix_is_caught():
    payload = wet_payload()
    payload["added_mass_matrix_kg"] = [[1.2e6, 3.0e5], [0.0, 0.9e6]]
    res = validate_wet_modal_handoff(payload)
    assert "M11_added_mass_matrix" in {c.check_id for c in res.failed_checks}


@pytest.mark.parametrize(
    "mutate, bad_check",
    [
        (lambda p: p.pop("units"), "M2_units"),
        (lambda p: p["fluid"].update({"c_m_s": 0.0}), "M3_fluid_properties"),
        (lambda p: p.pop("S_ref_m2"), "M4_reference_area"),
        (lambda p: p["modes"][0].update({"sigma": 4.0}), "M9_sigma_bounds"),
        (lambda p: p["modes"].reverse(), "M8_mode_ordering"),
        (lambda p: p["modes"][0].update({"added_mass_kg": None}), "M5_modes_present"),
    ],
)
def test_wet_modal_negative_controls(mutate, bad_check):
    payload = wet_payload()
    mutate(payload)
    res = validate_wet_modal_handoff(payload)
    assert not res.passed
    assert bad_check in {c.check_id for c in res.failed_checks}


# --------------------------------------------------------- (b) endpoint compatibility
def test_endpoint_compatibility_reports_missing_wind_endpoint():
    facts = {
        "n_paths": 443,
        "endpoints": {
            "/api/v1/calculations/wind_resistance": {
                "present": False, "methods": [], "related_paths": ["/api/v1/polaris/operational-window"],
                "schemas": {}},
            "/api/v1/fem/modal": {
                "present": True, "methods": ["post"],
                "related_paths": ["/api/v1/fem/modal"],
                "schemas": {"ModalInput": {"properties": {"length": {"type": "number"},
                                                          "boundary_type": {"type": "string"}}}}},
        },
    }
    results = {r.contract_id: r for r in validate_endpoint_compatibility(facts)}
    assert not results["wind_endpoint_v1"].passed
    assert {c.check_id for c in results["wind_endpoint_v1"].failed_checks} == {
        "E1_wind_endpoint_present", "E2_wind_endpoint_fields"}
    # dry-mode endpoint exists but cannot carry wet-modal data
    assert "E4_modal_endpoint_wet_fields" in {c.check_id for c in
                                              results["wet_modal_endpoint_v1"].failed_checks}


def test_endpoint_compatibility_passes_when_consumer_is_ready():
    facts = {"n_paths": 1, "endpoints": {
        "/api/v1/calculations/wind_resistance": {
            "present": True, "methods": ["post"], "related_paths": [],
            "schemas": {"WindInput": {"properties": {k: {} for k in
                                                     ("beta_deg", "U_ref_m_s", "rho_air_kg_m3",
                                                      "A_front_m2", "A_lateral_m2", "z_ref_m")}}}},
        "/api/v1/fem/modal": {
            "present": True, "methods": ["post"], "related_paths": [],
            "schemas": {"ModalInput": {"properties": {k: {} for k in
                                                      ("natural_frequencies", "fluid_rho", "sound_speed",
                                                       "added_mass", "added_mass_matrix")}}}}}}
    results = {r.contract_id: r for r in validate_endpoint_compatibility(facts)}
    assert results["wind_endpoint_v1"].passed
    assert results["wet_modal_endpoint_v1"].passed


# ------------------------------------------------------------------ (a) node verdicts
def test_parse_criteria_handles_uppercase_and_lowercase_ids():
    pairs = parse_criteria(["C1: err <= 0.3%", "a: turning diameter err <15%", "no id here"])
    assert [p[0] for p in pairs] == ["C1", "a", "#3"]


def test_node_without_artifact_is_withheld_not_passed():
    v = evaluate_node(node(), {"delivered": False, "accepted": False}, None)
    assert v.status == "WITHHELD"
    assert all(c.status == "WITHHELD" for c in v.criteria)
    assert "no criterion is evaluable" in v.reason


def test_node_with_passing_artifact_passes():
    v = evaluate_node(node(), {"delivered": True, "accepted": True}, artifact(PASS_CRIT))
    assert v.status == "PASS"
    assert [c.status for c in v.criteria] == ["PASS", "PASS"]
    assert v.criteria[0].value == pytest.approx(0.0021)
    assert v.criteria[0].limit == pytest.approx(0.003)


def test_node_attestation_mismatch_fails():
    crit = {"C1": {"value": 0.0051, "limit": 0.003, "op": "le", "pass": True}}
    v = evaluate_node(node(criteria=["C1: err <= 0.3%"]), {"delivered": True}, artifact(crit))
    assert v.status == "FAIL"
    assert "attestation mismatch" in v.criteria[0].note


def test_node_missing_criterion_is_withheld():
    v = evaluate_node(node(), {"delivered": True}, artifact({"C1": PASS_CRIT["C1"]}))
    assert v.status == "WITHHELD"
    assert v.criteria[1].criterion_id == "C2" and v.criteria[1].status == "WITHHELD"


def test_node_artifact_wrong_schema_is_failed():
    v = evaluate_node(node(), {"delivered": True}, artifact(PASS_CRIT, schema="whatever.v0"))
    assert v.status == "FAIL"
    assert "rejected artifacts" in v.reason


def test_node_artifact_mission_mismatch_is_failed():
    v = evaluate_node(node(mission=5184), {"delivered": True}, artifact(PASS_CRIT, mission=9999))
    assert v.status == "FAIL"


# ----------------------------------------------------------------- (c) regression
def test_parse_pytest_counts_variants():
    assert parse_pytest_counts("36 passed, 1 skipped in 12.3s")["passed"] == 36
    assert parse_pytest_counts("5 failed, 12 passed in 2s")["failed"] == 5
    assert parse_pytest_counts("19/19 passed")["passed"] == 19
    assert parse_pytest_counts("no tests ran") == {}


def _p15_node(passed=19, total=19):
    return {"ext_id": "IN14-D1", "repo": "TensorDEM",
            "probe": {"test_targets": ["tests/test_hertz_mindlin.py"],
                      "n_passed_recorded": passed, "n_total_recorded": total}}


def test_regression_clean_rerun_passes():
    v = evaluate_case(_p15_node(), {"counts": {"passed": 19}, "returncode": 0})
    assert v.status == "PASS" and abs(v.delta) < 0.01


def test_regression_one_failure_breaks_the_gate():
    v = evaluate_case(_p15_node(), {"counts": {"passed": 18, "failed": 1}, "returncode": 1})
    assert v.status == "FAIL"
    assert abs(v.delta) > 0.01


def test_regression_test_loss_is_flagged_even_when_green():
    v = evaluate_case(_p15_node(passed=19), {"counts": {"passed": 12}, "returncode": 0})
    assert v.test_count_preserved is False
    assert v.status == "FAIL"


def test_regression_case_dropped_at_fresh_head_fails():
    """A P15 verification case absent at GitHub main is a dropped deliverable."""
    v = evaluate_case(_p15_node(), None, fresh_absence={
        "IN14-D1": {"repo": "TensorDEM", "head": "03e857d",
                    "job_id": "job-1",
                    "targets_absent_at_fresh_head": ["tests/test_hertz_mindlin.py"]}})
    assert v.status == "FAIL"
    assert v.coverage == "none"
    assert "fresh head" in v.detail and "test_hertz_mindlin.py" in v.detail


def test_regression_partial_coverage_stays_withheld():
    """Green measured subset + unresolved target = evidence gap, not a pass."""
    probe = {"counts": {"passed": 34}, "returncode": 0,
             "unresolved_detail": [{"path": "tests/test_hertz_mindlin.py",
                                    "classification": "not_in_baseline_history"}]}
    v = evaluate_case(_p15_node(), probe)
    assert v.status == "WITHHELD"
    assert v.coverage == "partial"


def test_regression_names_the_failing_test():
    probe = {"counts": {"passed": 73, "failed": 1}, "returncode": 1, "resolvable": True,
             "stdout_tail": ("FAILED tests/test_doc_claim_audit.py::test_ratchet_against_baseline"
                             " - AssertionError\n1 failed, 73 passed in 12.0s")}
    v = evaluate_case(_p15_node(passed=74), probe)
    assert v.status == "FAIL"
    assert v.failing_tests and "test_ratchet_against_baseline" in v.failing_tests[0]
    assert "failing:" in v.detail


def test_regression_unresolvable_targets_get_a_specific_message():
    probe = {"counts": {}, "returncode": None, "resolvable": False,
             "unresolved_detail": [{"path": "tests/test_x.py",
                                    "classification": "not_in_baseline_history"}]}
    v = evaluate_case(_p15_node(), probe)
    assert v.status == "WITHHELD"
    assert "no P15 probe target resolvable" in v.detail


def test_regression_unexecuted_probe_is_withheld():
    v = evaluate_case(_p15_node(), None)
    assert v.status == "WITHHELD" and v.delta is None


def test_regression_unparsable_probe_is_withheld():
    v = evaluate_case(_p15_node(), {"counts": {}, "returncode": 4,
                                    "stdout_tail": "ERROR: file or directory not found"})
    assert v.status == "WITHHELD"
    assert "no parsable pytest summary" in v.detail


# ------------------------------------------------------------- end-to-end gate
def _write_tree(tmp_path, *, node_passes=True, wind_ok=True, probe_ok=True):
    d = tmp_path / "evidence"
    (d / "contracts").mkdir(parents=True)
    (d / "nodes").mkdir()
    campaign = {"schema": "p16_campaign_nodes.v1", "collected_utc": "T",
                "source": {"plan_title": "synthetic", "plan_status": "active",
                           "physics_scope": ["LBM"], "coupling_framework_requirements": {}},
                "nodes": [node("P16-N1", 5184, "done"), node("P16-N2", 5185, "done"),
                          {"ext_id": "P16-G1", "name": "gate", "repo": "TensorPhysics",
                           "mission_id": 5192, "status": "in-flight", "acceptance_criteria": []}]}
    (d / "campaign_nodes.json").write_text(json.dumps(campaign))
    (d / "node_evidence.json").write_text(json.dumps(
        {"nodes": {"P16-N1": {"delivered": True, "accepted": True},
                   "P16-N2": {"delivered": True, "accepted": True}}}))
    crit = dict(PASS_CRIT)
    if not node_passes:
        crit = {"C1": {"value": 0.0099, "limit": 0.003, "op": "le", "pass": True},
                "C2": dict(PASS_CRIT["C2"])}
    (d / "nodes" / "P16-N1.json").write_text(json.dumps(artifact(crit, "P16-N1", 5184)))
    (d / "nodes" / "P16-N2.json").write_text(json.dumps(artifact(PASS_CRIT, "P16-N2", 5185)))

    wind = wind_payload()
    if not wind_ok:
        wind["units"].pop("U_ref_m_s")
    (d / "contracts" / "wind_coefficients.json").write_text(json.dumps(wind))
    (d / "contracts" / "wet_modal_handoff.json").write_text(json.dumps(wet_payload()))
    (d / "ship_platform_facts.json").write_text(json.dumps(
        {"n_paths": 2, "endpoints": {
            "/api/v1/calculations/wind_resistance": {
                "present": True, "methods": ["post"], "related_paths": [],
                "schemas": {"WindInput": {"properties": {k: {} for k in
                                                         ("beta_deg", "U_ref_m_s", "rho_air_kg_m3",
                                                          "A_front_m2", "A_lateral_m2", "z_ref_m")}}}},
            "/api/v1/fem/modal": {
                "present": True, "methods": ["post"], "related_paths": [],
                "schemas": {"ModalInput": {"properties": {k: {} for k in
                                                          ("fluid_rho", "sound_speed", "added_mass",
                                                           "added_mass_matrix")}}}}}}))
    (d / "p15_baseline.json").write_text(json.dumps(
        {"nodes": [_p15_node()], "collected_utc": "T"}))
    counts = {"passed": 19} if probe_ok else {"passed": 17, "failed": 2}
    (d / "probe_results.json").write_text(json.dumps(
        {"schema": "p15_probe_results.v1",
         "results": {"IN14-D1": {"counts": counts, "returncode": 0 if probe_ok else 1,
                                 "command": "python -m pytest tests/test_hertz_mindlin.py -q"}}}))
    return d


def test_gate_passes_on_complete_evidence(tmp_path):
    d = _write_tree(tmp_path)
    verdict = run_gate(d, generated_utc="2026-01-01T00:00:00+00:00")
    assert verdict["schema"] == GATE_SCHEMA
    assert {c["criterion_id"]: c["status"] for c in verdict["criteria"]} == {
        "a": "PASS", "b": "PASS", "c": "PASS", "d": "PASS"}
    assert verdict["overall_status"] == "PASS" and verdict["blocking"] == []
    # provenance covers every input artifact with a real sha256
    prov = verdict["provenance"]
    assert prov["campaign"]["sha256"] and prov["node_artifact:P16-N1"]["sha256"]
    md = render_markdown(verdict)
    assert "overall_status" in md and "P15 回归基线" in md


@pytest.mark.parametrize(
    "flags, failing",
    [
        ({"node_passes": False}, "a"),
        ({"wind_ok": False}, "b"),
        ({"probe_ok": False}, "c"),
    ],
)
def test_gate_fails_when_any_criterion_breaks(tmp_path, flags, failing):
    d = _write_tree(tmp_path, **flags)
    verdict = run_gate(d)
    statuses = {c["criterion_id"]: c["status"] for c in verdict["criteria"]}
    assert statuses[failing] == "FAIL"
    assert verdict["overall_status"] == "FAIL"
    assert any(b.startswith(failing) for b in verdict["blocking"])


def test_gate_without_upstream_evidence_withholds_and_fails(tmp_path):
    """An empty evidence tree must never yield PASS (honest-FAIL discipline)."""
    d = tmp_path / "empty"
    d.mkdir()
    verdict = run_gate(d)
    statuses = {c["criterion_id"]: c["status"] for c in verdict["criteria"]}
    assert statuses["a"] == "WITHHELD" and statuses["c"] == "WITHHELD"
    assert verdict["overall_status"] == "FAIL"


def test_write_outputs_emits_json_and_markdown_with_digests(tmp_path):
    verdict = run_gate(_write_tree(tmp_path))
    out = write_outputs(verdict, tmp_path / "v.json", tmp_path / "v.md")
    assert sha256_file(tmp_path / "v.json") == out["json_sha256"]
    assert (tmp_path / "v.md").read_text().startswith("#")