"""Criteria for the measured-anchor *maturation* node (AG29W1-N02).

Covers the four additions of this node on top of the m9425 baseline:

* C6 governance block -- declared, machine-checked bench/scale/uncertainty/
  digitisation/applicability for **every** archived series, with negative cases;
* C7 criterion metrology -- what the truth can and cannot resolve;
* C8 external-validity probe -- the second, independent measured campaign, its
  exclusion from the 15 % verdict set, and the rule that a "good" error obtained
  by letting the domain limiter clip the ice thickness does not count;
* C9 evidence-gap registry -- dated attempts, machine-checked.

Plus the regression that motivated a real defect fix: a loaded series used to
drop its governance block on ``as_dict()`` round-trip, which silently erased the
applicability declaration.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from tensorphysics.ice.measured_anchor import (  # noqa: E402
    GATE_TOLERANCE,
    MeasuredSeries,
    double_entry_check,
    load_registry,
    sha256_file,
    validate_series,
)
from tensorphysics.ice.measured_criterion import (  # noqa: E402
    BREAKING_MODES,
    criterion_resolvability,
    evidence_gap_summary,
    governance_errors_for_registry,
    load_evidence_gaps,
    validate_evidence_gaps,
    validate_governance,
)

DATA_ROOT = REPO / "data" / "measured" / "ice_resistance"
V2_ARTIFACT = REPO / "artifacts" / "ice_resistance" / "measured_adjudication_v2.json"
SALMI_ID = "SALMI2014-UBIB"


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def loaded():
    return load_registry(DATA_ROOT, verify=True)


@pytest.fixture(scope="module")
def docs(loaded):
    return {sid: s.as_dict() for sid, s in loaded["series"].items()}


@pytest.fixture(scope="module")
def v2():
    assert V2_ARTIFACT.is_file(), f"run the maturation benchmark first: {V2_ARTIFACT}"
    return json.loads(V2_ARTIFACT.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# C6 -- governance: the fields the contract names, now declared and checked
# ---------------------------------------------------------------------------


def test_c6_three_series_registered_and_digest_verified(loaded):
    assert set(loaded["series"]) == {"MTUIKKU-HZ2016", "AGULHAS2-FS2012", SALMI_ID}
    assert loaded["errors"] == []
    for entry in loaded["registry"]["series"]:
        path = DATA_ROOT / entry["path"]
        assert sha256_file(path) == entry["sha256"]


def test_c6_every_series_declares_governance(docs):
    errors = governance_errors_for_registry(docs.values())
    assert errors and all(not v for v in errors.values()), errors


def test_c6_contract_fields_present(docs):
    for sid, doc in docs.items():
        assert doc["facility"]["kind"] in ("ice_basin", "full_scale_trial", "numerical_benchmark", "unknown")
        assert doc["facility"]["reported_frame"] == "full_scale"
        assert len(doc["digitisation"]["passes"]) >= 2
        assert isinstance(doc["uncertainty"]["reported"], bool)
        assert doc["applicability"]["ice_breaking_mode"] in BREAKING_MODES
        assert isinstance(doc["applicability"]["in_15pct_verdict_set"], bool)


def test_c6_negative_missing_facility(docs):
    doc = json.loads(json.dumps(docs["MTUIKKU-HZ2016"]))
    doc.pop("facility")
    errs = validate_governance(doc)
    assert any("facility" in e for e in errs)


def test_c6_negative_undeclared_facility_name(docs):
    doc = json.loads(json.dumps(docs["MTUIKKU-HZ2016"]))
    doc["facility"]["name"] = None
    doc["facility"]["known_gaps"] = []  # clearing the explanation must be rejected
    errs = validate_governance(doc)
    assert any("known_gaps" in e for e in errs)


def test_c6_negative_uncertainty_declaration_contradicts_data(docs):
    doc = json.loads(json.dumps(docs["MTUIKKU-HZ2016"]))
    doc["uncertainty"]["reported"] = False  # yet the points still carry R_unc_kN
    errs = validate_governance(doc)
    assert any("declaration and data disagree" in e for e in errs)


def test_c6_negative_single_digitisation_pass(docs):
    doc = json.loads(json.dumps(docs["AGULHAS2-FS2012"]))
    doc["digitisation"]["passes"] = ["pass_a"]
    errs = validate_governance(doc)
    assert any("passes" in e for e in errs)


def test_c6_negative_unknown_breaking_mode(docs):
    doc = json.loads(json.dumps(docs[SALMI_ID]))
    doc["applicability"]["ice_breaking_mode"] = "sideways"
    errs = validate_governance(doc)
    assert any("ice_breaking_mode" in e for e in errs)


def test_c6_governance_survives_round_trip(loaded, docs):
    """Regression: MeasuredSeries used to lose facility/applicability on as_dict()."""

    for sid, series in loaded["series"].items():
        rt = MeasuredSeries.from_dict(series.as_dict(), path=series.path)
        assert rt.applicability == series.applicability
        assert rt.facility == series.facility
        assert rt.uncertainty == series.uncertainty
        assert rt.digitisation == series.digitisation
        assert rt.as_dict()["applicability"]["in_15pct_verdict_set"] == docs[sid]["applicability"]["in_15pct_verdict_set"]


def test_c6_verdict_set_is_explicit(docs):
    in_set = {sid for sid, d in docs.items() if d["applicability"]["in_15pct_verdict_set"]}
    assert in_set == {"MTUIKKU-HZ2016"}, "only the measured-resistance series on the downward-bending " \
                                        "envelope may certify the 15 % criterion"


# ---------------------------------------------------------------------------
# C8 -- the second independent measured campaign
# ---------------------------------------------------------------------------


def test_c8_salmi_series_schema_and_logic(loaded):
    s = loaded["series"][SALMI_ID]
    assert len(s.points) == 4
    assert validate_series(s.as_dict(), root=DATA_ROOT / "series" / "salmi2014_ubib") == []
    # table 5 (channel-corrected measurement) must exceed table 8 (further friction-corrected)
    for p in s.points:
        assert p.R_alt_kN is not None and p.R_meas_kN > p.R_alt_kN
    # full-scale thickness 1.0 m from 40 mm model ice at 1:25
    assert {p.h_m for p in s.points} == {1.0}
    assert [round(p.V_ms, 3) for p in sorted(s.points, key=lambda p: p.V_ms)] == [1.543, 3.087, 4.63, 6.173]


def test_c8_salmi_double_entry(loaded):
    sdir = DATA_ROOT / "series" / "salmi2014_ubib"
    pa = json.loads((sdir / "pass_a.json").read_text(encoding="utf-8"))
    pb = json.loads((sdir / "pass_b.json").read_text(encoding="utf-8"))
    rec = double_entry_check(pa, pb, sorted(pa))
    assert rec["passed"] and rec["agreement_rate"] == 1.0
    assert rec["n_fields_compared"] >= 16


def test_c8_salmi_source_digest_matches_bytes(loaded):
    s = loaded["series"][SALMI_ID]
    src = s.provenance["sources"][0]
    path = DATA_ROOT / "series" / "salmi2014_ubib" / src["artifact"]
    assert sha256_file(path) == src["sha256"]
    assert path.stat().st_size == src["bytes"]


def test_c8_salmi_excluded_from_verdict_set(v2):
    ext = v2["C8_external_validity"]
    assert ext["series_id"] == SALMI_ID
    assert ext["in_15pct_verdict_set"] is False
    assert ext["ice_breaking_mode"] == "upward_bending"
    assert all(r["in_15pct_verdict_set"] is False for r in v2["per_point_table"] if r["series_id"] == SALMI_ID)


def test_c8_domain_clipping_never_counts_as_a_fit(v2):
    ext = v2["C8_external_validity"]
    clipped = [p for p in ext["sweep"] if p["domain_clipped"]]
    assert clipped, "expected some configurations to hit the h <= 0.1*B limiter"
    best = min(ext["sweep"], key=lambda p: p["median_rel_error_pct"])
    # the outright best configuration is only 'best' because the limiter clipped h
    assert best["median_rel_error_pct"] < ext["best_admissible_median_rel_error_pct"]
    assert best["admissible"] is False
    assert all(p["admissible"] for p in ext["sweep"] if p["config"] == ext["best_admissible_config"])


def test_c8_two_closure_families_bracket_the_measurement(v2):
    ext = v2["C8_external_validity"]
    model_ratio = ext["best_admissible_pred_over_meas"]
    theory = ext["published_ubib_theory"]
    assert model_ratio > 1.0, "the downward-bending closure over-predicts this upward-breaking hull"
    for key, rec in theory.items():
        assert rec["n_points"] == 4
        assert rec["all_underpredicted"], key
        assert rec["median_rel_error_pct"] > GATE_TOLERANCE


# ---------------------------------------------------------------------------
# C7 -- criterion metrology
# ---------------------------------------------------------------------------


def test_c7_resolvability_on_synthetic_clean_data():
    rows = [
        {"point_id": "a", "R_meas_kN": 100.0, "R_unc_kN": 2.0, "R_pred_kN": 102.0, "h_m": 1.0, "V_ms": 1.0},
        {"point_id": "b", "R_meas_kN": 200.0, "R_unc_kN": 4.0, "R_pred_kN": 196.0, "h_m": 1.0, "V_ms": 2.0},
    ]
    res = criterion_resolvability(rows, tolerance=0.15)
    assert res["fail_resolvable"] == 0.0
    assert res["pass_resolvable"] == 1.0
    assert res["criterion_state"] == "BOTH_DIRECTIONS_RESOLVABLE"


def test_c7_resolvability_without_uncertainty_is_withheld():
    rows = [{"point_id": "a", "R_meas_kN": 100.0, "R_unc_kN": None, "R_pred_kN": 300.0}]
    res = criterion_resolvability(rows)
    assert res["fail_resolvable"] is None
    assert res["criterion_state"] == "WITHHELD"


def test_c7_measured_truth_can_falsify_but_not_confirm(v2):
    c7 = v2["C7_criterion_metrology"]
    assert c7["n_points_with_uncertainty"] == 7, "3 of 10 points report no scatter" \
                                                 " (the source leaves them empty: missing is missing)"
    assert c7["fail_resolvable"] == 1.0
    assert c7["n_fail_resolvable_at_kmax"] == 7
    assert c7["pass_resolvable"] == 0.0
    assert c7["n_points_uncertainty_above_tolerance"] == 7
    assert c7["median_u_over_R"] > GATE_TOLERANCE
    assert c7["criterion_state"] == "FAIL_RESOLVABLE_PASS_WITHHELD"


# ---------------------------------------------------------------------------
# C9 -- evidence gaps
# ---------------------------------------------------------------------------


def test_c9_gap_registry_validates():
    doc = load_evidence_gaps(DATA_ROOT, verify=True)
    summary = evidence_gap_summary(doc)
    assert summary["n_gaps"] == 4
    assert summary["n_attempts"] >= summary["n_gaps"]
    assert "GAP-INDEPENDENT-DOWNWARD-SERIES" in summary["gap_ids"]
    for gap in doc["gaps"]:
        assert gap["attempts"], gap["gap_id"]
        for a in gap["attempts"]:
            assert a["utc"] and a["url"]


def test_c9_negative_gap_without_attempts_is_rejected():
    doc = {
        "kind": "measured_evidence_gap_registry",
        "gaps": [
            {
                "gap_id": "GAP-X",
                "target": "t",
                "why_needed": "w",
                "unblock": "u",
                "status": "OPEN",
                "attempts": [],
            }
        ],
    }
    errs = validate_evidence_gaps(doc)
    assert any("attempts" in e for e in errs)


def test_c9_negative_attempt_without_timestamp_or_error():
    doc = {
        "kind": "measured_evidence_gap_registry",
        "gaps": [
            {
                "gap_id": "GAP-X",
                "target": "t",
                "why_needed": "w",
                "unblock": "u",
                "status": "OPEN",
                "attempts": [{"url": "https://example.invalid"}],
            }
        ],
    }
    errs = validate_evidence_gaps(doc)
    assert any("utc" in e for e in errs) and any("http_status" in e for e in errs)


def test_c9_negative_attempt_log_digest_drift():
    doc = load_evidence_gaps(DATA_ROOT, verify=True)
    tampered = json.loads(json.dumps(doc))
    tampered["attempt_log"]["sha256"] = "0" * 64
    errs = validate_evidence_gaps(tampered, root=DATA_ROOT)
    assert any("digest mismatch" in e for e in errs)


def test_c9_closed_gap_must_point_at_the_asset_it_closed():
    doc = load_evidence_gaps(DATA_ROOT, verify=True)
    partial = [g for g in doc["gaps"] if g["gap_id"] == "GAP-INDEPENDENT-DOWNWARD-SERIES"][0]
    assert partial["status"] == "PARTIAL"
    prog = partial["partial_progress"]
    assert prog["series_added"] == SALMI_ID
    assert prog["sha256"] == sha256_file(DATA_ROOT / "series" / "salmi2014_ubib" / "series.json")


# ---------------------------------------------------------------------------
# the maturation artifact as a whole
# ---------------------------------------------------------------------------


def test_artifact_has_all_criteria_and_honest_fail(v2):
    crit = v2["criteria"]
    assert set(crit) == {
        "C1_schema_and_digest",
        "C2_double_entry",
        "C3_15pct_adjudication",
        "C4_physics_gates",
        "C5_legacy_full_spectrum",
        "C6_governance",
        "C7_criterion_metrology",
        "C8_external_validity_probe",
        "C9_evidence_gaps",
    }
    assert crit["C3_15pct_adjudication"] is False, "the 15 % FAIL is honest and must not be flipped"
    assert v2["verdict_15pct"] == "FAIL_15PCT"
    for key in ("C1_schema_and_digest", "C2_double_entry", "C4_physics_gates",
                "C5_legacy_full_spectrum", "C6_governance", "C7_criterion_metrology",
                "C8_external_validity_probe", "C9_evidence_gaps"):
        assert crit[key] is True, key


def test_artifact_numbers_come_from_the_baseline_artifact(v2):
    base = v2["baseline"]
    assert base["sha256"] == sha256_file(REPO / base["source_artifact"])
    assert abs(base["best_median_rel_error_pct"] - 198.27909035426337) < 1e-9
    assert base["ledger"]["C3_15pct_adjudication"]["honest_fail"] is True


def test_artifact_per_point_table_is_complete(v2):
    rows = v2["per_point_table"]
    verdict_rows = [r for r in rows if r["in_15pct_verdict_set"]]
    assert len(verdict_rows) == 10
    assert {r["series_id"] for r in verdict_rows} == {"MTUIKKU-HZ2016"}
    assert all(r["rel_error_pct"] > 15.0 for r in verdict_rows)
    assert {r["point_id"] for r in rows if r["series_id"] == SALMI_ID} == {
        "UBIB-3KT", "UBIB-6KT", "UBIB-9KT", "UBIB-12KT"
    }


def test_artifact_figures_are_on_disk_and_digested(v2):
    assert len(v2["figures"]) >= 3
    for fig in v2["figures"]:
        assert fig["caption"]
        path = REPO / fig["path"]
        assert path.is_file() and path.stat().st_size > 5000
        assert sha256_file(path) == fig["sha256"]