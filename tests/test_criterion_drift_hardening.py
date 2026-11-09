"""Maturity tests for the criterion freeze / drift-audit mechanism.

These tests measure the four machine acceptance properties of the node and are
themselves the reproducibility contract:

(a) drift detection on the labelled 24-case synthetic set: recall >= 0.92,
    precision >= 0.95;
(b) field-level extraction F1 >= 0.95 over the six structural fields;
(c) fail-closed: empty / unparseable goal -> ``unregistered`` (never a pass),
    and a frozen criterion missing from the report -> ``unregistered``;
(d) zero domain semantics: the mechanism sources contain no domain vocabulary
    (the lexicon is supplied here, not in the package).

All fixtures are synthetic (zero domain semantics).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_registry import (  # noqa: E402
    CriterionField,
    CriterionLedger,
    CriterionManifest,
    audit_delivery,
    extract_criteria_from_goal,
    register_goal,
)
from tensorphysics.campaign.criterion_registry import bench  # noqa: E402
from tensorphysics.campaign.criterion_registry.manifest import (  # noqa: E402
    CRITERION_FIELDS,
    freeze_manifest,
)

ARTIFACT = ROOT / "artifacts" / "criterion_drift_audit" / "bench_v2_hardened.json"
BASELINE_ARTIFACT = ROOT / "artifacts" / "criterion_drift_audit" / "bench_v1_legacy.json"

PACKAGE = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry"


# ---------------------------------------------------------------------------
# (a)+(b) benchmark gates
# ---------------------------------------------------------------------------

class TestBenchmarkGates:
    @pytest.fixture(scope="class")
    def record(self) -> dict:
        return bench.run_benchmark(label="pytest")

    def test_all_gates_pass(self, record):
        assert record["all_gates_pass"], json.dumps(record["gates"], indent=2)

    def test_detection_recall_and_precision(self, record):
        det = record["detection"]
        assert det["recall"] >= 0.92, det
        assert det["precision"] >= 0.95, det
        assert det["tp"] + det["fn"] == 12, det  # 12 drift cases
        assert det["tn"] + det["fp"] == 12, det  # 12 unchanged cases

    def test_extraction_field_f1(self, record):
        ext = record["extraction"]
        assert ext["f1"] >= 0.95, ext
        for name, m in ext["per_field"].items():
            assert m["precision"] >= 0.9 and m["recall"] >= 0.9, (name, m)

    def test_held_out_corpus_is_reported_and_above_gate(self, record):
        """Held-out set written after freezing the extractor (measured once)."""
        held = record["held_out"]
        assert held["cases"] == 20
        assert held["f1"] >= 0.90, held
        artefact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        assert artefact["held_out"]["f1"] == held["f1"]
        assert artefact["held_out"]["failures"] == held["failures"]

    def test_corpus_is_the_artefact_corpus(self, record):
        """Numbers quoted in the report must come from this exact corpus."""
        assert ARTIFACT.is_file(), f"missing artefact {ARTIFACT}"
        artefact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
        assert artefact["corpus_sha256"] == record["corpus_sha256"]
        for key in ("extraction", "detection"):
            for metric in ("precision", "recall", "f1"):
                assert artefact[key][metric] == record[key][metric], (key, metric)

    def test_baseline_is_recorded_and_worse(self, record):
        """Baseline (legacy extractor on the same corpus) validates the delta."""
        assert BASELINE_ARTIFACT.is_file(), f"missing baseline {BASELINE_ARTIFACT}"
        baseline = json.loads(BASELINE_ARTIFACT.read_text(encoding="utf-8"))
        assert baseline["corpus_sha256"] == record["corpus_sha256"]
        assert baseline["extraction"]["f1"] < record["extraction"]["f1"]
        assert baseline["detection"]["recall"] <= record["detection"]["recall"]
        assert not baseline["all_gates_pass"]


# ---------------------------------------------------------------------------
# (b) targeted extraction regressions found by the baseline measurement
# ---------------------------------------------------------------------------

class TestExtractionRegressions:
    @pytest.mark.parametrize("text,expected", [
        # case-count trap: level is anchored on the comparator
        ("on 24 cases recall >= 0.92", {"threshold": 0.92, "comparator": "ge"}),
        ("across 12 runs margin <= 5 pct",
         {"threshold": 5.0, "unit": "pct", "comparator": "le"}),
        # natural-language comparators
        ("at most 2.0 mm deviation",
         {"threshold": 2.0, "unit": "mm", "comparator": "le"}),
        ("no less than 3 runs",
         {"threshold": 3.0, "unit": "runs", "comparator": "ge"}),
        # scientific notation / thousands separators
        ("mean error of 1.5e-3 m relative to /data/ref_7",
         {"threshold": 1.5e-3, "unit": "m", "denominator": "/data/ref_7",
          "reference_path": "/data/ref_7"}),
        ("1,024 samples, error <= 5 pct",
         {"threshold": 5.0, "unit": "pct", "comparator": "le"}),
        # intervals use the upper bound
        ("error in 2-3 pct band",
         {"threshold": 3.0, "unit": "pct", "comparator": "in_range"}),
    ])
    def test_level_is_comparator_anchored(self, text, expected):
        got = bench.extracted_fields(text)
        for key, value in expected.items():
            assert got[key] == value, (text, key, got)

    def test_no_level_no_criterion(self):
        for text in ("", None, "   ", "use TBD as the level", "report in prose"):
            assert extract_criteria_from_goal(text) == [], text


# ---------------------------------------------------------------------------
# (c) fail-closed negative controls
# ---------------------------------------------------------------------------

class TestFailClosed:
    def test_empty_goal_registers_nothing(self):
        regs = register_goal("", "agent")
        assert len(regs) == 1
        assert regs[0].status == "unregistered"
        assert regs[0].manifest is None

    def test_placeholder_goal_registers_nothing(self):
        regs = register_goal("C1: error <= TBD pct", "agent")
        assert [r.status for r in regs] == ["unregistered"]

    def test_missing_criterion_in_report_is_unregistered(self):
        regs = register_goal("C1: error <= 5 pct\nC2: score >= 0.9 ratio", "agent")
        assert len(regs) == 2
        ledger = CriterionLedger()
        audit = audit_delivery(regs, "some prose without any criterion", "m-missing", ledger)
        assert [r.classification for r in audit.results] == ["unregistered", "unregistered"]
        assert audit.failed_closed is True
        assert audit.kpi.unregistered_rate == 1.0
        assert audit.kpi.registered_rate == 0.0

    def test_unparseable_report_never_unchanged(self):
        regs = register_goal("C1: error <= 5 pct", "agent")
        audit = audit_delivery(regs, "", "m-empty", CriterionLedger())
        assert audit.results[0].classification == "unregistered"

    def test_tampered_hash_is_not_trusted(self):
        regs = register_goal("C1: error <= 5 pct", "agent")
        frozen = regs[0].manifest
        tampered = CriterionManifest(
            criterion_id=frozen.criterion_id,
            fields={"threshold": CriterionField("threshold", 99.0, True)},
            status="registered",
            frozen_at=frozen.frozen_at,
            frozen_by=frozen.frozen_by,
            sha256=frozen.sha256,
        )
        forged = register_goal("C1: error <= 5 pct", "agent")[0]
        forged.manifest = tampered
        audit = audit_delivery([forged], "C1: error <= 5 pct", "m-tamper", CriterionLedger())
        assert audit.results[0].classification == "tampered"
        assert audit.failed_closed is True


# ---------------------------------------------------------------------------
# event stream + KPI view
# ---------------------------------------------------------------------------

class TestEventStreamAndKPI:
    def test_drift_events_land_in_ledger(self, tmp_path):
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = CriterionLedger(storage_path=ledger_path)
        regs = register_goal(
            "C1: error <= 5 pct of reference\nC2: score >= 0.9 ratio", "agent")
        audit = audit_delivery(
            regs,
            "C1: error <= 9 pct of reference\nC2: score >= 0.9 ratio",
            "m-events",
            ledger,
        )
        classes = [r.classification for r in audit.results]
        assert classes == ["loosened", "unchanged"]
        events = ledger.query(mission_id="m-events")
        assert [e.classification for e in events] == classes
        assert events[0].reason
        # persistence round-trip
        assert len(CriterionLedger(storage_path=ledger_path)) == 2

        kpi = audit.kpi
        assert kpi.registered_rate == 1.0
        assert kpi.loosened_rate == 0.5
        assert kpi.unregistered_rate == 0.0

    def test_tightened_and_scope_changed_are_distinguished(self):
        regs = register_goal("C1: error <= 5 pct of reference", "agent")
        tight = audit_delivery(regs, "C1: error <= 2 pct of reference", "m-t", CriterionLedger())
        scope = audit_delivery(regs, "C1: error <= 2 pct of baseline", "m-s", CriterionLedger())
        assert tight.results[0].classification == "tightened"
        assert scope.results[0].classification == "scope_changed"


# ---------------------------------------------------------------------------
# (d) zero domain semantics
# ---------------------------------------------------------------------------

DOMAIN_LEXICON = (
    "resistance", "hull", "propeller", "ship", "wave", "drag", "froude",
    "reynolds", "displacement", "waterline", "ice", "iceberg", "floe",
    "plate", "beam", "stiffener", "bulkhead", "holtrop", "lindqvist", "iacs",
    "ittc", "turbine", "combustion",
)


class TestZeroDomainSemantics:
    def test_mechanism_sources_are_domain_free(self):
        paths = sorted(PACKAGE.glob("*.py"))
        assert paths, "no sources found"
        hits = bench.scan_domain_semantics(paths, DOMAIN_LEXICON)
        assert hits == [], hits

    def test_synthetic_fixtures_are_domain_free(self):
        """The benchmark corpus itself must not smuggle domain words in."""
        texts = [c.text for c in bench.EXTRACTION_CASES]
        texts += [c.frozen_text for c in bench.DRIFT_CASES]
        texts += [c.report_text for c in bench.DRIFT_CASES]
        blob = "\n".join(texts).lower()
        hits = [t for t in DOMAIN_LEXICON
                if re.search(r"\b" + re.escape(t) + r"\b", blob)]
        assert hits == [], hits


# ---------------------------------------------------------------------------
# canonical hashing (pre-registration anchor)
# ---------------------------------------------------------------------------

class TestFreezeAnchor:
    def test_same_content_same_hash(self):
        a = freeze_manifest(
            CriterionManifest("c1", {"threshold": CriterionField("threshold", 5.0, True)}),
            "agent")
        b = freeze_manifest(
            CriterionManifest("c2", {"threshold": CriterionField("threshold", 5.0, True)}),
            "agent")
        assert a.sha256 == b.sha256
        assert set(json.loads(a.to_canonical_json())) == set(CRITERION_FIELDS)

    def test_one_field_change_changes_hash(self):
        a = freeze_manifest(
            CriterionManifest("c1", {"threshold": CriterionField("threshold", 5.0, True)}),
            "agent")
        b = freeze_manifest(
            CriterionManifest("c1", {
                "threshold": CriterionField("threshold", 5.0, True),
                "unit": CriterionField("unit", "pct", True),
            }), "agent")
        assert a.sha256 != b.sha256