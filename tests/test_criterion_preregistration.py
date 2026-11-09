"""Tests for criterion pre-registration and drift audit mechanism.

Covers the machine-checkable acceptance criteria:

* C1: 24 synthetic test cases (12 modified, 12 unchanged):
      recall ≥ 0.92, precision ≥ 0.95 for drift detection
* C2: Field-level F1 ≥ 0.95 on 12 synthetic criteria (six-field extraction)
* C3: Fail-closed negative controls:
      - empty goal → unregistered
      - no criterion block → unregistered
      - tampered sha256 → tampered
* C4: Real replay: 100 synthetic goals → report registered_rate & unregistered_rate
* C5: Zero domain semantics scan: no domain vocabulary in new code/tests

All fixtures are synthetic (zero domain semantics).
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.campaign.criterion_registry import (
    CriterionManifest,
    CriterionField,
    extract_criteria_from_goal,
    freeze_manifest,
    verify_frozen_hash,
    DriftClassifier,
    DriftResult,
    audit_drift,
    classify_drift,
    CriterionEvent,
    CriterionLedger,
    record_drift_event,
    query_kpi,
    KPIView,
    compute_kpi,
)
from tensorphysics.campaign.criterion_registry.manifest import CRITERION_FIELDS


# ---------------------------------------------------------------------------
# Synthetic fixtures (zero domain semantics)
# ---------------------------------------------------------------------------

def _make_goal(criteria_texts: list[str]) -> str:
    """Build a synthetic goal text from criterion blocks."""
    lines = []
    for i, text in enumerate(criteria_texts, 1):
        lines.append(f"C{i}: {text}")
    return "\n".join(lines)


def _make_criterion(
    cid: str,
    threshold: float | None = None,
    unit: str | None = None,
    denominator: str | None = None,
    exclusion: str | None = None,
    reference_path: str | None = None,
    comparator: str | None = None,
) -> CriterionManifest:
    """Build a synthetic CriterionManifest directly."""
    fields = {}
    if threshold is not None:
        fields["threshold"] = CriterionField("threshold", threshold, True)
    if unit is not None:
        fields["unit"] = CriterionField("unit", unit, True)
    if denominator is not None:
        fields["denominator"] = CriterionField("denominator", denominator, True)
    if exclusion is not None:
        fields["exclusion"] = CriterionField("exclusion", exclusion, True)
    if reference_path is not None:
        fields["reference_path"] = CriterionField("reference_path", reference_path, True)
    if comparator is not None:
        fields["comparator"] = CriterionField("comparator", comparator, True)
    return CriterionManifest(criterion_id=cid, fields=fields, status="unregistered")


# ---------------------------------------------------------------------------
# C1: Drift detection on 24 synthetic cases (12 modified, 12 unchanged)
# ---------------------------------------------------------------------------

class TestDriftDetection24:
    """24 synthetic test cases for drift detection accuracy."""

    @pytest.fixture
    def cases(self):
        """Generate 24 test cases: 12 modified, 12 unchanged."""
        cases = []

        # --- 12 UNCHANGED cases ---
        for i in range(12):
            frozen = _make_criterion(
                f"c{i:02d}",
                threshold=5.0 + i,
                unit="pct",
                comparator="le",
                denominator="ref",
                reference_path=f"/data/ref_{i}",
            )
            frozen = freeze_manifest(frozen, "test_agent")
            # Current is identical
            current = _make_criterion(
                f"c{i:02d}",
                threshold=5.0 + i,
                unit="pct",
                comparator="le",
                denominator="ref",
                reference_path=f"/data/ref_{i}",
            )
            cases.append((frozen, current, "unchanged"))

        # --- 12 MODIFIED cases ---
        # 3 cases: threshold loosened (3% → 10%)
        for i in range(3):
            frozen = _make_criterion(
                f"m{i:02d}",
                threshold=3.0,
                unit="pct",
                comparator="le",
            )
            frozen = freeze_manifest(frozen, "test_agent")
            current = _make_criterion(
                f"m{i:02d}",
                threshold=10.0,
                unit="pct",
                comparator="le",
            )
            cases.append((frozen, current, "loosened"))

        # 3 cases: denominator changed (scope_changed)
        for i in range(3, 6):
            frozen = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="pct",
                comparator="le",
                denominator="alpha",
            )
            frozen = freeze_manifest(frozen, "test_agent")
            current = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="pct",
                comparator="le",
                denominator="beta",
            )
            cases.append((frozen, current, "scope_changed"))

        # 3 cases: exclusion added (scope_changed)
        for i in range(6, 9):
            frozen = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="pct",
                comparator="le",
            )
            frozen = freeze_manifest(frozen, "test_agent")
            current = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="pct",
                comparator="le",
                exclusion="edge_case_1",
            )
            cases.append((frozen, current, "scope_changed"))

        # 3 cases: unit changed (scope_changed)
        for i in range(9, 12):
            frozen = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="pct",
                comparator="le",
            )
            frozen = freeze_manifest(frozen, "test_agent")
            current = _make_criterion(
                f"m{i:02d}",
                threshold=5.0,
                unit="ratio",
                comparator="le",
            )
            cases.append((frozen, current, "scope_changed"))

        return cases

    def test_recall_ge_092(self, cases):
        """Recall ≥ 0.92: of all truly modified cases, how many are detected."""
        truly_modified = [(f, c, expected) for f, c, expected in cases if expected != "unchanged"]
        detected_modified = 0
        for frozen, current, expected in truly_modified:
            result = classify_drift(frozen, current)
            if result.classification != "unchanged":
                detected_modified += 1
        recall = detected_modified / len(truly_modified) if truly_modified else 1.0
        assert recall >= 0.92, f"recall={recall:.3f} < 0.92"

    def test_precision_ge_095(self, cases):
        """Precision ≥ 0.95: of all detected-as-modified, how many truly are."""
        flagged_modified = 0
        true_positives = 0
        for frozen, current, expected in cases:
            result = classify_drift(frozen, current)
            if result.classification != "unchanged":
                flagged_modified += 1
                if expected != "unchanged":
                    true_positives += 1
        precision = true_positives / flagged_modified if flagged_modified else 1.0
        assert precision >= 0.95, f"precision={precision:.3f} < 0.95"

    def test_unchanged_cases_correctly_classified(self, cases):
        """All 12 unchanged cases should be classified as 'unchanged'."""
        unchanged_cases = [(f, c, e) for f, c, e in cases if e == "unchanged"]
        for frozen, current, expected in unchanged_cases:
            result = classify_drift(frozen, current)
            assert result.classification == "unchanged", (
                f"criterion {frozen.criterion_id}: expected unchanged, got {result.classification}"
            )


# ---------------------------------------------------------------------------
# C2: Field-level F1 ≥ 0.95 on 12 synthetic criteria
# ---------------------------------------------------------------------------

class TestFieldExtractionF1:
    """Test that field extraction achieves F1 ≥ 0.95 per field."""

    @pytest.fixture
    def extraction_cases(self):
        """12 synthetic criteria with known field values."""
        cases = []
        for i in range(12):
            goal_text = (
                f"C{i}: error <= {3.0 + i} pct of ref_value "
                f"(ref: /data/baseline_{i}, excluding case_{i})"
            )
            expected_fields = {
                "threshold": 3.0 + i,
                "unit": "pct",
                "comparator": "le",
                "denominator": "ref_value",
                "reference_path": f"/data/baseline_{i}",
                "exclusion": f"case_{i}",
            }
            cases.append((goal_text, expected_fields))
        return cases

    def test_field_f1_ge_095(self, extraction_cases):
        """Field-level F1 ≥ 0.95 across all 12 criteria and 6 fields."""
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for goal_text, expected in extraction_cases:
            criteria = extract_criteria_from_goal(goal_text)
            assert len(criteria) >= 1, f"No criteria extracted from: {goal_text}"

            extracted = criteria[0]
            for fname in CRITERION_FIELDS:
                expected_val = expected.get(fname)
                extracted_field = extracted.fields.get(fname)
                extracted_val = extracted_field.value if extracted_field and extracted_field.present else None

                if expected_val is not None and extracted_val is not None:
                    if expected_val == extracted_val:
                        total_tp += 1
                    else:
                        total_fp += 1
                        total_fn += 1
                elif expected_val is not None and extracted_val is None:
                    total_fn += 1
                elif expected_val is None and extracted_val is not None:
                    total_fp += 1
                # Both None: true negative, not counted in F1

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 1.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 1.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        assert f1 >= 0.95, f"field F1={f1:.3f} < 0.95 (precision={precision:.3f}, recall={recall:.3f})"


# ---------------------------------------------------------------------------
# C3: Fail-closed negative controls
# ---------------------------------------------------------------------------

class TestFailClosed:
    """Fail-closed: unparseable → unregistered, never pass."""

    def test_empty_goal_returns_empty(self):
        """Empty goal → no criteria extracted."""
        criteria = extract_criteria_from_goal("")
        assert criteria == []

    def test_none_goal_returns_empty(self):
        """None goal → no criteria extracted."""
        criteria = extract_criteria_from_goal(None)
        assert criteria == []

    def test_no_criterion_block_returns_empty(self):
        """Goal with no criterion-like content → no criteria."""
        criteria = extract_criteria_from_goal("This is just a description with no numbers or criteria.")
        assert criteria == []

    def test_unfrozen_manifest_is_unregistered(self):
        """Unfrozen manifest has status='unregistered'."""
        manifest = _make_criterion("test", threshold=5.0, comparator="le")
        assert manifest.status == "unregistered"

    def test_tampered_sha256_detected(self):
        """Tampered sha256 → verify_frozen_hash returns False."""
        manifest = _make_criterion("test", threshold=5.0, comparator="le")
        frozen = freeze_manifest(manifest, "test_agent")
        assert verify_frozen_hash(frozen) is True

        # Tamper: change threshold after freezing
        tampered = CriterionManifest(
            criterion_id=frozen.criterion_id,
            fields={
                "threshold": CriterionField("threshold", 99.0, True),  # changed!
                "comparator": CriterionField("comparator", "le", True),
            },
            status="registered",
            frozen_at=frozen.frozen_at,
            frozen_by=frozen.frozen_by,
            sha256=frozen.sha256,  # old hash
        )
        assert verify_frozen_hash(tampered) is False

    def test_drift_on_tampered_returns_tampered(self):
        """Drift classification on tampered frozen → 'tampered'."""
        manifest = _make_criterion("test", threshold=5.0, comparator="le")
        frozen = freeze_manifest(manifest, "test_agent")

        # Tamper the frozen manifest
        tampered = CriterionManifest(
            criterion_id=frozen.criterion_id,
            fields={
                "threshold": CriterionField("threshold", 99.0, True),
                "comparator": CriterionField("comparator", "le", True),
            },
            status="registered",
            frozen_at=frozen.frozen_at,
            frozen_by=frozen.frozen_by,
            sha256=frozen.sha256,
        )

        current = _make_criterion("test", threshold=5.0, comparator="le")
        result = classify_drift(tampered, current)
        assert result.classification == "tampered"

    def test_drift_on_unregistered_frozen(self):
        """Drift on unfrozen criterion → 'unregistered'."""
        frozen = _make_criterion("test", threshold=5.0, comparator="le")
        # Not frozen, status='unregistered'
        current = _make_criterion("test", threshold=5.0, comparator="le")
        result = classify_drift(frozen, current)
        assert result.classification == "unregistered"


# ---------------------------------------------------------------------------
# C4: Real replay - 100 synthetic goals
# ---------------------------------------------------------------------------

class TestRealReplay:
    """Replay 100 synthetic goals and report KPIs."""

    def test_replay_100_goals(self):
        """Generate 100 synthetic goals, freeze, audit, report KPIs."""
        ledger = CriterionLedger()
        registered_count = 0
        unregistered_count = 0

        for i in range(100):
            # Generate synthetic goal
            if i % 5 == 0:
                # 20% have no parseable criteria (truly no numbers)
                goal_text = f"Mission: optimize the design qualitatively without numeric targets for iteration {chr(65 + i % 26)}."
            else:
                goal_text = (
                    f"C1: achieve error <= {2.0 + (i % 10)} pct of reference "
                    f"(ref: /data/ref_{i}, excluding edge_{i % 3})"
                )

            criteria = extract_criteria_from_goal(goal_text)

            if not criteria:
                # No criteria extracted → record as unregistered
                record_drift_event(
                    ledger,
                    mission_id=f"mission_{i:03d}",
                    criterion_id=f"c_miss_{i:03d}",
                    classification="unregistered",
                    reason="no criteria extracted from goal",
                )
                unregistered_count += 1
            else:
                for c in criteria:
                    frozen = freeze_manifest(c, "replay_agent")
                    # Simulate audit: current is same as frozen (no drift)
                    record_drift_event(
                        ledger,
                        mission_id=f"mission_{i:03d}",
                        criterion_id=frozen.criterion_id,
                        classification="unchanged",
                    )
                    registered_count += 1

        total = registered_count + unregistered_count
        assert total == 100, f"expected 100 events, got {total}"

        kpi = compute_kpi(ledger)
        # Report the rates (no pass/fail threshold, just report facts)
        result = {
            "total_goals": 100,
            "registered_rate": kpi.registered_rate,
            "unregistered_rate": kpi.unregistered_rate,
            "registered_count": registered_count,
            "unregistered_count": unregistered_count,
        }
        # Verify the rates are sensible
        assert 0.0 <= kpi.registered_rate <= 1.0
        assert 0.0 <= kpi.unregistered_rate <= 1.0
        assert abs(kpi.registered_rate + kpi.unregistered_rate - 1.0) < 0.01

        # Verify expected split (80% registered, 20% unregistered)
        assert registered_count == 80, f"expected 80 registered, got {registered_count}"
        assert unregistered_count == 20, f"expected 20 unregistered, got {unregistered_count}"

        # Print for report
        print(f"\n=== Real Replay KPI (100 goals) ===")
        print(f"registered_rate: {kpi.registered_rate:.3f}")
        print(f"unregistered_rate: {kpi.unregistered_rate:.3f}")


# ---------------------------------------------------------------------------
# C5: Zero domain semantics scan
# ---------------------------------------------------------------------------

# Domain vocabulary that should NOT appear in the new code
DOMAIN_SCAN_TERMS = frozenset({
    # Ship/hydrodynamics
    "resistance", "hull", "propeller", "ship", "wave", "drag",
    "froude", "reynolds", "wetted", "displacement", "waterline",
    # Ice
    "ice", "iceberg", "floe", "ridge", "brash",
    # Structure
    "beam", "plate", "stiffener", "bulkhead", "frame",
    # Specific physics
    "holtrop", "lindqvist", "iacs", "ittc",
})


class TestZeroDomainSemantics:
    """Scan new code and tests for domain vocabulary."""

    def _scan_file(self, path: Path) -> list[str]:
        """Scan a file for domain terms. Returns list of hits."""
        hits = []
        content = path.read_text().lower()
        for term in DOMAIN_SCAN_TERMS:
            # Use word boundary matching
            pattern = r'\b' + re.escape(term) + r'\b'
            if re.search(pattern, content):
                hits.append(term)
        return hits

    def test_no_domain_terms_in_manifest(self):
        """manifest.py contains no domain vocabulary."""
        manifest_path = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry" / "manifest.py"
        hits = self._scan_file(manifest_path)
        assert hits == [], f"Domain terms found in manifest.py: {hits}"

    def test_no_domain_terms_in_drift(self):
        """drift.py contains no domain vocabulary."""
        drift_path = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry" / "drift.py"
        hits = self._scan_file(drift_path)
        assert hits == [], f"Domain terms found in drift.py: {hits}"

    def test_no_domain_terms_in_ledger(self):
        """ledger.py contains no domain vocabulary."""
        ledger_path = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry" / "ledger.py"
        hits = self._scan_file(ledger_path)
        assert hits == [], f"Domain terms found in ledger.py: {hits}"

    def test_no_domain_terms_in_kpi(self):
        """kpi.py contains no domain vocabulary."""
        kpi_path = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry" / "kpi.py"
        hits = self._scan_file(kpi_path)
        assert hits == [], f"Domain terms found in kpi.py: {hits}"

    def test_no_domain_terms_in_init(self):
        """__init__.py contains no domain vocabulary."""
        init_path = ROOT / "src" / "tensorphysics" / "campaign" / "criterion_registry" / "__init__.py"
        hits = self._scan_file(init_path)
        assert hits == [], f"Domain terms found in __init__.py: {hits}"

    def test_no_domain_terms_in_test_fixtures(self):
        """Test fixtures use synthetic names only (no domain vocabulary)."""
        test_path = ROOT / "tests" / "test_criterion_preregistration.py"
        if not test_path.exists():
            pytest.skip("test file not yet created")
        content = test_path.read_text()

        # Extract just the fixture/helper functions (not the DOMAIN_SCAN_TERMS definition)
        # Look for the _make_criterion function and test data
        fixture_section = content[content.find("def _make_criterion"):]
        # Stop before the domain scan terms definition
        domain_def_pos = fixture_section.find("DOMAIN_SCAN_TERMS")
        if domain_def_pos > 0:
            fixture_section = fixture_section[:domain_def_pos]

        for term in DOMAIN_SCAN_TERMS:
            pattern = r'\b' + re.escape(term) + r'\b'
            matches = re.findall(pattern, fixture_section, re.IGNORECASE)
            assert not matches, f"Domain term '{term}' found in test fixtures"


# ---------------------------------------------------------------------------
# Additional integration tests
# ---------------------------------------------------------------------------

class TestLedgerIntegration:
    """Integration tests for the full ledger workflow."""

    def test_full_workflow(self, tmp_path):
        """Test complete workflow: extract → freeze → audit → ledger → KPI."""
        ledger_path = tmp_path / "ledger.jsonl"
        ledger = CriterionLedger(storage_path=ledger_path)

        # 1. Extract criteria from goal
        goal = "C1: error <= 5 pct of reference (ref: /data/baseline, excluding edge_case)"
        criteria = extract_criteria_from_goal(goal)
        assert len(criteria) >= 1

        # 2. Freeze
        frozen = freeze_manifest(criteria[0], "test_agent")
        assert frozen.status == "registered"
        assert frozen.sha256 is not None
        assert verify_frozen_hash(frozen)

        # 3. Simulate audit (no drift)
        result = classify_drift(frozen, criteria[0])
        assert result.classification == "unchanged"

        # 4. Record event
        record_drift_event(
            ledger,
            mission_id="test_mission",
            criterion_id=frozen.criterion_id,
            classification=result.classification,
            reason=result.reason,
        )

        # 5. Compute KPI
        kpi = compute_kpi(ledger)
        assert kpi.total_criteria == 1
        assert kpi.registered == 1
        assert kpi.unchanged == 1

        # 6. Verify persistence
        ledger2 = CriterionLedger(storage_path=ledger_path)
        assert len(ledger2) == 1

    def test_loosened_requires_reason(self):
        """Loosened events must have a reason (fail-closed)."""
        ledger = CriterionLedger()
        with pytest.raises(ValueError, match="requires a non-empty reason"):
            record_drift_event(
                ledger,
                mission_id="test",
                criterion_id="c1",
                classification="loosened",
                reason=None,  # Missing reason!
            )

    def test_scope_changed_requires_reason(self):
        """Scope_changed events must have a reason (fail-closed)."""
        ledger = CriterionLedger()
        with pytest.raises(ValueError, match="requires a non-empty reason"):
            record_drift_event(
                ledger,
                mission_id="test",
                criterion_id="c1",
                classification="scope_changed",
                reason=None,
            )

    def test_kpi_view_properties(self):
        """Test KPI view rate calculations."""
        kpi = KPIView(
            total_criteria=10,
            registered=8,
            unregistered=2,
            unchanged=5,
            tightened=1,
            loosened=1,
            scope_changed=1,
        )
        assert kpi.registered_rate == 0.8
        assert kpi.unregistered_rate == 0.2
        assert kpi.loosened_rate == 0.125  # 1/8
        assert kpi.tightened_rate == 0.125
        assert kpi.scope_changed_rate == 0.125

    def test_canonical_json_deterministic(self):
        """Canonical JSON must be deterministic for hashing."""
        m1 = _make_criterion("c1", threshold=5.0, unit="pct", comparator="le")
        m2 = _make_criterion("c1", threshold=5.0, unit="pct", comparator="le")
        assert m1.to_canonical_json() == m2.to_canonical_json()

    def test_freeze_produces_same_hash_for_same_content(self):
        """Same content → same sha256."""
        m1 = _make_criterion("c1", threshold=5.0, comparator="le")
        m2 = _make_criterion("c1", threshold=5.0, comparator="le")
        f1 = freeze_manifest(m1, "agent")
        f2 = freeze_manifest(m2, "agent")
        assert f1.sha256 == f2.sha256
