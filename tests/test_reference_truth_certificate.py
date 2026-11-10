"""Tests for the Reference-Truth Certificate system.

Tests cover:
    - Schema construction and serialization
    - Type-specific validation rules
    - Gate enforcement (allow/block decisions)
    - Store persistence (save/load)
    - Backfill from existing benchmarks
"""
import json
import math
import tempfile
from pathlib import Path

import pytest

from tensorphysics.reference_truth import (
    CertificateType,
    CertificateStatus,
    ReferenceTruthCertificate,
    build_certificate,
    certificate_digest,
    validate_certificate,
    ReferenceGate,
    GateDecision,
    CertificateStore,
)
from tensorphysics.reference_truth.backfill import (
    backfill_ice_resistance_corpus,
    create_gold_standard_certificates,
)


# ============================================================================
# Schema tests
# ============================================================================

class TestSchema:
    """Test certificate schema construction and serialization."""

    def test_build_analytical_certificate(self):
        """Test building an analytical certificate."""
        cert = build_certificate(
            reference_id="TEST-001",
            reference_name="Test analytical reference",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.234,
            unit="m",
            quantity_description="Test quantity",
            analytical_evidence={
                "exact_formula": "x = a + b",
                "symbolic_equivalence": True,
            },
        )
        assert cert.reference_id == "TEST-001"
        assert cert.certificate_type == CertificateType.ANALYTICAL
        assert cert.reference_value == 1.234
        assert cert.version_sha256 != ""
        assert len(cert.version_sha256) == 64

    def test_build_self_convergence_certificate(self):
        """Test building a self-convergence certificate."""
        cert = build_certificate(
            reference_id="TEST-002",
            reference_name="Test convergence reference",
            certificate_type=CertificateType.SELF_CONVERGENCE,
            reference_value=3.14159,
            refinement_sequence=[
                {"level_index": 0, "resolution_parameter": 0.1, "quantity_of_interest": 3.10},
                {"level_index": 1, "resolution_parameter": 0.05, "quantity_of_interest": 3.13},
                {"level_index": 2, "resolution_parameter": 0.025, "quantity_of_interest": 3.14},
            ],
            richardson={
                "observed_order_p": 2.0,
                "extrapolated_value": 3.14159,
                "gci_fine": 0.001,
                "gci_coarse": 0.01,
            },
        )
        assert cert.certificate_type == CertificateType.SELF_CONVERGENCE
        assert len(cert.refinement_sequence) == 3
        assert cert.richardson["observed_order_p"] == 2.0

    def test_certificate_digest_changes_with_data(self):
        """Test that digest changes when evidence changes."""
        cert1 = build_certificate(
            reference_id="TEST-003",
            reference_name="Test",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1"},
        )
        cert2 = build_certificate(
            reference_id="TEST-003",
            reference_name="Test",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=2.0,  # changed
            analytical_evidence={"exact_formula": "x = 1"},
        )
        assert cert1.version_sha256 != cert2.version_sha256

    def test_certificate_serialization_roundtrip(self):
        """Test that certificates can be serialized and deserialized."""
        cert = build_certificate(
            reference_id="TEST-004",
            reference_name="Test roundtrip",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=42.0,
            analytical_evidence={"exact_formula": "x = 42"},
        )
        d = cert.to_dict()
        cert2 = ReferenceTruthCertificate.from_dict(d)
        assert cert2.reference_id == cert.reference_id
        assert cert2.reference_value == cert.reference_value
        assert cert2.certificate_type == cert.certificate_type


# ============================================================================
# Validator tests
# ============================================================================

class TestValidator:
    """Test type-specific validation rules."""

    def test_validate_analytical_pass(self):
        """Test analytical certificate validation passes with valid evidence."""
        cert = build_certificate(
            reference_id="VAL-001",
            reference_name="Valid analytical",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={
                "exact_formula": "R = a + b",
                "symbolic_equivalence": True,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is True
        assert result.status == CertificateStatus.VALID

    def test_validate_analytical_fail_no_evidence(self):
        """Test analytical certificate fails without evidence."""
        cert = build_certificate(
            reference_id="VAL-002",
            reference_name="No evidence",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
        )
        result = validate_certificate(cert)
        assert result.passed is False
        assert result.status == CertificateStatus.INSUFFICIENT

    def test_validate_analytical_fail_symbolic_false(self):
        """Test analytical certificate fails if symbolic check is False."""
        cert = build_certificate(
            reference_id="VAL-003",
            reference_name="Symbolic failed",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"symbolic_equivalence": False},
        )
        result = validate_certificate(cert)
        assert result.passed is False

    def test_validate_self_convergence_pass(self):
        """Test self-convergence validation passes with 3+ levels and good p."""
        cert = build_certificate(
            reference_id="VAL-004",
            reference_name="Valid convergence",
            certificate_type=CertificateType.SELF_CONVERGENCE,
            reference_value=3.14,
            refinement_sequence=[
                {"level_index": 0, "resolution_parameter": 0.1, "quantity_of_interest": 3.0},
                {"level_index": 1, "resolution_parameter": 0.05, "quantity_of_interest": 3.1},
                {"level_index": 2, "resolution_parameter": 0.025, "quantity_of_interest": 3.14},
            ],
            richardson={
                "observed_order_p": 2.0,
                "extrapolated_value": 3.14159,
                "gci_fine": 0.001,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is True
        assert result.richardson_details is not None
        assert result.richardson_details["observed_order_p"] == 2.0

    def test_validate_self_convergence_fail_too_few_levels(self):
        """Test self-convergence fails with < 3 levels."""
        cert = build_certificate(
            reference_id="VAL-005",
            reference_name="Too few levels",
            certificate_type=CertificateType.SELF_CONVERGENCE,
            reference_value=3.14,
            refinement_sequence=[
                {"level_index": 0, "resolution_parameter": 0.1, "quantity_of_interest": 3.0},
                {"level_index": 1, "resolution_parameter": 0.05, "quantity_of_interest": 3.1},
            ],
            richardson={"observed_order_p": 2.0, "gci_fine": 0.001},
        )
        result = validate_certificate(cert)
        assert result.passed is False
        assert "≥3 levels" in result.notes[0]

    def test_validate_self_convergence_fail_low_order(self):
        """Test self-convergence fails if observed order p is too low."""
        cert = build_certificate(
            reference_id="VAL-006",
            reference_name="Low order",
            certificate_type=CertificateType.SELF_CONVERGENCE,
            reference_value=3.14,
            refinement_sequence=[
                {"level_index": 0, "resolution_parameter": 0.1, "quantity_of_interest": 3.0},
                {"level_index": 1, "resolution_parameter": 0.05, "quantity_of_interest": 3.1},
                {"level_index": 2, "resolution_parameter": 0.025, "quantity_of_interest": 3.14},
            ],
            richardson={
                "observed_order_p": 0.3,  # below minimum 0.5
                "gci_fine": 0.001,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is False
        assert "below minimum" in result.notes[0]

    def test_validate_experimental_pass(self):
        """Test experimental certificate validation passes."""
        cert = build_certificate(
            reference_id="VAL-007",
            reference_name="Valid experimental",
            certificate_type=CertificateType.EXPERIMENTAL,
            reference_value=100.0,
            experimental={
                "source_reference": "Smith et al. (2020)",
                "measurement_method": "Towing tank",
                "stated_uncertainty": 0.02,
                "confidence_level": 0.95,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is True

    def test_validate_experimental_fail_no_uncertainty(self):
        """Test experimental certificate fails without uncertainty."""
        cert = build_certificate(
            reference_id="VAL-008",
            reference_name="No uncertainty",
            certificate_type=CertificateType.EXPERIMENTAL,
            reference_value=100.0,
            experimental={
                "source_reference": "Smith et al. (2020)",
                "measurement_method": "Towing tank",
            },
        )
        result = validate_certificate(cert)
        assert result.passed is False

    def test_validate_statistical_pass(self):
        """Test statistical certificate validation passes."""
        cert = build_certificate(
            reference_id="VAL-009",
            reference_name="Valid statistical",
            certificate_type=CertificateType.STATISTICAL,
            reference_value=50.0,
            statistical={
                "n_samples": 1000,
                "mean": 50.0,
                "standard_error": 0.5,
                "confidence_level": 0.95,
                "ci_lower": 49.0,
                "ci_upper": 51.0,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is True
        assert result.statistical_details is not None

    def test_validate_statistical_fail_too_few_samples(self):
        """Test statistical certificate fails with too few samples."""
        cert = build_certificate(
            reference_id="VAL-010",
            reference_name="Too few samples",
            certificate_type=CertificateType.STATISTICAL,
            reference_value=50.0,
            statistical={
                "n_samples": 10,  # below minimum 30
                "mean": 50.0,
                "standard_error": 0.5,
                "confidence_level": 0.95,
                "ci_lower": 49.0,
                "ci_upper": 51.0,
            },
        )
        result = validate_certificate(cert)
        assert result.passed is False


# ============================================================================
# Gate tests
# ============================================================================

class TestGate:
    """Test reference gate enforcement."""

    def test_gate_allows_valid_certificate(self):
        """Test gate allows citation of valid certificate."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="GATE-001",
            reference_name="Valid cert",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1", "symbolic_equivalence": True},
        )
        validate_certificate(cert)
        store.add(cert)

        gate = ReferenceGate(store)
        result = gate.check("GATE-001")
        assert result.decision == GateDecision.ALLOWED

    def test_gate_blocks_missing_certificate(self):
        """Test gate blocks citation when no certificate exists."""
        store = CertificateStore()
        gate = ReferenceGate(store)
        result = gate.check("NONEXISTENT")
        assert result.decision == GateDecision.BLOCKED_NO_CERT

    def test_gate_blocks_revoked_certificate(self):
        """Test gate blocks citation of revoked certificate."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="GATE-002",
            reference_name="Revoked cert",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1", "symbolic_equivalence": True},
        )
        validate_certificate(cert)
        store.add(cert)
        store.revoke("GATE-002", "Evidence found to be incorrect")

        gate = ReferenceGate(store)
        result = gate.check("GATE-002")
        assert result.decision == GateDecision.BLOCKED_REVOKED

    def test_gate_blocks_expired_certificate(self):
        """Test gate blocks citation of expired certificate."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="GATE-003",
            reference_name="Expired cert",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1", "symbolic_equivalence": True},
        )
        validate_certificate(cert)
        store.add(cert)
        store.expire("GATE-003", "Superseded by new measurement")

        gate = ReferenceGate(store)
        result = gate.check("GATE-003")
        assert result.decision == GateDecision.BLOCKED_EXPIRED

    def test_gate_blocks_insufficient_certificate(self):
        """Test gate blocks citation of insufficient certificate."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="GATE-004",
            reference_name="Insufficient cert",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            # No evidence provided
        )
        validate_certificate(cert)  # This will set status to INSUFFICIENT
        store.add(cert)

        gate = ReferenceGate(store)
        result = gate.check("GATE-004")
        assert result.decision == GateDecision.BLOCKED_INVALID

    def test_gate_check_multiple(self):
        """Test checking multiple references at once."""
        store = CertificateStore()
        for i in range(3):
            cert = build_certificate(
                reference_id=f"GATE-MULTI-{i}",
                reference_name=f"Cert {i}",
                certificate_type=CertificateType.ANALYTICAL,
                reference_value=float(i),
                analytical_evidence={"exact_formula": f"x = {i}", "symbolic_equivalence": True},
            )
            validate_certificate(cert)
            store.add(cert)

        gate = ReferenceGate(store)
        results = gate.check_multiple(["GATE-MULTI-0", "GATE-MULTI-1", "GATE-MULTI-2"])
        assert len(results) == 3
        assert all(r.decision == GateDecision.ALLOWED for r in results)


# ============================================================================
# Store tests
# ============================================================================

class TestStore:
    """Test certificate store persistence."""

    def test_store_add_and_get(self):
        """Test adding and retrieving certificates."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="STORE-001",
            reference_name="Test",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1"},
        )
        store.add(cert)
        retrieved = store.get("STORE-001")
        assert retrieved is not None
        assert retrieved.reference_id == "STORE-001"

    def test_store_save_and_load(self):
        """Test saving and loading certificates to/from JSON."""
        store = CertificateStore()
        for i in range(5):
            cert = build_certificate(
                reference_id=f"STORE-SAVE-{i}",
                reference_name=f"Cert {i}",
                certificate_type=CertificateType.ANALYTICAL,
                reference_value=float(i),
                analytical_evidence={"exact_formula": f"x = {i}"},
            )
            store.add(cert)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            store.save_to_file(temp_path)

            store2 = CertificateStore()
            count = store2.load_from_file(temp_path)
            assert count == 5
            assert len(store2) == 5
            assert "STORE-SAVE-3" in store2
        finally:
            Path(temp_path).unlink()

    def test_store_statistics(self):
        """Test computing store statistics."""
        store = CertificateStore()
        for i in range(3):
            cert = build_certificate(
                reference_id=f"STAT-{i}",
                reference_name=f"Cert {i}",
                certificate_type=CertificateType.ANALYTICAL,
                reference_value=float(i),
                analytical_evidence={"exact_formula": f"x = {i}", "symbolic_equivalence": True},
            )
            validate_certificate(cert)
            store.add(cert)

        stats = store.statistics()
        assert stats["total"] == 3
        assert stats["by_type"]["analytical"] == 3
        assert stats["by_status"]["valid"] == 3
        assert stats["valid_rate"] == 1.0

    def test_store_revoke(self):
        """Test revoking a certificate."""
        store = CertificateStore()
        cert = build_certificate(
            reference_id="REVOKE-001",
            reference_name="To be revoked",
            certificate_type=CertificateType.ANALYTICAL,
            reference_value=1.0,
            analytical_evidence={"exact_formula": "x = 1", "symbolic_equivalence": True},
        )
        validate_certificate(cert)
        store.add(cert)

        result = store.revoke("REVOKE-001", "Test revocation")
        assert result is True
        revoked = store.get("REVOKE-001")
        assert revoked.status == CertificateStatus.REVOKED


# ============================================================================
# Backfill tests
# ============================================================================

class TestBackfill:
    """Test backfilling existing benchmark references."""

    def test_backfill_ice_resistance_corpus(self):
        """Test backfilling ice resistance L2 anchors."""
        # This test requires the actual corpus file
        corpus_path = Path("benchmarks/ice_resistance/corpus/l2_anchors.json")
        if not corpus_path.exists():
            pytest.skip("Ice resistance corpus not available")

        certs = backfill_ice_resistance_corpus(corpus_path)
        assert len(certs) > 0
        # All should have reference_ids starting with "IR-"
        assert all(c.reference_id.startswith("IR-") for c in certs)

    def test_create_gold_standard_certificates(self):
        """Test creating the full gold-standard certificate set."""
        benchmarks_dir = Path("benchmarks")
        if not benchmarks_dir.exists():
            pytest.skip("Benchmarks directory not available")

        store = create_gold_standard_certificates(benchmarks_dir)
        assert len(store) > 0
        stats = store.statistics()
        assert stats["total"] > 0


# ============================================================================
# Integration tests
# ============================================================================

class TestIntegration:
    """Integration tests for the full certificate workflow."""

    def test_full_workflow(self):
        """Test the complete workflow: build → validate → store → gate."""
        # Build certificate
        cert = build_certificate(
            reference_id="INTEG-001",
            reference_name="Integration test reference",
            certificate_type=CertificateType.SELF_CONVERGENCE,
            reference_value=3.14159,
            unit="dimensionless",
            quantity_description="Pi approximation via mesh refinement",
            refinement_sequence=[
                {"level_index": 0, "resolution_parameter": 0.1, "quantity_of_interest": 3.0},
                {"level_index": 1, "resolution_parameter": 0.05, "quantity_of_interest": 3.1},
                {"level_index": 2, "resolution_parameter": 0.025, "quantity_of_interest": 3.14},
            ],
            richardson={
                "observed_order_p": 2.0,
                "extrapolated_value": 3.14159,
                "gci_fine": 0.001,
                "gci_coarse": 0.01,
            },
            caliber_declaration="Test convergence study",
            disclaimer_boundaries=["Not valid outside test scope"],
        )

        # Validate
        result = validate_certificate(cert)
        assert result.passed is True

        # Store
        store = CertificateStore()
        store.add(cert)
        assert "INTEG-001" in store

        # Gate check
        gate = ReferenceGate(store)
        gate_result = gate.check("INTEG-001")
        assert gate_result.decision == GateDecision.ALLOWED

        # Save and reload
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            store.save_to_file(temp_path)
            store2 = CertificateStore()
            store2.load_from_file(temp_path)

            # Gate should still work after reload
            gate2 = ReferenceGate(store2)
            gate_result2 = gate2.check("INTEG-001")
            assert gate_result2.decision == GateDecision.ALLOWED
        finally:
            Path(temp_path).unlink()
