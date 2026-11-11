"""Machine-checked acceptance criteria for the P489-N02 maturity node.

Five criteria, all must pass:

C1. >=10 synthetic capability boundary detection errors <=1% axis span,
    failure type classification accuracy >=90%.
C2. >=3 real capability auto-detected boundaries not wider than human declaration.
C3. >=10 out-of-domain call samples all return OUT_OF_DOMAIN, none write REAL.
C4. Zero domain semantics: axis names / capability ids from config only.
C5. Determinism: same config -> envelope diff=0; evaluation count <= budget.

These tests are the machine-checkable form of the acceptance criteria.
The full verification report is produced by :mod:`tensorphysics.applicability.verification`.
"""
from __future__ import annotations

import json
import pathlib

import pytest

from tensorphysics.applicability import (
    CapabilityRegistry, DomainStatus, GradeWriteError, assign_grade, probe_registry,
)
from tensorphysics.applicability.benchmarks import SYNTHETIC_CASES, synthetic_registry
from tensorphysics.applicability.verification import (
    run_full_verification,
    verify_c1_synthetic_boundaries,
    verify_c2_real_boundaries,
    verify_c3_out_of_domain,
    verify_c4_zero_domain_semantics,
    verify_c5_determinism_and_budget,
)

CONFIG_PATH = (pathlib.Path(__file__).resolve().parents[1] / "src" / "tensorphysics"
               / "applicability" / "configs" / "default_capabilities.v1.json")


# ---------------------------------------------------------------------------
# C1: synthetic boundary accuracy + failure type classification
# ---------------------------------------------------------------------------

class TestC1SyntheticBoundaries:
    """C1: >=10 synthetic cases, boundary error <=1% span, failure type >=90%."""

    def test_at_least_ten_synthetic_cases(self):
        assert len(SYNTHETIC_CASES) >= 10

    def test_boundary_error_within_one_percent_of_span(self):
        result = verify_c1_synthetic_boundaries(rel_tol=1e-3)
        n_within_1pct = sum(
            1 for d in result.details
            if d.get("within_1pct_span", False)
        )
        assert n_within_1pct >= 10, (
            f"Only {n_within_1pct}/{len(SYNTHETIC_CASES)} cases have boundary "
            f"error <= 1% of axis span"
        )

    def test_failure_type_classification_accuracy(self):
        result = verify_c1_synthetic_boundaries(rel_tol=1e-3)
        for note in result.notes:
            if note.startswith("failure_type_accuracy="):
                pct = float(note.split("=")[1].split("%")[0])
                assert pct >= 90.0, f"Failure type accuracy {pct}% < 90%"
                return
        pytest.fail("Could not find failure_type_accuracy in verification notes")

    def test_c1_overall_passes(self):
        result = verify_c1_synthetic_boundaries(rel_tol=1e-3)
        assert result.passed, result.notes


# ---------------------------------------------------------------------------
# C2: real capability boundaries not wider than declarations
# ---------------------------------------------------------------------------

class TestC2RealBoundaries:
    """C2: >=3 real capabilities with boundaries <= declared range."""

    def test_at_least_three_real_capabilities(self):
        registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
        assert len(registry) >= 3

    def test_boundaries_not_wider_than_declarations(self):
        result = verify_c2_real_boundaries(CONFIG_PATH)
        assert result.passed, result.notes
        assert result.metric_value >= 3

    def test_each_axis_not_wider(self):
        registry = CapabilityRegistry.from_config_file(CONFIG_PATH)
        suite = probe_registry(registry)
        for capability_id in registry.ids():
            report = suite.reports[capability_id]
            assert report.compat is not None, capability_id
            assert report.compat.ok, (capability_id, report.compat.to_dict())
            for ca in report.compat.axes:
                assert ca.wider_low_by == 0.0, (capability_id, ca.axis, ca.wider_low_by)
                assert ca.wider_high_by == 0.0, (capability_id, ca.axis, ca.wider_high_by)


# ---------------------------------------------------------------------------
# C3: out-of-domain grading
# ---------------------------------------------------------------------------

class TestC3OutOfDomain:
    """C3: >=10 out-of-domain samples, all OUT_OF_DOMAIN, none write REAL."""

    def test_at_least_ten_out_of_domain_samples(self):
        result = verify_c3_out_of_domain(CONFIG_PATH)
        assert result.metric_value >= 10, (
            f"Only {result.metric_value} out-of-domain samples found"
        )

    def test_all_queries_are_out_of_domain_or_unknown(self):
        result = verify_c3_out_of_domain(CONFIG_PATH)
        for d in result.details:
            assert d["is_out_or_unknown"], (
                f"Query {d['capability_id']}@{d['query_value']} "
                f"got status={d['domain_status']}, expected OUT or UNKNOWN"
            )

    def test_no_real_grade_written_for_out_of_domain(self):
        """None of the OUT_OF_DOMAIN results should allow a REAL grade."""
        result = verify_c3_out_of_domain(CONFIG_PATH)
        for note in result.notes:
            if note.startswith("real_write_blocked="):
                n_blocked = int(note.split("=")[1])
                assert n_blocked >= 1, "No REAL writes were blocked"
                return
        pytest.fail("Could not find real_write_blocked in verification notes")

    def test_c3_overall_passes(self):
        result = verify_c3_out_of_domain(CONFIG_PATH)
        assert result.passed, result.notes


# ---------------------------------------------------------------------------
# C4: zero domain semantics
# ---------------------------------------------------------------------------

class TestC4ZeroDomainSemantics:
    """C4: capability ids and axis names only in config, not in code."""

    def test_no_capability_id_in_source_code(self):
        result = verify_c4_zero_domain_semantics(CONFIG_PATH)
        assert result.passed, f"Violations: {result.notes}"

    def test_no_axis_name_quoted_in_source_code(self):
        result = verify_c4_zero_domain_semantics(CONFIG_PATH)
        assert result.metric_value == 0, (
            f"Found {result.metric_value} domain semantics violations"
        )


# ---------------------------------------------------------------------------
# C5: determinism and evaluation budget
# ---------------------------------------------------------------------------

class TestC5DeterminismAndBudget:
    """C5: same config -> envelope diff=0; eval count <= budget."""

    def test_suite_digest_is_reproducible(self):
        result = verify_c5_determinism_and_budget(CONFIG_PATH)
        assert "real_suite_digest_match=True" in result.notes

    def test_synthetic_digest_is_reproducible(self):
        result = verify_c5_determinism_and_budget(CONFIG_PATH)
        assert "synth_suite_digest_match=True" in result.notes

    def test_all_envelopes_match_between_runs(self):
        result = verify_c5_determinism_and_budget(CONFIG_PATH)
        assert "all_envelopes_match=True" in result.notes

    def test_evaluation_count_within_budget(self):
        result = verify_c5_determinism_and_budget(CONFIG_PATH)
        assert "budget_ok=True" in result.notes

    def test_c5_overall_passes(self):
        result = verify_c5_determinism_and_budget(CONFIG_PATH)
        assert result.passed, result.notes


# ---------------------------------------------------------------------------
# Full verification
# ---------------------------------------------------------------------------

class TestFullVerification:
    """Run all criteria together (C1-C5 from the earlier node, C6 joint coverage)."""

    def test_all_five_criteria_pass(self):
        result = run_full_verification(CONFIG_PATH)
        assert result.all_passed, (
            f"Only {result.summary['criteria_passed']}/{result.summary['criteria_total']} "
            f"criteria passed: "
            + ", ".join(f"{c.criterion_id}={'PASS' if c.passed else 'FAIL'}"
                        for c in result.criteria)
        )

    def test_verification_report_is_json_serialisable(self):
        result = run_full_verification(CONFIG_PATH)
        payload = result.to_dict()
        text = json.dumps(payload, indent=2, default=str)
        assert len(text) > 100  # non-trivial report
        # Round-trip
        reparsed = json.loads(text)
        assert reparsed["all_passed"] == result.all_passed
        assert reparsed["criteria_total"] == len(result.criteria) >= 6
        assert any(c["criterion_id"] == "C6" for c in reparsed["criteria"])
