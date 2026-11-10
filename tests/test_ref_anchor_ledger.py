"""
Tests for the Reference Anchor Uncertainty Ledger.

Covers:
- C1: Anchor coverage ≥30 entries with U_ref + provenance
- C2: Three-state judgment accuracy (10 known combinations)
- C3: Fail-closed when U_ref missing
- C4: Uncertainty propagation vs Monte Carlo (≤5% error)
- C5: ≥15 pytest cases covering ledger/judgment/propagation/fail-closed
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

# Import the module under test
from tensorphysics.metrology.ref_anchor_ledger.schema import (
    RefAnchor, AnchorSource, AnchorProvenance
)
from tensorphysics.metrology.ref_anchor_ledger.ledger import (
    RefAnchorLedger, load_ledger, get_ledger
)
from tensorphysics.metrology.ref_anchor_ledger.judgment import (
    ThreeStateJudgment, JudgmentResult, judge_criterion,
    compute_combined_uncertainty, batch_judge, judgment_statistics
)
from tensorphysics.metrology.ref_anchor_ledger.propagation import (
    propagate_uncertainty_analytical, monte_carlo_propagation,
    compare_propagation_methods, PropagationResult
)


# ============================================================
# Test data path
# ============================================================
DATA_PATH = Path(__file__).parent.parent / "src" / "tensorphysics" / "metrology" / "ref_anchor_ledger" / "data" / "ref_anchors.json"


# ============================================================
# C1: Anchor coverage tests
# ============================================================

class TestAnchorCoverage:
    """C1: ≥30 anchors with U_ref + provenance, 100% coverage."""

    def test_anchor_count_ge_30(self):
        """At least 30 anchors in the ledger."""
        ledger = load_ledger()
        assert len(ledger) >= 30, f"Expected ≥30 anchors, got {len(ledger)}"

    def test_all_anchors_have_U_ref(self):
        """Every anchor has a defined U_ref (may be 0 for regulatory values)."""
        ledger = load_ledger()
        for anchor in ledger.anchors:
            assert anchor.U_ref >= 0, f"Anchor {anchor.anchor_id} has negative U_ref"

    def test_all_anchors_have_source(self):
        """Every anchor has at least one source."""
        ledger = load_ledger()
        for anchor in ledger.anchors:
            assert anchor.source.has_any_source(), \
                f"Anchor {anchor.anchor_id} has no source"

    def test_all_anchors_have_provenance(self):
        """Every anchor has provenance metadata."""
        ledger = load_ledger()
        for anchor in ledger.anchors:
            assert anchor.provenance.file, \
                f"Anchor {anchor.anchor_id} missing provenance file"
            assert anchor.provenance.sha256, \
                f"Anchor {anchor.anchor_id} missing provenance sha256"
            assert anchor.provenance.locator, \
                f"Anchor {anchor.anchor_id} missing provenance locator"

    def test_all_anchors_have_U_ref_basis(self):
        """Every anchor has a valid U_ref_basis."""
        ledger = load_ledger()
        valid_bases = {"specification_tolerance", "experimental_scatter", "literature_range"}
        for anchor in ledger.anchors:
            assert anchor.U_ref_basis in valid_bases, \
                f"Anchor {anchor.anchor_id} has invalid U_ref_basis: {anchor.U_ref_basis}"

    def test_coverage_by_domain(self):
        """Anchors span multiple domains."""
        ledger = load_ledger()
        domains = ledger.domains()
        assert len(domains) >= 5, f"Expected ≥5 domains, got {len(domains)}: {domains}"

    def test_specific_anchors_present(self):
        """Key anchors from the task spec are present."""
        ledger = load_ledger()
        required_prefixes = ["ICE", "RES", "PROP"]
        for prefix in required_prefixes:
            matching = [a for a in ledger.anchors if a.anchor_id.startswith(prefix)]
            assert len(matching) >= 3, \
                f"Expected ≥3 anchors with prefix {prefix}, got {len(matching)}"


# ============================================================
# C2: Three-state judgment tests
# ============================================================

class TestThreeStateJudgment:
    """C2: 10 known Δ/U_num/U_ref combinations → correct classification."""

    # Pre-computed test cases: (delta, U_num, U_ref, expected_judgment)
    TEST_CASES = [
        # ROBUST cases: |Δ| ≤ k·√(U_num² + U_ref²)
        (0.5, 0.3, 0.2, ThreeStateJudgment.ROBUST),   # k·√(0.09+0.04) = 2·0.361 = 0.721 > 0.5
        (1.0, 0.5, 0.5, ThreeStateJudgment.ROBUST),   # k·√(0.25+0.25) = 2·0.707 = 1.414 > 1.0
        (0.1, 0.1, 0.1, ThreeStateJudgment.ROBUST),   # k·√(0.01+0.01) = 2·0.141 = 0.283 > 0.1
        (2.0, 1.0, 1.0, ThreeStateJudgment.ROBUST),   # k·√(1+1) = 2·1.414 = 2.828 > 2.0

        # CRITICAL cases: k·√(...) < |Δ| ≤ 2k·√(...)
        (1.5, 0.5, 0.3, ThreeStateJudgment.CRITICAL), # k·√(0.25+0.09) = 2·0.583 = 1.166 < 1.5 ≤ 2.332 ✓·√(0.04+0.01) = 2·0.224 = 0.447 < 1.0 ≤ 0.894? No, 1.0 > 0.894 → INDET
        (0.6, 0.2, 0.1, ThreeStateJudgment.CRITICAL), # 0.4472 < 0.6 ≤ 0.8944 ✓
        (0.8, 0.2, 0.1, ThreeStateJudgment.CRITICAL), # 0.4472 < 0.8 ≤ 0.8944 ✓
        (3.0, 1.0, 0.5, ThreeStateJudgment.CRITICAL), # k·√(1+0.25) = 2·1.118 = 2.236 < 3.0 ≤ 4.472 ✓

        # INDETERMINATE cases: |Δ| > 2k·√(...)
        (5.0, 0.5, 0.3, ThreeStateJudgment.INDETERMINATE),  # 2k·√(0.25+0.09) = 4·0.583 = 2.332 < 5.0
        (10.0, 1.0, 1.0, ThreeStateJudgment.INDETERMINATE), # 2k·√(1+1) = 4·1.414 = 5.657 < 10.0
    ]

    @pytest.mark.parametrize("delta,U_num,U_ref,expected", TEST_CASES)
    def test_judgment_classification(self, delta, U_num, U_ref, expected):
        """Test three-state classification for known combinations."""
        result = judge_criterion(delta, U_num, U_ref, k=2.0)
        assert result.judgment == expected, \
            f"delta={delta}, U_num={U_num}, U_ref={U_ref}: " \
            f"expected {expected.value}, got {result.judgment.value}. " \
            f"Reason: {result.reason}"

    def test_judgment_result_fields(self):
        """JudgmentResult has all required fields."""
        result = judge_criterion(1.0, 0.5, 0.3, k=2.0)
        assert hasattr(result, 'judgment')
        assert hasattr(result, 'delta')
        assert hasattr(result, 'U_combined')
        assert hasattr(result, 'U_num')
        assert hasattr(result, 'U_ref')
        assert hasattr(result, 'k')
        assert hasattr(result, 'threshold_robust')
        assert hasattr(result, 'threshold_critical')
        assert hasattr(result, 'reason')


# ============================================================
# C3: Fail-closed tests
# ============================================================

class TestFailClosed:
    """C3: U_ref missing → INDETERMINATE (fail-closed), 100% detection."""

    def test_none_U_ref_is_indeterminate(self):
        """U_ref=None → INDETERMINATE."""
        result = judge_criterion(delta=0.01, U_num=0.1, U_ref=None)
        assert result.judgment == ThreeStateJudgment.INDETERMINATE
        assert result.fail_closed is True

    def test_negative_U_ref_is_indeterminate(self):
        """U_ref < 0 → INDETERMINATE."""
        result = judge_criterion(delta=0.01, U_num=0.1, U_ref=-1.0)
        assert result.judgment == ThreeStateJudgment.INDETERMINATE
        assert result.fail_closed is True

    def test_zero_delta_with_none_U_ref_still_indeterminate(self):
        """Even with Δ=0, missing U_ref → INDETERMINATE (fail-closed)."""
        result = judge_criterion(delta=0.0, U_num=0.1, U_ref=None)
        assert result.judgment == ThreeStateJudgment.INDETERMINATE

    def test_batch_with_mixed_U_refs(self):
        """Batch judgment with some None U_refs."""
        deltas = [0.5, 1.0, 0.1]
        U_nums = [0.3, 0.5, 0.1]
        U_refs = [0.2, None, 0.1]
        results = batch_judge(deltas, U_nums, U_refs)
        assert results[1].judgment == ThreeStateJudgment.INDETERMINATE
        assert results[1].fail_closed is True

    def test_fail_closed_count(self):
        """judgment_statistics correctly counts fail-closed."""
        results = [
            judge_criterion(0.1, 0.5, 0.3),   # ROBUST
            judge_criterion(0.01, 0.1, None),  # INDETERMINATE
            judge_criterion(5.0, 0.5, 0.3),    # INDETERMINATE
        ]
        stats = judgment_statistics(results)
        assert stats["fail_closed_count"] == 2
        assert stats["indeterminate"] == 2


# ============================================================
# C4: Uncertainty propagation tests
# ============================================================

class TestUncertaintyPropagation:
    """C4: Propagation formula correct, vs Monte Carlo ≤5% error."""

    def test_analytical_simple_sum(self):
        """f(x,y) = x + y: U_result = √(U_x² + U_y²) × k."""
        def f(x, y):
            return x + y

        result = propagate_uncertainty_analytical(
            f,
            anchor_values={"x": 10.0, "y": 20.0},
            anchor_uncertainties={"x": 0.5, "y": 0.3},
            k=2.0,
        )
        # Expected: U_result = 2·√(0.25 + 0.09) = 2·0.5831 = 1.1662
        expected_U = 2.0 * math.sqrt(0.5**2 + 0.3**2)
        assert abs(result.U_result - expected_U) / expected_U < 0.01  # 1% tolerance

    def test_analytical_product(self):
        """f(x,y) = x*y: U_result via error propagation."""
        def f(x, y):
            return x * y

        result = propagate_uncertainty_analytical(
            f,
            anchor_values={"x": 10.0, "y": 5.0},
            anchor_uncertainties={"x": 1.0, "y": 0.5},
            k=2.0,
        )
        # ∂f/∂x = y = 5, ∂f/∂y = x = 10
        # U_result = 2·√((5·1)² + (10·0.5)²) = 2·√(25+25) = 2·7.071 = 14.142
        expected_U = 2.0 * math.sqrt((5 * 1.0)**2 + (10 * 0.5)**2)
        assert abs(result.U_result - expected_U) / expected_U < 0.02  # 2% tolerance

    def test_monte_carlo_vs_analytical_within_5pct(self):
        """Monte Carlo result within 5% of analytical for linear function."""
        def f(x, y):
            return 2.0 * x + 3.0 * y

        anchor_values = {"x": 10.0, "y": 5.0}
        anchor_uncertainties = {"x": 1.0, "y": 0.5}

        comparison = compare_propagation_methods(
            f, anchor_values, anchor_uncertainties,
            n_samples=5000, k=2.0, seed=42
        )
        assert comparison["U_relative_error"] < 0.05, \
            f"MC vs analytical U error: {comparison['U_relative_error']*100:.1f}% > 5%"

    def test_monte_carlo_convergence(self):
        """MC result converges with more samples."""
        def f(x):
            return x ** 2

        anchor_values = {"x": 5.0}
        anchor_uncertainties = {"x": 0.5}

        results = []
        for n in [100, 1000, 5000]:
            r = monte_carlo_propagation(
                f, anchor_values, anchor_uncertainties,
                n_samples=n, k=2.0, seed=42
            )
            results.append(r.U_result)

        # Later results should be closer to each other
        assert abs(results[2] - results[1]) < abs(results[1] - results[0]) + 0.1

    def test_propagation_contributions_sum(self):
        """Contributions sum to total variance."""
        def f(x, y, z):
            return x + 2*y + 0.5*z

        result = propagate_uncertainty_analytical(
            f,
            anchor_values={"x": 1.0, "y": 2.0, "z": 3.0},
            anchor_uncertainties={"x": 0.1, "y": 0.2, "z": 0.3},
            k=2.0,
        )
        total_var_from_contribs = sum(result.contributions.values())
        total_var_expected = (result.U_result / 2.0) ** 2  # Remove k factor
        assert abs(total_var_from_contribs - total_var_expected) / total_var_expected < 0.01

    def test_zero_uncertainty_anchor(self):
        """Anchor with U=0 contributes nothing."""
        def f(x, y):
            return x + y

        result = propagate_uncertainty_analytical(
            f,
            anchor_values={"x": 10.0, "y": 20.0},
            anchor_uncertainties={"x": 0.0, "y": 0.5},
            k=2.0,
        )
        assert result.contributions["x"] == 0.0
        assert result.contributions["y"] > 0.0


# ============================================================
# C5: Schema and ledger tests
# ============================================================

class TestSchemaAndLedger:
    """C5: Schema validation and ledger operations."""

    def test_schema_validation_negative_U_ref(self):
        """Negative U_ref raises ValueError."""
        with pytest.raises(ValueError, match="non-negative"):
            RefAnchor(
                anchor_id="TEST-001",
                name="Test",
                value=1.0,
                unit="m",
                source=AnchorSource(spec_clause="test"),
                U_ref=-1.0,
                U_ref_basis="experimental_scatter",
                U_ref_basis_detail="test",
                provenance=AnchorProvenance(file="test.pdf", sha256="abc", locator="p.1"),
                domain="test",
                category="test",
            )

    def test_schema_validation_no_source(self):
        """No source raises ValueError."""
        with pytest.raises(ValueError, match="at least one source"):
            RefAnchor(
                anchor_id="TEST-002",
                name="Test",
                value=1.0,
                unit="m",
                source=AnchorSource(),  # No source
                U_ref=0.1,
                U_ref_basis="experimental_scatter",
                U_ref_basis_detail="test",
                provenance=AnchorProvenance(file="test.pdf", sha256="abc", locator="p.1"),
                domain="test",
                category="test",
            )

    def test_schema_validation_invalid_basis(self):
        """Invalid U_ref_basis raises ValueError."""
        with pytest.raises(ValueError, match="U_ref_basis"):
            RefAnchor(
                anchor_id="TEST-003",
                name="Test",
                value=1.0,
                unit="m",
                source=AnchorSource(spec_clause="test"),
                U_ref=0.1,
                U_ref_basis="invalid_basis",
                U_ref_basis_detail="test",
                provenance=AnchorProvenance(file="test.pdf", sha256="abc", locator="p.1"),
                domain="test",
                category="test",
            )

    def test_ledger_roundtrip(self):
        """Ledger can be serialized and deserialized."""
        ledger = load_ledger()
        data = ledger.to_dict()
        # Reconstruct
        anchors = [RefAnchor.from_dict(d) for d in data["anchors"]]
        ledger2 = RefAnchorLedger(anchors, schema_version=data["schema_version"])
        assert len(ledger2) == len(ledger)

    def test_ledger_filter_by_domain(self):
        """Filter by domain works."""
        ledger = load_ledger()
        ice_anchors = ledger.filter_by_domain("ice_navigation")
        assert len(ice_anchors) > 0
        for a in ice_anchors:
            assert a.domain == "ice_navigation"

    def test_ledger_get_or_raise(self):
        """get_or_raise raises for missing anchor."""
        ledger = load_ledger()
        with pytest.raises(KeyError):
            ledger.get_or_raise("NONEXISTENT-999")

    def test_combined_uncertainty_formula(self):
        """Combined uncertainty formula is correct."""
        U_combined = compute_combined_uncertainty(0.3, 0.4, k=2.0)
        # Expected: 2·√(0.09 + 0.16) = 2·0.5 = 1.0
        assert abs(U_combined - 1.0) < 1e-10

    def test_relative_uncertainty_property(self):
        """Relative uncertainty computed correctly."""
        anchor = RefAnchor(
            anchor_id="TEST-004",
            name="Test",
            value=100.0,
            unit="kN",
            source=AnchorSource(spec_clause="test"),
            U_ref=10.0,
            U_ref_basis="experimental_scatter",
            U_ref_basis_detail="test",
            provenance=AnchorProvenance(file="test.pdf", sha256="abc", locator="p.1"),
            domain="test",
            category="test",
        )
        assert anchor.relative_uncertainty == 0.1  # 10%


# ============================================================
# Integration test: full pipeline
# ============================================================

class TestIntegration:
    """End-to-end integration test."""

    def test_full_pipeline(self):
        """Load ledger → pick anchors → propagate → judge."""
        # 1. Load ledger
        ledger = load_ledger()
        assert len(ledger) >= 30

        # 2. Pick two anchors for a simple calculation
        anchor1 = ledger.get("RES-001")  # ITTC-57 coefficient
        anchor2 = ledger.get("RES-002")  # Form factor

        assert anchor1 is not None
        assert anchor2 is not None

        # 3. Define a simple calculation: R = anchor1 * anchor2 * 1000
        def resistance_calc(c_f, form_factor):
            return c_f * form_factor * 1000.0

        # 4. Propagate uncertainty
        anchor_values = {"c_f": anchor1.value, "form_factor": anchor2.value}
        # Convert U_ref (k=2) to standard uncertainty (k=1)
        anchor_uncertainties = {
            "c_f": anchor1.U_ref / 2.0,
            "form_factor": anchor2.U_ref / 2.0,
        }

        result = propagate_uncertainty_analytical(
            resistance_calc, anchor_values, anchor_uncertainties, k=2.0
        )

        assert result.result_nominal > 0
        assert result.U_result > 0

        # 5. Judge against a reference value
        ref_value = result.result_nominal * 1.05  # 5% off
        delta = abs(result.result_nominal - ref_value)

        judgment = judge_criterion(
            delta=delta,
            U_num=result.U_result,
            U_ref=0.0,  # No additional reference uncertainty
            k=2.0,
        )

        # Should be ROBUST since 5% is within combined uncertainty
        assert judgment.judgment in (ThreeStateJudgment.ROBUST, ThreeStateJudgment.CRITICAL)
