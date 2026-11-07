"""Tests for the contribution measurement unit ontology, classifier, and metrics."""

import json
import pytest

from tensorphysics.contribution.classifier import (
    ContributionItem,
    ContributionClassifier,
    classify_item,
    _count_signals,
    _check_evidence_fields,
)
from tensorphysics.contribution.golden_set import GOLDEN_SET, load_golden_set
from tensorphysics.contribution.metrics import compute_f1, compute_productivity_rate
from tensorphysics.contribution.ontology import (
    ContributionCategory,
    CONTRIBUTION_ONTOLOGY,
    EVIDENCE_THRESHOLDS,
    COUNTER_EXAMPLES,
    get_category,
    list_categories,
)


# ===== Ontology Tests =====

class TestOntology:
    def test_six_categories_defined(self):
        """Exactly 6 contribution categories must be defined."""
        assert len(ContributionCategory) == 6
        assert len(CONTRIBUTION_ONTOLOGY) == 6

    def test_all_categories_have_definitions(self):
        """Each category must have a complete definition."""
        for cat, defn in CONTRIBUTION_ONTOLOGY.items():
            assert defn.name
            assert defn.label
            assert defn.description
            assert len(defn.min_evidence) > 0
            assert len(defn.positive_signals) > 0
            assert len(defn.counter_examples) > 0
            assert defn.evidence_threshold

    def test_evidence_thresholds_lookup(self):
        """EVIDENCE_THRESHOLDS must have entries for all categories."""
        for cat in ContributionCategory:
            assert cat.value in EVIDENCE_THRESHOLDS

    def test_counter_examples_lookup(self):
        """COUNTER_EXAMPLES must have entries for all categories."""
        for cat in ContributionCategory:
            assert cat.value in COUNTER_EXAMPLES

    def test_get_category(self):
        """get_category must resolve valid category strings."""
        assert get_category("C1_verified_case") == ContributionCategory.C1_VERIFIED_CASE
        assert get_category("C6_infra") == ContributionCategory.C6_INFRA
        assert get_category("nonexistent") is None

    def test_list_categories(self):
        """list_categories must return 6 summaries."""
        cats = list_categories()
        assert len(cats) == 6
        for cat in cats:
            assert "name" in cat
            assert "label" in cat
            assert "description" in cat
            assert "min_evidence" in cat
            assert "evidence_threshold" in cat


# ===== Classifier Tests =====

class TestClassifier:
    def test_classify_c1_verified_case(self):
        """C1 items with full evidence should be classified as C1."""
        item = ContributionItem(
            mission_id="test_c1",
            title="Holtrop resistance validation against experimental data",
            summary="Verified Holtrop 1984 resistance prediction. Geometry validated: Cb=0.65, L=230m, B=32.2m. Computed Ct=4.012e-3 vs experimental Ct=3.987e-3, error=0.63%.",
            tags=["resistance", "validation", "holtrop"],
            evidence={
                "geometry_parameters": {"Cb": 0.65, "L": 230.0, "B": 32.2},
                "reference_value": 3.987e-3,
                "computed_value": 4.012e-3,
                "error_percent": 0.63,
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C1_VERIFIED_CASE
        assert result.confidence >= 0.5

    def test_classify_c2_capability_delta(self):
        """C2 items with before/after metrics should be classified as C2."""
        item = ContributionItem(
            mission_id="test_c2",
            title="LBM GPU acceleration: 12x speedup over CPU",
            summary="Ported TensorLBM to CUDA. Baseline CPU: 45 min. GPU: 3.7 min. Speedup: 12.2x.",
            tags=["LBM", "GPU", "performance"],
            evidence={
                "baseline_metric": 2700.0,
                "improved_metric": 222.0,
                "delta_percent": 1120.0,
                "metric_name": "time_seconds",
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C2_CAPABILITY_DELTA
        assert result.confidence >= 0.5

    def test_classify_c3_defect_fix(self):
        """C3 items with bug/fix/regression should be classified as C3."""
        item = ContributionItem(
            mission_id="test_c3",
            title="Fix: BEM Green function singularity",
            summary="Fixed singularity handling. Bug: division by zero. Fix: added analytical limit. Regression test added.",
            tags=["BEM", "bug-fix", "green-function"],
            evidence={
                "bug_description": "Division by zero at coincident points",
                "fix_description": "Added analytical limit evaluation",
                "regression_test": "test_green_function_singularity_limit",
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C3_DEFECT_FIX
        assert result.confidence >= 0.5

    def test_classify_c4_method_doc(self):
        """C4 items with method/steps/output should be classified as C4."""
        item = ContributionItem(
            mission_id="test_c4",
            title="Tutorial: How to run TensorLBM lid-driven cavity benchmark",
            summary="Step-by-step tutorial for running the lid-driven cavity benchmark. Includes: grid setup, boundary conditions, relaxation time selection, convergence criteria, post-processing.",
            tags=["LBM", "tutorial", "benchmark", "documentation"],
            evidence={
                "method_description": "Lid-driven cavity benchmark setup",
                "steps_or_workflow": "5 steps: grid setup, BCs, tau, convergence, post-processing",
                "expected_output": "Velocity profiles matching Ghia 1982",
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C4_METHOD_DOC
        assert result.confidence >= 0.5

    def test_classify_c5_negative_result(self):
        """C5 items with attempt/failure/diagnostic should be classified as C5."""
        item = ContributionItem(
            mission_id="test_c5",
            title="Negative result: LBM turbulence model fails for high Re",
            summary="Attempted Smagorinsky LES in LBM for Re=10^7. Failed: numerical instability. Diagnostic: kinetic energy grows unbounded, CFL violated.",
            tags=["LBM", "turbulence", "negative-result"],
            evidence={
                "attempt_description": "Smagorinsky LES in LBM for Re=10^7",
                "failure_mode": "Numerical instability, unbounded kinetic energy",
                "diagnostic_data": "CFL violation at dt=1e-6",
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C5_NEGATIVE_RESULT
        assert result.confidence >= 0.5

    def test_classify_c6_infra(self):
        """C6 items with infra/improvement/impact should be classified as C6."""
        item = ContributionItem(
            mission_id="test_c6",
            title="CI pipeline: added automated regression testing",
            summary="Added CI pipeline for all solvers. Baseline: manual testing, 2 days. Improved: automated, 30 min.",
            tags=["CI", "regression-testing", "automation", "infrastructure"],
            evidence={
                "infra_component": "CI pipeline",
                "improvement_description": "Automated regression testing for all solvers",
                "impact_metric": "Turnaround from 2 days to 30 min",
            },
        )
        result = classify_item(item)
        assert result.predicted_category == ContributionCategory.C6_INFRA
        assert result.confidence >= 0.5

    def test_classifier_batch(self):
        """Batch classification must return results for all items."""
        classifier = ContributionClassifier()
        items = [
            ContributionItem(mission_id=f"m{i}", title=f"Test {i}")
            for i in range(5)
        ]
        results = classifier.classify_batch(items)
        assert len(results) == 5
        for r in results:
            assert r.mission_id.startswith("m")

    def test_count_signals(self):
        """Signal counting must be case-insensitive."""
        text = "This is a VALIDATION test with experimental DATA"
        signals = _count_signals(text, ["validation", "experimental data", "not present"])
        assert "validation" in signals
        assert "experimental data" in signals
        assert "not present" not in signals

    def test_check_evidence_fields(self):
        """Evidence field checking must find present fields."""
        item = ContributionItem(
            mission_id="test",
            evidence={"geometry_parameters": {"Cb": 0.6}, "reference_value": 1.0},
        )
        present, missing = _check_evidence_fields(item, ContributionCategory.C1_VERIFIED_CASE)
        assert "geometry_parameters" in present
        assert "reference_value" in present
        assert "computed_value" in missing
        assert "error_percent" in missing


# ===== Golden Set Tests =====

class TestGoldenSet:
    def test_golden_set_size(self):
        """Golden set must have exactly 60 items."""
        assert len(GOLDEN_SET) == 60

    def test_category_distribution(self):
        """Each category must have at least 8 items."""
        counts = {}
        for item in GOLDEN_SET:
            gt = item["ground_truth"]
            counts[gt] = counts.get(gt, 0) + 1

        for cat in ContributionCategory:
            assert counts.get(cat.value, 0) >= 8, f"{cat.value} has only {counts.get(cat.value, 0)} items"

    def test_adversarial_count(self):
        """Must have at least 10 adversarial samples."""
        adversarial = [d for d in GOLDEN_SET if d.get("is_adversarial")]
        assert len(adversarial) >= 10

    def test_load_golden_set(self):
        """load_golden_set must return ContributionItem objects."""
        items = load_golden_set()
        assert len(items) == 60
        for item in items:
            assert isinstance(item, ContributionItem)
            assert item.mission_id
            assert "ground_truth" in item.metadata

    def test_all_mission_ids_unique(self):
        """All mission IDs must be unique."""
        ids = [d["mission_id"] for d in GOLDEN_SET]
        assert len(ids) == len(set(ids))


# ===== Metrics Tests =====

class TestMetrics:
    def test_compute_f1(self):
        """F1 computation must return valid metrics."""
        metrics = compute_f1()
        assert "macro_f1" in metrics
        assert "weighted_f1" in metrics
        assert "per_category" in metrics
        assert "confusion_matrix" in metrics
        assert metrics["total_items"] == 60

    def test_f1_threshold(self):
        """Macro F1 must be >= 0.85 (acceptance criterion)."""
        metrics = compute_f1()
        assert metrics["macro_f1"] >= 0.85, f"Macro F1 = {metrics['macro_f1']} < 0.85"

    def test_per_category_f1(self):
        """Each category with support must have F1 computed."""
        metrics = compute_f1()
        for cat, vals in metrics["per_category"].items():
            if vals["support"] > 0:
                assert "precision" in vals
                assert "recall" in vals
                assert "f1" in vals
                assert 0.0 <= vals["precision"] <= 1.0
                assert 0.0 <= vals["recall"] <= 1.0
                assert 0.0 <= vals["f1"] <= 1.0

    def test_compute_productivity_rate(self):
        """Productivity rate computation must return valid metrics."""
        prod = compute_productivity_rate()
        assert "total_contributions" in prod
        assert "verified_contributions" in prod
        assert "productivity_rate" in prod
        assert "breakdown" in prod
        assert prod["total_contributions"] == 60
        assert prod["verified_contributions"] > 0

    def test_confusion_matrix_shape(self):
        """Confusion matrix must be 6x6."""
        metrics = compute_f1()
        cm = metrics["confusion_matrix"]
        assert len(cm) == 6
        for row in cm.values():
            assert len(row) == 6


# ===== CLI Tests =====

class TestCLI:
    def test_ontology_command(self):
        """Ontology command must list 6 categories."""
        from tensorphysics.contribution.cli import cmd_ontology
        import argparse
        result = cmd_ontology(argparse.Namespace())
        assert result == 0

    def test_report_command(self, tmp_path):
        """Report command must generate a JSON report."""
        from tensorphysics.contribution.cli import cmd_report
        import argparse
        output = str(tmp_path / "test_report.json")
        result = cmd_report(argparse.Namespace(output=output))
        assert result == 0
        with open(output) as f:
            report = json.load(f)
        assert "f1_metrics" in report
        assert "productivity_metrics" in report
        assert report["f1_metrics"]["total_items"] == 60
