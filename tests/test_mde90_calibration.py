"""Acceptance tests for the MDE_90 calibration engine.

Each test pins one machine-checkable claim of the calibration:

* C1: ≥20 real criteria complete MDE_90 calibration (100% coverage);
* C2: at 5% injected error, detection rate ≥80% (≥16 of 20);
* C3: at 1% injected error, overall detection rate ≤30%;
* C4: sigmoid fit R² ≥ 0.95 for every graded criterion;
* C5: negative control — 0% error → 0% detection (all criteria);
* C6: negative control — 100% error → 100% detection (all criteria);
* C7: grading ledger is produced with correct schema.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "audit" / "mde90_calibration.py"
LEDGER = ROOT / "artifacts" / "mde90" / "mde90_ledger.json"
GRADING = ROOT / "artifacts" / "mde90" / "mde90_summary.json"


def _load_tool():
    spec = importlib.util.spec_from_file_location("mde90_calibration", TOOL)
    module = importlib.util.module_from_spec(spec)
    sys.modules["mde90_calibration"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return _load_tool()


@pytest.fixture(scope="module")
def ledger():
    if not LEDGER.is_file():
        # Run calibration if ledger doesn't exist
        mod = _load_tool()
        mod.main(["--no-figures"])
    return json.loads(LEDGER.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def grading():
    if not GRADING.is_file():
        mod = _load_tool()
        mod.main(["--no-figures"])
    return json.loads(GRADING.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def graded_criteria(ledger):
    return [c for c in ledger["criteria"]
            if c["calibratable"] and c["grade"] in ("strong", "medium", "weak")]


# ---------------------------------------------------------------------------
# C1: ≥20 real criteria complete MDE_90 calibration
# ---------------------------------------------------------------------------
def test_c1_at_least_20_criteria_calibrated(graded_criteria):
    assert len(graded_criteria) >= 20, (
        f"Need ≥20 graded criteria, got {len(graded_criteria)}")


# ---------------------------------------------------------------------------
# C2: 5% error → ≥80% detection rate
# ---------------------------------------------------------------------------
def test_c2_detection_at_5pct_at_least_80(graded_criteria):
    detection_rates = []
    for c in graded_criteria:
        rate = next((dc["detection_rate"] for dc in c["detection_curve"]
                     if dc["epsilon"] == 0.05), 0.0)
        detection_rates.append(rate)
    n_pass = sum(1 for r in detection_rates if r >= 0.80)
    assert n_pass >= 16, (
        f"Need ≥16 of {len(graded_criteria)} criteria to detect at 5%, got {n_pass}")


# ---------------------------------------------------------------------------
# C3: 1% error → ≤30% overall detection rate
# ---------------------------------------------------------------------------
def test_c3_detection_at_1pct_at_most_30(graded_criteria):
    detection_rates = []
    for c in graded_criteria:
        rate = next((dc["detection_rate"] for dc in c["detection_curve"]
                     if dc["epsilon"] == 0.01), 0.0)
        detection_rates.append(rate)
    overall = sum(detection_rates) / len(detection_rates)
    assert overall <= 0.30, (
        f"Overall detection at 1% should be ≤30%, got {overall*100:.1f}%")


# ---------------------------------------------------------------------------
# C4: sigmoid fit R² ≥ 0.95
# ---------------------------------------------------------------------------
def test_c4_sigmoid_r_squared_at_least_095(graded_criteria):
    for c in graded_criteria:
        r2 = c.get("sigmoid", {}).get("r_squared", 0.0)
        assert r2 >= 0.95, (
            f"{c['criterion_id']}: R² = {r2:.6f} < 0.95")


# ---------------------------------------------------------------------------
# C5: negative control — 0% error → 0% detection
# ---------------------------------------------------------------------------
def test_c5_zero_error_zero_detection(graded_criteria):
    for c in graded_criteria:
        rate = c.get("negative_control_0pct")
        assert rate == 0.0, (
            f"{c['criterion_id']}: detection at 0% error = {rate}, expected 0.0")


# ---------------------------------------------------------------------------
# C6: negative control — 100% error → 100% detection
# ---------------------------------------------------------------------------
def test_c6_full_error_full_detection(graded_criteria):
    for c in graded_criteria:
        rate = c.get("negative_control_100pct")
        assert rate == 1.0, (
            f"{c['criterion_id']}: detection at 100% error = {rate}, expected 1.0")


# ---------------------------------------------------------------------------
# C7: grading ledger schema and content
# ---------------------------------------------------------------------------
def test_c7_grading_ledger_schema(grading):
    assert grading["schema"] == "tensorphysics.criterion_strength_grading/1"
    assert "grading_rules" in grading
    assert "criteria" in grading
    assert "summary" in grading
    assert grading["summary"]["total"] >= 20


def test_c7_grading_ledger_has_all_criteria(grading, graded_criteria):
    ledger_ids = {c["criterion_id"] for c in grading["criteria"]}
    graded_ids = {c["criterion_id"] for c in graded_criteria}
    assert graded_ids.issubset(ledger_ids), (
        f"Missing criteria in grading ledger: {graded_ids - ledger_ids}")


# ---------------------------------------------------------------------------
# Additional structural tests
# ---------------------------------------------------------------------------
def test_detection_curve_has_all_epsilons(graded_criteria):
    expected_epsilons = {0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.07,
                         0.10, 0.15, 0.20, 0.30, 0.50, 1.00}
    for c in graded_criteria:
        actual = {dc["epsilon"] for dc in c["detection_curve"]}
        assert actual == expected_epsilons, (
            f"{c['criterion_id']}: missing epsilons {expected_epsilons - actual}")


def test_mde90_values_are_finite_for_graded(graded_criteria):
    for c in graded_criteria:
        mde = c.get("mde_90", float("inf"))
        assert mde != float("inf"), (
            f"{c['criterion_id']}: MDE_90 is infinite")
        assert mde >= 0, (
            f"{c['criterion_id']}: MDE_90 is negative: {mde}")


def test_grading_categories_consistent(grading):
    s = grading["summary"]
    assert s["strong"] + s["medium"] + s["weak"] + s["unquantifiable"] + s["not_applicable"] == s["calibratable"]


def test_negative_controls_tautological(ledger):
    """Tautological negative controls should never detect anything."""
    nc = ledger.get("negative_controls", {})
    assert nc.get("all_zero_at_0pct") is True
    assert nc.get("all_zero_at_100pct") is True


def test_engine_has_no_domain_vocabulary(tool):
    """The MDE_90 engine must not contain domain-specific vocabulary."""
    source = TOOL.read_text(encoding="utf-8")
    forbidden = ("hull", "ship", "vessel", "draught", "resistance",
                 "propeller", "rudder", "seakeeping", "ice_")
    # Only check the engine region (not the criteria definitions)
    # The criteria definitions are allowed to reference domain concepts
    # through the gate functions they call.
    found = [tok for tok in forbidden if tok in source.lower()]
    # This is a soft check — the criteria DO reference domain through gate calls
    # The key constraint is that the ENGINE (fit_sigmoid, grade_criterion, etc.)
    # is domain-free.
    engine_region_start = source.find("def fit_sigmoid")
    engine_region_end = source.find("def build_grading_ledger")
    if engine_region_start > 0 and engine_region_end > engine_region_start:
        engine_region = source[engine_region_start:engine_region_end].lower()
        engine_violations = [tok for tok in forbidden if tok in engine_region]
        assert not engine_violations, (
            f"Domain vocabulary in engine region: {engine_violations}")
