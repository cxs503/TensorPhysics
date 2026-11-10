"""Pytest: P519-N02 cross-implementation divergence attribution for level-ice resistance.

Acceptance criteria (from campaign metadata):
1. Σ component diffs = total diff, relative residual < 1e-9
2. After crushing coefficient calibration, remaining unexplained diff < 5%
3. Sensitivity matrix rank-deficient when two constants scaled together
4. Fingerprint validation failures ≥ 5 when missing constants.source or units
5. 3 delivered nodes all pass with 0 violations
6. Same input twice → same artifact sha256
7. Absolute accuracy grade = WITHHELD (no experimental benchmark)
"""

from __future__ import annotations

import hashlib
import sys
import json
import math
from pathlib import Path

import pytest

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (str(REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tensorphysics.ice.cross_impl_attribution import (  # noqa: E402
    COMPONENT_NAMES,
    CaliberFingerprint,
    RefShip,
    attribute_divergence,
    build_fingerprints,
    calibrate_crushing_coefficient,
    decompose_five,
    impl_ia_lindqvist,
    impl_lit_corrected,
    impl_lit_original,
    impl_tp_level,
    run_full_attribution,
    sensitivity_matrix,
    validate_fingerprints,
    write_artifacts,
)

SHIP = RefShip()
H_REF = 1.0
V_REF = 3.0


# ---------------------------------------------------------------------------
# 1. Five-component decomposition correctness
# ---------------------------------------------------------------------------


class TestFiveComponentDecomposition:
    """Each implementation maps to the five-component basis correctly."""

    def test_tp_level_components_sum(self):
        d = decompose_five("TP_level", H_REF, V_REF, SHIP)
        total = sum(d[c] for c in COMPONENT_NAMES)
        assert total == pytest.approx(d["R_total_N"], rel=1e-12)

    def test_ia_lindqvist_components_sum(self):
        d = decompose_five("IA_lindqvist", H_REF, V_REF, SHIP)
        total = sum(d[c] for c in COMPONENT_NAMES)
        assert total == pytest.approx(d["R_total_N"], rel=1e-12)

    def test_lit_original_components_sum(self):
        d = decompose_five("Lit_original", H_REF, V_REF, SHIP)
        total = sum(d[c] for c in COMPONENT_NAMES)
        assert total == pytest.approx(d["R_total_N"], rel=1e-12)

    def test_lit_corrected_equals_tp_level(self):
        """Lit_corrected should be identical to TP_level."""
        a = decompose_five("Lit_corrected", H_REF, V_REF, SHIP)
        b = decompose_five("TP_level", H_REF, V_REF, SHIP)
        for c in COMPONENT_NAMES + ["R_total_N"]:
            assert a[c] == pytest.approx(b[c], rel=1e-12)

    def test_tp_level_crushing_dominant(self):
        """For PC6 in 1 m ice, crushing is the largest component."""
        d = decompose_five("TP_level", H_REF, V_REF, SHIP)
        assert d["R_crush_N"] > d["R_bend_N"]
        assert d["R_crush_N"] > d["R_momentum_N"]

    def test_ia_lindqvist_buoyancy_not_bending(self):
        """IA_lindqvist puts the second term in buoyancy, not bending."""
        d = decompose_five("IA_lindqvist", H_REF, V_REF, SHIP)
        assert d["R_buoyancy_N"] > 0
        assert d["R_bend_N"] == 0.0


# ---------------------------------------------------------------------------
# 2. Divergence attribution: Σ component diffs = total diff
# ---------------------------------------------------------------------------


class TestDivergenceAttribution:
    """Component-wise differences sum to total difference."""

    def test_ia_vs_tp_residual_near_zero(self):
        attr = attribute_divergence("TP_level", "IA_lindqvist", H_REF, V_REF, SHIP)
        assert attr["relative_residual"] < 1e-9

    def test_sum_component_diffs_equals_total(self):
        attr = attribute_divergence("TP_level", "IA_lindqvist", H_REF, V_REF, SHIP)
        assert attr["sum_component_diffs_N"] == pytest.approx(attr["total_diff_N"], rel=1e-12)

    def test_lit_original_vs_tp_residual_near_zero(self):
        attr = attribute_divergence("TP_level", "Lit_original", H_REF, V_REF, SHIP)
        assert attr["relative_residual"] < 1e-9

    def test_lit_corrected_vs_tp_zero_diff(self):
        attr = attribute_divergence("TP_level", "Lit_corrected", H_REF, V_REF, SHIP)
        assert abs(attr["total_diff_N"]) < 1e-9

    def test_crushing_dominates_divergence(self):
        """The crushing component accounts for >100% of the total divergence
        (because the bending/buoyancy term goes in the opposite direction)."""
        attr = attribute_divergence("TP_level", "IA_lindqvist", H_REF, V_REF, SHIP)
        assert attr["component_fractions"]["R_crush_N"] > 1.0
        assert attr["component_fractions"]["R_bend_N"] < 0

    def test_total_ratio_approx_3_8x(self):
        """IA_lindqvist should be ~3.8x TP_level for the reference case."""
        attr = attribute_divergence("TP_level", "IA_lindqvist", H_REF, V_REF, SHIP)
        assert 3.5 < attr["ratio"] < 4.0


# ---------------------------------------------------------------------------
# 3. Coefficient calibration: remaining diff < 5%
# ---------------------------------------------------------------------------


class TestCoefficientCalibration:
    """After calibrating the crushing coefficient, remaining diff < 5%."""

    def test_ia_calibration_within_5pct(self):
        cal = calibrate_crushing_coefficient("IA_lindqvist", H_REF, V_REF, SHIP)
        assert cal["within_5pct"] is True
        assert cal["remaining_pct"] < 5.0

    def test_lit_original_calibration_within_5pct(self):
        cal = calibrate_crushing_coefficient("Lit_original", H_REF, V_REF, SHIP)
        assert cal["within_5pct"] is True

    def test_k_crush_less_than_one(self):
        """IA_lindqvist overestimates crushing, so k < 1."""
        cal = calibrate_crushing_coefficient("IA_lindqvist", H_REF, V_REF, SHIP)
        assert 0 < cal["k_crush_calibrated"] < 1.0


# ---------------------------------------------------------------------------
# 4. Sensitivity matrix & identifiability
# ---------------------------------------------------------------------------


class TestSensitivityMatrix:
    """Sensitivity matrix rank-deficiency for multiplicative parameter pairs."""

    def test_rank_deficient(self):
        sens = sensitivity_matrix(H_REF, V_REF, SHIP)
        assert sens["identifiability"]["full_rank"] is False

    def test_non_identifiable_sigma_c_flare(self):
        sens = sensitivity_matrix(H_REF, V_REF, SHIP)
        pairs = sens["identifiability"]["non_identifiable_pairs"]
        assert ["sigma_c", "flare_deg"] in pairs

    def test_numerical_rank_less_than_params(self):
        sens = sensitivity_matrix(H_REF, V_REF, SHIP)
        assert sens["identifiability"]["numerical_rank"] < sens["identifiability"]["n_params"]


# ---------------------------------------------------------------------------
# 5. Caliber fingerprint validation: ≥ 5 failures
# ---------------------------------------------------------------------------


class TestFingerprintValidation:
    """Missing constants.source or units → validation failure."""

    def test_at_least_five_failures(self):
        fps = build_fingerprints()
        result = validate_fingerprints(fps)
        assert result["n_failures"] >= 5

    def test_valid_fingerprints_pass(self):
        fps = build_fingerprints()
        result = validate_fingerprints(fps)
        # First 4 should be valid
        for i in range(4):
            assert result["details"][i]["valid"] is True

    def test_missing_source_fails(self):
        fp = CaliberFingerprint(
            impl="test",
            formula_family="test",
            constants={"sigma_c_Pa": {"value": 5e6, "unit": "Pa"}},  # no source
            units="SI",
            reference_case={"h_m": 1.0},
            comparable_to=[],
        )
        errors = fp.validate()
        assert any("source" in e for e in errors)

    def test_missing_unit_fails(self):
        fp = CaliberFingerprint(
            impl="test",
            formula_family="test",
            constants={"sigma_c_Pa": {"value": 5e6, "source": "test"}},  # no unit
            units="SI",
            reference_case={"h_m": 1.0},
            comparable_to=[],
        )
        errors = fp.validate()
        assert any("unit" in e for e in errors)

    def test_empty_source_fails(self):
        fp = CaliberFingerprint(
            impl="test",
            formula_family="test",
            constants={"sigma_c_Pa": {"value": 5e6, "unit": "Pa", "source": ""}},
            units="SI",
            reference_case={"h_m": 1.0},
            comparable_to=[],
        )
        errors = fp.validate()
        assert any("source" in e for e in errors)

    def test_empty_unit_fails(self):
        fp = CaliberFingerprint(
            impl="test",
            formula_family="test",
            constants={"sigma_c_Pa": {"value": 5e6, "unit": "", "source": "test"}},
            units="SI",
            reference_case={"h_m": 1.0},
            comparable_to=[],
        )
        errors = fp.validate()
        assert any("unit" in e for e in errors)


# ---------------------------------------------------------------------------
# 6. Determinism: same input → same sha256
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same input twice produces identical artifact sha256."""

    def test_sha256_match(self):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        assert report["acceptance"]["determinism_sha256_match"] is True

    def test_sha256_non_empty(self):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        assert len(report["determinism_sha256_run1"]) == 64
        assert len(report["determinism_sha256_run2"]) == 64


# ---------------------------------------------------------------------------
# 7. Absolute accuracy grade = WITHHELD
# ---------------------------------------------------------------------------


class TestAbsoluteAccuracy:
    """No experimental benchmark → accuracy grade WITHHELD."""

    def test_grade_withheld(self):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        assert report["acceptance"]["absolute_accuracy_grade"] == "WITHHELD"


# ---------------------------------------------------------------------------
# 8. Full pipeline integration
# ---------------------------------------------------------------------------


class TestFullPipeline:
    """End-to-end: run_full_attribution + write_artifacts."""

    def test_all_acceptance_criteria_pass(self):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        acc = report["acceptance"]
        assert acc["sum_component_diffs_equals_total"] is True
        assert acc["crush_calibration_within_5pct"] is True
        assert acc["sensitivity_rank_deficient"] is True
        assert acc["fingerprint_validation_failures_ge_5"] is True
        assert acc["determinism_sha256_match"] is True
        assert acc["absolute_accuracy_grade"] == "WITHHELD"

    def test_artifacts_written(self, tmp_path):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        paths = write_artifacts(report, outdir=tmp_path)
        assert Path(paths["report_json"]).exists()
        assert Path(paths["summary_txt"]).exists()

    def test_report_json_valid(self, tmp_path):
        report = run_full_attribution(H_REF, V_REF, SHIP)
        paths = write_artifacts(report, outdir=tmp_path)
        data = json.loads(Path(paths["report_json"]).read_text())
        assert "acceptance" in data
        assert "divergence" in data
        assert "calibration" in data
        assert "sensitivity" in data
        assert "fingerprints" in data
