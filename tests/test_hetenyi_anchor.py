"""Pytest tests for Hetényi (1946) series solution anchor.

Machine acceptance criteria (all must pass):
    1. series_truncation_100_error < 0.01  (ratio)
    2. convergence_monotone == True
    3. pytest_test_hetenyi_anchor == PASS  (overall)

Reference:
    Hetényi, M. (1946). Beams on Elastic Foundation. Univ. Michigan Press.
    – Chapter X: Rectangular Plates on Elastic Foundations.
"""

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.ice_sheet.hetenyi_plate import (
    HetenyiPlateParams,
    hetenyi_center_deflection,
    hetenyi_convergence_study,
    hetenyi_deflection_field,
    ice_plate_default_params,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def square_plate_params():
    """Standard simply-supported square plate on Winkler foundation.

    Parameters chosen for ice-sheet relevance:
        a = b = 10 m,  h = 0.5 m,  E = 5 GPa,  ν = 0.3
        k_w = ρ_w g = 1025 × 9.81 ≈ 10055.25 Pa/m
        q = 1000 Pa (uniform)
    """
    return ice_plate_default_params(a=10.0, h=0.5, E=5e9, nu=0.3, q=1000.0)


@pytest.fixture
def small_plate_params():
    """Small plate for fast testing.

    a = b = 1 m, h = 0.1 m, E = 1 GPa, k_w = 1e5 Pa/m, q = 10000 Pa.
    """
    return HetenyiPlateParams(
        a=1.0, b=1.0, E=1e9, nu=0.3, h=0.1, k_w=1e5, q=10000.0
    )


# ---------------------------------------------------------------------------
# Test 1: Series truncation error at 100 terms < 0.01
# ---------------------------------------------------------------------------

class TestSeriesTruncation:
    """Verify that 100-term truncation gives < 1% error vs. 500-term reference."""

    def test_truncation_100_error_small_plate(self, small_plate_params):
        """series_truncation_100_error < 0.01 for small plate."""
        conv = hetenyi_convergence_study(
            small_plate_params,
            n_terms_list=[10, 50, 100, 200, 500],
        )
        idx_100 = conv["n_terms_list"].index(100)
        error_100 = conv["relative_to_reference"][idx_100]
        assert error_100 < 0.01, (
            f"100-term truncation error = {error_100:.6e}, expected < 0.01"
        )

    def test_truncation_100_error_ice_plate(self, square_plate_params):
        """series_truncation_100_error < 0.01 for ice-scale plate."""
        conv = hetenyi_convergence_study(
            square_plate_params,
            n_terms_list=[10, 50, 100, 200, 500],
        )
        idx_100 = conv["n_terms_list"].index(100)
        error_100 = conv["relative_to_reference"][idx_100]
        assert error_100 < 0.01, (
            f"100-term truncation error = {error_100:.6e}, expected < 0.01"
        )


# ---------------------------------------------------------------------------
# Test 2: Convergence is monotone (error decreases monotonically)
# ---------------------------------------------------------------------------

class TestConvergenceMonotone:
    """Verify that the relative error decreases monotonically with n_terms."""

    def test_monotone_small_plate(self, small_plate_params):
        """convergence_monotone == True for small plate."""
        conv = hetenyi_convergence_study(
            small_plate_params,
            n_terms_list=[1, 2, 3, 5, 10, 20, 50, 100, 200],
        )
        assert conv["is_monotone"] is True, (
            f"Convergence not monotone! errors = {conv['relative_to_reference']}"
        )

    def test_monotone_ice_plate(self, square_plate_params):
        """convergence_monotone == True for ice-scale plate."""
        conv = hetenyi_convergence_study(
            square_plate_params,
            n_terms_list=[1, 2, 3, 5, 10, 20, 50, 100, 200],
        )
        assert conv["is_monotone"] is True, (
            f"Convergence not monotone! errors = {conv['relative_to_reference']}"
        )


# ---------------------------------------------------------------------------
# Test 3: Physical sanity checks
# ---------------------------------------------------------------------------

class TestPhysicalSanity:
    """Verify physical correctness of the Hetényi solution."""

    def test_positive_deflection_for_positive_load(self, small_plate_params):
        """Positive uniform load → positive (downward) center deflection."""
        w = hetenyi_center_deflection(small_plate_params, n_terms=51)
        assert w > 0, f"Expected positive deflection, got {w}"

    def test_zero_load_gives_zero_deflection(self):
        """Zero load → zero deflection."""
        params = HetenyiPlateParams(
            a=1.0, b=1.0, E=1e9, nu=0.3, h=0.1, k_w=1e5, q=0.0
        )
        w = hetenyi_center_deflection(params, n_terms=51)
        assert abs(w) < 1e-30, f"Expected zero deflection, got {w}"

    def test_increasing_foundation_reduces_deflection(self):
        """Stiffer foundation → smaller deflection."""
        base = dict(a=1.0, b=1.0, E=1e9, nu=0.3, h=0.1, q=10000.0)
        w_soft = hetenyi_center_deflection(
            HetenyiPlateParams(k_w=1e3, **base), n_terms=51
        )
        w_stiff = hetenyi_center_deflection(
            HetenyiPlateParams(k_w=1e6, **base), n_terms=51
        )
        assert w_soft > w_stiff, (
            f"Soft foundation should give larger deflection: "
            f"w_soft={w_soft}, w_stiff={w_stiff}"
        )

    def test_boundary_deflection_is_zero(self, small_plate_params):
        """Simply-supported BC: w = 0 on all edges."""
        x, y, w = hetenyi_deflection_field(
            small_plate_params, nx=21, ny=21, n_terms=51
        )
        assert np.allclose(w[0, :], 0.0, atol=1e-14), "w(x=0) ≠ 0"
        assert np.allclose(w[-1, :], 0.0, atol=1e-14), "w(x=a) ≠ 0"
        assert np.allclose(w[:, 0], 0.0, atol=1e-14), "w(y=0) ≠ 0"
        assert np.allclose(w[:, -1], 0.0, atol=1e-14), "w(y=b) ≠ 0"

    def test_symmetry_of_deflection_field(self, small_plate_params):
        """Square plate under uniform load: w(x,y) = w(y,x) = w(a-x, y) etc."""
        x, y, w = hetenyi_deflection_field(
            small_plate_params, nx=21, ny=21, n_terms=51
        )
        assert np.allclose(w, w.T, atol=1e-12), "Symmetry w(x,y)≠w(y,x) violated"
        assert np.allclose(w, w[::-1, :], atol=1e-12), "Symmetry about x=a/2 violated"
        assert np.allclose(w, w[:, ::-1], atol=1e-12), "Symmetry about y=b/2 violated"


# ---------------------------------------------------------------------------
# Test 4: Cross-validation with known limiting cases
# ---------------------------------------------------------------------------

class TestLimitingCases:
    """Validate against known analytical limits."""

    def test_no_foundation_recovers_navier(self):
        """With k_w → 0, should recover classical Navier SS plate solution.

        For SS square plate under uniform load, Navier center deflection:
            w_c = 16 q a⁴ / (π⁶ D) · Σ_{m,n odd} (-1)^{(m-1)/2+(n-1)/2} / (mn(m²+n²)²)
        """
        a = 1.0
        E = 1e9
        nu = 0.3
        h = 0.1
        q = 10000.0
        D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))

        params = HetenyiPlateParams(
            a=a, b=a, E=E, nu=nu, h=h, k_w=1e-10, q=q  # effectively no foundation
        )
        w_hetenyi = hetenyi_center_deflection(params, n_terms=100)

        # Classical Navier reference (no foundation)
        pi = math.pi
        w_navier = 0.0
        for mi in range(100):
            m = 2 * mi + 1
            sign_m = (-1.0) ** mi
            for ni in range(100):
                n = 2 * ni + 1
                sign_n = (-1.0) ** ni
                w_navier += sign_m * sign_n / (m * n * (m ** 2 + n ** 2) ** 2)
        w_navier *= 16.0 * q * a ** 4 / (pi ** 6 * D)

        rel_err = abs(w_hetenyi - w_navier) / abs(w_navier)
        assert rel_err < 1e-6, (
            f"No-foundation limit: w_hetenyi={w_hetenyi:.8e}, "
            f"w_navier={w_navier:.8e}, rel_err={rel_err:.3e}"
        )

    def test_known_alpha_coefficient(self):
        """Validate against Timoshenko's α coefficient for SS square plate.

        For SS square plate under uniform load (no foundation):
            w_max = α q a⁴ / D,  where α ≈ 0.00406 (Timoshenko & Woinowsky-Krieger)
        """
        a = 1.0
        E = 1e9
        nu = 0.3
        h = 0.1
        q = 10000.0
        D = E * h ** 3 / (12.0 * (1.0 - nu ** 2))

        params = HetenyiPlateParams(
            a=a, b=a, E=E, nu=nu, h=h, k_w=1e-10, q=q
        )
        w = hetenyi_center_deflection(params, n_terms=100)
        alpha_computed = w * D / (q * a ** 4)
        alpha_reference = 0.00406

        rel_err = abs(alpha_computed - alpha_reference) / alpha_reference
        assert rel_err < 0.001, (
            f"α coefficient: computed={alpha_computed:.6f}, "
            f"reference={alpha_reference}, rel_err={rel_err:.3e}"
        )


# ---------------------------------------------------------------------------
# Test 5: Anchor report generation
# ---------------------------------------------------------------------------

class TestHetenyiAnchor:
    """Master anchor test: generates the validation report JSON."""

    def test_hetenyi_anchor(self, small_plate_params, tmp_path):
        """Master test: all acceptance criteria in one place.

        Produces artifacts/hetenyi_anchor_report.json with:
            - series_truncation_100_error
            - convergence_monotone
            - pytest_test_hetenyi_anchor
        """
        # --- Convergence study ---
        conv = hetenyi_convergence_study(
            small_plate_params,
            n_terms_list=[1, 2, 3, 5, 10, 20, 50, 100, 200, 500],
        )

        # Criterion 1: 100-term truncation error < 0.01
        idx_100 = conv["n_terms_list"].index(100)
        truncation_error_100 = conv["relative_to_reference"][idx_100]

        # Criterion 2: monotone convergence (error decreases monotonically)
        convergence_monotone = conv["is_monotone"]

        # Criterion 3: overall pass
        all_pass = (truncation_error_100 < 0.01) and convergence_monotone

        # Build report
        report = {
            "mission_id": 721,
            "scenario_key": "SIR-PC2|V01",
            "title": "Hetényi (1946) Series Solution Anchor – Simply-Supported Square Plate on Winkler Foundation",
            "timestamp": "2026-10-31T00:00:00Z",
            "parameters": {
                "a": small_plate_params.a,
                "b": small_plate_params.b,
                "E": small_plate_params.E,
                "nu": small_plate_params.nu,
                "h": small_plate_params.h,
                "k_w": small_plate_params.k_w,
                "q": small_plate_params.q,
                "D": small_plate_params.D,
                "char_length": small_plate_params.char_length,
                "relative_stiffness": small_plate_params.relative_stiffness,
            },
            "acceptance_criteria": {
                "series_truncation_100_error": {
                    "value": truncation_error_100,
                    "threshold": 0.01,
                    "passed": truncation_error_100 < 0.01,
                },
                "convergence_monotone": {
                    "value": convergence_monotone,
                    "threshold": True,
                    "passed": convergence_monotone,
                },
                "pytest_test_hetenyi_anchor": {
                    "value": all_pass,
                    "threshold": True,
                    "passed": all_pass,
                },
            },
            "convergence_data": {
                "n_terms": conv["n_terms_list"],
                "w_center": conv["w_center"],
                "relative_error": conv["relative_to_reference"],
            },
            "reference": {
                "n_terms": conv["reference_n"],
                "w_center": conv["reference_w"],
            },
        }

        # Save report
        report_path = tmp_path / "hetenyi_anchor_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2)

        # Also save to artifacts/ if it exists
        artifacts_dir = Path(__file__).resolve().parents[1] / "artifacts"
        if artifacts_dir.exists():
            with open(artifacts_dir / "hetenyi_anchor_report.json", "w") as f:
                json.dump(report, f, indent=2)

        # Assertions
        assert truncation_error_100 < 0.01, (
            f"FAIL: series_truncation_100_error = {truncation_error_100:.6e} >= 0.01"
        )
        assert convergence_monotone is True, (
            "FAIL: convergence is not monotone"
        )
        assert all_pass is True, "FAIL: overall anchor test failed"

        print(f"\n=== Hetényi Anchor Report ===")
        print(f"series_truncation_100_error = {truncation_error_100:.6e} (< 0.01: PASS)")
        print(f"convergence_monotone = {convergence_monotone} (== True: PASS)")
        print(f"pytest_test_hetenyi_anchor = {all_pass} (== True: PASS)")
        print(f"Reference w_center (n=500) = {conv['reference_w']:.10e}")
        print(f"100-term w_center = {conv['w_center'][idx_100]:.10e}")
