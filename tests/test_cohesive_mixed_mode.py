"""Test cohesive mixed-mode validation (SIR-PC2 | V04).

Tests:
1. BK law correctness (analytical)
2. Proportional mixed-mode paths: Gc error < 5%
3. Non-proportional mixed-mode paths: Gc error < 5%
4. Mixed-mode ratio follows BK law within 10%
5. Energy dissipation consistency
6. Full validation report generation
"""

import json
import numpy as np
import pytest
from pathlib import Path

from tensorphysics.fracture_mechanics.bk_law import (
    BKLaw,
    compute_mixed_mode_ratio,
    compute_effective_fracture_toughness,
)
from tensorphysics.fracture_mechanics.cohesive import (
    CohesiveElement,
    CohesiveProperties,
    MixedModeCohesiveModel,
)
from tensorphysics.fracture_mechanics.validation import (
    MixedModeValidator,
    ValidationReport,
)


# ---------------------------------------------------------------------------
# Test 1: BK law analytical correctness
# ---------------------------------------------------------------------------

class TestBKLawAnalytical:
    """Verify BK law against analytical values."""

    def test_pure_mode_I(self):
        """ψ=0 → G_c = G_Ic."""
        bk = BKLaw(G_Ic=200.0, G_IIc=600.0, eta=1.5)
        assert bk.effective_toughness(0.0) == pytest.approx(200.0, rel=1e-12)

    def test_pure_mode_II(self):
        """ψ=1 → G_c = G_IIc."""
        bk = BKLaw(G_Ic=200.0, G_IIc=600.0, eta=1.5)
        assert bk.effective_toughness(1.0) == pytest.approx(600.0, rel=1e-12)

    def test_midpoint(self):
        """ψ=0.5 → G_c = G_Ic + (G_IIc - G_Ic) * 0.5^eta."""
        G_Ic, G_IIc, eta = 200.0, 600.0, 1.5
        bk = BKLaw(G_Ic, G_IIc, eta)
        expected = G_Ic + (G_IIc - G_Ic) * (0.5 ** eta)
        assert bk.effective_toughness(0.5) == pytest.approx(expected, rel=1e-12)

    def test_monotonicity(self):
        """G_c should increase monotonically with ψ when G_IIc > G_Ic."""
        bk = BKLaw(G_Ic=200.0, G_IIc=600.0, eta=1.5)
        psi_vals = np.linspace(0, 1, 50)
        Gc_vals = bk.effective_toughness_array(psi_vals)
        assert all(np.diff(Gc_vals) >= 0)

    def test_mixed_mode_ratio_bounds(self):
        """Mixed-mode ratio should be in [0, 1]."""
        assert compute_mixed_mode_ratio(100.0, 0.0) == 0.0
        assert compute_mixed_mode_ratio(0.0, 100.0) == 1.0
        assert compute_mixed_mode_ratio(50.0, 50.0) == pytest.approx(0.5, rel=1e-12)

    def test_invalid_psi_raises(self):
        """Out-of-range ψ should raise ValueError."""
        bk = BKLaw(G_Ic=200.0, G_IIc=600.0, eta=1.5)
        with pytest.raises(ValueError):
            bk.effective_toughness(-0.1)
        with pytest.raises(ValueError):
            bk.effective_toughness(1.1)


# ---------------------------------------------------------------------------
# Test 2: Proportional mixed-mode paths
# ---------------------------------------------------------------------------

class TestProportionalMixedMode:
    """Test cohesive model under proportional loading paths."""

    @pytest.fixture
    def validator(self):
        return MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
            sigma_max=5e6,
            tau_max=5e6,
            K_n=1e10,
            K_s=1e10,
        )

    def test_pure_mode_I_Gc_error(self, validator):
        """Pure mode I: Gc error < 5%."""
        vp = validator.validate_point(0.0, nonproportional=False, n_steps=500)
        assert vp.Gc_error < 0.05, f"Gc_error = {vp.Gc_error:.4f} >= 0.05"

    def test_pure_mode_II_Gc_error(self, validator):
        """Pure mode II: Gc error < 5%."""
        vp = validator.validate_point(1.0, nonproportional=False, n_steps=500)
        assert vp.Gc_error < 0.05, f"Gc_error = {vp.Gc_error:.4f} >= 0.05"

    def test_equal_mixity_Gc_error(self, validator):
        """ψ=0.5: Gc error < 5%."""
        vp = validator.validate_point(0.5, nonproportional=False, n_steps=500)
        assert vp.Gc_error < 0.05, f"Gc_error = {vp.Gc_error:.4f} >= 0.05"

    def test_all_proportional_Gc_error(self, validator):
        """All proportional paths: max Gc error < 5%."""
        report = validator.validate_all(nonproportional=False, n_steps=500)
        assert report.Gc_error_vs_input < 0.05, (
            f"Gc_error_vs_input = {report.Gc_error_vs_input:.4f} >= 0.05"
        )


# ---------------------------------------------------------------------------
# Test 3: Non-proportional mixed-mode paths
# ---------------------------------------------------------------------------

class TestNonProportionalMixedMode:
    """Test cohesive model under non-proportional loading paths."""

    @pytest.fixture
    def validator(self):
        return MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
            sigma_max=5e6,
            tau_max=5e6,
            K_n=1e10,
            K_s=1e10,
        )

    def test_nonprop_pure_mode_I(self, validator):
        """Non-proportional pure mode I: Gc error < 5%."""
        vp = validator.validate_point(0.0, nonproportional=True, n_steps=500)
        assert vp.Gc_error < 0.05, f"Gc_error = {vp.Gc_error:.4f} >= 0.05"

    def test_nonprop_equal_mixity(self, validator):
        """Non-proportional ψ=0.5: Gc error < 5%."""
        vp = validator.validate_point(0.5, nonproportional=True, n_steps=500)
        assert vp.Gc_error < 0.05, f"Gc_error = {vp.Gc_error:.4f} >= 0.05"

    def test_nonprop_all_Gc_error(self, validator):
        """All non-proportional paths: max Gc error < 5%."""
        report = validator.validate_all(nonproportional=True, n_steps=500)
        assert report.Gc_error_vs_input < 0.05, (
            f"Gc_error_vs_input = {report.Gc_error_vs_input:.4f} >= 0.05"
        )


# ---------------------------------------------------------------------------
# Test 4: Mixed-mode ratio follows BK law
# ---------------------------------------------------------------------------

class TestMixedModeRatioBKLaw:
    """Verify that computed mixed-mode ratio matches BK law prediction."""

    @pytest.fixture
    def validator(self):
        return MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
        )

    def test_proportional_BK_law(self, validator):
        """Proportional paths: mixed_mode_ratio_BK_law < 0.1."""
        report = validator.validate_all(nonproportional=False, n_steps=500)
        assert report.mixed_mode_ratio_BK_law < 0.1, (
            f"mixed_mode_ratio_BK_law = {report.mixed_mode_ratio_BK_law:.4f} >= 0.1"
        )

    def test_nonproportional_BK_law(self, validator):
        """Non-proportional paths: mixed_mode_ratio_BK_law < 0.1."""
        report = validator.validate_all(nonproportional=True, n_steps=500)
        assert report.mixed_mode_ratio_BK_law < 0.1, (
            f"mixed_mode_ratio_BK_law = {report.mixed_mode_ratio_BK_law:.4f} >= 0.1"
        )


# ---------------------------------------------------------------------------
# Test 5: Energy dissipation consistency
# ---------------------------------------------------------------------------

class TestEnergyDissipation:
    """Verify energy dissipation matches fracture toughness."""

    def test_pure_mode_I_failure(self):
        """Pure mode I: element fails at correct ERR."""
        # Use consistent material properties
        # For bilinear law: need G_Ic > 0.5 * sigma_max^2 / K_n
        # With sigma_max=1e6, K_n=1e9: 0.5 * 1e12 / 1e9 = 500 J/m²
        # So use G_Ic = 1000 J/m² to be safe
        props = CohesiveProperties(
            K_n=1e9, K_s=1e9,
            sigma_max=1e6, tau_max=1e6,
            G_Ic=1000.0, G_IIc=3000.0, eta=1.5,
        )
        elem = CohesiveElement(props)
        
        # Apply monotonically increasing mode I displacement
        # delta_f = 2*G_Ic/sigma_max = 2*1000/1e6 = 2e-3 m
        delta_n_max = 4.0 * 2.0 * props.G_Ic / props.sigma_max
        for dn in np.linspace(0, delta_n_max, 500):
            elem.update(dn, 0.0)
        
        # Check that element failed
        assert elem.current_state.failed, "Element should have failed"
        
        # Check that ERR at failure matches G_Ic
        G_I_at_failure = elem.current_state.G_I
        assert abs(G_I_at_failure - props.G_Ic) / props.G_Ic < 0.05, (
            f"G_I at failure {G_I_at_failure:.2f} differs from G_Ic={props.G_Ic:.2f} by more than 5%"
        )

    def test_damage_irreversibility(self):
        """Damage should not decrease (no healing)."""
        props = CohesiveProperties()
        elem = CohesiveElement(props)
        
        # Load, then partially unload
        elem.update(1e-5, 0.0)
        d1 = elem.current_state.damage
        
        elem.update(0.5e-5, 0.0)  # partial unload
        d2 = elem.current_state.damage
        
        assert d2 >= d1 - 1e-15, "Damage should not decrease (no healing)"


# ---------------------------------------------------------------------------
# Test 6: Full validation report
# ---------------------------------------------------------------------------

class TestValidationReport:
    """Test full validation report generation."""

    def test_report_generation(self, tmp_path):
        """Generate and save validation report."""
        validator = MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
        )
        report = validator.validate_all(nonproportional=False, n_steps=500)
        
        # Check report structure
        assert report.n_points == 11
        assert isinstance(report.Gc_error_vs_input, float)
        assert isinstance(report.mixed_mode_ratio_BK_law, float)
        
        # Save to JSON
        report_path = tmp_path / "validation_report.json"
        report.to_json(str(report_path))
        assert report_path.exists()
        
        # Load and verify
        with open(report_path) as f:
            data = json.load(f)
        assert data["n_points"] == 11
        assert "validation_points" in data

    def test_acceptance_criteria(self):
        """Full acceptance criteria check."""
        validator = MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
        )
        
        # Proportional
        report_prop = validator.validate_all(nonproportional=False, n_steps=500)
        assert report_prop.Gc_error_vs_input < 0.05
        assert report_prop.mixed_mode_ratio_BK_law < 0.1
        
        # Non-proportional
        report_nonprop = validator.validate_all(nonproportional=True, n_steps=500)
        assert report_nonprop.Gc_error_vs_input < 0.05
        assert report_nonprop.mixed_mode_ratio_BK_law < 0.1


# ---------------------------------------------------------------------------
# Integration test: pytest_test_cohesive_mixed_mode == PASS
# ---------------------------------------------------------------------------

class TestIntegration:
    """Integration test matching the acceptance criterion name."""

    def test_pytest_test_cohesive_mixed_mode(self):
        """Master test: all acceptance criteria pass."""
        validator = MixedModeValidator(
            G_Ic=200.0,
            G_IIc=600.0,
            eta=1.5,
        )
        
        # Run full validation (proportional + non-proportional)
        report_prop = validator.validate_all(nonproportional=False, n_steps=500)
        report_nonprop = validator.validate_all(nonproportional=True, n_steps=500)
        
        # Criterion 1: Gc_error_vs_input < 0.05
        gc_err = max(report_prop.Gc_error_vs_input, report_nonprop.Gc_error_vs_input)
        assert gc_err < 0.05, f"Gc_error_vs_input = {gc_err:.4f} >= 0.05"
        
        # Criterion 2: mixed_mode_ratio_BK_law < 0.1
        bk_err = max(report_prop.mixed_mode_ratio_BK_law, report_nonprop.mixed_mode_ratio_BK_law)
        assert bk_err < 0.1, f"mixed_mode_ratio_BK_law = {bk_err:.4f} >= 0.1"
        
        # Criterion 3: pytest passes (we're in it!)
        assert True
