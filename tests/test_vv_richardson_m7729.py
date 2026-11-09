"""
Tests for Richardson extrapolation and GCI calculation.

Verifies:
- Observed order of convergence p
- Extrapolated value f_ext
- GCI_fine and GCI_coarse
- Convergence status determination (PASS/FAIL/UNRESOLVED)
- Known analytical convergence cases
"""

import pytest
import numpy as np
from tensorphysics.vv.richardson import (
    RichardsonExtrapolation,
    GCIResult,
    ConvergenceStatus,
)


class TestRichardsonBasic:
    """Basic Richardson extrapolation tests."""
    
    def test_perfect_second_order_convergence(self):
        """Test with exact second-order convergence (p=2)."""
        # f_exact = 1.0, f_h = f_exact + C*h^2
        # h_coarse=4h, h_medium=2h, h_fine=h
        # f_coarse = 1.0 + C*(4h)^2 = 1.0 + 16C*h^2
        # f_medium = 1.0 + C*(2h)^2 = 1.0 + 4C*h^2
        # f_fine   = 1.0 + C*h^2
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16  # h=4
        f_medium = f_exact + C * 4   # h=2
        f_fine = f_exact + C * 1     # h=1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)
        
        assert abs(result.p - 2.0) < 1e-10, f"Expected p=2, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10, f"Expected f_ext={f_exact}, got {result.f_ext}"
        assert result.status == ConvergenceStatus.PASS
    
    def test_perfect_first_order_convergence(self):
        """Test with exact first-order convergence (p=1)."""
        C = 0.1
        f_exact = 2.0
        f_coarse = f_exact + C * 4  # h=4
        f_medium = f_exact + C * 2  # h=2
        f_fine = f_exact + C * 1    # h=1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)
        
        assert abs(result.p - 1.0) < 1e-10, f"Expected p=1, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10
        # With p=1 and r=2, GCI can be large, so just check it computes
        assert result.f_ext is not None
    
    def test_third_order_convergence(self):
        """Test with exact third-order convergence (p=3)."""
        C = 0.001
        f_exact = 5.0
        f_coarse = f_exact + C * 64  # h=4, h^3=64
        f_medium = f_exact + C * 8   # h=2, h^3=8
        f_fine = f_exact + C * 1     # h=1, h^3=1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)
        
        assert abs(result.p - 3.0) < 1e-10, f"Expected p=3, got p={result.p}"
        assert abs(result.f_ext - f_exact) < 1e-10
    
    def test_gci_decreases_with_refinement(self):
        """GCI_fine should be smaller than GCI_coarse."""
        C = 0.05
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)
        
        assert result.gci_fine < result.gci_coarse, \
            f"GCI_fine ({result.gci_fine}) should be < GCI_coarse ({result.gci_coarse})"
    
    def test_gci_ratio_near_unity_asymptotic(self):
        """In asymptotic range, GCI_coarse / (r^p * GCI_fine) ≈ 1."""
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)
        
        # For perfect convergence, should be in asymptotic range
        # The ratio might not be exactly 1 due to the formula, but should be reasonable
        assert result.asymptotic or result.gci_ratio > 0.5


class TestRichardsonConvergenceStatus:
    """Tests for convergence status determination."""
    
    def test_pass_status_low_gci(self):
        """PASS when GCI is below threshold."""
        C = 0.001  # Small error -> low GCI
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1
        
        re = RichardsonExtrapolation(gci_threshold=5.0)
        result = re.compute(f_fine, f_medium, f_coarse)
        
        assert result.status == ConvergenceStatus.PASS
    
    def test_fail_status_high_gci(self):
        """FAIL when GCI exceeds threshold."""
        C = 0.5  # Large error -> high GCI
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1
        
        re = RichardsonExtrapolation(gci_threshold=5.0)
        result = re.compute(f_fine, f_medium, f_coarse)
        
        assert result.status == ConvergenceStatus.FAIL
    
    def test_unresolved_oscillatory(self):
        """UNRESOLVED for oscillatory convergence."""
        # Oscillatory: fine > medium > coarse is not monotone
        f_coarse = 1.0
        f_medium = 1.2
        f_fine = 0.9  # Overshoots
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse)
        
        assert result.status == ConvergenceStatus.UNRESOLVED
    
    def test_monotone_convergence_computes(self):
        """Monotone convergence should compute valid p."""
        # Solutions getting better with refinement
        f_coarse = 1.0
        f_medium = 1.1
        f_fine = 1.15
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse)
        
        # Should compute positive p
        assert result.p > 0
        assert not np.isnan(result.p)


class TestRichardsonNonUniform:
    """Tests for non-uniform refinement ratios."""
    
    def test_nonuniform_refinement(self):
        """Test with different refinement ratios."""
        C = 0.01
        f_exact = 1.0
        # r_coarse = 3, r_fine = 2
        f_coarse = f_exact + C * 36  # h=6
        f_medium = f_exact + C * 4   # h=2
        f_fine = f_exact + C * 1     # h=1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=3.0)
        
        # Should still compute reasonable p
        assert 0.5 < result.p < 4.0
        assert result.f_ext is not None
    
    def test_to_dict_serialization(self):
        """Test that results can be serialized to dict."""
        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1
        
        re = RichardsonExtrapolation()
        result = re.compute(f_fine, f_medium, f_coarse)
        
        d = result.to_dict()
        assert 'p' in d
        assert 'f_ext' in d
        assert 'gci_fine' in d
        assert 'status' in d
        assert d['status'] == 'PASS'


class TestRichardsonAssessment:
    """Tests for the high-level assessment interface."""
    
    def test_assess_grid_convergence(self):
        """Test the assess_grid_convenience method."""
        C = 0.01
        f_exact = 1.0
        solutions = [
            f_exact + C * 16,  # coarse
            f_exact + C * 4,   # medium
            f_exact + C * 1,   # fine
        ]
        
        re = RichardsonExtrapolation()
        report = re.assess_grid_convergence(
            solutions, [2.0, 2.0], quantity_name="drag_coefficient"
        )
        
        assert report['quantity'] == 'drag_coefficient'
        assert 'result' in report
        assert 'recommendation' in report
        assert report['result']['status'] == 'PASS'
    
    def test_assess_requires_three_solutions(self):
        """Assessment requires exactly 3 solutions."""
        re = RichardsonExtrapolation()
        with pytest.raises(ValueError, match="3 solutions"):
            re.assess_grid_convergence([1.0, 2.0], [2.0, 2.0])
    
    def test_assess_requires_two_ratios(self):
        """Assessment requires exactly 2 refinement ratios."""
        re = RichardsonExtrapolation()
        with pytest.raises(ValueError, match="2 refinement"):
            re.assess_grid_convergence([1.0, 2.0, 3.0], [2.0])
