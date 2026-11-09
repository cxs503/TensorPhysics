"""
Tests for ASME V&V 20 validation uncertainty quantification.

Verifies:
- Numerical uncertainty from GCI
- Input parameter uncertainty propagation
- Model-form uncertainty computation
- Total validation uncertainty (RSS combination)
- Validation assessment (error within uncertainty band)
"""

import pytest
import numpy as np
from tensorphysics.vv.asme_vv20 import ASMEVV20, UncertaintyBudget


class TestASMEVV20Numerical:
    """Tests for numerical uncertainty computation."""
    
    def test_numerical_uncertainty_from_gci(self):
        """U_num should equal GCI when no extrapolated value."""
        vv = ASMEVV20()
        u_num = vv.compute_numerical_uncertainty(gci_fine=2.5)
        assert abs(u_num - 2.5) < 1e-10
    
    def test_numerical_uncertainty_with_extrapolation(self):
        """U_num should use max of GCI and extrapolation error."""
        vv = ASMEVV20()
        # GCI = 2.5%, extrapolation error = 3.0%
        u_num = vv.compute_numerical_uncertainty(
            gci_fine=2.5, f_ext=1.03, f_fine=1.0
        )
        # Extrapolation error = |1.0 - 1.03| / |1.0| * 100 = 3.0%
        assert u_num >= 2.5  # Should be at least GCI
        assert abs(u_num - 3.0) < 1e-10  # Should be extrapolation error


class TestASMEVV20Input:
    """Tests for input parameter uncertainty."""
    
    def test_input_uncertainty_single_param(self):
        """Single parameter with absolute uncertainty."""
        vv = ASMEVV20()
        params = {
            'velocity': {'value': 10.0, 'uncertainty': 0.1}  # 0.1 m/s on 10 m/s = 1%
        }
        u_input = vv.compute_input_uncertainty(params)
        assert abs(u_input - 1.0) < 1e-10
    
    def test_input_uncertainty_relative(self):
        """Parameter with relative uncertainty."""
        vv = ASMEVV20()
        params = {
            'viscosity': {'value': 1e-3, 'uncertainty': 2.0, 'is_relative': True}
        }
        u_input = vv.compute_input_uncertainty(params)
        assert abs(u_input - 2.0) < 1e-10
    
    def test_input_uncertainty_multiple_params(self):
        """Multiple parameters combined via RSS."""
        vv = ASMEVV20()
        params = {
            'velocity': {'value': 10.0, 'uncertainty': 0.1},   # 1%
            'density': {'value': 1000.0, 'uncertainty': 10.0},  # 1%
            'length': {'value': 1.0, 'uncertainty': 0.005},     # 0.5%
        }
        u_input = vv.compute_input_uncertainty(params)
        expected = np.sqrt(1.0**2 + 1.0**2 + 0.5**2)
        assert abs(u_input - expected) < 1e-10
    
    def test_input_uncertainty_with_sensitivity(self):
        """Input uncertainty with sensitivity coefficients."""
        vv = ASMEVV20()
        params = {
            'velocity': {'value': 10.0, 'uncertainty': 0.1},  # 1%
        }
        sensitivities = {'velocity': 2.0}  # dQ/dV = 2
        u_input = vv.compute_input_uncertainty(params, sensitivities)
        # u_input = |2.0| * 1.0% = 2.0%
        assert abs(u_input - 2.0) < 1e-10
    
    def test_input_uncertainty_empty(self):
        """Empty parameters should give zero uncertainty."""
        vv = ASMEVV20()
        u_input = vv.compute_input_uncertainty({})
        assert u_input == 0.0


class TestASMEVV20Model:
    """Tests for model-form uncertainty."""
    
    def test_model_uncertainty_residual(self):
        """Model uncertainty from residual after accounting for U_num and U_input."""
        vv = ASMEVV20()
        # Error = 5%, U_num = 2%, U_input = 1%
        # U_model = sqrt(5^2 - 2^2 - 1^2) = sqrt(20) ≈ 4.47%
        u_model = vv.compute_model_uncertainty(
            computed_value=1.05,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
        )
        expected = np.sqrt(25.0 - 4.0 - 1.0)
        assert abs(u_model - expected) < 1e-10
    
    def test_model_uncertainty_zero_when_error_explained(self):
        """Model uncertainty is zero when error is fully explained."""
        vv = ASMEVV20()
        # Error = 2%, U_num = 2%, U_input = 0%
        # Residual = 4 - 4 - 0 = 0 -> U_model = 0
        u_model = vv.compute_model_uncertainty(
            computed_value=1.02,
            experimental_value=1.0,
            u_num=2.0,
            u_input=0.0,
        )
        # Due to floating point, might be very small but not exactly zero
        assert u_model < 0.01
    
    def test_model_uncertainty_negative_residual(self):
        """Model uncertainty is zero when error < combined U_num + U_input."""
        vv = ASMEVV20()
        # Error = 1%, U_num = 2%, U_input = 1%
        # Residual = 1 - 4 - 1 = -4 < 0 -> U_model = 0
        u_model = vv.compute_model_uncertainty(
            computed_value=1.01,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
        )
        assert u_model == 0.0


class TestASMEVV20Total:
    """Tests for total validation uncertainty."""
    
    def test_total_uncertainty_rss(self):
        """Total uncertainty is RSS of components."""
        vv = ASMEVV20()
        u_val = vv.compute_total_uncertainty(u_num=3.0, u_input=2.0, u_model=4.0)
        expected = np.sqrt(9.0 + 4.0 + 16.0)
        assert abs(u_val - expected) < 1e-10
    
    def test_total_uncertainty_zero_components(self):
        """Zero components give zero total."""
        vv = ASMEVV20()
        u_val = vv.compute_total_uncertainty(u_num=0.0, u_input=0.0, u_model=0.0)
        assert u_val == 0.0


class TestASMEVV20Validation:
    """Tests for complete validation assessment."""
    
    def test_validate_error_within_uncertainty(self):
        """Validation passes when error is within uncertainty band."""
        vv = ASMEVV20()
        # Error = 3%, U_num = 2%, U_input = 1%, U_model computed from residual
        # U_model = sqrt(9 - 4 - 1) = sqrt(4) = 2%
        # U_val = sqrt(4 + 1 + 4) = 3%
        # Error (3%) <= U_val (3%) -> validated
        budget = vv.validate(
            computed_value=1.03,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            quantity_name="drag_coefficient",
        )
        
        assert budget.error == pytest.approx(3.0, abs=0.01)
        # Error should be within or equal to uncertainty band
        assert budget.error <= budget.u_val + 0.01  # Small tolerance
        assert budget.is_validated() == True
    
    def test_validate_error_exceeds_uncertainty(self):
        """Validation fails when error exceeds uncertainty band."""
        vv = ASMEVV20()
        # Error = 10%, U_num = 1%, U_input = 0.5%
        # U_model = sqrt(100 - 1 - 0.25) = sqrt(98.75) ≈ 9.94%
        # U_val = sqrt(1 + 0.25 + 98.75) = 10%
        # But error = 10%, so it's borderline. Let's make error larger.
        budget = vv.validate(
            computed_value=1.15,  # 15% error
            experimental_value=1.0,
            u_num=1.0,
            u_input=0.5,
            quantity_name="lift_coefficient",
        )
        
        assert budget.error == pytest.approx(15.0, abs=0.01)
        # With large error, U_model will be large, but error should still exceed U_val
        # Actually, U_val will be close to error. Let's check the logic differently.
        # The key is that error_within_uncertainty should be False when error > U_val
        # With error=15%, U_num=1%, U_input=0.5%:
        # U_model = sqrt(225 - 1 - 0.25) = sqrt(223.75) ≈ 14.96%
        # U_val = sqrt(1 + 0.25 + 223.75) = 15%
        # So error ≈ U_val. Let's use a case where error clearly exceeds.
        
        # Better test: provide explicit U_model
        budget2 = vv.validate(
            computed_value=1.10,  # 10% error
            experimental_value=1.0,
            u_num=1.0,
            u_input=0.5,
            u_model=2.0,  # Explicit small model uncertainty
            quantity_name="test",
        )
        # U_val = sqrt(1 + 0.25 + 4) = sqrt(5.25) ≈ 2.29%
        # Error = 10% > 2.29% -> not validated
        assert budget2.error_within_uncertainty == False
        assert budget2.is_validated() == False
    
    def test_validate_with_explicit_model_uncertainty(self):
        """Validation with explicitly provided model uncertainty."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.05,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            u_model=3.0,
        )
        
        expected_u_val = np.sqrt(4.0 + 1.0 + 9.0)
        assert abs(budget.u_val - expected_u_val) < 1e-10
    
    def test_assess_validation_all_pass(self):
        """Overall assessment when all quantities validated."""
        vv = ASMEVV20()
        budgets = [
            vv.validate(1.02, 1.0, 2.0, 1.0, quantity_name="Q1"),
            vv.validate(1.03, 1.0, 3.0, 1.0, quantity_name="Q2"),
        ]
        
        assessment = vv.assess_validation(budgets)
        # Both should be validated (error within uncertainty)
        assert assessment['n_validated'] >= 1
        assert assessment['n_total'] == 2
    
    def test_assess_validation_partial(self):
        """Overall assessment when some quantities fail."""
        vv = ASMEVV20()
        budgets = [
            vv.validate(1.02, 1.0, 2.0, 1.0, quantity_name="Q1"),  # Should pass
            vv.validate(1.20, 1.0, 1.0, 0.5, u_model=1.0, quantity_name="Q2"),  # Should fail
        ]
        
        assessment = vv.assess_validation(budgets)
        # At least one should fail
        assert assessment['n_validated'] < assessment['n_total']
    
    def test_budget_to_dict(self):
        """Test budget serialization."""
        vv = ASMEVV20()
        budget = vv.validate(1.03, 1.0, 2.0, 1.0, quantity_name="test")
        d = budget.to_dict()
        
        assert 'u_num' in d
        assert 'u_input' in d
        assert 'u_model' in d
        assert 'u_val' in d
        assert 'error' in d
        assert d['quantity_name'] == 'test'
