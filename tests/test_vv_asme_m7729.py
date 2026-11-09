"""Tests for the high-level ASME V&V 20 validation uncertainty engine.

Verifies the RSS combination of ``U_num`` / ``U_input`` / ``U_model`` into
``U_val`` and the ``|E| <= U_val`` acceptance test.
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv import (
    ASMEVV20,
    UncertaintyBudget,
)


class TestASMEVV20Numerical:
    """Numerical uncertainty from GCI."""

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
    """Input parameter uncertainty."""

    def test_input_uncertainty_single_param(self):
        """Single parameter with absolute uncertainty."""
        vv = ASMEVV20()
        params = {
            "velocity": {"value": 10.0, "uncertainty": 0.1}  # 0.1 m/s on 10 m/s = 1%
        }
        u_input = vv.compute_input_uncertainty(params)
        assert abs(u_input - 1.0) < 1e-10

    def test_input_uncertainty_relative(self):
        """Parameter with relative uncertainty."""
        vv = ASMEVV20()
        params = {
            "viscosity": {"value": 1e-3, "uncertainty": 2.0, "is_relative": True}
        }
        u_input = vv.compute_input_uncertainty(params)
        assert abs(u_input - 2.0) < 1e-10

    def test_input_uncertainty_multiple_params(self):
        """Multiple parameters combined via RSS."""
        vv = ASMEVV20()
        params = {
            "velocity": {"value": 10.0, "uncertainty": 0.1},   # 1%
            "density": {"value": 1000.0, "uncertainty": 10.0},  # 1%
            "length": {"value": 1.0, "uncertainty": 0.005},     # 0.5%
        }
        u_input = vv.compute_input_uncertainty(params)
        expected = math.sqrt(1.0**2 + 1.0**2 + 0.5**2)
        assert abs(u_input - expected) < 1e-10

    def test_input_uncertainty_with_sensitivity(self):
        """Input uncertainty with sensitivity coefficients."""
        vv = ASMEVV20()
        params = {
            "velocity": {"value": 10.0, "uncertainty": 0.1},  # 1%
        }
        sensitivities = {"velocity": 2.0}  # dQ/dV = 2
        u_input = vv.compute_input_uncertainty(params, sensitivities)
        # u_input = |2.0| * 1.0% = 2.0%
        assert abs(u_input - 2.0) < 1e-10

    def test_input_uncertainty_empty(self):
        """Empty parameters should give zero uncertainty."""
        vv = ASMEVV20()
        u_input = vv.compute_input_uncertainty({})
        assert u_input == 0.0


class TestASMEVV20Model:
    """Model-form uncertainty."""

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
        expected = math.sqrt(25.0 - 4.0 - 1.0)
        assert abs(u_model - expected) < 1e-10

    def test_model_uncertainty_zero_when_covered(self):
        """U_model should be zero when U_num + U_input already cover the error."""
        vv = ASMEVV20()
        # Error = 2%, U_num = 3% -> already covered
        u_model = vv.compute_model_uncertainty(
            computed_value=1.02,
            experimental_value=1.0,
            u_num=3.0,
            u_input=0.0,
        )
        assert u_model == 0.0


class TestASMEVV20Validation:
    """Full validation workflow."""

    def test_validate_within_uncertainty(self):
        """Error within uncertainty band should validate."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.05,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            quantity_name="test_quantity",
        )
        # U_val = sqrt(2^2 + 1^2) = sqrt(5) ≈ 2.24%
        # Error = 5% -> NOT within uncertainty
        assert budget.u_val == pytest.approx(math.sqrt(5.0), rel=1e-10)
        assert budget.error == pytest.approx(5.0, rel=1e-10)
        assert not budget.is_validated()

    def test_validate_within_uncertainty_small_error(self):
        """Small error should validate."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.01,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
        )
        # Error = 1%, U_val = 2.24% -> within
        assert budget.is_validated()

    def test_validate_with_model_uncertainty(self):
        """U_model should be included in U_val."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.05,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            u_model=3.0,
        )
        # U_val = sqrt(2^2 + 3^2 + 1^2) = sqrt(14) ≈ 3.74%
        expected = math.sqrt(4.0 + 9.0 + 1.0)
        assert budget.u_val == pytest.approx(expected, rel=1e-10)

    def test_validate_returns_budget(self):
        """validate() should return an UncertaintyBudget."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.01,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
        )
        assert isinstance(budget, UncertaintyBudget)
        assert budget.quantity_name == ""
        assert budget.notes == ""

    def test_validate_with_metadata(self):
        """validate() should carry quantity name and notes."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.01,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            quantity_name="drag_coefficient",
            notes="Re=40 cylinder",
        )
        assert budget.quantity_name == "drag_coefficient"
        assert budget.notes == "Re=40 cylinder"

    def test_budget_to_dict(self):
        """UncertaintyBudget should serialise to a JSON-friendly dict."""
        vv = ASMEVV20()
        budget = vv.validate(
            computed_value=1.01,
            experimental_value=1.0,
            u_num=2.0,
            u_input=1.0,
            quantity_name="drag",
        )
        d = budget.to_dict()
        assert d["u_num"] == 2.0
        assert d["u_input"] == 1.0
        assert d["u_val"] == budget.u_val
        assert d["quantity_name"] == "drag"
        assert d["error"] == budget.error
        assert d["error_within_uncertainty"] == budget.error_within_uncertainty


class TestASMEVV20EdgeCases:
    """Edge cases and validation."""

    def test_invalid_max_uncertainty(self):
        """Max acceptable uncertainty must be positive."""
        with pytest.raises(ValueError):
            ASMEVV20(max_acceptable_uncertainty=0.0)

    def test_invalid_gci(self):
        """GCI must be non-negative."""
        vv = ASMEVV20()
        with pytest.raises(ValueError):
            vv.compute_numerical_uncertainty(gci_fine=-1.0)

    def test_invalid_input_param(self):
        """Input parameter must have value and uncertainty."""
        vv = ASMEVV20()
        with pytest.raises(ValueError):
            vv.compute_input_uncertainty({"bad": {}})

    def test_zero_experimental_value(self):
        """Zero experimental value should raise."""
        vv = ASMEVV20()
        with pytest.raises(ValueError):
            vv.compute_model_uncertainty(
                computed_value=1.0,
                experimental_value=0.0,
                u_num=2.0,
            )
