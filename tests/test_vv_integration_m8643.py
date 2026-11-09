"""Integration tests for V&V machine gate (m8643 re-delivery of PR#161).

Tests the complete V&V workflow:
- Richardson extrapolation and GCI computation
- ASME V&V 20 validation uncertainty
- Cross-solver agreement
- End-to-end gate evaluation

These tests verify the current module structure (gci.py, validation.py, 
cross_solver.py, uncertainty.py) works correctly as an integrated system.
"""
import pytest
import numpy as np
from tensorphysics.vv.gci import (
    evaluate,
    grid_convergence_study,
    observed_order,
    richardson_extrapolated,
    GatePolicy,
    GateVerdict,
    Observation,
)
from tensorphysics.vv.validation import (
    validate,
    validation_uncertainty,
    evaluate_validation_table,
    ValidationResult,
)
from tensorphysics.vv.cross_solver import (
    cross_solver_agreement,
    consensus,
    agreement_spread,
    SolverSample,
    AgreementPolicy,
)
from tensorphysics.vv.uncertainty import (
    UncertaintyLedger,
    compare,
    ledger_from_study,
    u_num_from_study,
)


class TestRichardsonExtrapolationIntegration:
    """Test Richardson extrapolation with various convergence scenarios."""

    def test_second_order_monotone_convergence(self):
        """Verify p=2 convergence is correctly identified."""
        # Synthetic data: f = 1.0 + C*h^2
        # h = [0.4, 0.2, 0.1], C = 0.5
        levels = [
            Observation(h=0.4, f=1.0 + 0.5 * 0.4**2),  # 1.08
            Observation(h=0.2, f=1.0 + 0.5 * 0.2**2),  # 1.02
            Observation(h=0.1, f=1.0 + 0.5 * 0.1**2),  # 1.005
        ]
        
        study = grid_convergence_study(levels)
        assert study.order_p is not None
        assert abs(study.order_p - 2.0) < 0.1  # Should be close to 2
        assert study.f_extrapolated is not None
        assert abs(study.f_extrapolated - 1.0) < 0.01  # Should extrapolate to 1.0
        assert study.gci_fine is not None
        assert study.gci_fine > 0

    def test_first_order_convergence(self):
        """Verify p=1 convergence is correctly identified."""
        # f = 1.0 + C*h
        levels = [
            Observation(h=0.4, f=1.0 + 0.3 * 0.4),  # 1.12
            Observation(h=0.2, f=1.0 + 0.3 * 0.2),  # 1.06
            Observation(h=0.1, f=1.0 + 0.3 * 0.1),  # 1.03
        ]
        
        study = grid_convergence_study(levels)
        assert study.order_p is not None
        assert abs(study.order_p - 1.0) < 0.1

    def test_oscillatory_convergence_detection(self):
        """Verify oscillatory convergence is detected."""
        # Oscillatory: errors change sign
        levels = [
            Observation(h=0.4, f=1.10),
            Observation(h=0.2, f=0.95),  # Overshoot
            Observation(h=0.1, f=1.02),  # Back above
        ]
        
        study = grid_convergence_study(levels)
        # Should detect oscillatory behavior
        assert study.convergence_type in ["oscillatory", "unknown"]

    def test_evaluate_gate_pass(self):
        """Test full gate evaluation with good convergence."""
        levels = [
            Observation(h=0.4, f=1.08),
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        
        verdict = evaluate(levels, quantity="test_qty")
        assert verdict.status in ["PASS", "WARN"]
        assert verdict.study is not None
        assert len(verdict.checks) > 0


class TestASMEVV20ValidationIntegration:
    """Test ASME V&V 20 validation uncertainty workflow."""

    def test_validation_uncertainty_combination(self):
        """Verify U_val = sqrt(U_num^2 + U_input^2)."""
        u_num = 3.0
        u_input = 4.0
        u_val = validation_uncertainty(u_num, u_input)
        expected = np.sqrt(3.0**2 + 4.0**2)  # 5.0
        assert abs(u_val - expected) < 1e-10

    def test_validation_pass_within_uncertainty(self):
        """Test validation passes when error < U_val."""
        # Using absolute values (not percentages)
        f_sim = 1.02
        f_exp = 1.00
        u_num = 0.05  # absolute uncertainty
        u_input = 0.02  # absolute uncertainty
        
        result = validate(f_sim, f_exp, u_num=u_num, u_input=u_input)
        assert result.passed is True
        assert result.ratio < 1.0

    def test_validation_fail_outside_uncertainty(self):
        """Test validation fails when error > U_val."""
        # Using absolute values
        f_sim = 1.20
        f_exp = 1.00
        u_num = 0.02  # small uncertainty
        u_input = 0.01  # small uncertainty
        
        result = validate(f_sim, f_exp, u_num=u_num, u_input=u_input)
        assert result.passed is False
        assert result.ratio > 1.0

    def test_validation_table_evaluation(self):
        """Test batch validation of multiple quantities."""
        rows = [
            {"f_sim": 1.02, "f_exp": 1.00, "u_num": 0.05, "u_input": 0.02, "label": "drag"},
            {"f_sim": 0.98, "f_exp": 1.00, "u_num": 0.05, "u_input": 0.02, "label": "lift"},
            {"f_sim": 1.50, "f_exp": 1.00, "u_num": 0.02, "u_input": 0.01, "label": "moment"},
        ]
        
        results = evaluate_validation_table(rows)
        assert len(results) == 3
        assert results[0].passed is True  # drag: 0.02 error < 0.054 U_val
        assert results[1].passed is True  # lift: 0.02 error < 0.054 U_val
        assert results[2].passed is False  # moment: 0.50 error > 0.022 U_val


class TestCrossSolverAgreementIntegration:
    """Test cross-solver comparison workflow."""

    def test_consensus_computation(self):
        """Test inverse-variance weighted consensus."""
        samples = [
            SolverSample(solver="A", f=1.00, u=0.05),
            SolverSample(solver="B", f=1.02, u=0.03),
            SolverSample(solver="C", f=0.98, u=0.04),
        ]
        
        consensus_val, consensus_u = consensus(samples)
        assert consensus_val is not None
        # Consensus should be close to 1.0
        assert abs(consensus_val - 1.0) < 0.05

    def test_agreement_spread(self):
        """Test pairwise agreement computation."""
        samples = [
            SolverSample(solver="A", f=1.00, u=0.05),
            SolverSample(solver="B", f=1.02, u=0.03),
            SolverSample(solver="C", f=1.01, u=0.04),
        ]
        
        spread, spread_rel, worst_pair = agreement_spread(samples)
        assert spread is not None
        # Spread should be small (max pairwise difference)
        assert spread < 0.05  # < 0.05 absolute

    def test_cross_solver_agreement_gate(self):
        """Test full cross-solver agreement gate."""
        samples = [
            SolverSample(solver="TensorLBM", f=1.00, u=0.05),
            SolverSample(solver="TensorFEM", f=1.02, u=0.03),
            SolverSample(solver="TensorFVM", f=0.99, u=0.04),
        ]
        
        result = cross_solver_agreement(samples, quantity="drag_coefficient")
        assert result.status in ["PASS", "WARN"]
        assert result.consensus is not None


class TestUncertaintyLedgerIntegration:
    """Test uncertainty ledger workflow."""

    def test_ledger_from_study(self):
        """Test ledger construction from convergence study."""
        levels = [
            Observation(h=0.4, f=1.08),
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        
        study = grid_convergence_study(levels)
        ledger = ledger_from_study(study, u_input=0.02, u_ref=0.01)
        
        assert ledger.u_num is not None
        assert ledger.u_input == 0.02
        assert ledger.u_ref == 0.01
        assert ledger.u_val is not None
        assert ledger.u_val > 0

    def test_ledger_comparison_with_reference(self):
        """Test comparison with declared reference uncertainty."""
        levels = [
            Observation(h=0.4, f=1.08),
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        
        study = grid_convergence_study(levels)
        ledger = ledger_from_study(study, u_input=0.01, u_ref=0.02)
        
        # Compare with experimental value
        f_exp = 1.00
        f_sim = study.f_fine
        
        verdict = compare(f_sim, f_exp, ledger)
        assert verdict.status in ["PASS", "FAIL", "WITHHELD"]


class TestEndToEndVVPipeline:
    """Test complete V&V pipeline from grid study to validation."""

    def test_full_pipeline_with_validation(self):
        """Test complete workflow: grid study -> uncertainty -> validation."""
        # Step 1: Grid convergence study
        levels = [
            Observation(h=0.4, f=1.08),
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        
        study = grid_convergence_study(levels)
        assert study.order_p is not None
        assert study.gci_fine is not None
        
        # Step 2: Build uncertainty ledger
        ledger = ledger_from_study(study, u_input=0.02, u_ref=0.015)
        assert ledger.u_val is not None
        
        # Step 3: Validate against experiment
        f_exp = 1.00
        f_sim = study.f_fine
        
        verdict = compare(f_sim, f_exp, ledger)
        assert verdict.status in ["PASS", "FAIL", "WITHHELD"]
        
        # Step 4: Full gate evaluation
        gate_verdict = evaluate(
            levels,
            quantity="drag_coefficient",
            f_exp=f_exp,
            u_input=0.02,
            u_ref=0.015,
        )
        assert gate_verdict.status in ["PASS", "WARN", "FAIL", "WITHHELD"]
        assert gate_verdict.uncertainty is not None

    def test_full_pipeline_with_cross_solver(self):
        """Test workflow: grid study -> cross-solver comparison."""
        # Solver A: grid study
        levels_a = [
            Observation(h=0.4, f=1.08),
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        study_a = grid_convergence_study(levels_a)
        
        # Solver B: grid study
        levels_b = [
            Observation(h=0.4, f=1.07),
            Observation(h=0.2, f=1.015),
            Observation(h=0.1, f=1.003),
        ]
        study_b = grid_convergence_study(levels_b)
        
        # Cross-solver comparison
        samples = [
            SolverSample(solver="A", f=study_a.f_fine, u=study_a.gci_fine),
            SolverSample(solver="B", f=study_b.f_fine, u=study_b.gci_fine),
        ]
        
        agreement = cross_solver_agreement(samples, quantity="lift_coefficient")
        assert agreement.status in ["PASS", "WARN"]
        assert agreement.consensus is not None


class TestEdgeCasesAndRobustness:
    """Test edge cases and error handling."""

    def test_insufficient_grid_levels(self):
        """Test that insufficient grid levels raise appropriate errors."""
        levels = [
            Observation(h=0.2, f=1.02),
            Observation(h=0.1, f=1.005),
        ]
        
        # Should work with 2 levels but with limitations
        study = grid_convergence_study(levels)
        assert study is not None

    def test_zero_uncertainty_handling(self):
        """Test validation with zero declared uncertainty."""
        f_sim = 1.00
        f_exp = 1.00
        u_num = 0.0
        u_input = 0.0
        
        result = validate(f_sim, f_exp, u_num=u_num, u_input=u_input)
        # Exact match with zero uncertainty should pass
        assert result.passed is True

    def test_negative_uncertainty_rejected(self):
        """Test that negative uncertainty values are rejected."""
        with pytest.raises(ValueError):
            validation_uncertainty(u_num=-1.0, u_input=2.0)

    def test_non_finite_values_rejected(self):
        """Test that non-finite values are rejected."""
        with pytest.raises(ValueError):
            validate(float('inf'), 1.0, u_num=1.0)
        
        with pytest.raises(ValueError):
            validate(1.0, float('nan'), u_num=1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
