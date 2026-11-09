"""
Integration test for V&V machine gate.

Demonstrates complete workflow:
1. Richardson extrapolation for grid convergence
2. ASME V&V 20 uncertainty quantification
3. Cross-solver comparison
4. Report generation
"""

import pytest
import numpy as np
from tensorphysics.vv import (
    RichardsonExtrapolation,
    ASMEVV20,
    CrossSolverAdapter,
    create_boundary_layer_case,
    create_cylinder_drag_case,
)


class TestVVIntegration:
    """Integration tests for complete V&V workflow."""
    
    def test_complete_vv_workflow_boundary_layer(self):
        """Test complete V&V workflow for flat plate boundary layer."""
        # Step 1: Grid convergence study
        # Blasius solution: delta ~ 4.64/sqrt(Re) * L
        # Simulate solutions at different grid resolutions
        delta_exact = 0.00464  # Reference value
        
        # Simulated solutions with second-order convergence
        C = 1e-5  # Error coefficient
        delta_coarse = delta_exact + C * 16  # h=4
        delta_medium = delta_exact + C * 4   # h=2
        delta_fine = delta_exact + C * 1     # h=1
        
        re = RichardsonExtrapolation()
        gci_result = re.compute(delta_fine, delta_medium, delta_coarse)
        
        assert gci_result.status.value == "PASS"
        assert abs(gci_result.p - 2.0) < 0.1
        assert gci_result.gci_fine < 5.0  # Should be small
        
        # Step 2: ASME V&V 20 uncertainty quantification
        vv = ASMEVV20()
        
        # Numerical uncertainty from GCI
        u_num = vv.compute_numerical_uncertainty(gci_result.gci_fine)
        
        # Input uncertainties (Reynolds number, velocity, etc.)
        input_params = {
            'Re_L': {'value': 1e5, 'uncertainty': 1e3},  # 1% uncertainty
            'U_inf': {'value': 1.0, 'uncertainty': 0.01},  # 1% uncertainty
        }
        u_input = vv.compute_input_uncertainty(input_params)
        
        # Validate against reference
        budget = vv.validate(
            computed_value=delta_fine,
            experimental_value=delta_exact,
            u_num=u_num,
            u_input=u_input,
            quantity_name="boundary_layer_thickness",
        )
        
        assert budget.u_val < 10.0  # Total uncertainty should be acceptable
        assert budget.is_validated() == True
        
        # Step 3: Cross-solver comparison
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        case = create_boundary_layer_case()
        adapter.register_canonical_case(case)
        
        # Simulate results from different solvers
        adapter.register_solver_result(
            "TensorLBM", "flat_plate_bl",
            {'boundary_layer_thickness': delta_fine * 1.00}
        )
        adapter.register_solver_result(
            "TensorFEM", "flat_plate_bl",
            {'boundary_layer_thickness': delta_fine * 1.02}
        )
        adapter.register_solver_result(
            "TensorFVM", "flat_plate_bl",
            {'boundary_layer_thickness': delta_fine * 1.01}
        )
        
        results = adapter.compare_case("flat_plate_bl")
        assert len(results) >= 1
        assert results[0].max_deviation < 5.0
        assert results[0].status.value == "PASS"
        
        # Step 4: Generate report
        report = adapter.generate_report()
        assert report['overall_status'] == 'PASS'
        assert report['n_quantities'] >= 1
    
    def test_complete_vv_workflow_cylinder_drag(self):
        """Test complete V&V workflow for cylinder drag."""
        # Reference drag coefficient at Re=40
        cd_exact = 1.54
        
        # Simulated solutions
        C = 0.01
        cd_coarse = cd_exact + C * 16
        cd_medium = cd_exact + C * 4
        cd_fine = cd_exact + C * 1
        
        # Grid convergence
        re = RichardsonExtrapolation()
        gci_result = re.compute(cd_fine, cd_medium, cd_coarse)
        
        assert gci_result.p > 1.5  # Should be around 2
        
        # Uncertainty quantification
        vv = ASMEVV20()
        u_num = vv.compute_numerical_uncertainty(gci_result.gci_fine)
        
        input_params = {
            'Re_D': {'value': 40.0, 'uncertainty': 0.4},  # 1%
        }
        u_input = vv.compute_input_uncertainty(input_params)
        
        budget = vv.validate(
            computed_value=cd_fine,
            experimental_value=cd_exact,
            u_num=u_num,
            u_input=u_input,
            quantity_name="drag_coefficient",
        )
        
        assert budget.u_val < 10.0
        
        # Cross-solver comparison
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        case = create_cylinder_drag_case()
        adapter.register_canonical_case(case)
        
        adapter.register_solver_result("TensorLBM", "cylinder_drag_Re40",
                                       {'drag_coefficient': cd_fine * 1.00})
        adapter.register_solver_result("TensorFEM", "cylinder_drag_Re40",
                                       {'drag_coefficient': cd_fine * 1.03})
        
        results = adapter.compare_case("cylinder_drag_Re40")
        assert results[0].max_deviation < 5.0
    
    def test_vv_machine_gate_all_pass(self):
        """Test that V&V machine gate passes for well-converged case."""
        # Create a scenario where everything should pass
        re = RichardsonExtrapolation(gci_threshold=5.0)
        vv = ASMEVV20(max_acceptable_uncertainty=10.0)
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        # Perfect convergence
        f_exact = 1.0
        C = 0.001
        solutions = [f_exact + C * 16, f_exact + C * 4, f_exact + C * 1]
        
        # Grid convergence: PASS
        gci_result = re.compute(solutions[2], solutions[1], solutions[0])
        assert gci_result.status.value == "PASS"
        
        # Uncertainty: acceptable
        u_num = vv.compute_numerical_uncertainty(gci_result.gci_fine)
        u_input = 0.5
        u_model = 1.0
        u_val = vv.compute_total_uncertainty(u_num, u_input, u_model)
        assert u_val < 10.0
        
        # Cross-solver: PASS
        case = create_boundary_layer_case()
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "flat_plate_bl",
                                       {'boundary_layer_thickness': 0.00464})
        adapter.register_solver_result("B", "flat_plate_bl",
                                       {'boundary_layer_thickness': 0.00470})
        
        results = adapter.compare_case("flat_plate_bl")
        assert results[0].status.value == "PASS"
    
    def test_vv_machine_gate_detects_failure(self):
        """Test that V&V machine gate detects convergence failure."""
        re = RichardsonExtrapolation(gci_threshold=5.0)
        
        # Poor convergence (large errors)
        f_exact = 1.0
        C = 0.5  # Large error coefficient
        solutions = [f_exact + C * 16, f_exact + C * 4, f_exact + C * 1]
        
        gci_result = re.compute(solutions[2], solutions[1], solutions[0])
        
        # Should detect failure
        assert gci_result.status.value == "FAIL"
        assert gci_result.gci_fine > 5.0
    
    def test_vv_machine_gate_detects_oscillatory(self):
        """Test that V&V machine gate detects oscillatory convergence."""
        re = RichardsonExtrapolation()
        
        # Oscillatory convergence
        f_coarse = 1.0
        f_medium = 1.2
        f_fine = 0.9  # Overshoots
        
        gci_result = re.compute(f_fine, f_medium, f_coarse)
        
        assert gci_result.status.value == "UNRESOLVED"


class TestVVMultipleQuantities:
    """Test V&V for multiple quantities simultaneously."""
    
    def test_multiple_quantity_validation(self):
        """Validate multiple quantities in one workflow."""
        vv = ASMEVV20()
        
        quantities = {
            'drag': {'computed': 1.05, 'reference': 1.0, 'u_num': 2.0, 'u_input': 1.0},
            'lift': {'computed': 0.52, 'reference': 0.5, 'u_num': 1.5, 'u_input': 0.5},
            'moment': {'computed': 0.105, 'reference': 0.1, 'u_num': 3.0, 'u_input': 1.0},
        }
        
        budgets = []
        for name, data in quantities.items():
            budget = vv.validate(
                computed_value=data['computed'],
                experimental_value=data['reference'],
                u_num=data['u_num'],
                u_input=data['u_input'],
                quantity_name=name,
            )
            budgets.append(budget)
        
        # Assess overall validation
        assessment = vv.assess_validation(budgets)
        
        assert assessment['n_total'] == 3
        assert assessment['n_validated'] >= 2  # At least some should pass
        assert 'details' in assessment
