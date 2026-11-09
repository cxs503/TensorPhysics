"""Integration test for the high-level V&V machine gate.

Demonstrates the complete workflow:
1. Richardson extrapolation for grid convergence
2. ASME V&V 20 uncertainty quantification
3. Cross-solver comparison on canonical cases
4. Report-template alignment
"""
from __future__ import annotations

import math

import pytest

from tensorphysics.vv import (
    ASMEVV20,
    ComparisonStatus,
    CrossSolverAdapter,
    RichardsonExtrapolation,
    create_boundary_layer_case,
    create_cylinder_drag_case,
    create_lid_driven_cavity_case,
)


class TestVVIntegration:
    """Integration tests for the complete V&V workflow."""

    def test_complete_vv_workflow_boundary_layer(self):
        """Complete V&V workflow for flat-plate boundary layer."""
        # Step 1: Grid convergence study.
        # Blasius: delta ~ 4.64 / sqrt(Re_L) * L
        delta_exact = 4.64e-3  # Reference value

        # Simulated solutions with second-order convergence.
        C = 1e-5  # Error coefficient
        delta_coarse = delta_exact + C * 16  # h=4
        delta_medium = delta_exact + C * 4   # h=2
        delta_fine = delta_exact + C * 1     # h=1

        re = RichardsonExtrapolation()
        gci_result = re.compute(delta_fine, delta_medium, delta_coarse)

        assert gci_result.status.value == "PASS"
        assert abs(gci_result.p - 2.0) < 0.1
        assert gci_result.gci_fine < 5.0  # Should be small

        # Step 2: ASME V&V 20 uncertainty quantification.
        vv = ASMEVV20()

        # Numerical uncertainty from GCI.
        u_num = vv.compute_numerical_uncertainty(gci_result.gci_fine)

        # Input uncertainties (Reynolds number, velocity, etc.).
        input_params = {
            "Re_L": {"value": 1e5, "uncertainty": 1e3},   # 1% uncertainty
            "U_inf": {"value": 1.0, "uncertainty": 0.01},  # 1% uncertainty
        }
        u_input = vv.compute_input_uncertainty(input_params)

        # Validate against reference.
        budget = vv.validate(
            computed_value=delta_fine,
            experimental_value=delta_exact,
            u_num=u_num,
            u_input=u_input,
            quantity_name="boundary_layer_thickness",
        )

        assert budget.u_val < 10.0  # Total uncertainty should be acceptable
        assert budget.is_validated()

        # Step 3: Cross-solver comparison.
        adapter = CrossSolverAdapter(deviation_threshold=5.0)

        case = create_boundary_layer_case()
        adapter.register_canonical_case(case)

        # Simulate results from different solvers.
        adapter.register_solver_result(
            "TensorLBM", "flat_plate_bl",
            {"boundary_layer_thickness": delta_fine * 1.00},
        )
        adapter.register_solver_result(
            "TensorFEM", "flat_plate_bl",
            {"boundary_layer_thickness": delta_fine * 1.02},
        )

        results = adapter.compare_case("flat_plate_bl")
        assert len(results) == 1
        assert results[0].status == ComparisonStatus.PASS

    def test_complete_vv_workflow_cylinder_drag(self):
        """Complete V&V workflow for cylinder drag at Re=40."""
        # Step 1: Grid convergence study.
        # Reference: C_D = 1.54 (Tritton 1959)
        cd_exact = 1.54

        # Simulated solutions with second-order convergence.
        C = 0.01
        cd_coarse = cd_exact + C * 16
        cd_medium = cd_exact + C * 4
        cd_fine = cd_exact + C * 1

        re = RichardsonExtrapolation()
        gci_result = re.compute(cd_fine, cd_medium, cd_coarse)

        assert gci_result.status.value == "PASS"
        assert abs(gci_result.p - 2.0) < 0.1

        # Step 2: ASME V&V 20.
        vv = ASMEVV20()
        u_num = vv.compute_numerical_uncertainty(gci_result.gci_fine)

        budget = vv.validate(
            computed_value=cd_fine,
            experimental_value=cd_exact,
            u_num=u_num,
            quantity_name="drag_coefficient",
        )
        assert budget.is_validated()

        # Step 3: Cross-solver.
        adapter = CrossSolverAdapter()
        adapter.register_canonical_case(create_cylinder_drag_case())
        adapter.register_solver_result("TensorLBM", "cylinder_drag_re40", {"drag_coefficient": 1.54})
        adapter.register_solver_result("TensorFEM", "cylinder_drag_re40", {"drag_coefficient": 1.56})

        results = adapter.compare_case("cylinder_drag_re40")
        assert results[0].status == ComparisonStatus.PASS

    def test_complete_vv_workflow_cavity(self):
        """Complete V&V workflow for lid-driven cavity at Re=100."""
        # Reference: u_center = 0.0547 (Ghia et al. 1982)
        u_exact = 0.0547

        # Simulated solutions with second-order convergence.
        C = 1e-4
        u_coarse = u_exact + C * 16
        u_medium = u_exact + C * 4
        u_fine = u_exact + C * 1

        re = RichardsonExtrapolation()
        gci_result = re.compute(u_fine, u_medium, u_coarse)

        assert gci_result.status.value == "PASS"

        # Cross-solver comparison.
        adapter = CrossSolverAdapter()
        adapter.register_canonical_case(create_lid_driven_cavity_case())
        adapter.register_solver_result("TensorLBM", "lid_driven_cavity_re100", {"centerline_u": 0.0547})
        adapter.register_solver_result("TensorFVM", "lid_driven_cavity_re100", {"centerline_u": 0.0550})

        results = adapter.compare_case("lid_driven_cavity_re100")
        assert results[0].status == ComparisonStatus.PASS

    def test_high_level_api_consistency(self):
        """High-level API should be consistent with the functional gate."""
        # The high-level RichardsonExtrapolation should agree with the
        # functional gci.evaluate on the same three-grid study.
        from tensorphysics.vv import Observation, evaluate

        C = 0.01
        f_exact = 1.0
        f_coarse = f_exact + C * 16
        f_medium = f_exact + C * 4
        f_fine = f_exact + C * 1

        # High-level.
        re = RichardsonExtrapolation()
        hl = re.compute(f_fine, f_medium, f_coarse, r_fine=2.0, r_coarse=2.0)

        # Functional.
        levels = (
            Observation(h=1.0, f=f_coarse),
            Observation(h=0.5, f=f_medium),
            Observation(h=0.25, f=f_fine),
        )
        verdict = evaluate(levels, quantity="f")

        # Both should recover p=2 and f_ext=1.0.
        assert abs(hl.p - 2.0) < 1e-6
        assert abs(verdict.study.order_p - 2.0) < 1e-6
        assert abs(hl.f_ext - 1.0) < 1e-6
        assert abs(verdict.study.f_extrapolated - 1.0) < 1e-6
