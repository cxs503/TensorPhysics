"""
Tests for cross-solver benchmark adapter.

Verifies:
- Registration of canonical cases and solver results
- Cross-solver comparison with deviation detection
- PASS/WARN/FAIL status determination
- Predefined canonical cases (boundary layer, cylinder, cavity)
- Report generation
"""

import pytest
import numpy as np
from tensorphysics.vv.cross_solver import (
    CrossSolverAdapter,
    ComparisonResult,
    ComparisonStatus,
    CanonicalCase,
    create_boundary_layer_case,
    create_cylinder_drag_case,
    create_lid_driven_cavity_case,
)


class TestCrossSolverBasic:
    """Basic cross-solver comparison tests."""
    
    def test_register_and_compare(self):
        """Register results from two solvers and compare."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        case = CanonicalCase(
            name="test_case",
            description="Test",
            physics="test",
            geometry={'type': 'box'},
            parameters={'Re': 100},
            reference_quantities=['drag'],
            reference_values={'drag': 1.0},
        )
        adapter.register_canonical_case(case)
        
        adapter.register_solver_result("SolverA", "test_case", {'drag': 1.00})
        adapter.register_solver_result("SolverB", "test_case", {'drag': 1.02})
        
        results = adapter.compare_case("test_case")
        assert len(results) == 1
        assert results[0].quantity_name == 'drag'
        assert results[0].status == ComparisonStatus.PASS
    
    def test_deviation_pass(self):
        """Small deviation should PASS."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=3.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
            reference_values={'q': 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q': 1.0})
        adapter.register_solver_result("B", "test", {'q': 1.02})  # 2% deviation
        
        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.PASS
    
    def test_deviation_warn(self):
        """Medium deviation should WARN."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=3.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
            reference_values={'q': 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q': 1.0})
        adapter.register_solver_result("B", "test", {'q': 1.04})  # 4% deviation
        
        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.WARN
    
    def test_deviation_fail(self):
        """Large deviation should FAIL."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
            reference_values={'q': 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q': 1.0})
        adapter.register_solver_result("B", "test", {'q': 1.10})  # 10% deviation
        
        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.FAIL
    
    def test_three_solver_comparison(self):
        """Compare three solvers."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=2.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
            reference_values={'q': 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("LBM", "test", {'q': 1.00})
        adapter.register_solver_result("FEM", "test", {'q': 1.01})  # 1%
        adapter.register_solver_result("DEM", "test", {'q': 1.015})  # 1.5%
        
        results = adapter.compare_case("test")
        assert len(results) == 1
        assert results[0].max_deviation < 5.0
        assert results[0].status == ComparisonStatus.PASS
    
    def test_pairwise_deviations(self):
        """Check pairwise deviation computation."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
            reference_values={'q': 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q': 1.0})
        adapter.register_solver_result("B", "test", {'q': 1.05})
        
        results = adapter.compare_case("test")
        assert 'A_vs_B' in results[0].pairwise_deviations
        assert abs(results[0].pairwise_deviations['A_vs_B'] - 5.0) < 0.1


class TestCrossSolverErrors:
    """Tests for error handling."""
    
    def test_compare_unregistered_case(self):
        """Comparing unregistered case should raise."""
        adapter = CrossSolverAdapter()
        with pytest.raises(ValueError, match="No results"):
            adapter.compare_case("nonexistent")
    
    def test_compare_insufficient_solvers(self):
        """Need at least 2 solvers for comparison."""
        adapter = CrossSolverAdapter()
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q'],
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q': 1.0})
        
        with pytest.raises(ValueError, match="at least 2"):
            adapter.compare_case("test")


class TestCrossSolverCanonicalCases:
    """Tests for predefined canonical cases."""
    
    def test_boundary_layer_case(self):
        """Test flat plate boundary layer canonical case."""
        case = create_boundary_layer_case()
        assert case.name == "flat_plate_bl"
        assert 'boundary_layer_thickness' in case.reference_quantities
        assert case.reference_values is not None
        assert case.reference_values['boundary_layer_thickness'] > 0
    
    def test_cylinder_drag_case(self):
        """Test cylinder drag canonical case."""
        case = create_cylinder_drag_case()
        assert case.name == "cylinder_drag_Re40"
        assert 'drag_coefficient' in case.reference_quantities
        assert case.reference_values['drag_coefficient'] > 0
    
    def test_lid_driven_cavity_case(self):
        """Test lid-driven cavity canonical case."""
        case = create_lid_driven_cavity_case()
        assert case.name == "lid_driven_cavity_Re100"
        assert 'center_u_velocity' in case.reference_quantities


class TestCrossSolverReport:
    """Tests for report generation."""
    
    def test_generate_report(self):
        """Generate a complete comparison report."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=2.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q1', 'q2'],
            reference_values={'q1': 1.0, 'q2': 2.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q1': 1.01, 'q2': 2.01})
        adapter.register_solver_result("B", "test", {'q1': 1.015, 'q2': 2.015})
        
        report = adapter.generate_report()
        
        assert 'overall_status' in report
        assert 'n_cases' in report
        assert 'n_quantities' in report
        assert report['n_cases'] == 1
        assert report['n_quantities'] == 2
        assert report['overall_status'] == 'PASS'
    
    def test_report_fail_overall(self):
        """Report should show FAIL if any quantity fails."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0)
        
        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=['q1', 'q2'],
            reference_values={'q1': 1.0, 'q2': 2.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {'q1': 1.01, 'q2': 2.0})
        adapter.register_solver_result("B", "test", {'q1': 1.03, 'q2': 2.5})  # 25% on q2
        
        report = adapter.generate_report()
        assert report['overall_status'] == 'FAIL'
        assert report['n_fail'] >= 1
    
    def test_comparison_result_to_dict(self):
        """Test ComparisonResult serialization."""
        result = ComparisonResult(
            quantity_name="test",
            solver_values={"A": 1.0, "B": 1.05},
            reference_value=1.0,
            mean_value=1.025,
            std_value=0.025,
            max_deviation=5.0,
            max_deviation_solver="B",
            status=ComparisonStatus.WARN,
        )
        d = result.to_dict()
        assert d['quantity_name'] == 'test'
        assert d['status'] == 'WARN'
        assert 'solver_values' in d
