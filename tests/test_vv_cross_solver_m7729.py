"""Tests for the high-level cross-solver benchmark adapter.

Verifies registration of canonical cases, per-solver result comparison,
PASS/WARN/FAIL status determination and the predefined canonical cases
(boundary layer, cylinder drag, lid-driven cavity).
"""
from __future__ import annotations

import pytest

from tensorphysics.vv import (
    CanonicalCase,
    ComparisonResult,
    ComparisonStatus,
    CrossSolverAdapter,
    create_boundary_layer_case,
    create_cylinder_drag_case,
    create_lid_driven_cavity_case,
)


class TestCrossSolverBasic:
    """Basic cross-solver comparison."""

    def test_register_and_compare(self):
        """Register results from two solvers and compare."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0)

        case = CanonicalCase(
            name="test_case",
            description="Test",
            physics="test",
            geometry={"type": "box"},
            parameters={"Re": 100},
            reference_quantities=["drag"],
            reference_values={"drag": 1.0},
        )
        adapter.register_canonical_case(case)

        adapter.register_solver_result("SolverA", "test_case", {"drag": 1.00})
        adapter.register_solver_result("SolverB", "test_case", {"drag": 1.02})

        results = adapter.compare_case("test_case")
        assert len(results) == 1
        assert results[0].quantity_name == "drag"
        assert results[0].status == ComparisonStatus.PASS

    def test_deviation_pass(self):
        """Small deviation should PASS."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=3.0)

        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=["q"],
            reference_values={"q": 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {"q": 1.0})
        adapter.register_solver_result("B", "test", {"q": 1.02})  # 2% deviation

        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.PASS

    def test_deviation_warn(self):
        """Medium deviation should WARN."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=3.0)

        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=["q"],
            reference_values={"q": 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {"q": 1.0})
        adapter.register_solver_result("B", "test", {"q": 1.04})  # 4% deviation

        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.WARN

    def test_deviation_fail(self):
        """Large deviation should FAIL."""
        adapter = CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=3.0)

        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=["q"],
            reference_values={"q": 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {"q": 1.0})
        adapter.register_solver_result("B", "test", {"q": 1.10})  # 10% deviation

        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.FAIL

    def test_insufficient_data(self):
        """Single solver should give INSUFFICIENT_DATA."""
        adapter = CrossSolverAdapter()

        case = CanonicalCase(
            name="test", description="Test", physics="test",
            geometry={}, parameters={}, reference_quantities=["q"],
            reference_values={"q": 1.0},
        )
        adapter.register_canonical_case(case)
        adapter.register_solver_result("A", "test", {"q": 1.0})

        results = adapter.compare_case("test")
        assert results[0].status == ComparisonStatus.INSUFFICIENT_DATA


class TestCanonicalCases:
    """Predefined canonical cases."""

    def test_boundary_layer_case(self):
        """Flat-plate boundary layer case should have correct reference."""
        case = create_boundary_layer_case()
        assert case.name == "flat_plate_bl"
        assert case.physics == "laminar boundary layer"
        assert "boundary_layer_thickness" in case.reference_quantities
        assert case.reference_values["boundary_layer_thickness"] == pytest.approx(4.64e-3)

    def test_cylinder_drag_case(self):
        """Cylinder drag case should have correct reference."""
        case = create_cylinder_drag_case()
        assert case.name == "cylinder_drag_re40"
        assert case.physics == "cylinder drag"
        assert "drag_coefficient" in case.reference_quantities
        assert case.reference_values["drag_coefficient"] == pytest.approx(1.54)

    def test_lid_driven_cavity_case(self):
        """Lid-driven cavity case should have correct reference."""
        case = create_lid_driven_cavity_case()
        assert case.name == "lid_driven_cavity_re100"
        assert case.physics == "incompressible cavity flow"
        assert "centerline_u" in case.reference_quantities
        assert case.reference_values["centerline_u"] == pytest.approx(0.0547)

    def test_cases_are_canonical(self):
        """All factory cases should be CanonicalCase instances."""
        for factory in (
            create_boundary_layer_case,
            create_cylinder_drag_case,
            create_lid_driven_cavity_case,
        ):
            assert isinstance(factory(), CanonicalCase)


class TestCrossSolverAdapter:
    """CrossSolverAdapter behaviour."""

    def test_register_unregistered_case(self):
        """Registering results for an unregistered case should raise."""
        adapter = CrossSolverAdapter()
        with pytest.raises(ValueError):
            adapter.register_solver_result("A", "nonexistent", {"q": 1.0})

    def test_compare_unregistered_case(self):
        """Comparing an unregistered case should raise."""
        adapter = CrossSolverAdapter()
        with pytest.raises(ValueError):
            adapter.compare_case("nonexistent")

    def test_compare_all(self):
        """compare_all() should return results for every registered case."""
        adapter = CrossSolverAdapter()
        adapter.register_canonical_case(create_boundary_layer_case())
        adapter.register_canonical_case(create_cylinder_drag_case())

        adapter.register_solver_result("A", "flat_plate_bl", {"boundary_layer_thickness": 4.64e-3})
        adapter.register_solver_result("B", "flat_plate_bl", {"boundary_layer_thickness": 4.70e-3})
        adapter.register_solver_result("A", "cylinder_drag_re40", {"drag_coefficient": 1.54})
        adapter.register_solver_result("B", "cylinder_drag_re40", {"drag_coefficient": 1.60})

        all_results = adapter.compare_all()
        assert set(all_results.keys()) == {"flat_plate_bl", "cylinder_drag_re40"}
        assert len(all_results["flat_plate_bl"]) == 1
        assert len(all_results["cylinder_drag_re40"]) == 1

    def test_result_to_dict(self):
        """ComparisonResult should serialise to a JSON-friendly dict."""
        adapter = CrossSolverAdapter()
        adapter.register_canonical_case(create_boundary_layer_case())
        adapter.register_solver_result("A", "flat_plate_bl", {"boundary_layer_thickness": 4.64e-3})
        adapter.register_solver_result("B", "flat_plate_bl", {"boundary_layer_thickness": 4.70e-3})

        results = adapter.compare_case("flat_plate_bl")
        d = results[0].to_dict()
        assert d["quantity_name"] == "boundary_layer_thickness"
        assert "A" in d["solver_values"]
        assert "B" in d["solver_values"]
        assert d["status"] in ("PASS", "WARN", "FAIL", "INSUFFICIENT_DATA")

    def test_invalid_thresholds(self):
        """Thresholds must be valid."""
        with pytest.raises(ValueError):
            CrossSolverAdapter(deviation_threshold=0.0)
        with pytest.raises(ValueError):
            CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=5.0)
        with pytest.raises(ValueError):
            CrossSolverAdapter(deviation_threshold=5.0, warning_threshold=6.0)
