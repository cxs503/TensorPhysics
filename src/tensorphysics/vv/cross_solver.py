"""
Cross-Solver Benchmark Adapter.

Provides infrastructure for comparing outputs from different solvers
(TensorLBM, TensorFEM, TensorDEM) on canonical test cases.
Supports automatic deviation detection and alerting.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


class SolverType(Enum):
    """Supported solver types."""
    TENSORLBM = "TensorLBM"
    TENSORFEM = "TensorFEM"
    TENSORDEM = "TensorDEM"
    TENSORBEM = "TensorBEM"
    TENSORFVM = "TensorFVM"
    ANALYTICAL = "Analytical"
    EXPERIMENTAL = "Experimental"


class ComparisonStatus(Enum):
    """Status of cross-solver comparison."""
    PASS = "PASS"           # All deviations within threshold
    WARN = "WARN"           # Some deviations near threshold
    FAIL = "FAIL"           # Deviations exceed threshold
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


@dataclass
class ComparisonResult:
    """Result of comparing a quantity across solvers."""
    quantity_name: str
    solver_values: Dict[str, float]  # solver_name -> value
    reference_value: Optional[float] = None
    reference_source: str = ""
    
    # Statistics
    mean_value: float = 0.0
    std_value: float = 0.0
    max_deviation: float = 0.0  # Maximum deviation from reference (%)
    max_deviation_solver: str = ""
    pairwise_deviations: Dict[str, float] = field(default_factory=dict)
    
    # Assessment
    status: ComparisonStatus = ComparisonStatus.INSUFFICIENT_DATA
    threshold: float = 5.0  # Deviation threshold (%)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            'quantity_name': self.quantity_name,
            'solver_values': self.solver_values,
            'reference_value': self.reference_value,
            'reference_source': self.reference_source,
            'mean_value': self.mean_value,
            'std_value': self.std_value,
            'max_deviation': self.max_deviation,
            'max_deviation_solver': self.max_deviation_solver,
            'pairwise_deviations': self.pairwise_deviations,
            'status': self.status.value,
            'threshold': self.threshold,
        }


@dataclass
class CanonicalCase:
    """Definition of a canonical test case for cross-solver comparison."""
    name: str
    description: str
    physics: str  # e.g., "laminar boundary layer", "cylinder drag"
    geometry: Dict[str, Any]
    parameters: Dict[str, float]
    reference_quantities: List[str]  # Quantities to compare
    reference_values: Optional[Dict[str, float]] = None  # Analytical/experimental
    reference_source: str = ""


class CrossSolverAdapter:
    """
    Cross-solver benchmark adapter for comparing solver outputs.
    
    Supports comparison of results from different Tensor* solvers
    on canonical test cases with automatic deviation detection.
    
    Parameters
    ----------
    deviation_threshold : float
        Maximum acceptable deviation (%) between solvers. Default 5%.
    warning_threshold : float
        Warning threshold (%) for near-threshold deviations. Default 3%.
    """
    
    def __init__(
        self,
        deviation_threshold: float = 5.0,
        warning_threshold: float = 3.0,
    ):
        self.deviation_threshold = deviation_threshold
        self.warning_threshold = warning_threshold
        self._registered_cases: Dict[str, CanonicalCase] = {}
        self._solver_results: Dict[str, Dict[str, Dict[str, float]]] = {}
    
    def register_canonical_case(self, case: CanonicalCase) -> None:
        """Register a canonical test case."""
        self._registered_cases[case.name] = case
    
    def register_solver_result(
        self,
        solver_name: str,
        case_name: str,
        quantities: Dict[str, float],
    ) -> None:
        """
        Register results from a solver for a canonical case.
        
        Parameters
        ----------
        solver_name : str
            Name of the solver (e.g., "TensorLBM", "TensorFEM")
        case_name : str
            Name of the canonical case
        quantities : dict
            Dictionary of quantity_name -> value
        """
        if case_name not in self._solver_results:
            self._solver_results[case_name] = {}
        if solver_name not in self._solver_results[case_name]:
            self._solver_results[case_name][solver_name] = {}
        
        self._solver_results[case_name][solver_name].update(quantities)
    
    def compare_case(self, case_name: str) -> List[ComparisonResult]:
        """
        Compare solver results for a canonical case.
        
        Parameters
        ----------
        case_name : str
            Name of the canonical case
            
        Returns
        -------
        list of ComparisonResult
            Comparison results for each quantity
        """
        if case_name not in self._solver_results:
            raise ValueError(f"No results registered for case '{case_name}'")
        
        case = self._registered_cases.get(case_name)
        solver_data = self._solver_results[case_name]
        
        if len(solver_data) < 2:
            raise ValueError(f"Need at least 2 solvers for comparison, got {len(solver_data)}")
        
        results = []
        quantities = case.reference_quantities if case else list(
            next(iter(solver_data.values())).keys()
        )
        
        for qty in quantities:
            result = self._compare_quantity(
                qty, solver_data,
                reference_value=case.reference_values.get(qty) if case and case.reference_values else None,
                reference_source=case.reference_source if case else "",
            )
            results.append(result)
        
        return results
    
    def _compare_quantity(
        self,
        quantity_name: str,
        solver_data: Dict[str, Dict[str, float]],
        reference_value: Optional[float] = None,
        reference_source: str = "",
    ) -> ComparisonResult:
        """Compare a single quantity across solvers."""
        values = {}
        for solver, data in solver_data.items():
            if quantity_name in data:
                values[solver] = data[quantity_name]
        
        if len(values) < 2:
            return ComparisonResult(
                quantity_name=quantity_name,
                solver_values=values,
                reference_value=reference_value,
                reference_source=reference_source,
                status=ComparisonStatus.INSUFFICIENT_DATA,
                threshold=self.deviation_threshold,
            )
        
        # Compute statistics
        vals_array = np.array(list(values.values()))
        mean_val = np.mean(vals_array)
        std_val = np.std(vals_array)
        
        # Compute deviations from reference or mean
        max_dev = 0.0
        max_dev_solver = ""
        
        ref = reference_value if reference_value is not None else mean_val
        
        for solver, val in values.items():
            if abs(ref) > 1e-14:
                dev = 100.0 * abs(val - ref) / abs(ref)
            else:
                dev = abs(val - ref)
            
            if dev > max_dev:
                max_dev = dev
                max_dev_solver = solver
        
        # Compute pairwise deviations
        pairwise = {}
        solvers = list(values.keys())
        for i, s1 in enumerate(solvers):
            for s2 in solvers[i+1:]:
                v1, v2 = values[s1], values[s2]
                if abs(v1) > 1e-14:
                    dev = 100.0 * abs(v1 - v2) / abs(v1)
                else:
                    dev = abs(v1 - v2)
                pairwise[f"{s1}_vs_{s2}"] = dev
        
        # Determine status
        if max_dev > self.deviation_threshold:
            status = ComparisonStatus.FAIL
        elif max_dev > self.warning_threshold:
            status = ComparisonStatus.WARN
        else:
            status = ComparisonStatus.PASS
        
        return ComparisonResult(
            quantity_name=quantity_name,
            solver_values=values,
            reference_value=reference_value,
            reference_source=reference_source,
            mean_value=float(mean_val),
            std_value=float(std_val),
            max_deviation=max_dev,
            max_deviation_solver=max_dev_solver,
            pairwise_deviations=pairwise,
            status=status,
            threshold=self.deviation_threshold,
        )
    
    def compare_all_cases(self) -> Dict[str, List[ComparisonResult]]:
        """Compare all registered cases."""
        results = {}
        for case_name in self._solver_results:
            try:
                results[case_name] = self.compare_case(case_name)
            except ValueError as e:
                results[case_name] = [ComparisonResult(
                    quantity_name="ERROR",
                    solver_values={},
                    status=ComparisonStatus.INSUFFICIENT_DATA,
                )]
        return results
    
    def generate_report(self) -> dict:
        """Generate a complete cross-solver comparison report."""
        all_results = self.compare_all_cases()
        
        # Overall assessment
        n_pass = 0
        n_warn = 0
        n_fail = 0
        n_total = 0
        
        for case_results in all_results.values():
            for r in case_results:
                n_total += 1
                if r.status == ComparisonStatus.PASS:
                    n_pass += 1
                elif r.status == ComparisonStatus.WARN:
                    n_warn += 1
                elif r.status == ComparisonStatus.FAIL:
                    n_fail += 1
        
        if n_fail > 0:
            overall = "FAIL"
        elif n_warn > 0:
            overall = "WARN"
        else:
            overall = "PASS"
        
        return {
            'overall_status': overall,
            'n_cases': len(all_results),
            'n_quantities': n_total,
            'n_pass': n_pass,
            'n_warn': n_warn,
            'n_fail': n_fail,
            'threshold': self.deviation_threshold,
            'details': {
                case: [r.to_dict() for r in results]
                for case, results in all_results.items()
            },
        }


# Predefined canonical cases for common validation scenarios

def create_boundary_layer_case() -> CanonicalCase:
    """Create canonical flat plate boundary layer case."""
    return CanonicalCase(
        name="flat_plate_bl",
        description="Laminar boundary layer over flat plate (Blasius solution)",
        physics="laminar boundary layer",
        geometry={
            'type': 'flat_plate',
            'length': 1.0,  # m
            'width': 0.1,   # m
        },
        parameters={
            'Re_L': 1e5,     # Reynolds number based on plate length
            'U_inf': 1.0,    # m/s
            'nu': 1e-5,      # kinematic viscosity (m^2/s)
        },
        reference_quantities=[
            'boundary_layer_thickness',
            'displacement_thickness',
            'momentum_thickness',
            'skin_friction_coefficient',
        ],
        reference_values={
            'boundary_layer_thickness': 0.00464,  # delta ~ 4.64/sqrt(Re) * L
            'displacement_thickness': 0.00162,    # delta* ~ 1.72/sqrt(Re) * L
            'momentum_thickness': 0.000664,       # theta ~ 0.664/sqrt(Re) * L
            'skin_friction_coefficient': 0.00283, # Cf ~ 0.664/sqrt(Re)
        },
        reference_source="Blasius analytical solution",
    )


def create_cylinder_drag_case() -> CanonicalCase:
    """Create canonical cylinder drag case (Re=40, laminar)."""
    return CanonicalCase(
        name="cylinder_drag_Re40",
        description="Flow past circular cylinder at Re=40 (steady, laminar)",
        physics="cylinder wake",
        geometry={
            'type': 'cylinder',
            'diameter': 1.0,  # m
        },
        parameters={
            'Re_D': 40.0,
            'U_inf': 1.0,
        },
        reference_quantities=[
            'drag_coefficient',
            'recirculation_length',
            'separation_angle',
        ],
        reference_values={
            'drag_coefficient': 1.54,    # From literature (Nieuwstadt & Keller 1974)
            'recirculation_length': 0.52, # L_r/D
            'separation_angle': 115.0,    # degrees from front stagnation
        },
        reference_source="Nieuwstadt & Keller (1974), Coutanceau & Bouard (1977)",
    )


def create_lid_driven_cavity_case() -> CanonicalCase:
    """Create canonical lid-driven cavity case (Re=100)."""
    return CanonicalCase(
        name="lid_driven_cavity_Re100",
        description="Lid-driven cavity flow at Re=100",
        physics="cavity flow",
        geometry={
            'type': 'square_cavity',
            'side_length': 1.0,
        },
        parameters={
            'Re': 100.0,
            'U_lid': 1.0,
        },
        reference_quantities=[
            'center_u_velocity',
            'center_v_velocity',
            'primary_vortex_x',
            'primary_vortex_y',
        ],
        reference_values={
            'center_u_velocity': 0.0,      # Ghia et al. (1982)
            'center_v_velocity': 0.0,
            'primary_vortex_x': 0.46875,
            'primary_vortex_y': 0.50781,
        },
        reference_source="Ghia, Ghia & Shin (1982), J. Comp. Phys.",
    )
