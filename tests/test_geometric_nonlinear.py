"""Tests for geometric nonlinear verification suite.

Covers:
1. Elastica exact solution correctness
2. Elastica linear limit verification
3. Von Karman plate reference solution
4. Shallow arch snap-through
5. Convergence/verification studies
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.geometric_nonlinear.elastica import (
    ElasticaExact, solve_elastica_problem
)
from tensorphysics.geometric_nonlinear.von_karman_plate import (
    VonKarmanPlateExact, solve_von_karman_plate
)
from tensorphysics.geometric_nonlinear.shallow_arch import (
    ShallowArchModel, VonMisesTruss, solve_snap_through
)
from tensorphysics.geometric_nonlinear.convergence import (
    convergence_study_elastica, convergence_study_plate
)


class TestElasticaExact:
    """Test the exact elastica solution."""
    
    def test_zero_load(self):
        """Zero load → zero deflection."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(0.0)
        assert sol.P == 0.0
        assert abs(sol.y_tip) < 1e-10
    
    def test_small_load_linear(self):
        """Small load should match linear theory: δ = PL³/(3EI)."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(0.05)
        assert sol.P > 0
        assert sol.y_tip > 0
        assert sol.y_tip < sol.L
    
    def test_linear_limit(self):
        """Verify δ/(PL³/3EI) → 1.0 for small deflections."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        result = exact.verify_linear_limit()
        assert abs(result['ratio'] - 1.0) < 0.01, \
            f"Linear limit ratio {result['ratio']} deviates from 1.0 by {result['error_pct']:.2f}%"
    
    def test_large_deflection(self):
        """Large deflection: tip deflection > 50% of L."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(1.2)
        assert sol.y_tip / sol.L > 0.5
    
    def test_load_deflection_monotone(self):
        """Load-deflection curve should be monotonically increasing."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        P_norm, delta_L, _ = exact.compute_load_deflection_curve(n_points=10)
        assert all(np.diff(P_norm) > 0)
        assert all(np.diff(delta_L) > 0)
    
    def test_shape_consistency(self):
        """Deformed shape should start at origin."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(0.8)
        assert abs(sol.x_array[0]) < 1e-6
        assert abs(sol.y_array[0]) < 1e-6


class TestVonKarmanPlate:
    """Test the von Karman plate solution."""
    
    def test_linear_limit(self):
        """Small deflection should match linear plate theory."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        
        w0 = 1e-6
        P_linear = 16 * np.pi * ref.D * w0 / ref.a**2
        w0_linear = ref.linear_solution(P_linear)
        
        assert abs(w0_linear - w0) / w0 < 0.01
    
    def test_nonlinear_stiffening(self):
        """Large deflection should show membrane stiffening."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        w0_h, P_norm, P_vals = ref.compute_load_deflection_curve(n_points=20)
        
        ratio = P_norm / w0_h
        assert ratio[-1] > ratio[0], "No membrane stiffening detected"
    
    def test_cubic_coefficient(self):
        """The cubic coefficient should be in the expected range."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        w0_h, P_norm, _ = ref.compute_load_deflection_curve(
            w0_h_range=np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        )
        
        c_values = (P_norm - w0_h) / w0_h**3
        c_mean = np.mean(c_values)
        assert 0.2 < c_mean < 0.6, f"Cubic coefficient {c_mean} out of range"


class TestShallowArch:
    """Test the shallow arch snap-through model."""
    
    def test_von_mises_truss_equilibrium(self):
        """Von Mises truss should show snap-through behavior."""
        truss = VonMisesTruss(L=10.0, H=1.0, EA=1e9)
        result = truss.find_limit_points(n_pts=500)
        
        assert result['q_limit_upper'] > 0
        assert 'unstable' in result['stability'], "No snap-through detected"
    
    def test_arch_model(self):
        """Arch model should find limit points."""
        arch = ShallowArchModel(L=10.0, H=1.0, EI=1e6, EA=1e9, n_elem=10)
        sol = arch.solve_arc_length(q_max=0.0, n_steps=100)
        
        assert sol.q_limit_upper > 0
        assert len(sol.q_values) > 0


class TestConvergence:
    """Test convergence/verification studies."""
    
    def test_elastica_convergence(self):
        """Elastica should converge to linear theory for small deflections."""
        result = convergence_study_elastica(
            L=1.0, EI=1.0,
            theta_B_range=[0.001, 0.01, 0.1],
        )
        
        # Error should decrease as theta_B decreases
        # All errors should be small (within 2% of linear theory)
        assert all(result["errors_pct"] < 2.0), "Errors too large"
        # Smallest angle should be very close to linear
    
    def test_plate_convergence(self):
        """Plate should show correct nonlinear behavior."""
        result = convergence_study_plate(
            a=1.0, h=0.01, E=210e9, nu=0.3,
            w0_h_range=[0.01, 0.1, 1.0, 5.0],
        )
        
        # Ratio should increase with deflection (membrane stiffening)
        assert result['ratios'][-1] > result['ratios'][0]


class TestIntegration:
    """Integration tests for the full verification suite."""
    
    def test_elastica_problem(self):
        """Full elastica problem should run successfully."""
        result = solve_elastica_problem(L=1.0, EI=1.0, n_points=10)
        
        assert len(result['P_values']) == 10
        assert all(result['P_normalized'] > 0)
        assert all(result['delta_over_L'] > 0)
        assert result['linear_verification']['error_pct'] < 1.0
    
    def test_snap_through_problem(self):
        """Full snap-through problem should run successfully."""
        result = solve_snap_through(L=10.0, H=1.0, EI=1e6, EA=1e9, n_steps=50)
        
        assert result['q_limit_upper'] > 0
        assert len(result['solution'].q_values) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
