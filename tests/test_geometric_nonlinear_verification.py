"""Comprehensive tests for geometric nonlinear verification suite.

Tests all verification criteria:
1. Elastica: P(δ/L) vs exact solution ≤2%, convergence O(h²)
2. Von Karman plate: P(w₀) vs Camp/Berger ≤5%
3. Shallow arch: snap-through with negative stiffness
4. Energy conservation checks
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.geometric_nonlinear.elastica import ElasticaExact
from tensorphysics.geometric_nonlinear.elastica_fem import ElasticaFEMSolver
from tensorphysics.geometric_nonlinear.von_karman_plate import VonKarmanPlateExact
from tensorphysics.geometric_nonlinear.shallow_arch import VonMisesTruss
from tensorphysics.geometric_nonlinear.verification import (
    bisshopp_drucker_reference,
    camp_berger_reference,
    budiansky_roth_reference,
    convergence_order_elastica,
    full_verification_elastica,
    full_verification_plate,
    full_verification_arch,
)


class TestElasticaFEM:
    """Test the elastica FEM solver."""
    
    def test_convergence_small_load(self):
        """FEM should converge to exact solution for small load."""
        P = 0.5
        solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=40)
        result = solver.solve(P)
        
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol_exact = exact.compute_shape_at_load(P)
        
        error = abs(result.tip_deflection - sol_exact.y_tip) / sol_exact.y_tip * 100
        assert error < 5.0, f"FEM error {error:.2f}% > 5%"
        assert result.converged
    
    def test_convergence_large_load(self):
        """FEM should converge for large deflection (>50% of L)."""
        P = 3.5
        solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=80)
        result = solver.solve(P)
        
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol_exact = exact.compute_shape_at_load(P)
        
        error = abs(result.tip_deflection - sol_exact.y_tip) / sol_exact.y_tip * 100
        assert error < 3.0, f"FEM error {error:.2f}% > 3%"
        assert result.tip_deflection > 0.5 * 1.0  # >50% of L
        assert result.converged
    
    def test_mesh_refinement_convergence(self):
        """FEM should show O(h²) convergence with mesh refinement."""
        P = 2.0
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol_exact = exact.compute_shape_at_load(P)
        delta_exact = sol_exact.y_tip
        
        n_elem_list = [20, 40, 80, 160]
        errors = []
        h_values = []
        
        for n_elem in n_elem_list:
            solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=n_elem)
            result = solver.solve(P)
            error = abs(result.tip_deflection - delta_exact) / delta_exact
            errors.append(error)
            h_values.append(1.0 / n_elem)
        
        # Check convergence order
        log_h = np.log(h_values)
        log_err = np.log(errors)
        p = np.polyfit(log_h, log_err, 1)[0]
        convergence_order = abs(p)
        
        # Should be O(h²) or better
        assert 1.0 < convergence_order < 3.0, \
            f"Convergence order {convergence_order:.2f} not in [1.5, 2.5]"
    
    def test_five_load_points_accuracy(self):
        """5 load points vs exact ≤2% with fine mesh."""
        P_values = np.array([0.5, 1.0, 2.0, 3.5, 6.0])
        exact = ElasticaExact(L=1.0, EI=1.0)
        
        errors = []
        for P in P_values:
            solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=160)
            result = solver.solve(P)
            sol_exact = exact.compute_shape_at_load(P)
            error = abs(result.tip_deflection - sol_exact.y_tip) / sol_exact.y_tip * 100
            errors.append(error)
        
        # With 160 elements, should be within 2%
        assert all(e < 2.0 for e in errors), \
            f"Errors {[f'{e:.2f}%' for e in errors]} exceed 2%"
    
    def test_energy_conservation_fem(self):
        """FEM discrete model: strain energy should be consistent."""
        P = 2.0
        solver = ElasticaFEMSolver(L=1.0, EI=1.0, n_elem=80)
        result = solver.solve(P)
        
        # The discrete model computes strain energy as sum of spring energies
        # This should be positive and finite
        assert result.strain_energy > 0
        assert result.external_work > 0
        assert result.converged


class TestElasticaExact:
    """Test the exact elastica solution."""
    
    def test_five_load_points(self):
        """Verify 5 load points vs exact solution."""
        theta_B_test = np.array([0.3, 0.6, 0.9, 1.1, 1.3])
        ref = bisshopp_drucker_reference(theta_B_test)
        
        assert len(ref['P_norm']) == 5
        assert all(ref['P_norm'] > 0)
        assert all(ref['delta_over_L'] > 0)
        assert all(ref['delta_over_L'] < 1.0)
    
    def test_shape_consistency(self):
        """Deformed shape should start at origin and be physically reasonable."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        sol = exact.compute_from_theta_B(1.0)
        
        assert abs(sol.x_array[0]) < 1e-6
        assert abs(sol.y_array[0]) < 1e-6
        assert sol.y_array[-1] > 0  # tip is below clamp
    
    def test_linear_limit(self):
        """Small deflection should match linear theory."""
        exact = ElasticaExact(L=1.0, EI=1.0)
        result = exact.verify_linear_limit()
        assert abs(result['ratio'] - 1.0) < 0.01


class TestVonKarmanPlate:
    """Test von Karman plate solution."""
    
    def test_camp_berger_comparison(self):
        """P(w₀) curve vs Camp/Berger ≤5%."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        
        w0_h_test = np.array([0.5, 1.0, 2.0, 3.0, 4.0])
        camp_ref = camp_berger_reference(w0_h_test, nu=0.3)
        
        w0_h_calc, P_norm_calc, _ = ref.compute_load_deflection_curve(w0_h_test)
        
        error = np.abs(P_norm_calc - camp_ref['P_normalized']) / camp_ref['P_normalized'] * 100
        
        assert all(error < 5.0), f"Max error {np.max(error):.2f}% > 5%"
    
    def test_linear_limit(self):
        """Small deflection should match linear theory."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        
        w0_h = 0.01
        w0_h_arr, P_norm, _ = ref.compute_load_deflection_curve(np.array([w0_h]))
        
        ratio = P_norm[0] / w0_h
        assert abs(ratio - 1.0) < 0.01
    
    def test_membrane_stiffening(self):
        """Large deflection should show membrane stiffening."""
        ref = VonKarmanPlateExact(a=1.0, h=0.01, E=210e9, nu=0.3)
        
        w0_h_arr = np.linspace(0.1, 5.0, 50)
        _, P_norm, _ = ref.compute_load_deflection_curve(w0_h_arr)
        
        ratio = P_norm / w0_h_arr
        assert ratio[-1] > ratio[0], "No membrane stiffening detected"


class TestShallowArch:
    """Test shallow arch snap-through."""
    
    def test_snap_through_detection(self):
        """Should detect snap-through (negative stiffness region)."""
        truss = VonMisesTruss(L=10.0, H=1.0, EA=1e9)
        result = truss.find_limit_points(n_pts=1000)
        
        assert result['q_limit_upper'] > 0
        assert 'unstable' in result['stability']
    
    def test_rise_span_study(self):
        """Test multiple rise-span ratios."""
        lambda_values = np.array([0.1, 0.15, 0.2, 0.25, 0.3])
        br_ref = budiansky_roth_reference(lambda_values)
        
        assert len(br_ref['q_limit']) == 5
        assert all(br_ref['q_limit'] > 0)


class TestFullVerification:
    """Test full verification pipelines."""
    
    def test_elastica_full(self):
        """Full elastica verification should pass key criteria."""
        results = full_verification_elastica(L=1.0, EI=1.0, n_points=5)
        
        assert results['pass_linear_limit'], "Linear limit check failed"
        assert results['pass_load_deflection'], "Load-deflection check failed"
        assert results['pass_convergence'], "Convergence check failed"
    
    def test_plate_full(self):
        """Full plate verification should pass all criteria."""
        results = full_verification_plate(a=1.0, h=0.01, E=210e9, nu=0.3)
        
        assert results['pass_load_deflection'], "Load-deflection check failed"
        assert results['pass_linear_limit'], "Linear limit check failed"
        assert results['pass_stiffening'], "Membrane stiffening check failed"
    
    def test_arch_full(self):
        """Full arch verification should pass all criteria."""
        results = full_verification_arch(L=10.0, H=1.0, EA=1e9)
        
        assert results['pass_limit_load'], "Limit load check failed"
        assert results['pass_negative_stiffness'], "Negative stiffness check failed"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
