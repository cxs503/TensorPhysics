"""Tests for ice-structure partitioned coupling stability module.

Tests cover:
- 1-DOF and 2-DOF model physics
- Partitioned coupling schemes
- Aitken relaxation
- Stability analysis
- Analytical vs numerical spectral radius
"""
from __future__ import annotations

import numpy as np
import pytest
import sys
from pathlib import Path

# Add src to path
WORKTREE = Path(__file__).resolve().parents[1]
SRC = WORKTREE / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tensorphysics.ice_structure import (
    OneDOFModel,
    TwoDOFModel,
    IceStructureSystem,
    PartitionedCoupling,
    CouplingScheme,
    StabilityAnalyzer,
    compute_analytical_spectral_radius,
    compute_critical_mu,
    AitkenRelaxation,
    DynamicRelaxation,
)


class TestOneDOFModel:
    """Tests for 1-DOF mass-spring-damper model."""
    
    def test_basic_properties(self):
        """Test basic model properties."""
        model = OneDOFModel(m_s=1.0, k=100.0, c=1.0, m_a=0.5)
        
        assert model.m_total == 1.5
        assert model.mu == 0.5
        assert model.omega_n > 0
        assert model.zeta >= 0
    
    def test_added_mass_ratio(self):
        """Test added mass ratio calculation."""
        model = OneDOFModel(m_s=2.0, k=100.0, c=1.0, m_a=1.0)
        assert model.mu == 0.5
        
        model = OneDOFModel(m_s=1.0, k=100.0, c=1.0, m_a=2.0)
        assert model.mu == 2.0
    
    def test_state_space(self):
        """Test state-space matrix dimensions."""
        model = OneDOFModel(m_s=1.0, k=100.0, c=1.0, m_a=0.5)
        A, B = model.state_space_matrices()
        
        assert A.shape == (2, 2)
        assert B.shape == (2, 1)
    
    def test_reference_solution(self):
        """Test that reference solution is physically reasonable."""
        model = OneDOFModel(m_s=1.0, k=100.0, c=1.0, m_a=0.0)
        system = IceStructureSystem(model=model, dt=0.001, t_end=0.5)
        
        result = system.solve_reference()
        
        assert 't' in result
        assert 'x' in result
        assert 'v' in result
        assert len(result['t']) == len(result['x'])
        # Solution should be bounded (damped oscillator)
        assert np.max(np.abs(result['x'])) < 10.0


class TestTwoDOFModel:
    """Tests for 2-DOF ice-structure model."""
    
    def test_basic_properties(self):
        """Test basic model properties."""
        model = TwoDOFModel(
            m_i=0.5, k_i=50.0, c_i=0.5,
            m_s=1.0, k_s=100.0, c_s=1.0,
            k_c=20.0
        )
        
        assert model.mu == 0.5
    
    def test_coupled_solution(self):
        """Test fully coupled solution."""
        model = TwoDOFModel(
            m_i=0.5, k_i=50.0, c_i=0.5,
            m_s=1.0, k_s=100.0, c_s=1.0,
            k_c=20.0
        )
        
        result = model.solve_coupled(t_end=0.5, dt=0.001)
        
        assert 't' in result
        assert 'x_i' in result
        assert 'x_s' in result
        assert len(result['t']) == len(result['x_i'])
        # Solutions should be bounded
        assert np.max(np.abs(result['x_i'])) < 10.0
        assert np.max(np.abs(result['x_s'])) < 10.0


class TestPartitionedCoupling:
    """Tests for partitioned coupling schemes."""
    
    def test_loose_coupling(self):
        """Test loose (one-way) coupling."""
        coupling = PartitionedCoupling(
            m_s=1.0, k_s=100.0, c_s=1.0, m_a=0.0,
            scheme=CouplingScheme.LOOSE
        )
        
        result = coupling.run(t_end=0.1, dt=0.001)
        
        assert result.converged
        assert not result.diverged
        assert result.scheme == "loose"
    
    def test_tight_coupling_stable(self):
        """Test tight coupling in stable regime (mu < mu_c)."""
        # At omega=1, mu_c=1, so mu=0.5 should be stable
        coupling = PartitionedCoupling(
            m_s=1.0, k_s=100.0, c_s=1.0, m_a=0.5,  # mu=0.5 < mu_c=1
            scheme=CouplingScheme.TIGHT_FIXED,
            omega=1.0
        )
        
        result = coupling.run(t_end=0.1, dt=0.001)
        
        assert result.converged
        assert not result.diverged
    
    def test_tight_coupling_unstable(self):
        """Test tight coupling in unstable regime (mu > mu_c)."""
        # At omega=1, mu_c=1, so mu=2 should be unstable
        coupling = PartitionedCoupling(
            m_s=1.0, k_s=100.0, c_s=1.0, m_a=2.0,  # mu=2 > mu_c=1
            scheme=CouplingScheme.TIGHT_FIXED,
            omega=1.0,
            max_iter=20
        )
        
        result = coupling.run(t_end=0.1, dt=0.001)
        
        # Should fail to converge or diverge
        assert not result.converged or result.diverged
    
    def test_aitken_coupling(self):
        """Test Aitken relaxation coupling."""
        coupling = PartitionedCoupling(
            m_s=1.0, k_s=100.0, c_s=1.0, m_a=1.5,  # mu=1.5 > mu_c=1
            scheme=CouplingScheme.TIGHT_AITKEN,
            omega=1.0,
            max_iter=50
        )
        
        result = coupling.run(t_end=0.1, dt=0.001)
        
        # Aitken should help convergence
        assert result.scheme == "tight_aitken"


class TestAnalyticalFormulas:
    """Tests for analytical stability formulas."""
    
    def test_spectral_radius_omega1(self):
        """Test spectral radius at omega=1."""
        # At omega=1: rho = |1 - (1+mu)| = mu
        mu = 0.5
        rho = compute_analytical_spectral_radius(mu, 1.0)
        expected = mu  # |1 - 1*(1+0.5)| = |1 - 1.5| = 0.5
        
        assert abs(rho - expected) < 1e-10
    
    def test_spectral_radius_stability(self):
        """Test spectral radius stability criterion."""
        # Stable: rho < 1 (mu < mu_c = 1 at omega=1)
        rho_stable = compute_analytical_spectral_radius(0.5, 1.0)
        assert rho_stable < 1.0
        
        # Unstable: rho > 1 (mu > mu_c = 1 at omega=1)
        rho_unstable = compute_analytical_spectral_radius(2.0, 1.0)
        # rho = |1 - 1*(1+2)| = |1 - 3| = 2
        assert rho_unstable > 1.0
    
    def test_critical_mu(self):
        """Test critical mu formula."""
        # At omega=1: mu_c = 1
        mu_c = compute_critical_mu(1.0)
        assert abs(mu_c - 1.0) < 1e-10
        
        # At omega=0.5: mu_c = 3
        mu_c = compute_critical_mu(0.5)
        assert abs(mu_c - 3.0) < 1e-10
    
    def test_critical_mu_monotonic(self):
        """Test that critical mu decreases with omega."""
        omega_values = [0.3, 0.5, 0.7, 1.0, 1.2]
        mu_c_values = [compute_critical_mu(w) for w in omega_values]
        
        # mu_c should decrease as omega increases
        for i in range(len(mu_c_values) - 1):
            assert mu_c_values[i] > mu_c_values[i+1]
    
    def test_spectral_radius_formula(self):
        """Test the spectral radius formula at various points."""
        test_cases = [
            # (mu, omega, expected_rho)
            (0.0, 1.0, 0.0),   # No added mass: rho=0
            (0.5, 1.0, 0.5),   # rho = mu at omega=1
            (1.0, 1.0, 1.0),   # At stability boundary
            (2.0, 1.0, 2.0),   # Unstable: rho = 2
            (0.5, 0.5, 0.25),  # rho = |1 - 0.5*1.5| = 0.25
            (3.0, 0.5, 1.0),   # At stability boundary for omega=0.5
        ]
        
        for mu, omega, expected in test_cases:
            rho = compute_analytical_spectral_radius(mu, omega)
            assert abs(rho - expected) < 1e-10, f"mu={mu}, omega={omega}: got {rho}, expected {expected}"


class TestStabilityAnalyzer:
    """Tests for stability analyzer."""
    
    def test_single_point(self):
        """Test stability at a single point."""
        analyzer = StabilityAnalyzer(
            m_s=1.0, k_s=100.0, c_s=1.0,
            dt=0.001, t_end=0.05
        )
        
        result = analyzer.test_stability(mu=0.5, omega=1.0)
        
        assert result.mu == 0.5
        assert result.omega == 1.0
        assert isinstance(result.stable, bool)
    
    def test_1d_scan(self):
        """Test 1D stability scan."""
        analyzer = StabilityAnalyzer(
            m_s=1.0, k_s=100.0, c_s=1.0,
            dt=0.001, t_end=0.05
        )
        
        mu_values = np.array([0.5, 1.0, 1.5])
        results = analyzer.scan_1d(mu_values, omega=1.0)
        
        assert len(results) == 3
        assert all(hasattr(r, 'stable') for r in results)
    
    def test_find_critical_mu(self):
        """Test finding critical mu."""
        analyzer = StabilityAnalyzer(
            m_s=1.0, k_s=100.0, c_s=1.0,
            dt=0.001, t_end=0.05, tol=1e-8, max_iter=50
        )
        
        mu_c = analyzer.find_critical_mu(omega=1.0, tol=0.1)
        mu_c_analytical = compute_critical_mu(1.0)  # = 1.0
        
        # Should be within 50% of analytical (numerical has some tolerance)
        assert abs(mu_c - mu_c_analytical) / mu_c_analytical < 0.5


class TestAitkenRelaxation:
    """Tests for Aitken relaxation."""
    
    def test_basic_update(self):
        """Test basic Aitken update."""
        aitken = AitkenRelaxation(omega_init=1.0)
        aitken.reset()
        
        # First update (no history)
        omega1 = aitken.update(1.0)
        assert omega1 == 1.0
        
        # Second update
        omega2 = aitken.update(0.5)
        # Should update based on residual change
    
    def test_reset(self):
        """Test reset functionality."""
        aitken = AitkenRelaxation()
        aitken.update(1.0)
        aitken.update(0.5)
        
        aitken.reset()
        
        assert len(aitken._residual_history) == 0
        assert aitken.omega == aitken.omega_init
    
    def test_convergence_rate(self):
        """Test convergence rate estimation."""
        aitken = AitkenRelaxation()
        aitken.reset()
        
        # Simulate converging residuals
        for r in [1.0, 0.5, 0.25, 0.125]:
            aitken.update(r)
        
        rate = aitken.get_convergence_rate()
        # Rate should be around 0.5 (halving each iteration)
        assert 0.3 < rate < 0.7


class TestDynamicRelaxation:
    """Tests for dynamic relaxation."""
    
    def test_fixed_strategy(self):
        """Test fixed relaxation strategy."""
        relax = DynamicRelaxation(strategy="fixed", omega_init=0.8)
        
        omega = relax.update(1.0)
        assert omega == 0.8
    
    def test_aitken_strategy(self):
        """Test Aitken strategy."""
        relax = DynamicRelaxation(strategy="aitken", omega_init=1.0)
        relax.reset()
        
        omega = relax.update(1.0)
        assert omega > 0
    
    def test_statistics(self):
        """Test statistics output."""
        relax = DynamicRelaxation(strategy="aitken")
        relax.reset()
        
        relax.update(1.0)
        relax.update(0.5)
        
        stats = relax.get_statistics()
        
        assert "strategy" in stats
        assert "omega_history" in stats
        assert "convergence_rate" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
