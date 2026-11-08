"""
Pytest tests for POD-Galerkin-DEIM ROM module.
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tensorphysics.rom import (
    NS2DSolver, PODDecomposition, GalerkinProjector,
    DEIMInterpolator, ROMIntegrator, EnergyVerifier
)


class TestNS2DSolver:
    """Tests for 2D Navier-Stokes solver."""
    
    def test_initialization(self):
        """Test solver initialization."""
        solver = NS2DSolver(nx=16, ny=16, nu=0.1)
        solver.initialize(seed=42)
        assert solver.u is not None
        assert solver.v is not None
        assert solver.u.shape == (16, 16)
        assert solver.v.shape == (16, 16)
    
    def test_energy_computation(self):
        """Test kinetic energy computation."""
        solver = NS2DSolver(nx=16, ny=16, nu=0.1)
        solver.u = np.ones((16, 16))
        solver.v = np.ones((16, 16))
        energy = solver.compute_energy()
        assert energy == pytest.approx(1.0, rel=1e-10)
    
    def test_divergence_free(self):
        """Test that initial condition is divergence-free."""
        solver = NS2DSolver(nx=16, ny=16, nu=0.1)
        solver.initialize(seed=42)
        div = solver.compute_divergence()
        assert div < 1e-10
    
    def test_time_step(self):
        """Test single time step."""
        solver = NS2DSolver(nx=16, ny=16, nu=0.1, forcing_amp=0.0)
        solver.initialize(seed=42)
        solver.u *= 0.1
        solver.v *= 0.1
        
        u_old = solver.u.copy()
        v_old = solver.v.copy()
        
        solver.step_rk4(dt=0.001)
        
        # Solution should change
        assert not np.allclose(solver.u, u_old)
        assert not np.allclose(solver.v, v_old)


class TestPOD:
    """Tests for POD decomposition."""
    
    def test_fit(self):
        """Test POD fitting."""
        # Create simple snapshots
        n_snapshots = 10
        n_dof = 100
        snapshots = [np.random.randn(n_dof) for _ in range(n_snapshots)]
        
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        assert pod.modes is not None
        assert pod.singular_values is not None
        assert len(pod.singular_values) == n_snapshots
    
    def test_energy_spectrum(self):
        """Test energy spectrum sums to 1."""
        snapshots = [np.random.randn(100) for _ in range(20)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        assert np.sum(pod.energy_spectrum) == pytest.approx(1.0, rel=1e-10)
    
    def test_cumulative_energy(self):
        """Test cumulative energy is monotonically increasing."""
        snapshots = [np.random.randn(100) for _ in range(20)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        assert np.all(np.diff(pod.cumulative_energy) >= 0)
        assert pod.cumulative_energy[-1] == pytest.approx(1.0, rel=1e-10)
    
    def test_optimal_rank(self):
        """Test optimal rank selection."""
        snapshots = [np.random.randn(100) for _ in range(20)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        k_star = pod.get_optimal_rank(threshold=0.99)
        assert k_star >= 1
        assert pod.cumulative_energy[k_star - 1] >= 0.99
    
    def test_projection_reconstruction(self):
        """Test projection and reconstruction."""
        snapshots = [np.random.randn(100) for _ in range(20)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        # Project and reconstruct first snapshot
        coeffs = pod.project(snapshots[0])
        recon = pod.reconstruct(coeffs)
        
        # Should be close to original
        error = np.linalg.norm(snapshots[0] - recon) / np.linalg.norm(snapshots[0])
        assert error < 0.1  # 10% error


class TestGalerkinProjector:
    """Tests for Galerkin projection."""
    
    def test_precompute(self):
        """Test operator precomputation."""
        # Create simple POD
        snapshots = [np.random.randn(200) for _ in range(10)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        # Create projector
        kx = np.fft.fftfreq(10) * 2 * np.pi
        ky = np.fft.fftfreq(10) * 2 * np.pi
        projector = GalerkinProjector(pod, nu=0.1, kx=kx, ky=ky)
        projector.precompute(n_modes=3)
        
        assert projector.A_uu is not None
        assert projector.A_vv is not None
        assert projector.A_uu.shape == (3, 3)
    
    def test_skew_symmetry(self):
        """Test skew-symmetry of nonlinear term."""
        snapshots = [np.random.randn(200) for _ in range(10)]
        pod = PODDecomposition()
        pod.fit(snapshots)
        
        kx = np.fft.fftfreq(10) * 2 * np.pi
        ky = np.fft.fftfreq(10) * 2 * np.pi
        projector = GalerkinProjector(pod, nu=0.1, kx=kx, ky=ky)
        projector.precompute(n_modes=3)
        
        a_test = np.random.randn(3) * 0.01
        is_skew, residual = projector.check_skew_symmetry(a_test)
        
        # Residual should be small (not necessarily zero due to discretization)
        assert residual < 1e-3


class TestDEIM:
    """Tests for DEIM interpolation."""
    
    def test_fit(self):
        """Test DEIM fitting."""
        n_snapshots = 10
        n_dof = 100
        snapshots = [np.random.randn(n_dof) for _ in range(n_snapshots)]
        
        deim = DEIMInterpolator(n_points=5)
        deim.fit(snapshots, n_points=5)
        
        assert deim.U_deim is not None
        assert deim.points is not None
        assert len(deim.points) == 5
    
    def test_interpolation(self):
        """Test DEIM interpolation."""
        snapshots = [np.random.randn(100) for _ in range(10)]
        deim = DEIMInterpolator(n_points=5)
        deim.fit(snapshots, n_points=5)
        
        # Test interpolation
        N_at_points = np.random.randn(5)
        N_full = deim.interpolate(N_at_points)
        
        assert N_full.shape == (100,)


class TestEnergyVerifier:
    """Tests for energy verification."""
    
    def test_kinetic_energy(self):
        """Test kinetic energy computation."""
        kx = np.fft.fftfreq(10) * 2 * np.pi
        ky = np.fft.fftfreq(10) * 2 * np.pi
        verifier = EnergyVerifier(nu=0.1, kx=kx, ky=ky)
        
        u = np.ones((10, 10))
        v = np.ones((10, 10))
        energy = verifier.compute_kinetic_energy(u, v)
        
        assert energy == pytest.approx(1.0, rel=1e-10)
    
    def test_dissipation(self):
        """Test dissipation computation."""
        kx = np.fft.fftfreq(10) * 2 * np.pi
        ky = np.fft.fftfreq(10) * 2 * np.pi
        verifier = EnergyVerifier(nu=0.1, kx=kx, ky=ky)
        
        u = np.zeros((10, 10))
        v = np.zeros((10, 10))
        diss = verifier.compute_dissipation(u, v)
        
        # Zero velocity → zero dissipation
        assert diss == pytest.approx(0.0, abs=1e-10)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
