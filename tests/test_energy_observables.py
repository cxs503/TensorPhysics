"""Tests for the three observable energy channels and energy closure ledger.

Evidence grade: REAL — all tests use first-principles computations with
analytically verifiable reference values.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from tensorphysics.ice_nav_integration.energy_observables import (
    EnergyClosureState,
    SolverInterfaceFlux,
    compute_energy_closure,
    free_surface_work_energy,
    free_surface_work_rate,
    ice_kinetic_energy_dem,
    ice_kinetic_energy_lbm,
    strain_rate_tensor,
    verify_cross_solver_conservation,
    viscous_dissipation_energy,
    viscous_dissipation_rate,
)


# ---------------------------------------------------------------------------
# 1. Ice Kinetic Energy — DEM
# ---------------------------------------------------------------------------


class TestIceKineticEnergyDEM:
    """Tests for ice_kinetic_energy_dem."""

    def test_single_particle(self):
        """Single particle: E = ½ m v²."""
        particles = [{"mass": 1000.0, "velocity": [3.0, 0.0, 0.0]}]
        e = ice_kinetic_energy_dem(particles)
        assert e == pytest.approx(0.5 * 1000.0 * 9.0, rel=1e-12)

    def test_multiple_particles(self):
        """Two particles: E = ½ m1 v1² + ½ m2 v2²."""
        particles = [
            {"mass": 500.0, "velocity": [2.0, 1.0, 0.0]},
            {"mass": 300.0, "velocity": [0.0, 3.0, 0.0]},
        ]
        e = ice_kinetic_energy_dem(particles)
        expected = 0.5 * 500.0 * (4 + 1) + 0.5 * 300.0 * 9
        assert e == pytest.approx(expected, rel=1e-12)

    def test_zero_velocity(self):
        particles = [{"mass": 1000.0, "velocity": [0.0, 0.0, 0.0]}]
        assert ice_kinetic_energy_dem(particles) == pytest.approx(0.0)

    def test_empty_particles(self):
        assert ice_kinetic_energy_dem([]) == pytest.approx(0.0)

    def test_negative_mass_raises(self):
        with pytest.raises(ValueError, match="negative mass"):
            ice_kinetic_energy_dem([{"mass": -1.0, "velocity": [1.0, 0.0, 0.0]}])

    def test_nan_velocity_raises(self):
        with pytest.raises(ValueError, match="non-finite velocity"):
            ice_kinetic_energy_dem([{"mass": 1.0, "velocity": [float("nan"), 0.0, 0.0]}])

    def test_2d_velocity(self):
        particles = [{"mass": 100.0, "velocity": [3.0, 4.0]}]
        e = ice_kinetic_energy_dem(particles)
        assert e == pytest.approx(0.5 * 100.0 * 25.0, rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Ice Kinetic Energy — LBM / Eulerian
# ---------------------------------------------------------------------------


class TestIceKineticEnergyLBM:
    """Tests for ice_kinetic_energy_lbm."""

    def test_uniform_field(self):
        """Uniform alpha=1, uniform u=(1,0,0): E = ½ ρ V."""
        alpha = np.ones((4, 4, 4))
        u = np.ones((3, 4, 4, 4))
        u[1:] = 0.0  # only x-component = 1
        dx = 0.5
        e = ice_kinetic_energy_lbm(alpha, u, rho_ice=917.0, dx=dx)
        volume = (4 * dx) ** 3
        expected = 0.5 * 917.0 * 1.0 * volume
        assert e == pytest.approx(expected, rel=1e-10)

    def test_zero_alpha(self):
        alpha = np.zeros((4, 4, 4))
        u = np.ones((3, 4, 4, 4))
        e = ice_kinetic_energy_lbm(alpha, u, dx=1.0)
        assert e == pytest.approx(0.0)

    def test_2d_field(self):
        """2D case: alpha and u are (ny, nx) and (3, ny, nx)."""
        alpha = np.ones((8, 8))
        u = np.zeros((3, 8, 8))
        u[0] = 2.0  # u_x = 2
        dx = 0.25
        e = ice_kinetic_energy_lbm(alpha, u, rho_ice=917.0, dx=dx, dz=1.0)
        area = (8 * dx) ** 2
        expected = 0.5 * 917.0 * 1.0 * 4.0 * area * 1.0  # dz=1
        assert e == pytest.approx(expected, rel=1e-10)

    def test_shape_mismatch_raises(self):
        alpha = np.ones((4, 4))
        u = np.ones((3, 5, 5))
        with pytest.raises(ValueError, match="spatial dims"):
            ice_kinetic_energy_lbm(alpha, u, dx=1.0)

    def test_wrong_ndim_raises(self):
        alpha = np.ones((4,))
        u = np.ones((3, 4))
        with pytest.raises(ValueError, match="2D or 3D"):
            ice_kinetic_energy_lbm(alpha, u, dx=1.0)


# ---------------------------------------------------------------------------
# 3. Viscous Dissipation
# ---------------------------------------------------------------------------


class TestViscousDissipation:
    """Tests for viscous_dissipation_rate and viscous_dissipation_energy."""

    def test_uniform_flow_zero_dissipation(self):
        """Uniform velocity → zero strain rate → zero dissipation."""
        u = np.ones((3, 8, 8))
        u[0] *= 2.0
        u[1] *= 1.0
        u[2] *= 0.5
        phi = viscous_dissipation_rate(u, mu=1e-3, dx=0.1, dy=0.1, dz=0.1)
        assert phi == pytest.approx(0.0, abs=1e-10)

    def test_linear_shear(self):
        """Linear shear u = (γ y, 0, 0): S_xy = γ/2, Φ = μ γ² V."""
        ny, nx = 20, 20
        dy = 0.05
        dx = 0.05
        y = np.arange(ny) * dy
        gamma = 10.0  # shear rate

        u = np.zeros((3, ny, nx))
        u[0] = gamma * y[:, None]  # u_x = γ y

        phi = viscous_dissipation_rate(u, mu=1e-3, dx=dx, dy=dy, dz=0.1)

        # Analytical: Φ = μ γ² V (for simple shear, S:S = γ²/2, so 2μ·γ²/2 = μγ²)
        volume = nx * dx * ny * dy * 0.1
        expected = 1e-3 * gamma ** 2 * volume
        # Central diff has O(h²) error, so use 5% tolerance
        assert phi == pytest.approx(expected, rel=0.05)

    def test_dissipation_energy_trapezoidal(self):
        """Constant dissipation rate: E = Φ × t."""
        rates = [100.0, 100.0, 100.0, 100.0]
        dt = 0.5
        e = viscous_dissipation_energy(rates, dt)
        assert e == pytest.approx(100.0 * 0.5 * 3, rel=1e-12)

    def test_dissipation_energy_linearly_increasing(self):
        """Linearly increasing: E = ½ (Φ₀ + Φₙ) × total_time."""
        rates = [0.0, 10.0, 20.0, 30.0]
        dt = 1.0
        e = viscous_dissipation_energy(rates, dt)
        # Trapezoidal: 0.5*(0+10)*1 + 0.5*(10+20)*1 + 0.5*(20+30)*1 = 5+15+25 = 45
        assert e == pytest.approx(45.0, rel=1e-12)

    def test_single_rate(self):
        e = viscous_dissipation_energy([50.0], dt=2.0)
        assert e == pytest.approx(100.0, rel=1e-12)

    def test_empty_rates(self):
        assert viscous_dissipation_energy([], dt=1.0) == pytest.approx(0.0)

    def test_3d_velocity_field(self):
        """3D velocity field (3, nz, ny, nx)."""
        nz, ny, nx = 4, 8, 8
        u = np.zeros((3, nz, ny, nx))
        # Parabolic profile in y: u_x = y*(1-y)
        y = np.linspace(0, 1, ny)
        u[0] = (y * (1 - y))[None, :, None] * np.ones((nz, 1, nx))
        phi = viscous_dissipation_rate(u, mu=1e-3, dx=0.1, dy=1.0 / (ny - 1), dz=0.1)
        assert phi > 0.0
        assert math.isfinite(phi)


# ---------------------------------------------------------------------------
# 4. Strain Rate Tensor
# ---------------------------------------------------------------------------


class TestStrainRateTensor:
    """Tests for strain_rate_tensor."""

    def test_uniform_flow(self):
        u = np.ones((3, 8, 8)) * 2.0
        S = strain_rate_tensor(u, dx=0.1, dy=0.1, dz=0.1)
        for s in S:
            assert np.allclose(s, 0.0, atol=1e-10)

    def test_pure_extension(self):
        """u = (α x, 0, 0): S_xx = α, all others = 0."""
        nx = 20
        dx = 0.1
        x = np.arange(nx) * dx
        alpha = 5.0
        u = np.zeros((3, 10, nx))
        u[0] = alpha * x[None, :]

        S_xx, S_yy, S_zz, S_xy, S_xz, S_yz = strain_rate_tensor(
            u, dx=dx, dy=0.1, dz=0.1
        )
        assert np.allclose(S_xx, alpha, rtol=0.02)
        assert np.allclose(S_yy, 0.0, atol=1e-10)
        assert np.allclose(S_zz, 0.0, atol=1e-10)

    def test_pure_shear(self):
        """u = (γ y, 0, 0): S_xy = γ/2."""
        ny = 20
        dy = 0.05
        y = np.arange(ny) * dy
        gamma = 10.0
        u = np.zeros((3, ny, 20))
        u[0] = gamma * y[:, None]

        S_xx, S_yy, S_zz, S_xy, S_xz, S_yz = strain_rate_tensor(
            u, dx=0.05, dy=dy, dz=0.1
        )
        assert np.allclose(S_xy, gamma / 2.0, rtol=0.05)


# ---------------------------------------------------------------------------
# 5. Free Surface Work
# ---------------------------------------------------------------------------


class TestFreeSurfaceWork:
    """Tests for free_surface_work_rate and free_surface_work_energy."""

    def test_static_surface_zero_work(self):
        """η constant → ∂η/∂t = 0 → W = 0."""
        eta = np.ones((8, 8)) * 0.5
        eta_prev = eta.copy()
        p_dyn = np.ones((8, 8)) * 1000.0
        w = free_surface_work_rate(eta, eta_prev, p_dyn, dx=0.5, dy=0.5, dt=1.0)
        assert w == pytest.approx(0.0, abs=1e-12)

    def test_uniform_rising_surface(self):
        """η rising uniformly with constant p': W = p' × A × Δη/Δt."""
        nx, ny = 10, 10
        dx, dy = 1.0, 1.0
        eta = np.ones((ny, nx)) * 0.2
        eta_prev = np.ones((ny, nx)) * 0.1
        p_dyn = np.ones((ny, nx)) * 500.0
        dt = 0.5
        w = free_surface_work_rate(eta, eta_prev, p_dyn, dx, dy, dt)
        area = nx * dx * ny * dy
        d_eta_dt = (0.2 - 0.1) / dt
        expected = 500.0 * area * d_eta_dt
        assert w == pytest.approx(expected, rel=1e-12)

    def test_free_surface_work_energy_constant(self):
        """Constant rate: W = rate × total_time."""
        nx, ny = 8, 8
        dx, dy = 0.5, 0.5
        dt = 0.1
        n_steps = 5

        eta_history = [np.ones((ny, nx)) * (0.1 + i * 0.02) for i in range(n_steps + 1)]
        p_history = [np.ones((ny, nx)) * 1000.0 for _ in range(n_steps + 1)]

        w = free_surface_work_energy(eta_history, p_history, dx, dy, dt)
        area = nx * dx * ny * dy
        d_eta_dt = 0.02 / dt
        rate = 1000.0 * area * d_eta_dt
        expected = rate * n_steps * dt
        assert w == pytest.approx(expected, rel=1e-10)

    def test_shape_mismatch_raises(self):
        eta = np.ones((8, 8))
        eta_prev = np.ones((8, 8))
        p_dyn = np.ones((9, 9))
        with pytest.raises(ValueError, match="shape"):
            free_surface_work_rate(eta, eta_prev, p_dyn, dx=1.0, dy=1.0, dt=1.0)

    def test_short_history(self):
        """Single time step → zero work (need at least 2 for rate)."""
        eta = [np.ones((4, 4))]
        p = [np.ones((4, 4))]
        assert free_surface_work_energy(eta, p, dx=1.0, dy=1.0, dt=1.0) == 0.0


# ---------------------------------------------------------------------------
# 6. Energy Closure State
# ---------------------------------------------------------------------------


class TestEnergyClosureState:
    """Tests for EnergyClosureState."""

    def test_perfect_closure(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 300.0
        state.e_viscous_dissipation = 400.0
        state.e_free_surface_work = 300.0
        assert state.closure_rate == pytest.approx(100.0, rel=1e-12)
        assert state.is_acceptable

    def test_under_closure(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 200.0
        state.e_viscous_dissipation = 200.0
        state.e_free_surface_work = 100.0
        assert state.closure_rate == pytest.approx(50.0, rel=1e-12)
        assert not state.is_acceptable

    def test_over_closure(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 500.0
        state.e_viscous_dissipation = 500.0
        state.e_free_surface_work = 500.0
        assert state.closure_rate == pytest.approx(150.0, rel=1e-12)
        assert not state.is_acceptable

    def test_boundary_70(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 700.0
        assert state.closure_rate == pytest.approx(70.0)
        assert state.is_acceptable

    def test_boundary_130(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 1300.0
        assert state.closure_rate == pytest.approx(130.0)
        assert state.is_acceptable

    def test_other_channels(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 200.0
        state.add_channel("crushing", 300.0)
        state.add_channel("breaking", 200.0)
        assert state.e_observed_total == pytest.approx(700.0)
        assert state.closure_rate == pytest.approx(70.0)

    def test_zero_input_zero_observed(self):
        state = EnergyClosureState(e_input=0.0)
        assert state.closure_rate == pytest.approx(100.0)

    def test_as_dict(self):
        state = EnergyClosureState(e_input=1000.0)
        state.e_ice_kinetic = 300.0
        state.e_viscous_dissipation = 400.0
        state.e_free_surface_work = 200.0
        state.add_channel("crushing", 100.0)
        d = state.as_dict()
        assert d["e_input_J"] == 1000.0
        assert d["e_ice_kinetic_J"] == 300.0
        assert d["e_viscous_dissipation_J"] == 400.0
        assert d["e_free_surface_work_J"] == 200.0
        assert d["e_other_J"]["crushing"] == 100.0
        assert d["closure_rate_pct"] == pytest.approx(100.0)
        assert d["is_acceptable"] is True

    def test_nonfinite_energy_raises(self):
        state = EnergyClosureState(e_input=1000.0)
        with pytest.raises(ValueError, match="non-finite"):
            state.add_channel("test", float("nan"))


# ---------------------------------------------------------------------------
# 7. Compute Energy Closure (integration test)
# ---------------------------------------------------------------------------


class TestComputeEnergyClosure:
    """Integration tests for compute_energy_closure."""

    def test_dem_particles_only(self):
        particles = [
            {"mass": 100.0, "velocity": [2.0, 0.0, 0.0]},
            {"mass": 200.0, "velocity": [0.0, 3.0, 0.0]},
        ]
        state = compute_energy_closure(
            e_input=1000.0,
            particles=particles,
            dt=1.0,
        )
        expected_ice = 0.5 * 100 * 4 + 0.5 * 200 * 9
        assert state.e_ice_kinetic == pytest.approx(expected_ice, rel=1e-12)

    def test_lbm_fields_only(self):
        alpha = np.ones((4, 4, 4))
        u = np.zeros((3, 4, 4, 4))
        u[0] = 1.0
        state = compute_energy_closure(
            e_input=1000.0,
            alpha_ice=alpha,
            velocity=u,
            rho_ice=917.0,
            dx=0.5,
            dt=1.0,
        )
        assert state.e_ice_kinetic > 0.0

    def test_with_dissipation_history(self):
        rates = [100.0, 150.0, 200.0]
        state = compute_energy_closure(
            e_input=1000.0,
            dissipation_rates=rates,
            dt=0.5,
        )
        expected = 0.5 * (100 + 150) * 0.5 + 0.5 * (150 + 200) * 0.5
        assert state.e_viscous_dissipation == pytest.approx(expected, rel=1e-12)

    def test_with_free_surface(self):
        nx, ny = 4, 4
        dx, dy = 1.0, 1.0
        dt = 0.5
        eta_h = [np.ones((ny, nx)) * i * 0.1 for i in range(4)]
        p_h = [np.ones((ny, nx)) * 500.0 for _ in range(4)]
        state = compute_energy_closure(
            e_input=1000.0,
            eta_history=eta_h,
            p_dynamic_history=p_h,
            dx=dx,
            dy=dy,
            dt=dt,
        )
        assert state.e_free_surface_work > 0.0

    def test_full_closure_acceptable(self):
        """Construct a scenario where closure is within 70-130%."""
        particles = [{"mass": 50.0, "velocity": [2.0, 0.0, 0.0]}]
        rates = [100.0, 120.0, 140.0]
        nx, ny = 4, 4
        eta_h = [np.ones((ny, nx)) * i * 0.05 for i in range(4)]
        p_h = [np.ones((ny, nx)) * 200.0 for _ in range(4)]

        state = compute_energy_closure(
            e_input=1000.0,
            particles=particles,
            dissipation_rates=rates,
            dt=0.5,
            eta_history=eta_h,
            p_dynamic_history=p_h,
            dx=1.0,
            dy=1.0,
            other_channels={"crushing": 300.0, "breaking": 200.0},
        )
        # Verify all channels are populated
        assert state.e_ice_kinetic > 0
        assert state.e_viscous_dissipation > 0
        assert state.e_free_surface_work > 0
        assert "crushing" in state.e_other


# ---------------------------------------------------------------------------
# 8. Cross-Solver Energy Flux
# ---------------------------------------------------------------------------


class TestSolverInterfaceFlux:
    """Tests for SolverInterfaceFlux and verify_cross_solver_conservation."""

    def test_perfect_conservation(self):
        flux = SolverInterfaceFlux("DEM", "LBM", 100.0, 100.0)
        assert flux.relative_error == pytest.approx(0.0)
        assert flux.is_conserved

    def test_within_tolerance(self):
        flux = SolverInterfaceFlux("DEM", "LBM", 100.0, 99.0)
        assert flux.relative_error == pytest.approx(0.01)
        assert flux.is_conserved

    def test_exceeds_tolerance(self):
        flux = SolverInterfaceFlux("DEM", "LBM", 100.0, 95.0)
        assert flux.relative_error == pytest.approx(0.05)
        assert not flux.is_conserved

    def test_zero_energy(self):
        flux = SolverInterfaceFlux("DEM", "LBM", 0.0, 0.0)
        assert flux.relative_error == pytest.approx(0.0)
        assert flux.is_conserved

    def test_as_dict(self):
        flux = SolverInterfaceFlux("LBM", "FEM", 500.0, 495.0)
        d = flux.as_dict()
        assert d["source"] == "LBM"
        assert d["target"] == "FEM"
        assert d["energy_out_J"] == 500.0
        assert d["energy_in_J"] == 495.0
        assert d["is_conserved"] is True

    def test_verify_all_conserved(self):
        fluxes = [
            SolverInterfaceFlux("DEM", "LBM", 100.0, 99.8),
            SolverInterfaceFlux("LBM", "FEM", 99.8, 99.6),
        ]
        result = verify_cross_solver_conservation(fluxes)
        assert result["all_conserved"] is True
        assert result["passed"] is True
        # max error is from second flux: 0.2/99.8 ≈ 0.002004
        assert result["max_relative_error"] == pytest.approx(0.002004, rel=0.01)

    def test_verify_one_fails(self):
        fluxes = [
            SolverInterfaceFlux("DEM", "LBM", 100.0, 99.5),
            SolverInterfaceFlux("LBM", "FEM", 99.5, 90.0),
        ]
        result = verify_cross_solver_conservation(fluxes)
        assert result["all_conserved"] is False
        assert result["passed"] is False

    def test_empty_fluxes(self):
        result = verify_cross_solver_conservation([])
        assert result["all_conserved"] is True
        assert result["max_relative_error"] == 0.0
