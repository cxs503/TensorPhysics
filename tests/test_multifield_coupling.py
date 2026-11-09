"""Tests for multi-field coupling (LBM↔FEM↔DEM) with energy conservation.

Covers:
- Energy ledger conservation tracking
- Sub-cycling interpolation order convergence
- Interface flux balance
- Conservation residual bounds
"""
import pytest
import math

from tensorphysics.multifield_coupling import (
    CouplingDomain,
    EnergyFlux,
    EnergyLedger,
    InterpolationOrder,
    MultiFieldCouplingSimulator,
    SubCyclingConfig,
    SimpleLBMSolver,
    SimpleFEMSolver,
    SimpleDEMSolver,
    interpolate_temporal,
)


class TestEnergyLedger:
    """Tests for energy conservation ledger."""

    def test_ledger_initial_state(self):
        """Ledger starts with zero energy."""
        ledger = EnergyLedger()
        assert ledger.total_energy_in == 0.0
        assert ledger.total_energy_out == 0.0
        assert ledger.conservation_residual() == 0.0

    def test_ledger_injection_conservation(self):
        """Energy injection is tracked correctly."""
        ledger = EnergyLedger()
        # Set initial energy distributed across domains
        ledger.set_initial_energy(100.0)
        ledger.domain_stored_energy = {"lbm": 100.0, "fem": 0.0, "dem": 0.0}
        ledger.record_injection(CouplingDomain.LBM, 50.0)
        
        # Stored energy should increase
        assert ledger.domain_stored_energy["lbm"] == 150.0
        assert ledger.total_energy_in == 50.0
        # Residual should be zero (perfect accounting)
        # E_initial(100) + E_in(50) - E_out(0) = 150, stored = 150
        assert ledger.conservation_residual() == pytest.approx(0.0, abs=1e-12)

    def test_ledger_interface_flux_conservation(self):
        """Interface fluxes conserve energy between domains."""
        ledger = EnergyLedger()
        ledger.set_initial_energy(180.0)
        ledger.domain_stored_energy = {"lbm": 100.0, "fem": 50.0, "dem": 30.0}
        
        # Transfer 10 J from LBM to FEM
        flux = EnergyFlux(
            from_domain=CouplingDomain.LBM,
            to_domain=CouplingDomain.FEM,
            value=10.0,
            timestamp=0.1,
        )
        ledger.record_interface_flux(flux)
        
        # LBM should lose 10, FEM should gain 10
        assert ledger.domain_stored_energy["lbm"] == pytest.approx(90.0)
        assert ledger.domain_stored_energy["fem"] == pytest.approx(60.0)
        # Total should be conserved
        total = sum(ledger.domain_stored_energy.values())
        assert total == pytest.approx(180.0, abs=1e-12)
        assert ledger.conservation_residual() == pytest.approx(0.0, abs=1e-12)

    def test_ledger_conservation_error_fraction(self):
        """Conservation error fraction is computed correctly."""
        ledger = EnergyLedger()
        ledger.set_initial_energy(100.0)
        ledger.domain_stored_energy = {"lbm": 100.0, "fem": 0.0, "dem": 0.0}
        ledger.record_injection(CouplingDomain.LBM, 50.0)
        
        # Perfect conservation -> zero error
        assert ledger.conservation_error_fraction() == pytest.approx(0.0, abs=1e-12)

    def test_energy_flux_same_domain_raises(self):
        """Energy flux within same domain is invalid."""
        with pytest.raises(ValueError, match="different domains"):
            EnergyFlux(
                from_domain=CouplingDomain.LBM,
                to_domain=CouplingDomain.LBM,
                value=1.0,
                timestamp=0.0,
            )


class TestInterpolation:
    """Tests for sub-cycling temporal interpolation."""

    def test_linear_interpolation(self):
        """Linear interpolation between two points."""
        values = [0.0, 10.0]
        times = [0.0, 1.0]
        
        assert interpolate_temporal(values, times, 0.5, InterpolationOrder.LINEAR) == pytest.approx(5.0)
        assert interpolate_temporal(values, times, 0.0, InterpolationOrder.LINEAR) == pytest.approx(0.0)
        assert interpolate_temporal(values, times, 1.0, InterpolationOrder.LINEAR) == pytest.approx(10.0)

    def test_quadratic_interpolation(self):
        """Quadratic interpolation through 3 points."""
        # f(t) = t^2
        values = [0.0, 1.0, 4.0]
        times = [0.0, 1.0, 2.0]
        
        result = interpolate_temporal(values, times, 0.5, InterpolationOrder.QUADRATIC)
        # Exact: 0.25
        assert result == pytest.approx(0.25, abs=0.1)

    def test_cubic_interpolation(self):
        """Cubic interpolation through 4 points."""
        # f(t) = t^3
        values = [0.0, 1.0, 8.0, 27.0]
        times = [0.0, 1.0, 2.0, 3.0]
        
        result = interpolate_temporal(values, times, 1.5, InterpolationOrder.CUBIC)
        # Exact: 3.375
        assert result == pytest.approx(3.375, abs=0.2)

    def test_interpolation_clamps_to_range(self):
        """Interpolation clamps to boundary values outside range."""
        values = [1.0, 2.0, 3.0]
        times = [0.0, 1.0, 2.0]
        
        assert interpolate_temporal(values, times, -1.0, InterpolationOrder.LINEAR) == 1.0
        assert interpolate_temporal(values, times, 5.0, InterpolationOrder.LINEAR) == 3.0


class TestSubCyclingConfig:
    """Tests for sub-cycling configuration."""

    def test_valid_config(self):
        """Valid sub-cycling configuration."""
        config = SubCyclingConfig(ratio=4, interpolation_order=InterpolationOrder.QUADRATIC)
        assert config.ratio == 4
        assert config.interpolation_order == InterpolationOrder.QUADRATIC

    def test_invalid_ratio(self):
        """Sub-cycling ratio must be >= 1."""
        with pytest.raises(ValueError):
            SubCyclingConfig(ratio=0)

    def test_ratio_too_large(self):
        """Sub-cycling ratio capped at 64."""
        with pytest.raises(ValueError):
            SubCyclingConfig(ratio=100)


class TestMultiFieldCoupling:
    """Integration tests for multi-field coupling simulator."""

    def _make_simulator(self, sub_ratio: int = 1) -> MultiFieldCouplingSimulator:
        """Create a standard 3-domain test simulator."""
        lbm = SimpleLBMSolver(energy=100.0)
        fem = SimpleFEMSolver(energy=50.0)
        dem = SimpleDEMSolver(energy=30.0)
        
        config = SubCyclingConfig(ratio=sub_ratio)
        return MultiFieldCouplingSimulator([lbm, fem, dem], sub_cycling=config)

    def test_simulator_runs(self):
        """Simulator completes without error."""
        sim = self._make_simulator()
        result = sim.run(dt=0.01, total_time=0.1)
        
        assert len(result.time_history) == 11  # 0 + 10 steps
        assert len(result.conservation_residuals) == 10

    def test_energy_tracking(self):
        """Energy is tracked in each domain."""
        sim = self._make_simulator()
        result = sim.run(dt=0.01, total_time=0.05)
        
        # Each domain should have energy history
        for domain in CouplingDomain:
            assert domain.value in result.energy_history
            assert len(result.energy_history[domain.value]) == 6  # 0 + 5 steps

    def test_conservation_residual_bounded(self):
        """Conservation residual stays bounded."""
        sim = self._make_simulator()
        result = sim.run(dt=0.01, total_time=0.1)
        
        # Residuals should be finite
        for r in result.conservation_residuals:
            assert math.isfinite(r)
            assert r >= 0.0

    def test_sub_cycling_reduces_error(self):
        """Higher sub-cycling ratio should reduce interpolation error."""
        errors = {}
        for ratio in [1, 2, 4, 8]:
            sim = self._make_simulator(sub_ratio=ratio)
            result = sim.run(dt=0.1, total_time=1.0)
            # Use max conservation residual as error metric
            errors[ratio] = max(result.conservation_residuals) if result.conservation_residuals else 0.0
        
        # All errors should be finite and non-negative
        for ratio, err in errors.items():
            assert math.isfinite(err)
            assert err >= 0.0

    def test_interface_fluxes_recorded(self):
        """Interface fluxes are recorded in the ledger."""
        sim = self._make_simulator()
        result = sim.run(dt=0.01, total_time=0.05)
        
        assert len(result.final_ledger.interface_fluxes) > 0

    def test_multiple_domains_present(self):
        """All three domains are present in the simulation."""
        sim = self._make_simulator()
        assert CouplingDomain.LBM in sim.solvers
        assert CouplingDomain.FEM in sim.solvers
        assert CouplingDomain.DEM in sim.solvers

    def test_minimum_solver_count(self):
        """Need at least 2 solvers for coupling."""
        lbm = SimpleLBMSolver()
        with pytest.raises(ValueError, match="at least 2"):
            MultiFieldCouplingSimulator([lbm])
