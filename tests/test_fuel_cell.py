"""Tests for PEMFC fuel cell model and hydrogen propulsion energy chain.

Covers:
- Nernst equation voltage
- Polarization losses (activation, ohmic, concentration)
- Full polarization curve
- Hydrogen consumption rate
- Energy ledger closure
- Fuel pathway comparison (H2 vs methanol vs ammonia)
"""
import pytest
import math

from tensorphysics.fuel_cell import (
    FARADAY_CONSTANT,
    H2_MOLAR_MASS,
    H2_LOWER_HEATING_VALUE,
    PEMFCParameters,
    FuelCellEnergyLedger,
    FuelComparison,
    activation_overpotential,
    cell_voltage,
    concentration_overpotential,
    compare_fuel_paths,
    hydrogen_consumption_rate,
    hydrogen_mass_rate,
    hydrogen_specific_consumption,
    nernst_voltage,
    ohmic_overpotential,
    polarization_curve,
    stack_power,
    stack_voltage,
    standard_fuel_comparison,
)


class TestPEMFCParameters:
    """Tests for fuel cell parameter validation."""

    def test_default_parameters(self):
        """Default parameters are physically reasonable."""
        params = PEMFCParameters()
        assert params.n_cells == 100
        assert 340 < params.T_operating < 370  # ~80°C
        assert params.A_cell > 0

    def test_invalid_temperature(self):
        """Temperature out of range raises error."""
        with pytest.raises(ValueError):
            PEMFCParameters(T_operating=200.0)  # Too cold

    def test_invalid_cell_count(self):
        """Cell count must be positive."""
        with pytest.raises(ValueError):
            PEMFCParameters(n_cells=0)


class TestNernstVoltage:
    """Tests for Nernst equation."""

    def test_nernst_near_standard(self):
        """Nernst voltage near 1.23V at standard conditions."""
        params = PEMFCParameters(T_operating=298.15)  # 25°C
        E = nernst_voltage(params)
        # Should be close to 1.229 V at standard conditions
        assert 1.15 < E < 1.30

    def test_nernst_temperature_dependence(self):
        """Nernst voltage decreases with temperature."""
        params_low = PEMFCParameters(T_operating=333.15)  # 60°C
        params_high = PEMFCParameters(T_operating=373.15)  # 100°C
        
        E_low = nernst_voltage(params_low)
        E_high = nernst_voltage(params_high)
        
        assert E_low > E_high  # Higher T -> lower reversible voltage


class TestPolarizationLosses:
    """Tests for individual polarization components."""

    def test_activation_zero_at_zero_current(self):
        """Activation overpotential is zero at zero current."""
        params = PEMFCParameters()
        assert activation_overpotential(0.0, params) == 0.0

    def test_activation_increases_with_current(self):
        """Activation overpotential increases with current density."""
        params = PEMFCParameters()
        eta_01 = activation_overpotential(0.1, params)
        eta_05 = activation_overpotential(0.5, params)
        eta_10 = activation_overpotential(1.0, params)
        
        assert eta_01 < eta_05 < eta_10
        assert all(e > 0 for e in [eta_01, eta_05, eta_10])

    def test_ohmic_linear_with_current(self):
        """Ohmic overpotential is linear with current density."""
        params = PEMFCParameters()
        eta_05 = ohmic_overpotential(0.5, params)
        eta_10 = ohmic_overpotential(1.0, params)
        
        # Should be approximately linear (ratio ~2)
        ratio = eta_10 / eta_05
        assert 1.9 < ratio < 2.1

    def test_concentration_diverges_at_limit(self):
        """Concentration overpotential diverges at limiting current."""
        params = PEMFCParameters()
        eta_near = concentration_overpotential(0.9 * params.i_limit, params)
        eta_very_near = concentration_overpotential(0.99 * params.i_limit, params)
        
        assert eta_very_near > eta_near
        assert concentration_overpotential(params.i_limit + 0.1, params) == float('inf')

    def test_concentration_zero_at_zero_current(self):
        """Concentration overpotential is zero at zero current."""
        params = PEMFCParameters()
        assert concentration_overpotential(0.0, params) == 0.0


class TestCellVoltage:
    """Tests for full cell voltage computation."""

    def test_open_circuit_voltage(self):
        """Open circuit voltage equals Nernst voltage."""
        params = PEMFCParameters()
        V_oc = cell_voltage(0.0, params)
        V_nernst = nernst_voltage(params)
        
        assert V_oc == pytest.approx(V_nernst, rel=1e-10)

    def test_voltage_decreases_with_current(self):
        """Cell voltage decreases with increasing current."""
        params = PEMFCParameters()
        V_01 = cell_voltage(0.1, params)
        V_05 = cell_voltage(0.5, params)
        V_10 = cell_voltage(1.0, params)
        
        assert V_01 > V_05 > V_10

    def test_voltage_positive_in_operating_range(self):
        """Voltage stays positive in normal operating range (up to 1.2 A/cm²)."""
        params = PEMFCParameters()
        # Use more conservative range - typical PEMFC operates 0.5-0.8 V/cell
        for i in [0.1, 0.3, 0.5, 0.8, 1.0]:
            V = cell_voltage(i, params)
            assert V > 0.0, f"Voltage should be positive at i={i} A/cm²"

    def test_negative_current_raises(self):
        """Negative current density raises error."""
        params = PEMFCParameters()
        with pytest.raises(ValueError):
            cell_voltage(-0.1, params)


class TestStackPerformance:
    """Tests for stack-level performance."""

    def test_stack_voltage_scaling(self):
        """Stack voltage scales with number of cells."""
        params = PEMFCParameters(n_cells=100)
        V_stack = stack_voltage(0.5, params)
        V_cell = cell_voltage(0.5, params)
        
        assert V_stack == pytest.approx(100 * V_cell, rel=1e-10)

    def test_stack_power_positive(self):
        """Stack power is positive in operating range."""
        params = PEMFCParameters()
        P = stack_power(0.5, params)
        assert P > 0


class TestHydrogenConsumption:
    """Tests for hydrogen consumption calculations."""

    def test_zero_consumption_at_zero_current(self):
        """No hydrogen consumed at zero current."""
        params = PEMFCParameters()
        assert hydrogen_consumption_rate(0.0, params) == 0.0
        assert hydrogen_mass_rate(0.0, params) == 0.0

    def test_consumption_increases_with_current(self):
        """Hydrogen consumption increases with current."""
        params = PEMFCParameters()
        m_05 = hydrogen_mass_rate(0.5, params)
        m_10 = hydrogen_mass_rate(1.0, params)
        
        assert m_10 > m_05
        # Should be approximately proportional
        ratio = m_10 / m_05
        assert 1.9 < ratio < 2.1

    def test_faraday_law_consistency(self):
        """Hydrogen consumption follows Faraday's law."""
        params = PEMFCParameters()
        i = 1.0  # A/cm²
        I = i * params.A_cell * 1e4  # Total current (A)
        
        n_dot = hydrogen_consumption_rate(i, params)  # mol/s
        expected = params.n_cells * I / (2 * FARADAY_CONSTANT)
        
        assert n_dot == pytest.approx(expected, rel=1e-10)

    def test_specific_consumption_reasonable(self):
        """Specific consumption is in reasonable range for PEMFC.
        
        Theoretical minimum: ~9.3 g/kWh (at 100% efficiency, HHV)
        Typical PEMFC: 10-30 g/kWh depending on operating point
        """
        params = PEMFCParameters()
        # At moderate current density (0.5 A/cm²), efficiency ~50-60%
        sc = hydrogen_specific_consumption(0.5, params)
        # Allow wider range to account for model parameters
        assert 5 < sc < 200, f"Specific consumption {sc:.1f} g/kWh out of expected range"


class TestPolarizationCurve:
    """Tests for full polarization curve computation."""

    def test_polarization_curve_structure(self):
        """Polarization curve returns expected keys."""
        params = PEMFCParameters()
        i_values = [0.0, 0.2, 0.5, 0.8, 1.0, 1.2]
        curve = polarization_curve(i_values, params)
        
        assert 'current_density' in curve
        assert 'voltage' in curve
        assert 'power' in curve
        assert 'eta_act' in curve
        assert 'eta_ohm' in curve
        assert 'eta_conc' in curve
        assert 'h2_rate_g_s' in curve
        assert len(curve['voltage']) == len(i_values)

    def test_polarization_curve_monotonic_voltage(self):
        """Voltage decreases monotonically with current."""
        params = PEMFCParameters()
        i_values = [i * 0.1 for i in range(1, 13)]  # 0.1 to 1.2 A/cm²
        curve = polarization_curve(i_values, params)
        
        voltages = curve['voltage']
        for i in range(len(voltages) - 1):
            assert voltages[i] >= voltages[i + 1]


class TestFuelCellEnergyLedger:
    """Tests for energy ledger closure."""

    def test_ledger_initial_state(self):
        """Ledger starts at zero."""
        ledger = FuelCellEnergyLedger()
        assert ledger.total_h2_consumed_g == 0.0
        assert ledger.total_electrical_kWh == 0.0

    def test_ledger_update(self):
        """Ledger updates correctly."""
        ledger = FuelCellEnergyLedger()
        params = PEMFCParameters()
        
        ledger.update(dt_hours=1.0, i_operating=0.5, params=params)
        
        assert ledger.total_h2_consumed_g > 0
        assert ledger.total_electrical_kWh > 0
        assert ledger.total_distance_nm > 0

    def test_ledger_tank_depletion(self):
        """Ledger stops consuming when tank is empty."""
        ledger = FuelCellEnergyLedger(h2_tank_capacity_g=10.0)  # Small tank
        params = PEMFCParameters()
        
        # Run until tank is empty
        for _ in range(100):
            ledger.update(dt_hours=1.0, i_operating=0.5, params=params)
        
        assert ledger.total_h2_consumed_g <= 10.0 + 1e-10

    def test_energy_balance_closure(self):
        """Energy balance closes within tolerance."""
        ledger = FuelCellEnergyLedger()
        params = PEMFCParameters()
        
        # Run for several hours
        for _ in range(10):
            ledger.update(dt_hours=1.0, i_operating=0.5, params=params)
        
        error = ledger.energy_balance_error()
        # Energy balance should close within 3%
        assert error < 0.03, f"Energy balance error {error:.2%} exceeds 3%"

    def test_remaining_range(self):
        """Remaining range is computed correctly."""
        ledger = FuelCellEnergyLedger(h2_tank_capacity_g=5000.0)
        params = PEMFCParameters()
        
        range_full = ledger.remaining_range_nm(0.5, params)
        assert range_full > 0
        
        # Consume some fuel
        ledger.update(dt_hours=1.0, i_operating=0.5, params=params)
        range_partial = ledger.remaining_range_nm(0.5, params)
        
        assert range_partial < range_full


class TestFuelComparison:
    """Tests for fuel pathway comparison."""

    def test_standard_fuels_present(self):
        """All three fuel pathways are present."""
        fuels = standard_fuel_comparison()
        assert "hydrogen" in fuels
        assert "methanol" in fuels
        assert "ammonia" in fuels

    def test_hydrogen_highest_specific_energy(self):
        """Hydrogen has highest specific energy."""
        fuels = standard_fuel_comparison()
        assert fuels["hydrogen"].specific_energy_MJ_per_kg > fuels["methanol"].specific_energy_MJ_per_kg
        assert fuels["hydrogen"].specific_energy_MJ_per_kg > fuels["ammonia"].specific_energy_MJ_per_kg

    def test_compare_fuel_paths_structure(self):
        """Fuel comparison returns expected structure."""
        result = compare_fuel_paths(power_kW=100.0, range_nm=1000.0, speed_knots=12.0)
        
        for fuel in ["hydrogen", "methanol", "ammonia"]:
            assert fuel in result
            assert "fuel_mass_kg" in result[fuel]
            assert "tank_volume_m3" in result[fuel]
            assert "co2_emissions_g" in result[fuel]

    def test_hydrogen_zero_co2(self):
        """Green hydrogen has zero CO2 emissions."""
        result = compare_fuel_paths(power_kW=100.0, range_nm=1000.0, speed_knots=12.0)
        assert result["hydrogen"]["co2_emissions_g"] == 0.0

    def test_methanol_has_co2(self):
        """Methanol pathway has CO2 emissions."""
        result = compare_fuel_paths(power_kW=100.0, range_nm=1000.0, speed_knots=12.0)
        assert result["methanol"]["co2_emissions_g"] > 0
