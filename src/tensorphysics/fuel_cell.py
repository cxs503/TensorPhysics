"""PEMFC (Proton Exchange Membrane Fuel Cell) model for hydrogen propulsion.

Implements:
- Nernst equation for reversible cell voltage
- Activation polarization (Butler-Volmer kinetics)
- Ohmic polarization (membrane + contact resistance)
- Concentration polarization (mass transport limitations)
- Polarization curve (V vs. current density)
- Hydrogen consumption rate
- Range and emissions accounting vs. methanol/ammonia pathways

References:
- Larminie & Ellis, "Fuel Cell Systems Explained" (2003)
- Barbir, "PEM Fuel Cells: Theory and Practice" (2012)
- IACS/IMO green ship energy guidelines
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import math


# Physical constants
FARADAY_CONSTANT = 96485.33212  # C/mol
UNIVERSAL_GAS_CONSTANT = 8.314462618  # J/(mol·K)
H2_MOLAR_MASS = 2.016e-3  # kg/mol
H2_LOWER_HEATING_VALUE = 120.0e6  # J/kg (LHV)
H2_HIGHER_HEATING_VALUE = 141.86e6  # J/kg (HHV)


@dataclass
class PEMFCParameters:
    """PEM fuel cell physical and operating parameters.
    
    Default values representative of a marine-grade PEMFC stack
    (e.g., Ballard FCmove-HD or similar).
    """
    # Thermodynamic
    n_cells: int = 100  # Number of cells in stack
    T_operating: float = 353.15  # Operating temperature (K), ~80°C
    P_H2: float = 101325.0  # Hydrogen partial pressure (Pa)
    P_O2: float = 21260.0  # Oxygen partial pressure (Pa, ~0.21 atm)
    
    # Electrochemical
    A_cell: float = 250.0e-4  # Active cell area (m²), 250 cm²
    i_0_anode: float = 1.0e-1  # Anode exchange current density (A/cm²) - Pt/C for HOR
    i_0_cathode: float = 1.0e-4  # Cathode exchange current density (A/cm²) - Pt for ORR
    alpha_anode: float = 0.5  # Anode charge transfer coefficient
    alpha_cathode: float = 1.0  # Cathode charge transfer coefficient
    
    # Membrane
    membrane_thickness: float = 25.0e-6  # m (25 μm thin Nafion for low resistance)
    membrane_conductivity: float = 10.0  # S/m (hydrated Nafion)
    contact_resistance: float = 0.02e-4  # Ω·m² (contact + bipolar plates)
    
    # Mass transport
    i_limit: float = 3.0  # Limiting current density (A/cm²)
    C_conc: float = 0.05  # Concentration loss coefficient (V)
    
    def __post_init__(self) -> None:
        if self.n_cells < 1:
            raise ValueError("n_cells must be >= 1")
        if self.T_operating < 273.15 or self.T_operating > 473.15:
            raise ValueError("Operating temperature out of range [0, 200]°C")
        if self.A_cell <= 0:
            raise ValueError("Cell area must be positive")


def nernst_voltage(params: PEMFCParameters) -> float:
    """Compute reversible (Nernst) cell voltage.
    
    E = E° + (RT/2F) * ln(P_H2 * P_O2^0.5)
    
    where E° = 1.229 V at 25°C, corrected for temperature.
    
    Returns:
        Reversible voltage per cell (V)
    """
    T = params.T_operating
    # Standard potential temperature correction
    E0 = 1.229 - 0.85e-3 * (T - 298.15)
    
    # Nernst correction for partial pressures
    R = UNIVERSAL_GAS_CONSTANT
    nernst_term = (R * T) / (2 * FARADAY_CONSTANT) * math.log(
        (params.P_H2 / 101325.0) * math.sqrt(params.P_O2 / 101325.0)
    )
    
    return E0 + nernst_term


def activation_overpotential(i: float, params: PEMFCParameters) -> float:
    """Compute activation overpotential (anode + cathode).
    
    Simplified Butler-Volmer: η_act = (RT/αF) * asinh(i / (2*i_0))
    
    Args:
        i: Current density (A/cm²)
        params: Cell parameters
    
    Returns:
        Total activation overpotential (V), positive value
    """
    if i <= 0:
        return 0.0
    
    R = UNIVERSAL_GAS_CONSTANT
    T = params.T_operating
    
    # Anode contribution
    eta_a = (R * T) / (params.alpha_anode * FARADAY_CONSTANT) * math.asinh(
        i / (2 * params.i_0_anode)
    )
    
    # Cathode contribution (dominant)
    eta_c = (R * T) / (params.alpha_cathode * FARADAY_CONSTANT) * math.asinh(
        i / (2 * params.i_0_cathode)
    )
    
    return eta_a + eta_c


def ohmic_overpotential(i: float, params: PEMFCParameters) -> float:
    """Compute ohmic overpotential.
    
    η_ohm = i * (t_mem/σ_mem + R_contact)
    
    Args:
        i: Current density (A/cm²)
        params: Cell parameters
    
    Returns:
        Ohmic overpotential (V)
    """
    if i <= 0:
        return 0.0
    
    # Convert i from A/cm² to A/m² for resistance calculation
    i_SI = i * 1e4  # A/m²
    
    # Membrane resistance
    R_mem = params.membrane_thickness / params.membrane_conductivity  # Ω·m²
    
    # Total area-specific resistance
    ASR = R_mem + params.contact_resistance  # Ω·m²
    
    # Overpotential
    eta_ohm = i_SI * ASR
    
    return eta_ohm


def concentration_overpotential(i: float, params: PEMFCParameters) -> float:
    """Compute concentration overpotential.
    
    η_conc = C * ln(i_limit / (i_limit - i))
    
    Args:
        i: Current density (A/cm²)
        params: Cell parameters
    
    Returns:
        Concentration overpotential (V)
    """
    if i <= 0:
        return 0.0
    if i >= params.i_limit:
        return float('inf')  # Mass transport limit
    
    return params.C_conc * math.log(params.i_limit / (params.i_limit - i))


def cell_voltage(i: float, params: PEMFCParameters) -> float:
    """Compute fuel cell voltage at given current density.
    
    V_cell = E_nernst - η_act - η_ohm - η_conc
    
    Args:
        i: Current density (A/cm²)
        params: Stack parameters
    
    Returns:
        Cell voltage (V)
    """
    if i < 0:
        raise ValueError("Current density must be non-negative")
    if i == 0:
        return nernst_voltage(params)
    
    E_rev = nernst_voltage(params)
    eta_act = activation_overpotential(i, params)
    eta_ohm = ohmic_overpotential(i, params)
    eta_conc = concentration_overpotential(i, params)
    
    V = E_rev - eta_act - eta_ohm - eta_conc
    return max(V, 0.0)  # Voltage cannot go negative


def stack_voltage(i: float, params: PEMFCParameters) -> float:
    """Compute stack voltage (n_cells * cell_voltage)."""
    return params.n_cells * cell_voltage(i, params)


def stack_power(i: float, params: PEMFCParameters) -> float:
    """Compute stack electrical power output.
    
    P = V_stack * I = V_stack * i * A_cell
    
    Args:
        i: Current density (A/cm²)
        params: Stack parameters
    
    Returns:
        Stack power (W)
    """
    V = stack_voltage(i, params)
    I = i * params.A_cell * 1e4  # A (convert cm² to m²)
    return V * I


def hydrogen_consumption_rate(i: float, params: PEMFCParameters) -> float:
    """Compute hydrogen molar consumption rate.
    
    From Faraday's law: ṅ_H2 = n_cells * I / (2F)
    
    Args:
        i: Current density (A/cm²)
        params: Stack parameters
    
    Returns:
        H2 consumption rate (mol/s)
    """
    if i <= 0:
        return 0.0
    
    I = i * params.A_cell * 1e4  # Total current (A)
    n_dot = params.n_cells * I / (2 * FARADAY_CONSTANT)
    return n_dot


def hydrogen_mass_rate(i: float, params: PEMFCParameters) -> float:
    """Compute hydrogen mass consumption rate.
    
    Returns:
        H2 consumption rate (g/s)
    """
    return hydrogen_consumption_rate(i, params) * H2_MOLAR_MASS * 1e3  # g/s


def hydrogen_specific_consumption(i: float, params: PEMFCParameters) -> float:
    """Compute specific hydrogen consumption.
    
    Returns:
        H2 consumption (g/kWh_electrical)
    """
    if i <= 0:
        return 0.0
    
    P_elec = stack_power(i, params)  # W
    if P_elec <= 0:
        return float('inf')
    
    m_dot = hydrogen_mass_rate(i, params)  # g/s
    # Convert to g/kWh
    return m_dot * 3600.0 / (P_elec * 1e-3)


def polarization_curve(i_values: list[float], params: PEMFCParameters
                       ) -> dict[str, list[float]]:
    """Compute full polarization curve.
    
    Args:
        i_values: List of current densities (A/cm²)
        params: Stack parameters
    
    Returns:
        Dictionary with keys: 'current_density', 'voltage', 'power',
        'eta_act', 'eta_ohm', 'eta_conc', 'h2_rate_g_s'
    """
    voltages = []
    powers = []
    eta_acts = []
    eta_ohms = []
    eta_concs = []
    h2_rates = []
    
    for i in i_values:
        V = cell_voltage(i, params)
        voltages.append(V)
        powers.append(stack_power(i, params))
        eta_acts.append(activation_overpotential(i, params))
        eta_ohms.append(ohmic_overpotential(i, params))
        eta_concs.append(concentration_overpotential(i, params))
        h2_rates.append(hydrogen_mass_rate(i, params))
    
    return {
        'current_density': list(i_values),
        'voltage': voltages,
        'power': powers,
        'eta_act': eta_acts,
        'eta_ohm': eta_ohms,
        'eta_conc': eta_concs,
        'h2_rate_g_s': h2_rates,
    }


@dataclass
class FuelCellEnergyLedger:
    """Energy and emissions ledger for hydrogen fuel cell propulsion.
    
    Tracks:
    - Hydrogen consumed (g)
    - Electrical energy produced (kWh)
    - Waste heat (kWh)
    - CO2 emissions (g) — zero for green H2
    - Range and endurance
    """
    total_h2_consumed_g: float = 0.0
    total_electrical_kWh: float = 0.0
    total_heat_kWh: float = 0.0
    total_distance_nm: float = 0.0
    h2_tank_capacity_g: float = 5000.0  # 5 kg tank
    ship_power_demand_W: float = 100e3  # 100 kW ship demand
    ship_speed_knots: float = 12.0  # Cruise speed
    
    def update(self, dt_hours: float, i_operating: float, params: PEMFCParameters) -> None:
        """Update ledger for one time step.
        
        Args:
            dt_hours: Time step (hours)
            i_operating: Operating current density (A/cm²)
            params: Fuel cell parameters
        """
        if self.total_h2_consumed_g >= self.h2_tank_capacity_g:
            return  # Tank empty
        
        P_elec = stack_power(i_operating, params)  # W
        m_dot_h2 = hydrogen_mass_rate(i_operating, params)  # g/s
        
        # Actual consumption limited by remaining tank
        h2_available = self.h2_tank_capacity_g - self.total_h2_consumed_g
        h2_consumed = min(m_dot_h2 * dt_hours * 3600, h2_available)
        
        self.total_h2_consumed_g += h2_consumed
        
        # Electrical energy
        self.total_electrical_kWh += P_elec * dt_hours * 1e-3
        
        # Heat (energy balance: H2_LHV * m = P_elec + Q)
        energy_in = h2_consumed * 1e-3 * H2_LOWER_HEATING_VALUE  # J
        energy_elec = P_elec * dt_hours * 3600  # J
        heat = max(energy_in - energy_elec, 0)
        self.total_heat_kWh += heat / 3.6e6
        
        # Distance
        self.total_distance_nm += self.ship_speed_knots * dt_hours
    
    def remaining_range_nm(self, i_operating: float, params: PEMFCParameters) -> float:
        """Estimate remaining range at current operating point."""
        remaining_h2 = self.h2_tank_capacity_g - self.total_h2_consumed_g
        if remaining_h2 <= 0:
            return 0.0
        
        m_dot = hydrogen_mass_rate(i_operating, params)  # g/s
        if m_dot <= 0:
            return float('inf')
        
        endurance_s = remaining_h2 / m_dot
        endurance_h = endurance_s / 3600
        return self.ship_speed_knots * endurance_h
    
    def specific_consumption_g_per_kWh(self) -> float:
        """Overall specific hydrogen consumption (g/kWh)."""
        if self.total_electrical_kWh <= 0:
            return 0.0
        return self.total_h2_consumed_g / self.total_electrical_kWh
    
    def energy_balance_error(self) -> float:
        """Check energy balance closure.
        
        Error = |E_H2_in - E_elec_out - E_heat_out| / E_H2_in
        """
        E_h2 = self.total_h2_consumed_g * 1e-3 * H2_LOWER_HEATING_VALUE / 3.6e6  # kWh
        E_out = self.total_electrical_kWh + self.total_heat_kWh
        if E_h2 <= 0:
            return 0.0
        return abs(E_h2 - E_out) / E_h2


@dataclass
class FuelComparison:
    """Comparison of hydrogen vs. methanol vs. ammonia fuel paths."""
    fuel_type: str  # "hydrogen", "methanol", "ammonia"
    specific_energy_MJ_per_kg: float
    tank_to_wire_efficiency: float  # Overall efficiency
    co2_g_per_kWh: float  # Well-to-wake CO2
    h2_equivalent: bool = False  # Whether this is a hydrogen carrier


def standard_fuel_comparison() -> dict[str, FuelComparison]:
    """Return standard fuel pathway comparison data.
    
    Values representative of marine propulsion:
    - Green hydrogen (PEMFC): zero tank-to-wake CO2
    - Green methanol (MFC/reformer): some CO2 from carbon cycle
    - Green ammonia (cracked to H2): zero CO2 but cracking energy
    """
    return {
        "hydrogen": FuelComparison(
            fuel_type="hydrogen",
            specific_energy_MJ_per_kg=120.0,  # LHV
            tank_to_wire_efficiency=0.55,  # PEMFC system efficiency
            co2_g_per_kWh=0.0,  # Green H2
            h2_equivalent=True,
        ),
        "methanol": FuelComparison(
            fuel_type="methanol",
            specific_energy_MJ_per_kg=19.9,  # LHV
            tank_to_wire_efficiency=0.40,  # Methanol fuel cell / engine
            co2_g_per_kWh=430.0,  # Green methanol (carbon-neutral cycle)
            h2_equivalent=False,
        ),
        "ammonia": FuelComparison(
            fuel_type="ammonia",
            specific_energy_MJ_per_kg=18.6,  # LHV
            tank_to_wire_efficiency=0.35,  # Cracking + fuel cell
            co2_g_per_kWh=0.0,  # Green NH3
            h2_equivalent=False,  # Requires cracking to H2
        ),
    }


def compare_fuel_paths(power_kW: float, range_nm: float, speed_knots: float
                       ) -> dict[str, dict[str, float]]:
    """Compare fuel consumption and emissions for different pathways.
    
    Args:
        power_kW: Required propulsion power (kW)
        range_nm: Required range (nautical miles)
        speed_knots: Cruise speed (knots)
    
    Returns:
        Dictionary with fuel mass (kg), tank volume (m³), CO2 (g) per pathway
    """
    endurance_h = range_nm / speed_knots
    energy_kWh = power_kW * endurance_h
    
    fuels = standard_fuel_comparison()
    results: dict[str, dict[str, float]] = {}
    
    for name, fuel in fuels.items():
        energy_needed = energy_kWh / fuel.tank_to_wire_efficiency  # kWh thermal
        fuel_mass_kg = energy_needed * 3.6 / fuel.specific_energy_MJ_per_kg
        co2_total = energy_kWh * fuel.co2_g_per_kWh
        
        # Approximate tank volume (density-based)
        densities = {"hydrogen": 70.8, "methanol": 792.0, "ammonia": 682.0}  # kg/m³
        tank_volume = fuel_mass_kg / densities.get(name, 700.0)
        
        results[name] = {
            "fuel_mass_kg": fuel_mass_kg,
            "tank_volume_m3": tank_volume,
            "co2_emissions_g": co2_total,
            "endurance_hours": endurance_h,
            "energy_kWh": energy_kWh,
        }
    
    return results
