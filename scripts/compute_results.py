"""Generate numerical results for the report."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import json
import numpy as np

from tensorphysics.fuel_cell import (
    PEMFCParameters, polarization_curve, cell_voltage, nernst_voltage,
    hydrogen_specific_consumption, FuelCellEnergyLedger, compare_fuel_paths
)
from tensorphysics.multifield_coupling import (
    MultiFieldCouplingSimulator, SubCyclingConfig,
    SimpleLBMSolver, SimpleFEMSolver, SimpleDEMSolver, CouplingDomain
)


def compute_polarization_results():
    """Compute polarization curve results."""
    params = PEMFCParameters()
    
    # Key operating points
    operating_points = [0.1, 0.3, 0.5, 0.8, 1.0, 1.2, 1.5]
    results = {}
    
    for i in operating_points:
        V = cell_voltage(i, params)
        sc = hydrogen_specific_consumption(i, params)
        results[f"i_{i:.1f}"] = {
            "current_density_A_cm2": i,
            "voltage_V": round(V, 4),
            "specific_consumption_g_kWh": round(sc, 2) if sc < 1e6 else None
        }
    
    # Open circuit voltage
    results["OCV"] = {
        "voltage_V": round(nernst_voltage(params), 4)
    }
    
    return results


def compute_coupling_results():
    """Compute multi-field coupling results."""
    results = {}
    
    # Test different sub-cycling ratios
    ratios = [1, 2, 4, 8, 16]
    conservation_errors = {}
    
    for ratio in ratios:
        lbm = SimpleLBMSolver(energy=100.0)
        fem = SimpleFEMSolver(energy=50.0)
        dem = SimpleDEMSolver(energy=30.0)
        sim = MultiFieldCouplingSimulator([lbm, fem, dem], sub_cycling=SubCyclingConfig(ratio=ratio))
        result = sim.run(dt=0.1, total_time=2.0)
        
        max_residual = max(result.conservation_residuals)
        initial_energy = 180.0  # 100 + 50 + 30
        error_fraction = max_residual / initial_energy if initial_energy > 0 else 0
        conservation_errors[f"ratio_{ratio}"] = {
            "max_residual_J": round(max_residual, 6),
            "error_fraction": round(error_fraction, 8),
            "error_percent": round(error_fraction * 100, 4)
        }
    
    results["conservation_errors"] = conservation_errors
    
    # Interface flux balance check
    lbm = SimpleLBMSolver(energy=100.0)
    fem = SimpleFEMSolver(energy=50.0)
    dem = SimpleDEMSolver(energy=30.0)
    sim = MultiFieldCouplingSimulator([lbm, fem, dem], sub_cycling=SubCyclingConfig(ratio=4))
    result = sim.run(dt=0.05, total_time=1.0)
    
    results["interface_flux_count"] = len(result.final_ledger.interface_fluxes)
    results["final_total_energy_J"] = round(sum(
        result.energy_history[d.value][-1] for d in CouplingDomain
    ), 4)
    results["initial_total_energy_J"] = 180.0
    
    return results


def compute_energy_ledger_results():
    """Compute fuel cell energy ledger results."""
    params = PEMFCParameters()
    ledger = FuelCellEnergyLedger(h2_tank_capacity_g=5000.0)
    
    # Simulate 10 hours of operation
    for _ in range(10):
        ledger.update(dt_hours=1.0, i_operating=0.6, params=params)
    
    results = {
        "total_h2_consumed_g": round(ledger.total_h2_consumed_g, 2),
        "total_electrical_kWh": round(ledger.total_electrical_kWh, 2),
        "total_heat_kWh": round(ledger.total_heat_kWh, 2),
        "total_distance_nm": round(ledger.total_distance_nm, 1),
        "specific_consumption_g_kWh": round(ledger.specific_consumption_g_per_kWh(), 2),
        "energy_balance_error_fraction": round(ledger.energy_balance_error(), 6),
        "remaining_range_nm": round(ledger.remaining_range_nm(0.6, params), 1)
    }
    
    return results


def compute_fuel_comparison_results():
    """Compute fuel pathway comparison results."""
    result = compare_fuel_paths(power_kW=500.0, range_nm=2000.0, speed_knots=14.0)
    
    # Format for report
    formatted = {}
    for fuel, data in result.items():
        formatted[fuel] = {
            "fuel_mass_kg": round(data["fuel_mass_kg"], 1),
            "tank_volume_m3": round(data["tank_volume_m3"], 2),
            "co2_emissions_kg": round(data["co2_emissions_g"] / 1000, 1),
            "endurance_hours": round(data["endurance_hours"], 1)
        }
    
    return formatted


def main():
    """Generate all numerical results."""
    results = {
        "polarization": compute_polarization_results(),
        "coupling": compute_coupling_results(),
        "energy_ledger": compute_energy_ledger_results(),
        "fuel_comparison": compute_fuel_comparison_results()
    }
    
    # Save results
    output_path = '/root/tp-agent/work/m7382/artifacts/numerical_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to: {output_path}")
    print(json.dumps(results, indent=2))
    
    return results


if __name__ == '__main__':
    main()
