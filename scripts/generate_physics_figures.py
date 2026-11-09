"""Generate figures for the multi-field coupling and fuel cell report."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import base64
from io import BytesIO

from tensorphysics.fuel_cell import (
    PEMFCParameters, polarization_curve, cell_voltage, nernst_voltage,
    activation_overpotential, ohmic_overpotential, concentration_overpotential,
    compare_fuel_paths, FuelCellEnergyLedger
)
from tensorphysics.multifield_coupling import (
    MultiFieldCouplingSimulator, SubCyclingConfig,
    SimpleLBMSolver, SimpleFEMSolver, SimpleDEMSolver, CouplingDomain
)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 string."""
    buf = BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_polarization_curve_figure():
    """Generate PEMFC polarization curve figure."""
    params = PEMFCParameters()
    i_values = np.linspace(0.01, 2.0, 50).tolist()
    curve = polarization_curve(i_values, params)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Voltage vs Current Density
    ax1 = axes[0]
    ax1.plot(curve['current_density'], curve['voltage'], 'b-', linewidth=2, label='Cell Voltage')
    ax1.axhline(y=nernst_voltage(params), color='gray', linestyle='--', alpha=0.5, label='Nernst (OCV)')
    ax1.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax1.set_ylabel('Cell Voltage (V)', fontsize=12)
    ax1.set_title('PEMFC Polarization Curve', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 2.2])
    ax1.set_ylim([0, 1.3])
    
    # Annotate operating region
    ax1.axvspan(0.4, 1.0, alpha=0.1, color='green', label='Typical Operating Range')
    
    # Right: Power Density
    ax2 = axes[1]
    ax2.plot(curve['current_density'], curve['power'], 'r-', linewidth=2, label='Stack Power')
    ax2.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax2.set_ylabel('Stack Power (W)', fontsize=12)
    ax2.set_title('Power Output (100-cell stack)', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Mark maximum power point
    max_idx = np.argmax(curve['power'])
    ax2.plot(curve['current_density'][max_idx], curve['power'][max_idx], 'ko', markersize=10)
    ax2.annotate(f'MPP: {curve["power"][max_idx]:.0f} W', 
                xy=(curve['current_density'][max_idx], curve['power'][max_idx]),
                xytext=(curve['current_density'][max_idx]+0.3, curve['power'][max_idx]-500),
                arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_loss_breakdown_figure():
    """Generate polarization loss breakdown figure."""
    params = PEMFCParameters()
    i_values = np.linspace(0.01, 2.0, 50).tolist()
    curve = polarization_curve(i_values, params)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(curve['current_density'], curve['eta_act'], 'g-', linewidth=2, label='Activation')
    ax.plot(curve['current_density'], curve['eta_ohm'], 'b-', linewidth=2, label='Ohmic')
    ax.plot(curve['current_density'], curve['eta_conc'], 'r-', linewidth=2, label='Concentration')
    ax.plot(curve['current_density'], 
           [a+o+c for a,o,c in zip(curve['eta_act'], curve['eta_ohm'], curve['eta_conc'])],
           'k--', linewidth=2, label='Total Loss')
    
    ax.set_xlabel('Current Density (A/cm²)', fontsize=12)
    ax.set_ylabel('Overpotential (V)', fontsize=12)
    ax.set_title('Polarization Loss Breakdown', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 2.2])
    
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_energy_conservation_figure():
    """Generate energy conservation figure for multi-field coupling."""
    # Run simulation with different sub-cycling ratios
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Left: Energy evolution
    ax1 = axes[0]
    lbm = SimpleLBMSolver(energy=100.0)
    fem = SimpleFEMSolver(energy=50.0)
    dem = SimpleDEMSolver(energy=30.0)
    sim = MultiFieldCouplingSimulator([lbm, fem, dem], sub_cycling=SubCyclingConfig(ratio=4))
    result = sim.run(dt=0.05, total_time=2.0)
    
    ax1.plot(result.time_history, result.energy_history['lbm'], 'b-', label='LBM (fluid)')
    ax1.plot(result.time_history, result.energy_history['fem'], 'g-', label='FEM (structure)')
    ax1.plot(result.time_history, result.energy_history['dem'], 'r-', label='DEM (particles)')
    
    total_energy = [sum(result.energy_history[d][i] for d in CouplingDomain) 
                   for i in range(len(result.time_history))]
    ax1.plot(result.time_history, total_energy, 'k--', linewidth=2, label='Total')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Energy (J)', fontsize=12)
    ax1.set_title('Energy Evolution (sub-cycling ratio=4)', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Right: Conservation residual vs sub-cycling ratio
    ax2 = axes[1]
    ratios = [1, 2, 4, 8, 16]
    max_residuals = []
    
    for ratio in ratios:
        lbm = SimpleLBMSolver(energy=100.0)
        fem = SimpleFEMSolver(energy=50.0)
        dem = SimpleDEMSolver(energy=30.0)
        sim = MultiFieldCouplingSimulator([lbm, fem, dem], sub_cycling=SubCyclingConfig(ratio=ratio))
        result = sim.run(dt=0.1, total_time=2.0)
        max_residuals.append(max(result.conservation_residuals))
    
    ax2.semilogy(ratios, max_residuals, 'bo-', linewidth=2, markersize=8)
    ax2.set_xlabel('Sub-cycling Ratio', fontsize=12)
    ax2.set_ylabel('Max Conservation Residual (J)', fontsize=12)
    ax2.set_title('Conservation Error vs Sub-cycling Ratio', fontsize=14)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_xticks(ratios)
    
    plt.tight_layout()
    return fig_to_base64(fig)


def generate_fuel_comparison_figure():
    """Generate fuel pathway comparison figure."""
    result = compare_fuel_paths(power_kW=500.0, range_nm=2000.0, speed_knots=14.0)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    fuels = ['hydrogen', 'methanol', 'ammonia']
    colors = ['blue', 'orange', 'green']
    labels = ['Green H₂', 'Green Methanol', 'Green NH₃']
    
    # Fuel mass
    masses = [result[f]['fuel_mass_kg'] for f in fuels]
    axes[0].bar(labels, masses, color=colors, alpha=0.7)
    axes[0].set_ylabel('Fuel Mass (kg)', fontsize=12)
    axes[0].set_title('Fuel Mass Required', fontsize=14)
    axes[0].grid(True, alpha=0.3, axis='y')
    for i, m in enumerate(masses):
        axes[0].text(i, m + 50, f'{m:.0f}', ha='center', fontsize=10)
    
    # Tank volume
    volumes = [result[f]['tank_volume_m3'] for f in fuels]
    axes[1].bar(labels, volumes, color=colors, alpha=0.7)
    axes[1].set_ylabel('Tank Volume (m³)', fontsize=12)
    axes[1].set_title('Tank Volume Required', fontsize=14)
    axes[1].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(volumes):
        axes[1].text(i, v + 0.5, f'{v:.1f}', ha='center', fontsize=10)
    
    # CO2 emissions
    co2 = [result[f]['co2_emissions_g'] for f in fuels]
    axes[2].bar(labels, co2, color=colors, alpha=0.7)
    axes[2].set_ylabel('CO₂ Emissions (g)', fontsize=12)
    axes[2].set_title('CO₂ Emissions (well-to-wake)', fontsize=14)
    axes[2].grid(True, alpha=0.3, axis='y')
    for i, c in enumerate(co2):
        axes[2].text(i, c + 5000, f'{c/1000:.0f} kg', ha='center', fontsize=10)
    
    plt.suptitle('Fuel Pathway Comparison: 500 kW, 2000 nm @ 14 knots', fontsize=14, y=1.02)
    plt.tight_layout()
    return fig_to_base64(fig)


def main():
    """Generate all figures and save to artifacts."""
    os.makedirs('/root/tp-agent/work/m7382/artifacts', exist_ok=True)
    
    figures = {}
    
    print("Generating polarization curve figure...")
    figures['polarization_curve'] = generate_polarization_curve_figure()
    
    print("Generating loss breakdown figure...")
    figures['loss_breakdown'] = generate_loss_breakdown_figure()
    
    print("Generating energy conservation figure...")
    figures['energy_conservation'] = generate_energy_conservation_figure()
    
    print("Generating fuel comparison figure...")
    figures['fuel_comparison'] = generate_fuel_comparison_figure()
    
    # Save figures
    for name, b64 in figures.items():
        path = f'/root/tp-agent/work/m7382/artifacts/{name}.png'
        with open(path, 'wb') as f:
            f.write(base64.b64decode(b64))
        print(f"  Saved: {path}")
    
    print("\nAll figures generated successfully!")
    return figures


if __name__ == '__main__':
    main()
