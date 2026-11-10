"""Generate figures for wake wash propagation verification.

This script generates three figures:
1. Kelvin wake pattern geometry (transverse and divergent waves)
2. Wave height decay with distance (deep water vs shallow water)
3. PIANC compliance assessment for multiple receptor types
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorphysics.wake_wash import (
    transverse_wave_angle,
    divergent_wave_angle,
    kelvin_wake_coordinates,
    deep_water_decay,
    shallow_water_decay,
    wave_height_with_depth,
    ReceptorType,
    check_compliance,
    get_wave_height_limit,
    predict_wave_height,
)


def figure1_kelvin_wake_pattern():
    """Generate Figure 1: Kelvin wake pattern geometry."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Ship parameters
    U = 10.0  # m/s
    g = 9.81
    
    # Generate wake coordinates
    x, y = kelvin_wake_coordinates(U, n_points=200)
    
    # Plot wake pattern (both sides)
    ax.plot(x, y, 'b-', linewidth=2, label='Divergent wave cusp')
    ax.plot(x, [-yi for yi in y], 'b-', linewidth=2)
    
    # Plot transverse wave line
    theta_t = math.radians(transverse_wave_angle())
    L = U**2 / g
    x_trans = np.linspace(-3*L, 0, 100)
    y_trans = np.full_like(x_trans, L * math.tan(theta_t))
    ax.plot(x_trans, y_trans, 'r--', linewidth=2, label='Transverse wave')
    ax.plot(x_trans, -y_trans, 'r--', linewidth=2)
    
    # Mark angles
    r_mark = 2 * L
    ax.annotate(f'θ_t = {transverse_wave_angle():.2f}°',
                xy=(r_mark * math.cos(theta_t), r_mark * math.sin(theta_t)),
                xytext=(r_mark * 0.5, r_mark * 0.8),
                arrowprops=dict(arrowstyle='->', color='red'),
                fontsize=12, color='red')
    
    theta_d = math.radians(divergent_wave_angle())
    ax.annotate(f'θ_d = {divergent_wave_angle():.2f}°',
                xy=(r_mark * math.cos(theta_d), r_mark * math.sin(theta_d)),
                xytext=(r_mark * 0.3, r_mark * 1.2),
                arrowprops=dict(arrowstyle='->', color='blue'),
                fontsize=12, color='blue')
    
    # Ship position
    ax.plot(0, 0, 'ko', markersize=10)
    ax.annotate('Ship', xy=(0, 0), xytext=(0.5, -0.5),
                fontsize=12, fontweight='bold')
    
    # Ship track
    ax.axhline(y=0, color='gray', linestyle=':', linewidth=1)
    ax.annotate('Ship track', xy=(L, 0), xytext=(L, -0.3),
                fontsize=10, color='gray')
    
    ax.set_xlabel('Distance along track (m)', fontsize=12)
    ax.set_ylabel('Distance from track (m)', fontsize=12)
    ax.set_title('Kelvin Wake Pattern Geometry\n' +
                 f'Transverse angle = {transverse_wave_angle():.2f}°, ' +
                 f'Divergent angle = {divergent_wave_angle():.2f}°',
                 fontsize=14)
    ax.legend(loc='upper right', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-3*L, 0.5*L)
    
    plt.tight_layout()
    plt.savefig('docs/figures/fig1_kelvin_wake_pattern.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 1 saved: docs/figures/fig1_kelvin_wake_pattern.png")


def figure2_wave_height_decay():
    """Generate Figure 2: Wave height decay with distance."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Parameters
    H_ref = 1.0  # m
    r_ref = 100.0  # m
    distances = np.linspace(100, 2000, 100)
    
    # Deep water decay
    H_deep = [deep_water_decay(H_ref, r_ref, r) for r in distances]
    
    # Shallow water decay
    H_shallow = [shallow_water_decay(H_ref, r_ref, r) for r in distances]
    
    # Intermediate depth (with depth effect)
    wavelength = 50.0  # m
    water_depth = 5.0  # m (shallow)
    H_intermediate = [wave_height_with_depth(H_ref, r_ref, r, water_depth, wavelength) 
                      for r in distances]
    
    # Plot 1: Linear scale
    ax1.plot(distances, H_deep, 'b-', linewidth=2, label='Deep water (H ∝ 1/√r)')
    ax1.plot(distances, H_shallow, 'r-', linewidth=2, label='Shallow water (H ∝ 1/r)')
    ax1.plot(distances, H_intermediate, 'g--', linewidth=2, 
             label=f'Intermediate (h={water_depth}m, λ={wavelength}m)')
    
    ax1.set_xlabel('Distance from ship (m)', fontsize=12)
    ax1.set_ylabel('Wave height (m)', fontsize=12)
    ax1.set_title('Wave Height Decay with Distance', fontsize=14)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 2000)
    ax1.set_ylim(0, 1.1)
    
    # Plot 2: Log-log scale
    ax2.loglog(distances, H_deep, 'b-', linewidth=2, label='Deep water')
    ax2.loglog(distances, H_shallow, 'r-', linewidth=2, label='Shallow water')
    
    # Add reference lines
    r_fit = distances[distances > 200]
    ax2.loglog(r_fit, H_ref * np.sqrt(r_ref / r_fit), 'b:', linewidth=1, 
               alpha=0.5, label='H ∝ 1/√r')
    ax2.loglog(r_fit, H_ref * (r_ref / r_fit), 'r:', linewidth=1, 
               alpha=0.5, label='H ∝ 1/r')
    
    ax2.set_xlabel('Distance from ship (m)', fontsize=12)
    ax2.set_ylabel('Wave height (m)', fontsize=12)
    ax2.set_title('Log-Log Scale: Decay Laws', fontsize=14)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    plt.savefig('docs/figures/fig2_wave_height_decay.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 2 saved: docs/figures/fig2_wave_height_decay.png")


def figure3_pianc_compliance():
    """Generate Figure 3: PIANC compliance assessment."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Ship parameters
    ship_speed = 10.0  # m/s
    ship_draft = 5.0  # m
    
    # Distances to assess
    distances = np.linspace(100, 1000, 50)
    
    # Receptor types
    receptors = [
        (ReceptorType.SWIMMERS, 'Swimmers', 'red'),
        (ReceptorType.MOORED_SMALL_VESSEL, 'Moored small vessel', 'orange'),
        (ReceptorType.MOORED_LARGE_VESSEL, 'Moored large vessel', 'green'),
        (ReceptorType.ERODIBLE_SHORELINE, 'Erodible shoreline', 'purple'),
    ]
    
    # Plot 1: Wave height vs distance
    for receptor, label, color in receptors:
        H_values = [predict_wave_height(ship_speed, ship_draft, r) for r in distances]
        ax1.plot(distances, H_values, color=color, linewidth=2, label=label)
        
        # Add PIANC limit line
        limit = get_wave_height_limit(receptor)
        ax1.axhline(y=limit, color=color, linestyle='--', alpha=0.5, linewidth=1)
    
    ax1.set_xlabel('Distance from ship (m)', fontsize=12)
    ax1.set_ylabel('Wave height (m)', fontsize=12)
    ax1.set_title('Wake Wash Wave Height vs PIANC Limits', fontsize=14)
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(100, 1000)
    ax1.set_ylim(0, 1.5)
    
    # Plot 2: Compliance status at different distances
    test_distances = [200, 400, 600, 800]
    receptor_names = [r[1] for r in receptors]
    x_pos = np.arange(len(receptor_names))
    width = 0.2
    
    for i, dist in enumerate(test_distances):
        compliance_status = []
        for receptor, _, _ in receptors:
            H = predict_wave_height(ship_speed, ship_draft, dist)
            level, ratio = check_compliance(H, receptor)
            if level.value == 'compliant':
                compliance_status.append(1)  # Green
            elif level.value == 'marginal':
                compliance_status.append(2)  # Yellow
            else:
                compliance_status.append(3)  # Red
        
        colors = ['green' if s == 1 else 'orange' if s == 2 else 'red' for s in compliance_status]
        bars = ax2.bar(x_pos + i*width, compliance_status, width, 
                       label=f'{dist}m', color=colors, alpha=0.7, edgecolor='black')
    
    ax2.set_xlabel('Receptor Type', fontsize=12)
    ax2.set_ylabel('Compliance Status', fontsize=12)
    ax2.set_title('PIANC Compliance Status\n(1=Compliant, 2=Marginal, 3=Non-compliant)', 
                  fontsize=14)
    ax2.set_xticks(x_pos + 1.5*width)
    ax2.set_xticklabels(receptor_names, rotation=15, ha='right', fontsize=9)
    ax2.legend(fontsize=10)
    ax2.set_ylim(0, 3.5)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('docs/figures/fig3_pianc_compliance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Figure 3 saved: docs/figures/fig3_pianc_compliance.png")


if __name__ == '__main__':
    # Create figures directory
    os.makedirs('docs/figures', exist_ok=True)
    
    print("Generating wake wash verification figures...")
    figure1_kelvin_wake_pattern()
    figure2_wave_height_decay()
    figure3_pianc_compliance()
    print("\nAll figures generated successfully!")
