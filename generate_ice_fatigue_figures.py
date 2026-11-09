"""Generate figures for the ice-induced structural fatigue verification report.

Produces:
1. S-N curves for B1, F, W3 details (log-log)
2. Ice load time series and PSD
3. Structural FRF and response PSD
4. Stress time history and rainflow histogram
5. Thickness correction effect
6. Miner damage accumulation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from tensorphysics.ice_fatigue import (
    SNCurveOS401,
    DNV_OS_C401_CURVES,
    thickness_correction,
    compute_load_psd,
    compute_frf,
    compute_response_psd,
    ice_load_time_series,
    response_time_series_from_psd,
    run_ice_fatigue_assessment,
)
from tensorphysics.fatigue.rainflow import rainflow_counting_stack


def figure_sn_curves(output_path: str = "artifacts/fig_sn_curves.png"):
    """Figure 1: S-N curves for B1, F, W3 details per DNV-OS-C401."""
    fig, ax = plt.subplots(figsize=(10, 7))

    stress_ranges = np.logspace(0.5, 3.0, 200)
    colors = {'B1': '#1f77b4', 'F': '#ff7f0e', 'W3': '#2ca02c'}
    markers = {'B1': '-', 'F': '--', 'W3': '-.'}

    for detail in ['B1', 'F', 'W3']:
        sn = SNCurveOS401(detail=detail)
        n_cycles = sn.evaluate_range(stress_ranges)
        params = DNV_OS_C401_CURVES[detail]

        ax.loglog(n_cycles, stress_ranges, markers[detail],
                  color=colors[detail], linewidth=2,
                  label=f"{detail}: log a={params.log_a:.3f}, m={params.m:.1f}\n"
                        f"  ({params.description})")

    ax.set_xlabel('Cycles to failure N', fontsize=12)
    ax.set_ylabel('Stress range ΔS (MPa)', fontsize=12)
    ax.set_title('DNV-OS-C401 S-N Curves for Welded Details\n(Table 2-1)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, which='both', alpha=0.3)
    ax.set_xlim(1e3, 1e10)
    ax.set_ylim(10, 1000)

    # Add fatigue limit annotation
    ax.axvline(x=1e7, color='gray', linestyle=':', alpha=0.5)
    ax.text(1.2e7, 20, 'N = 10⁷\n(fatigue limit)', fontsize=9, color='gray')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_ice_load_spectrum(output_path: str = "artifacts/fig_ice_load_spectrum.png"):
    """Figure 2: Ice load time series and power spectral density."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1, 1])

    # Generate ice load
    time, force = ice_load_time_series(
        duration_s=600.0, fs_hz=10.0,
        mean_force_n=3.0e6, amplitude_n=1.5e6,
        primary_freq_hz=0.5, secondary_freq_hz=1.2,
        seed=20261109
    )

    # Time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time, force / 1e6, 'b-', linewidth=0.5, alpha=0.8)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Ice Force (MN)', fontsize=11)
    ax1.set_title('Ice Load Time Series (10 min record)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 600)

    # Compute PSD
    load_spec = compute_load_psd(time, force)

    # PSD
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.semilogy(load_spec.frequencies, load_spec.psd, 'b-', linewidth=1.5)
    ax2.axvline(x=load_spec.peak_frequency, color='r', linestyle='--',
                label=f'Peak: {load_spec.peak_frequency:.2f} Hz')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('PSD (N²/Hz)', fontsize=11)
    ax2.set_title('Load Power Spectral Density', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)

    # Zoom near peak
    ax3 = fig.add_subplot(gs[1, 1])
    mask = (load_spec.frequencies > 0.2) & (load_spec.frequencies < 1.5)
    ax3.semilogy(load_spec.frequencies[mask], load_spec.psd[mask], 'b-', linewidth=1.5)
    ax3.axvline(x=load_spec.peak_frequency, color='r', linestyle='--',
                label=f'Peak: {load_spec.peak_frequency:.2f} Hz')
    ax3.axvline(x=0.5, color='g', linestyle=':', alpha=0.7,
                label='Input: 0.50 Hz')
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('PSD (N²/Hz)', fontsize=11)
    ax3.set_title('PSD Zoom (near primary frequency)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_frf_response(output_path: str = "artifacts/fig_frf_response.png"):
    """Figure 3: Structural FRF and response PSD."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Structural parameters
    fn = 1.2  # Hz
    mass = 5e6  # kg
    stiffness = mass * (2 * np.pi * fn) ** 2
    zeta = 0.02
    damping = 2 * zeta * np.sqrt(stiffness * mass)

    # Generate ice load
    time, force = ice_load_time_series(
        duration_s=600.0, fs_hz=10.0,
        primary_freq_hz=0.5, noise_ratio=0.3,
        seed=20261109
    )
    load_spec = compute_load_psd(time, force)

    # Compute FRF
    freqs = np.linspace(0.01, 5.0, 1000)
    frf = compute_frf(freqs, mass, stiffness, damping)

    # FRF magnitude
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.semilogy(frf.frequencies, frf.h_mag * 1e6, 'b-', linewidth=1.5)  # μm/N
    ax1.axvline(x=fn, color='r', linestyle='--', label=f'fn = {fn:.1f} Hz')
    ax1.set_xlabel('Frequency (Hz)', fontsize=11)
    ax1.set_ylabel('|H(f)| (μm/N)', fontsize=11)
    ax1.set_title('Structural Frequency Response Function', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(0, 3)

    # FRF phase
    ax2 = fig.add_subplot(gs[0, 1])
    phase = np.degrees(np.arctan2(frf.h_imag, frf.h_real))
    ax2.plot(frf.frequencies, phase, 'b-', linewidth=1.5)
    ax2.axvline(x=fn, color='r', linestyle='--', label=f'fn = {fn:.1f} Hz')
    ax2.set_xlabel('Frequency (Hz)', fontsize=11)
    ax2.set_ylabel('Phase (degrees)', fontsize=11)
    ax2.set_title('FRF Phase', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0, 3)
    ax2.set_ylim(-200, 20)

    # Response PSD
    resp_spec = compute_response_psd(load_spec, frf)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.loglog(resp_spec.frequencies[resp_spec.frequencies > 0.01],
               resp_spec.response_psd[resp_spec.frequencies > 0.01],
               'b-', linewidth=1.5, label='Response PSD')
    ax3.semilogy(load_spec.frequencies, load_spec.psd / np.max(load_spec.psd) * np.max(resp_spec.response_psd),
                 'g--', alpha=0.5, label='Load PSD (normalized)')
    ax3.axvline(x=fn, color='r', linestyle=':', label=f'Structure fn = {fn:.1f} Hz')
    ax3.set_xlabel('Frequency (Hz)', fontsize=11)
    ax3.set_ylabel('Response PSD (m²/Hz)', fontsize=11)
    ax3.set_title('Structural Response PSD = |H(f)|² × S_FF(f)', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0.01, 5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_rainflow_histogram(output_path: str = "artifacts/fig_rainflow_histogram.png"):
    """Figure 4: Stress time history and rainflow histogram."""
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)

    # Run assessment to get stress time series
    result = run_ice_fatigue_assessment(
        duration_s=600.0, fs_hz=10.0,
        mean_ice_force_n=5.0e6, dynamic_amplitude_n=2.0e6,
        primary_freq_hz=0.5,
        structural_mass_kg=5e6, natural_freq_hz=1.2,
        damping_ratio=0.02,
        stress_factor=5.0e-6,  # Higher stress for visible cycles
        sn_detail="F",
        hotspot_kt=1.2,
        seed=20261109
    )

    # Stress time series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(result.time_vector, result.stress_time_series, 'b-', linewidth=0.5)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_ylabel('Hot-spot Stress (MPa)', fontsize=11)
    ax1.set_title('Stress Time History (from spectral synthesis)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Rainflow histogram
    ranges = result.rainflow_ranges
    counts = result.rainflow_counts

    ax2 = fig.add_subplot(gs[1, 0])
    if len(ranges) > 0:
        n_bins = 30
        max_range = np.percentile(ranges, 99)
        bins = np.linspace(0, max_range, n_bins + 1)
        hist_counts = np.zeros(n_bins)
        for r, c in zip(ranges, counts):
            idx = min(int(r / max_range * n_bins), n_bins - 1)
            if idx >= 0:
                hist_counts[idx] += c
        centers = (bins[:-1] + bins[1:]) / 2
        ax2.bar(centers, hist_counts, width=bins[1] - bins[0],
                color='steelblue', alpha=0.7, edgecolor='navy')
    ax2.set_xlabel('Stress Range Δσ (MPa)', fontsize=11)
    ax2.set_ylabel('Cycle Count', fontsize=11)
    ax2.set_title('Rainflow Cycle Histogram', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Range-mean plot
    ax3 = fig.add_subplot(gs[1, 1])
    if len(ranges) > 0:
        ax3.scatter(ranges, np.zeros_like(ranges), c=counts, cmap='hot_r',
                    s=10, alpha=0.5)
    ax3.set_xlabel('Stress Range Δσ (MPa)', fontsize=11)
    ax3.set_ylabel('Mean Stress (MPa)', fontsize=11)
    ax3.set_title('Rainflow Range Distribution', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_thickness_effect(output_path: str = "artifacts/fig_thickness_effect.png"):
    """Figure 5: Effect of plate thickness on fatigue life."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Thickness correction factor
    thicknesses = np.linspace(10, 60, 100)
    ax1 = axes[0]
    for k, label in [(0.2, 'k=0.2 (m≤3)'), (0.25, 'k=0.25 (3<m≤4)'), (0.3, 'k=0.3 (m>4)')]:
        factors = [thickness_correction(t, 25.0, k) for t in thicknesses]
        ax1.plot(thicknesses, factors, linewidth=2, label=label)
    ax1.axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    ax1.axvline(x=25.0, color='gray', linestyle=':', alpha=0.5, label='t_ref = 25 mm')
    ax1.set_xlabel('Plate Thickness t (mm)', fontsize=11)
    ax1.set_ylabel('Thickness Correction Factor', fontsize=11)
    ax1.set_title('DNV-OS-C401 Thickness Correction\n(t/t_ref)^k', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Effect on fatigue life
    stress = 100.0  # MPa
    ax2 = axes[1]
    for detail in ['B1', 'F', 'W3']:
        lives = []
        for t in thicknesses:
            sn = SNCurveOS401(detail=detail, thickness_mm=t)
            n = sn.n_to_failure(stress)
            lives.append(n)
        ax2.semilogy(thicknesses, lives, linewidth=2, label=f'{detail} detail')

    ax2.axvline(x=25.0, color='gray', linestyle=':', alpha=0.5, label='t_ref = 25 mm')
    ax2.set_xlabel('Plate Thickness t (mm)', fontsize=11)
    ax2.set_ylabel('Cycles to Failure N', fontsize=11)
    ax2.set_title(f'Effect of Thickness on Fatigue Life\n(Δσ = {stress} MPa)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_miner_damage(output_path: str = "artifacts/fig_miner_damage.png"):
    """Figure 6: Miner cumulative damage accumulation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Constant amplitude damage accumulation
    sn = SNCurveOS401(detail="F")
    stress = 80.0  # MPa
    n_f = sn.n_to_failure(stress)

    ax1 = axes[0]
    n_applied = np.logspace(3, np.log10(n_f * 2), 100)
    damage = n_applied / n_f
    ax1.loglog(n_applied, damage, 'b-', linewidth=2)
    ax1.axhline(y=1.0, color='r', linestyle='--', label='Failure (D = 1)')
    ax1.axvline(x=n_f, color='g', linestyle=':', label=f'N_f = {n_f:.2e}')
    ax1.set_xlabel('Number of Cycles Applied n', fontsize=11)
    ax1.set_ylabel('Cumulative Damage D', fontsize=11)
    ax1.set_title(f'Miner Damage Accumulation\n(Constant Amplitude Δσ = {stress} MPa)',
                  fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(1e-4, 10)

    # Multi-level loading
    ax2 = axes[1]
    stresses = np.array([60, 80, 100, 120, 140])
    colors = plt.cm.viridis(np.linspace(0, 1, len(stresses)))

    for i, s in enumerate(stresses):
        n_f_i = sn.n_to_failure(s)
        n_app = np.logspace(2, np.log10(n_f_i * 1.5), 50)
        d = n_app / n_f_i
        ax2.loglog(n_app, d, color=colors[i], linewidth=2,
                   label=f'Δσ = {s} MPa (N_f={n_f_i:.1e})')

    ax2.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='D = 1 (failure)')
    ax2.set_xlabel('Number of Cycles Applied n', fontsize=11)
    ax2.set_ylabel('Cumulative Damage D', fontsize=11)
    ax2.set_title('Miner Damage for Various Stress Levels\n(F detail, DNV-OS-C401)',
                  fontsize=12, fontweight='bold')
    ax2.legend(fontsize=8, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(1e-4, 10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def figure_dnv_check(output_path: str = "artifacts/fig_dnv_check.png"):
    """Figure 7: DNV-OS-C401 fatigue check summary."""
    fig, ax = plt.subplots(figsize=(10, 7))

    # Run assessment for different DFL values
    dfl_values = [2, 3, 5, 10]
    design_life = 20.0  # years

    results = []
    for dfl in dfl_values:
        result = run_ice_fatigue_assessment(
            duration_s=600.0, fs_hz=10.0,
            mean_ice_force_n=3.0e6, dynamic_amplitude_n=1.0e6,
            primary_freq_hz=0.5,
            structural_mass_kg=5e6, natural_freq_hz=1.2,
            damping_ratio=0.02,
            stress_factor=3.0e-6,
            sn_detail="F",
            design_life_years=design_life,
            dfl=dfl,
            service_years=1.0,
            seed=20261109
        )
        results.append(result)

    # Plot utilisation
    x = np.arange(len(dfl_values))
    utilisations = [r.dnv_check.utilisation for r in results]
    colors = ['green' if u <= 1.0 else 'red' for u in utilisations]

    bars = ax.bar(x, utilisations, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=1.0, color='r', linestyle='--', linewidth=2, label='Acceptance limit (U = 1)')

    ax.set_xticks(x)
    ax.set_xticklabels([f'DFL={d}\nT_req={d*design_life}yr' for d in dfl_values])
    ax.set_xlabel('Design Fatigue Factor (DFL)', fontsize=11)
    ax.set_ylabel('Utilisation Factor U = T_req / T_life', fontsize=11)
    ax.set_title('DNV-OS-C401 Fatigue Check\nU = T_req / T_life ≤ 1.0 for acceptance',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, u in zip(bars, utilisations):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{u:.3f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    os.makedirs("artifacts", exist_ok=True)

    figure_sn_curves()
    figure_ice_load_spectrum()
    figure_frf_response()
    figure_rainflow_histogram()
    figure_thickness_effect()
    figure_miner_damage()
    figure_dnv_check()

    print("\nAll figures generated successfully!")
