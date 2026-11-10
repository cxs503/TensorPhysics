"""Generate figures for P519-N10 ice multi-physics chain report."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import json
import os

from tensorphysics.ice_multiphysics_chain.chain import run_ice_multiphysics_assessment
from tensorphysics.ice_multiphysics_chain.acoustics import thorp_attenuation_db_per_km

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'artifacts', 'm9222')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fig_to_base64(fig):
    """Convert matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_vibration_spectrum(result):
    """Figure 1: 1/3-octave acceleration spectrum with ISO 19906 envelope."""
    v = result.vibration
    fig, ax = plt.subplots(figsize=(10, 6))

    # 1/3-octave RMS acceleration
    ax.bar(v.one_third_octave_freqs, v.one_third_octave_rms,
           width=np.diff(np.append(v.one_third_octave_freqs, v.one_third_octave_freqs[-1] * 2**(1/3)))[0] * 0.8,
           alpha=0.7, label='1/3-octave RMS acceleration', color='steelblue')

    # ISO 19906 envelope
    ax.plot(v.frequencies_hz, v.iso19906_envelope, 'r--', linewidth=2,
            label='ISO 19906 envelope')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Acceleration (m/s²)', fontsize=12)
    ax.set_title('Ice-Induced Hull Vibration: 1/3-Octave Acceleration Spectrum', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, f'Energy conservation error: {v.energy_conservation_error*100:.2f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_fatigue_damage(result):
    """Figure 2: Fatigue damage breakdown."""
    f = result.fatigue
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rainflow histogram
    ax = axes[0]
    if len(f.rainflow_ranges) > 0:
        n_bins = min(30, len(f.rainflow_ranges))
        counts, bins, patches = ax.hist(f.rainflow_ranges, bins=n_bins,
                                         weights=f.rainflow_counts,
                                         color='steelblue', alpha=0.7, edgecolor='black')
        ax.set_xlabel('Stress Range (MPa)', fontsize=11)
        ax.set_ylabel('Cycle Count', fontsize=11)
        ax.set_title('Rainflow Cycle Distribution', fontsize=12)
    else:
        ax.text(0.5, 0.5, 'No cycles detected', ha='center', va='center',
                transform=ax.transAxes, fontsize=14)
    ax.grid(True, alpha=0.3)

    # Damage summary
    ax = axes[1]
    categories = ['Miner Damage', 'Spectral Damage']
    values = [f.miner_damage, f.spectral_damage]
    colors = ['steelblue', 'coral']
    bars = ax.bar(categories, values, color=colors, alpha=0.7, edgecolor='black')

    # Add D=1 line
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, label='D = 1 (failure)')

    ax.set_ylabel('Cumulative Damage D', fontsize=11)
    ax.set_title('Fatigue Damage Assessment', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add text
    ax.text(0.02, 0.98, f'DNV Check: {"PASS" if f.dnv_check.is_acceptable else "FAIL"}',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen' if f.dnv_check.is_acceptable else 'lightcoral', alpha=0.5))

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_acoustics_spl(result):
    """Figure 3: 1/3-octave SPL spectrum with marine mammal thresholds."""
    a = result.acoustics
    fig, ax = plt.subplots(figsize=(10, 6))

    valid = np.isfinite(a.spl_db)
    if np.any(valid):
        ax.bar(a.frequencies_hz[valid], a.spl_db[valid],
               width=np.diff(np.append(a.frequencies_hz, a.frequencies_hz[-1] * 2**(1/3)))[0] * 0.8,
               alpha=0.7, label='Received SPL', color='steelblue')

    # Marine mammal thresholds
    mm = a.marine_mammal
    thresholds = {
        'Behavioural (MF)': 160,
        'TTS onset (MF)': 170,
        'PTS onset (MF)': 185,
        'Injury peak (MF)': 230,
    }
    colors_thresh = ['green', 'orange', 'red', 'darkred']
    for (name, val), color in zip(thresholds.items(), colors_thresh):
        ax.axhline(y=val, color=color, linestyle='--', linewidth=1.5, alpha=0.7, label=name)

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('SPL (dB re 1 μPa)', fontsize=12)
    ax.set_title('Underwater Radiated Noise: 1/3-Octave SPL Spectrum', fontsize=14)
    ax.set_xscale('log')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)

    ax.text(0.02, 0.98, f'Source Level: {a.source_level_db:.1f} dB re 1 μPa @ 1 m',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_thorp_curve():
    """Figure 4: Thorp absorption coefficient vs frequency."""
    freqs_hz = np.logspace(1, 5, 500)
    alpha = thorp_attenuation_db_per_km(freqs_hz)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs_hz, alpha, 'b-', linewidth=2)
    ax.axvline(x=10000, color='red', linestyle='--', alpha=0.7, label='10 kHz')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.7, label='1 dB/km')

    ax.set_xlabel('Frequency (Hz)', fontsize=12)
    ax.set_ylabel('Absorption Coefficient α (dB/km)', fontsize=12)
    ax.set_title('Thorp Empirical Absorption Formula', fontsize=14)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    # Annotate 10 kHz point
    alpha_10k = thorp_attenuation_db_per_km(10000.0)
    ax.annotate(f'α(10 kHz) = {alpha_10k:.3f} dB/km',
                xy=(10000, alpha_10k), xytext=(20000, alpha_10k * 1.5),
                fontsize=10, arrowprops=dict(arrowstyle='->', color='red'))

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def generate_cargo_shift(result):
    """Figure 5: Cargo shift assessment diagram."""
    c = result.cargo_shift
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Sliding check
    ax = axes[0]
    forces = ['Transverse\nInertial', 'Longitudinal\nInertial', 'Friction\nResistance']
    values = [c.sliding_force_transverse_n / 1000, c.sliding_force_longitudinal_n / 1000,
              c.friction_resistance_n / 1000]
    colors = ['coral', 'coral', 'steelblue']
    ax.bar(forces, values, color=colors, alpha=0.7, edgecolor='black')
    ax.set_ylabel('Force (kN)', fontsize=11)
    ax.set_title('Cargo Sliding Check', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    status = 'SAFE' if c.is_sliding_safe else 'UNSAFE'
    color_status = 'lightgreen' if c.is_sliding_safe else 'lightcoral'
    ax.text(0.02, 0.98, f'Sliding: {status}', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color_status, alpha=0.7))

    # Overturning check
    ax = axes[1]
    moments = ['Overturning\nMoment', 'Restoring\nMoment']
    moment_values = [c.overturning_moment_n_m / 1000, c.restoring_moment_n_m / 1000]
    ax.bar(moments, moment_values, color=['coral', 'steelblue'], alpha=0.7, edgecolor='black')
    ax.set_ylabel('Moment (kN·m)', fontsize=11)
    ax.set_title('Cargo Overturning Check', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

    status_ot = 'SAFE' if c.is_overturning_safe else 'UNSAFE'
    color_ot = 'lightgreen' if c.is_overturning_safe else 'lightcoral'
    ax.text(0.02, 0.98, f'Overturning: {status_ot}', transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor=color_ot, alpha=0.7))

    b64 = fig_to_base64(fig)
    plt.close(fig)
    return b64


def main():
    """Generate all figures and save."""
    print("Running multi-physics assessment...")
    result = run_ice_multiphysics_assessment(seed=20261110)

    figures = []

    # Figure 1: Vibration spectrum
    print("Generating vibration spectrum figure...")
    b64 = generate_vibration_spectrum(result)
    figures.append({
        "name": "vibration_spectrum",
        "caption": "1/3-octave acceleration spectrum with ISO 19906 envelope. Energy conservation error: {:.2f}%".format(
            result.vibration.energy_conservation_error * 100),
        "png_base64": b64,
    })

    # Figure 2: Fatigue damage
    print("Generating fatigue damage figure...")
    b64 = generate_fatigue_damage(result)
    figures.append({
        "name": "fatigue_damage",
        "caption": "Rainflow cycle distribution and cumulative Miner damage assessment. DNV-OS-C401 check: {}".format(
            "PASS" if result.fatigue.dnv_check.is_acceptable else "FAIL"),
        "png_base64": b64,
    })

    # Figure 3: Acoustics SPL
    print("Generating acoustics SPL figure...")
    b64 = generate_acoustics_spl(result)
    figures.append({
        "name": "acoustics_spl",
        "caption": "1/3-octave received SPL spectrum at {} m range with marine mammal thresholds (MF group).".format(
            result.acoustics.range_m),
        "png_base64": b64,
    })

    # Figure 4: Thorp curve
    print("Generating Thorp absorption curve...")
    b64 = generate_thorp_curve()
    figures.append({
        "name": "thorp_absorption",
        "caption": "Thorp empirical absorption coefficient vs frequency. α(10 kHz) ≈ 1.187 dB/km.",
        "png_base64": b64,
    })

    # Figure 5: Cargo shift
    print("Generating cargo shift figure...")
    b64 = generate_cargo_shift(result)
    figures.append({
        "name": "cargo_shift",
        "caption": "Cargo sliding and overturning check per IMO CSS Code Annex 13.",
        "png_base64": b64,
    })

    # Save figures
    for fig in figures:
        png_path = os.path.join(OUTPUT_DIR, f"{fig['name']}.png")
        with open(png_path, 'wb') as f:
            f.write(base64.b64decode(fig['png_base64']))
        print(f"Saved {png_path}")

    # Save figure metadata
    meta_path = os.path.join(OUTPUT_DIR, 'figures.json')
    with open(meta_path, 'w') as f:
        json.dump(figures, f, indent=2)
    print(f"Saved {meta_path}")

    # Print verification summary
    print("\n=== Verification Summary ===")
    v = result.verification
    for key, val in v.items():
        status = "PASS" if val else "FAIL"
        print(f"  {key}: {status} ({val})")

    print(f"\nSHA-256: {result.sha256}")

    return figures


if __name__ == '__main__':
    main()
