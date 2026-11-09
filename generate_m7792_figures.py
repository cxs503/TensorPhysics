"""Generate figures for P350W1-N03 report: plate-beam elastoplastic analysis."""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
import io
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from tensorphysics.ice_plate_beam.analysis import run_full_analysis, generate_load_displacement_data
from tensorphysics.ice_plate_beam.plate_beam import default_plate_beam_section
from tensorphysics.ice_plate_beam.elastoplastic import ElasticPlateBeam
from tensorphysics.ice_plate_beam.ice_load import HalfSineIcePulse


def encode_fig(fig) -> str:
    """Encode matplotlib figure to base64 PNG."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')


def generate_load_displacement_curve(result) -> str:
    """Generate load-displacement curve with stage annotations."""
    data = generate_load_displacement_data(result)
    w = np.array(data["displacement_m"]) * 1000  # mm
    P = np.array(data["load_N"]) / 1e3  # kN
    stages = data["stages"]
    plate_yielded = data["plate_yielded"]
    beam_yielded = data["beam_yielded"]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Color by stage
    stage_colors = {
        "initial": "gray",
        "elastic": "blue",
        "plate_yielding": "orange",
        "beam_approaching_yield": "red",
        "collapse": "darkred"
    }

    # Plot segments by stage
    current_stage = stages[0]
    start_idx = 0
    for i in range(1, len(stages)):
        if stages[i] != current_stage:
            color = stage_colors.get(current_stage, "gray")
            ax.plot(w[start_idx:i], P[start_idx:i], color=color, linewidth=2,
                    label=current_stage.replace('_', ' ').title())
            start_idx = i
            current_stage = stages[i]
    # Last segment
    color = stage_colors.get(current_stage, "gray")
    ax.plot(w[start_idx:], P[start_idx:], color=color, linewidth=2,
            label=current_stage.replace('_', ' ').title())

    # Mark key points
    P_yield = data["P_yield_N"] / 1e3
    P_ult = data["P_ult_N"] / 1e3
    w_ult = data["w_at_P_ult_m"] * 1000

    # Find yield point
    yield_idx = 0
    for i, py in enumerate(plate_yielded):
        if py:
            yield_idx = i
            break
    w_yield = w[yield_idx] if yield_idx > 0 else w[1]

    ax.plot(w_yield, P_yield, 'go', markersize=10, label=f'Yield (P={P_yield:.0f} kN)')
    ax.plot(w_ult, P_ult, 'r*', markersize=15, label=f'Ultimate (P={P_ult:.0f} kN)')

    ax.set_xlabel('Deflection [mm]', fontsize=12)
    ax.set_ylabel('Load [kN]', fontsize=12)
    ax.set_title('Plate-Beam Combination: Load-Displacement Curve\n'
                 'Elastic → Plate Yield → Beam Yield → Collapse', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return encode_fig(fig)


def generate_ice_pulse_figure() -> str:
    """Generate half-sine ice pulse figure."""
    pulse = HalfSineIcePulse(p0=1.35e6, T=0.2)  # PC5 typical
    t, p = pulse.time_history(200)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1000, p / 1e6, 'b-', linewidth=2)
    ax.fill_between(t * 1000, 0, p / 1e6, alpha=0.2, color='blue')
    ax.axhline(y=1.35, color='r', linestyle='--', alpha=0.5, label=f'p₀ = 1.35 MPa')
    ax.axvline(x=100, color='g', linestyle=':', alpha=0.5, label='t = T/2 (peak)')

    ax.set_xlabel('Time [ms]', fontsize=12)
    ax.set_ylabel('Ice Pressure [MPa]', fontsize=12)
    ax.set_title('Half-Sine Ice Pulse Model\np(t) = p₀·sin(πt/T)', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return encode_fig(fig)


def generate_section_figure(section) -> str:
    """Generate cross-section diagram."""
    fig, ax = plt.subplots(figsize=(8, 6))

    t_p = section.plate.t_p * 1000  # mm
    h_w = section.stiffener.h_w * 1000
    t_w = section.stiffener.t_w * 1000
    b_f = section.stiffener.b_f * 1000
    t_f = section.stiffener.t_f * 1000
    b_eff = section.b_eff * 1000

    # Draw plate (top)
    plate_y = h_w + t_f
    ax.add_patch(plt.Rectangle((-b_eff/2, plate_y), b_eff, t_p,
                                facecolor='lightblue', edgecolor='blue', linewidth=2))

    # Draw flange
    ax.add_patch(plt.Rectangle((-b_f/2, h_w), b_f, t_f,
                                facecolor='lightcoral', edgecolor='red', linewidth=2))

    # Draw web
    ax.add_patch(plt.Rectangle((-t_w/2, 0), t_w, h_w,
                                facecolor='lightgreen', edgecolor='green', linewidth=2))

    # Labels
    ax.text(0, plate_y + t_p/2, f'Plate\nt={t_p:.1f}mm', ha='center', va='center', fontsize=9)
    ax.text(0, h_w + t_f/2, f'Flange\n{b_f:.0f}×{t_f:.1f}', ha='center', va='center', fontsize=9)
    ax.text(0, h_w/2, f'Web\n{t_w:.1f}×{h_w:.0f}', ha='center', va='center', fontsize=9)

    # Dimensions
    ax.annotate('', xy=(-b_eff/2 - 10, 0), xytext=(-b_eff/2 - 10, plate_y + t_p),
                arrowprops=dict(arrowstyle='<->', color='black'))
    ax.text(-b_eff/2 - 20, (plate_y + t_p)/2, f'H={h_w+t_f+t_p:.0f}mm',
            ha='right', va='center', fontsize=9, rotation=90)

    ax.set_xlim(-b_eff/2 - 40, b_eff/2 + 40)
    ax.set_ylim(-20, h_w + t_f + t_p + 20)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'T-Stiffener with Effective Plate (b_eff = {b_eff:.0f} mm)\n'
                 f'Shape Factor = {section.shape_factor:.2f}', fontsize=12)

    plt.tight_layout()
    return encode_fig(fig)


def generate_failure_mode_figure(result) -> str:
    """Generate failure mode progression figure."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    stages = [
        ("Elastic Stage", "Both plate and beam\nrespond elastically", "blue"),
        ("Plate Yielding", "Plate forms plastic\nhinge lines", "orange"),
        ("Collapse", "Full mechanism:\nplate + beam plastic", "red")
    ]

    for ax, (title, desc, color) in zip(axes, stages):
        # Draw simplified panel
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        # Plate
        ax.add_patch(plt.Rectangle((-1, 0.5), 2, 0.15, facecolor=color, alpha=0.3, edgecolor=color))
        # Stiffener
        ax.add_patch(plt.Rectangle((-0.05, 0), 0.1, 0.5, facecolor=color, alpha=0.5, edgecolor=color))

        # Deflection arrows
        if title == "Elastic Stage":
            ax.annotate('', xy=(0, 0.4), xytext=(0, 0.6),
                        arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        elif title == "Plate Yielding":
            ax.annotate('', xy=(-0.5, 0.3), xytext=(-0.5, 0.65),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))
            ax.annotate('', xy=(0.5, 0.3), xytext=(0.5, 0.65),
                        arrowprops=dict(arrowstyle='->', color='orange', lw=2))
        else:
            ax.annotate('', xy=(0, -0.2), xytext=(0, 0.65),
                        arrowprops=dict(arrowstyle='->', color='red', lw=3))
            # Plastic hinge symbols
            ax.plot([-0.3, -0.2], [0.55, 0.65], 'r-', lw=2)
            ax.plot([0.2, 0.3], [0.55, 0.65], 'r-', lw=2)

        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.text(0, -0.35, desc, ha='center', fontsize=9, style='italic')
        ax.axis('off')

    plt.suptitle('Failure Mode Progression: Plate Yield First → Beam Yield → Collapse',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    return encode_fig(fig)


def main():
    """Generate all figures and save to artifacts."""
    os.makedirs('artifacts', exist_ok=True)

    # Run analysis
    result = run_full_analysis(pc=5, n_points=200)

    # Generate figures
    fig1 = generate_load_displacement_curve(result)
    fig2 = generate_ice_pulse_figure()
    fig3 = generate_section_figure(result.section)
    fig4 = generate_failure_mode_figure(result)

    # Save base64 to files
    figures = {
        "load_displacement": fig1,
        "ice_pulse": fig2,
        "section_diagram": fig3,
        "failure_mode": fig4
    }

    import json
    with open('artifacts/figures_m7792.json', 'w') as f:
        json.dump(figures, f)

    # Print summary
    print("Figures generated successfully:")
    for name, b64 in figures.items():
        print(f"  {name}: {len(b64)} bytes base64")

    # Print key results
    print(f"\nKey Results:")
    print(f"  PC Class: {result.ice_load['pc']}")
    print(f"  p0 = {result.ice_load['p0_MPa']:.3f} MPa")
    print(f"  P_yield = {result.response.P_yield/1e3:.1f} kN")
    print(f"  P_ult = {result.response.P_ult/1e3:.1f} kN")
    print(f"  Failure mode: {result.failure_mode}")
    print(f"  Acceptance all_pass: {result.acceptance_criteria['all_pass']}")


if __name__ == '__main__':
    main()
