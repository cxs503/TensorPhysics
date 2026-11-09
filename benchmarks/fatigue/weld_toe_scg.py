#!/usr/bin/env python3
"""Weld toe stress concentration benchmark — detail fatigue validation.

Validates the progression from global beam fatigue to detail fatigue:
1. Weld toe geometry parameterization (T-joint, butt joint, fillet joint)
2. Stress concentration factor Kt vs analytical anchors (Nii, Maddox)
3. Hot-spot stress extrapolation (linear & quadratic)
4. S-N curve fatigue life (DNV-RP-C203 bilinear)
5. Mesh convergence verification

Acceptance criteria:
(a) T-joint Kt error ≤ 3% vs Nii formula
(b) Butt joint Kt error ≤ 3% vs Maddox formula
(c) Hot-spot stress mesh convergence: 2x refinement → change ≤ 2%
(d) Fatigue life N error ≤ 15% vs S-N analytical solution

Usage:
    python benchmarks/fatigue/weld_toe_scg.py [--output-dir DIR]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from tensorphysics.fatigue.detail_fatigue import (
    WeldToeGeometry,
    kt_nii_t_joint,
    kt_maddox_butt_joint,
    kt_fillet_joint_approx,
    linear_extrapolation,
    quadratic_extrapolation,
    fem_stress_field_t_joint,
    fem_stress_field_butt_joint,
    fem_stress_field_fillet_joint,
    mesh_convergence_analysis,
    detail_fatigue_assessment,
)
from tensorphysics.fatigue.sn_curve import DNCurve


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_kt_validation(output_dir: Path) -> dict:
    """Validate Kt formulas against analytical anchors."""
    results = {'t_joint': [], 'butt_joint': [], 'fillet_joint': []}
    t = 12.0
    nominal_stress = 100.0
    radii = [0.5, 1.0, 1.5, 2.0]

    a = 60.0
    for r in radii:
        kt_analytical = kt_nii_t_joint(t, r, a)
        x_near = np.array([r * 0.01])
        stress_near = fem_stress_field_t_joint(x_near, nominal_stress, t, r, a)
        kt_fem = float(stress_near[0] / nominal_stress)
        error_pct = abs(kt_fem - kt_analytical) / kt_analytical * 100
        results['t_joint'].append({
            'weld_toe_radius_mm': r, 'kt_analytical': round(kt_analytical, 4),
            'kt_fem': round(kt_fem, 4), 'error_pct': round(error_pct, 4),
            'pass': bool(error_pct <= 3.0),
        })

    h = 3.0
    for r in radii:
        kt_analytical = kt_maddox_butt_joint(t, r, h)
        x_near = np.array([r * 0.01])
        stress_near = fem_stress_field_butt_joint(x_near, nominal_stress, t, r, h)
        kt_fem = float(stress_near[0] / nominal_stress)
        error_pct = abs(kt_fem - kt_analytical) / kt_analytical * 100
        results['butt_joint'].append({
            'weld_toe_radius_mm': r, 'kt_analytical': round(kt_analytical, 4),
            'kt_fem': round(kt_fem, 4), 'error_pct': round(error_pct, 4),
            'pass': bool(error_pct <= 3.0),
        })

    z = 8.0
    for r in radii:
        kt_analytical = kt_fillet_joint_approx(t, r, z)
        x_near = np.array([r * 0.01])
        stress_near = fem_stress_field_fillet_joint(x_near, nominal_stress, t, r, z)
        kt_fem = float(stress_near[0] / nominal_stress)
        error_pct = abs(kt_fem - kt_analytical) / kt_analytical * 100
        results['fillet_joint'].append({
            'weld_toe_radius_mm': r, 'kt_analytical': round(kt_analytical, 4),
            'kt_fem': round(kt_fem, 4), 'error_pct': round(error_pct, 4),
            'pass': bool(error_pct <= 3.0),
        })

    return results


def run_hotspot_extrapolation(output_dir: Path) -> dict:
    """Validate hot-spot stress extrapolation methods."""
    t, r, a, nominal_stress = 12.0, 1.0, 60.0, 100.0
    x = np.array([0.4 * t, 0.9 * t, 1.4 * t])
    stresses = fem_stress_field_t_joint(x, nominal_stress, t, r, a)
    s_04t, s_09t, s_14t = stresses

    hs_linear = linear_extrapolation(s_04t, s_09t, t)
    hs_quadratic = quadratic_extrapolation(s_04t, s_09t, s_14t, t)
    kt_analytical = kt_nii_t_joint(t, r, a)
    hs_analytical = nominal_stress * kt_analytical

    return {
        'plate_thickness_mm': t, 'weld_toe_radius_mm': r,
        'nominal_stress_MPa': nominal_stress,
        'stress_at_04t_MPa': round(float(s_04t), 4),
        'stress_at_09t_MPa': round(float(s_09t), 4),
        'stress_at_14t_MPa': round(float(s_14t), 4),
        'linear_extrapolation': {
            'hot_spot_stress_MPa': round(float(hs_linear.hot_spot_stress), 4),
            'error_vs_analytical_pct': round(
                abs(hs_linear.hot_spot_stress - hs_analytical) / hs_analytical * 100, 4),
        },
        'quadratic_extrapolation': {
            'hot_spot_stress_MPa': round(float(hs_quadratic.hot_spot_stress), 4),
            'error_vs_analytical_pct': round(
                abs(hs_quadratic.hot_spot_stress - hs_analytical) / hs_analytical * 100, 4),
        },
        'analytical_hotspot_MPa': round(float(hs_analytical), 4),
    }


def run_mesh_convergence(output_dir: Path) -> dict:
    """Validate mesh convergence for hot-spot stress."""
    t, r, a, nominal_stress = 12.0, 1.0, 60.0, 100.0
    elem_sizes = [t / 2, t / 4, t / 8, t / 16, t / 32]
    conv = mesh_convergence_analysis(
        t, r, nominal_stress, attachment_length=a, joint_type='t_joint',
        element_sizes=elem_sizes, tolerance=0.02,
    )
    last_two_change = (abs(conv.hot_spot_stresses[-1] - conv.hot_spot_stresses[-2])
                       / conv.reference_stress * 100) if len(conv.hot_spot_stresses) >= 2 else float('inf')

    return {
        'element_sizes_mm': [round(s, 4) for s in conv.element_sizes],
        'hot_spot_stresses_MPa': [round(s, 4) for s in conv.hot_spot_stresses],
        'reference_stress_MPa': round(conv.reference_stress, 4),
        'max_relative_change_pct': round(conv.max_relative_change * 100, 4),
        'converged': bool(conv.converged),
        'convergence_order': round(float(conv.convergence_order), 4),
        'last_2x_refinement_change_pct': round(last_two_change, 4),
        'pass_2x_convergence': bool(last_two_change <= 2.0),
    }


def run_fatigue_life(output_dir: Path) -> dict:
    """Validate fatigue life calculation vs S-N analytical solution."""
    t, r, a, nominal_stress = 12.0, 1.0, 60.0, 100.0
    geometry = WeldToeGeometry(joint_type='t_joint', plate_thickness=t,
                               weld_toe_radius=r, attachment_length=a)
    result = detail_fatigue_assessment(geometry=geometry, nominal_stress=nominal_stress,
                                       sn_curve_class='D', environment='seawater_cp')

    return {
        'geometry': {'joint_type': geometry.joint_type, 'plate_thickness_mm': geometry.plate_thickness,
                     'weld_toe_radius_mm': geometry.weld_toe_radius, 'attachment_length_mm': geometry.attachment_length},
        'kt_analytical': round(result.kt_analytical, 4),
        'kt_from_fem': round(result.kt_from_fem, 4),
        'kt_error_pct': round(result.kt_error_pct, 4),
        'hot_spot_stress_MPa': round(result.hot_spot_stress, 4),
        'sn_curve_class': result.sn_curve_class, 'environment': result.environment,
        'n_to_failure': round(float(result.n_to_failure), 2),
        'n_analytical': round(float(result.n_analytical), 2),
        'n_error_pct': round(float(result.n_error_pct), 4),
        'pass_fatigue_life': bool(result.n_error_pct <= 15.0),
        'mesh_converged': bool(result.mesh_convergence.converged),
        'is_acceptable': bool(result.is_acceptable),
    }


def generate_figures(output_dir, kt_results, hotspot_results, mesh_results, fatigue_results):
    """Generate validation figures."""
    figures = {}

    # Figure 1: Kt vs weld toe radius
    fig, ax = plt.subplots(figsize=(10, 6))
    radii = [0.5, 1.0, 1.5, 2.0]
    for joint_type, label, color in [
        ('t_joint', 'T-joint (Nii)', '#1f77b4'),
        ('butt_joint', 'Butt joint (Maddox)', '#ff7f0e'),
        ('fillet_joint', 'Fillet joint', '#2ca02c'),
    ]:
        data = kt_results[joint_type]
        ax.plot(radii, [d['kt_analytical'] for d in data], 'o-', color=color,
                label=f'{label} (analytical)', linewidth=2)
        ax.plot(radii, [d['kt_fem'] for d in data], 's--', color=color, alpha=0.6,
                label=f'{label} (FEM surrogate)')
    ax.set_xlabel('Weld toe radius r (mm)', fontsize=12)
    ax.set_ylabel('Stress concentration factor Kt', fontsize=12)
    ax.set_title('Kt vs Weld Toe Radius: Analytical vs FEM Surrogate', fontsize=14)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3)
    fig_path = output_dir / 'kt_vs_radius.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['kt_vs_radius'] = str(fig_path)

    # Figure 2: Hot-spot stress extrapolation
    fig, ax = plt.subplots(figsize=(8, 5))
    t = hotspot_results['plate_thickness_mm']
    x_positions = [0, 0.4 * t, 0.9 * t, 1.4 * t]
    stresses = [hotspot_results['analytical_hotspot_MPa'],
                hotspot_results['stress_at_04t_MPa'],
                hotspot_results['stress_at_09t_MPa'],
                hotspot_results['stress_at_14t_MPa']]
    ax.plot(x_positions, stresses, 'o-', color='#1f77b4', linewidth=2, label='FEM stress field')
    ax.axhline(y=hotspot_results['linear_extrapolation']['hot_spot_stress_MPa'],
               color='#ff7f0e', linestyle='--',
               label=f"Linear ({hotspot_results['linear_extrapolation']['hot_spot_stress_MPa']:.1f} MPa)")
    ax.axhline(y=hotspot_results['quadratic_extrapolation']['hot_spot_stress_MPa'],
               color='#2ca02c', linestyle='-.',
               label=f"Quadratic ({hotspot_results['quadratic_extrapolation']['hot_spot_stress_MPa']:.1f} MPa)")
    ax.axhline(y=hotspot_results['analytical_hotspot_MPa'], color='red', linestyle=':',
               alpha=0.7, label=f"Analytical Kt ({hotspot_results['analytical_hotspot_MPa']:.1f} MPa)")
    ax.set_xlabel('Distance from weld toe (mm)', fontsize=12)
    ax.set_ylabel('Stress (MPa)', fontsize=12)
    ax.set_title('Hot-Spot Stress Extrapolation: Linear vs Quadratic', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig_path = output_dir / 'hotspot_extrapolation.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['hotspot_extrapolation'] = str(fig_path)

    # Figure 3: Mesh convergence
    fig, ax = plt.subplots(figsize=(8, 5))
    elem_sizes = mesh_results['element_sizes_mm']
    hs_stresses = mesh_results['hot_spot_stresses_MPa']
    ref = mesh_results['reference_stress_MPa']
    ax.semilogx(elem_sizes, hs_stresses, 'o-', color='#1f77b4', linewidth=2, label='Hot-spot stress')
    ax.axhline(y=ref, color='red', linestyle='--', alpha=0.7, label=f'Reference ({ref:.2f} MPa)')
    ax.axhspan(ref * 0.98, ref * 1.02, alpha=0.1, color='green', label='±2% tolerance')
    ax.set_xlabel('Element size (mm)', fontsize=12)
    ax.set_ylabel('Hot-spot stress (MPa)', fontsize=12)
    ax.set_title('Mesh Convergence: Hot-Spot Stress vs Element Size', fontsize=14)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()
    fig_path = output_dir / 'mesh_convergence.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['mesh_convergence'] = str(fig_path)

    # Figure 4: S-N curve
    fig, ax = plt.subplots(figsize=(8, 6))
    sn = DNCurve(curve_class='D', environment='seawater_cp')
    log_S = np.linspace(1.0, 3.0, 200)
    S = 10 ** log_S
    N = [sn.n_to_failure(s) for s in S]
    ax.loglog(S, N, '-', color='#1f77b4', linewidth=2, label='DNV D-curve (seawater CP)')
    hs = fatigue_results['hot_spot_stress_MPa']
    n_comp = fatigue_results['n_to_failure']
    n_anal = fatigue_results['n_analytical']
    ax.plot(hs, n_comp, 'ro', markersize=10, label=f'Computed (N={n_comp:.0f})')
    ax.plot(hs, n_anal, 'g^', markersize=10, label=f'Analytical (N={n_anal:.0f})')
    ax.set_xlabel('Stress range (MPa)', fontsize=12)
    ax.set_ylabel('Cycles to failure N', fontsize=12)
    ax.set_title('S-N Curve: Computed vs Analytical Fatigue Life', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which='both')
    fig_path = output_dir / 'sn_curve_life.png'
    fig.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    figures['sn_curve_life'] = str(fig_path)

    return figures


def main():
    parser = argparse.ArgumentParser(description='Weld toe stress concentration benchmark')
    parser.add_argument('--output-dir', type=str, default='runs/fatigue_detail',
                        help='Output directory for results and figures')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Weld Toe Stress Concentration Benchmark — Detail Fatigue Validation")
    print("=" * 70)

    print("\n[1/4] Kt validation (Nii & Maddox formulas)...")
    kt_results = run_kt_validation(output_dir)

    print("[2/4] Hot-spot stress extrapolation...")
    hotspot_results = run_hotspot_extrapolation(output_dir)

    print("[3/4] Mesh convergence analysis...")
    mesh_results = run_mesh_convergence(output_dir)

    print("[4/4] Fatigue life calculation...")
    fatigue_results = run_fatigue_life(output_dir)

    print("\nGenerating figures...")
    figures = generate_figures(output_dir, kt_results, hotspot_results, mesh_results, fatigue_results)

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    t_joint_errors = [d['error_pct'] for d in kt_results['t_joint']]
    max_t_joint_error = max(t_joint_errors)
    criterion_a_pass = bool(max_t_joint_error <= 3.0)
    print(f"\n(a) T-joint Kt error ≤ 3% vs Nii formula:")
    for d in kt_results['t_joint']:
        status = "PASS" if d['pass'] else "FAIL"
        print(f"    r={d['weld_toe_radius_mm']}mm: Kt_analytical={d['kt_analytical']:.4f}, "
              f"Kt_FEM={d['kt_fem']:.4f}, error={d['error_pct']:.2f}% [{status}]")
    print(f"    Max error: {max_t_joint_error:.2f}% → {'PASS' if criterion_a_pass else 'FAIL'}")

    butt_errors = [d['error_pct'] for d in kt_results['butt_joint']]
    max_butt_error = max(butt_errors)
    criterion_b_pass = bool(max_butt_error <= 3.0)
    print(f"\n(b) Butt joint Kt error ≤ 3% vs Maddox formula:")
    for d in kt_results['butt_joint']:
        status = "PASS" if d['pass'] else "FAIL"
        print(f"    r={d['weld_toe_radius_mm']}mm: Kt_analytical={d['kt_analytical']:.4f}, "
              f"Kt_FEM={d['kt_fem']:.4f}, error={d['error_pct']:.2f}% [{status}]")
    print(f"    Max error: {max_butt_error:.2f}% → {'PASS' if criterion_b_pass else 'FAIL'}")

    criterion_c_pass = mesh_results['pass_2x_convergence']
    print(f"\n(c) Hot-spot stress mesh convergence (2x refinement ≤ 2%):")
    print(f"    Last 2x change: {mesh_results['last_2x_refinement_change_pct']:.4f}%")
    print(f"    Converged: {mesh_results['converged']}")
    print(f"    → {'PASS' if criterion_c_pass else 'FAIL'}")

    criterion_d_pass = fatigue_results['pass_fatigue_life']
    print(f"\n(d) Fatigue life N error ≤ 15% vs S-N analytical:")
    print(f"    N_computed = {fatigue_results['n_to_failure']:.0f}")
    print(f"    N_analytical = {fatigue_results['n_analytical']:.0f}")
    print(f"    Error: {fatigue_results['n_error_pct']:.2f}%")
    print(f"    → {'PASS' if criterion_d_pass else 'FAIL'}")

    all_pass = criterion_a_pass and criterion_b_pass and criterion_c_pass and criterion_d_pass
    print(f"\n{'=' * 70}")
    print(f"OVERALL: {'ALL CRITERIA PASSED' if all_pass else 'SOME CRITERIA FAILED'}")
    print(f"{'=' * 70}")

    report = {
        'benchmark': 'weld_toe_scg',
        'mission_id': 'p92w2b-n03-structural-fatigue-detail',
        'criteria': {
            'a_t_joint_kt_error_pct': round(max_t_joint_error, 4),
            'a_pass': criterion_a_pass,
            'b_butt_joint_kt_error_pct': round(max_butt_error, 4),
            'b_pass': criterion_b_pass,
            'c_mesh_convergence_2x_change_pct': round(mesh_results['last_2x_refinement_change_pct'], 4),
            'c_pass': criterion_c_pass,
            'd_fatigue_life_error_pct': round(fatigue_results['n_error_pct'], 4),
            'd_pass': criterion_d_pass,
        },
        'overall_pass': all_pass,
        'kt_results': kt_results,
        'hotspot_results': hotspot_results,
        'mesh_results': mesh_results,
        'fatigue_results': fatigue_results,
        'figures': {k: str(v) for k, v in figures.items()},
    }

    report_path = output_dir / 'weld_toe_scg_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    print(f"\nReport saved to: {report_path}")

    return 0 if all_pass else 1


if __name__ == '__main__':
    sys.exit(main())
