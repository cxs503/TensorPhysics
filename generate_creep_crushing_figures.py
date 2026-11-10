#!/usr/bin/env python3
"""Generate figures and report for the creep-crushing module (P270W1-N09).

Produces:
  artifacts/creep_crushing_m9579/
    fig1_pressure_history.png    — pressure vs time for different stiffnesses
    fig2_scaling_law.png         — pressure–velocity scaling
    fig3_relaxation.png          — stress relaxation vs analytical
    fig4_phase_space.png         — pressure vs effective velocity
    results.json                 — all acceptance-criterion numbers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT / "src"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.ice.creep_crushing import (
    NortonCreepParams,
    CreepCrushingConfig,
    solve_creep_crushing,
    steady_state_pressure,
    rigid_limit_pressure,
    relaxation_time_constant,
    analytical_relaxation,
    iacs_pc_sustained_pressure,
    check_iacs_monotonicity,
)

ARTIFACT_DIR = REPO_ROOT / "artifacts" / "creep_crushing_m9579"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)


def fig1_pressure_history():
    """Figure 1: Pressure vs time for different structural stiffnesses."""
    creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
    v = 0.01
    L_c = 0.5
    p_rigid = rigid_limit_pressure(v, creep, L_c)

    fig, ax = plt.subplots(figsize=(8, 5))

    stiffnesses = [1e6, 1e8, 1e10, 1e12]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for k_s, color in zip(stiffnesses, colors):
        cfg = CreepCrushingConfig(
            v=v, k_s=k_s, A=0.1, L_c=L_c,
            creep=creep, t_end=10.0, n_steps=2000,
        )
        res = solve_creep_crushing(cfg, compute_m=False)
        tau = res.tau
        label = f'$k_s$={k_s:.0e} N/m (τ={tau:.3f} s)'
        ax.plot(res.t, res.p / 1e3, color=color, linewidth=1.5, label=label)

    ax.axhline(y=p_rigid / 1e3, color='k', linestyle='--', linewidth=1,
               label=f'Rigid limit: {p_rigid/1e3:.1f} kPa')
    ax.set_xlabel('Time [s]', fontsize=12)
    ax.set_ylabel('Contact pressure [kPa]', fontsize=12)
    ax.set_title('Pressure–time history: effect of structural stiffness', fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "fig1_pressure_history.png", dpi=150)
    plt.close(fig)
    return {"p_rigid_kPa": p_rigid / 1e3}


def fig2_scaling_law():
    """Figure 2: Pressure–velocity scaling law."""
    creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
    L_c = 0.5

    v_arr = np.logspace(-3, 1, 50)
    p_rigid = np.array([rigid_limit_pressure(v, creep, L_c) for v in v_arr])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(v_arr, p_rigid / 1e3, 'b-', linewidth=2, label='Rigid limit')

    # Theoretical slope: m = 1/n = 1/3
    v_ref = 0.01
    p_ref = rigid_limit_pressure(v_ref, creep, L_c)
    v_fit = np.array([v_ref / 10, v_ref * 10])
    p_fit = p_ref * (v_fit / v_ref) ** (1.0 / 3.0)
    ax.loglog(v_fit, p_fit / 1e3, 'r--', linewidth=1.5,
              label=f'Theory: $p \\propto v^{{1/n}}$, m=1/{creep.n:.0f}')

    ax.set_xlabel('Indentation velocity [m/s]', fontsize=12)
    ax.set_ylabel('Steady-state pressure [kPa]', fontsize=12)
    ax.set_title('Pressure–velocity scaling law', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, which='both')

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "fig2_scaling_law.png", dpi=150)
    plt.close(fig)
    return {"m_theory": 1.0 / creep.n}


def fig3_relaxation():
    """Figure 3: Stress relaxation vs analytical solution."""
    B = 1e-20
    n = 3.0
    k_s = 1e8
    A = 0.1
    L_c = 0.5
    p0 = 1e6

    tau0 = relaxation_time_constant(k_s, A, L_c, B, n, p0)
    t_end = 5.0 * tau0

    t_ana = np.linspace(0, t_end, 500)
    p_ana = analytical_relaxation(p0, k_s, A, L_c, B, n, t_ana)

    cfg = CreepCrushingConfig(
        v=0.0, k_s=k_s, A=A, L_c=L_c,
        creep=NortonCreepParams(B_ref=B, Q=0.0, n=n, T_K=263.15),
        p0=p0, t_end=t_end, n_steps=5000, relaxation_mode=True,
    )
    res = solve_creep_crushing(cfg, compute_m=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: linear scale
    ax1.plot(t_ana / tau0, p_ana / 1e3, 'b-', linewidth=2, label='Analytical')
    ax1.plot(res.t / tau0, res.p / 1e3, 'ro', markersize=3, label='Backward Euler')
    ax1.set_xlabel('$t / \\tau_0$', fontsize=12)
    ax1.set_ylabel('Pressure [kPa]', fontsize=12)
    ax1.set_title('Stress relaxation', fontsize=13)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Right: log scale
    ax2.semilogy(t_ana / tau0, np.maximum(p_ana, 1e-10) / 1e3, 'b-', linewidth=2, label='Analytical')
    ax2.semilogy(res.t / tau0, np.maximum(res.p, 1e-10) / 1e3, 'ro', markersize=3, label='Backward Euler')
    ax2.set_xlabel('$t / \\tau_0$', fontsize=12)
    ax2.set_ylabel('Pressure [kPa]', fontsize=12)
    ax2.set_title('Stress relaxation (log scale)', fontsize=13)
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3, which='both')

    fig.suptitle(f'$\\tau_0$ = {tau0:.4f} s, n = {n:.0f}', fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "fig3_relaxation.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    return {"tau0_s": tau0}


def fig4_iacs_monotonicity():
    """Figure 4: IACS PC sustained pressure monotonicity."""
    ok, values = check_iacs_monotonicity(area=1.0)

    classes = [v[0] for v in values]
    pressures = [v[1] / 1e6 for v in values]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(classes, pressures, color='#2ca02c', edgecolor='black', alpha=0.8)
    ax.set_xlabel('Polar Class', fontsize=12)
    ax.set_ylabel('Sustained pressure [MPa]', fontsize=12)
    ax.set_title('IACS Polar Class sustained ice pressure (A = 1 m²)', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, p in zip(bars, pressures):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{p:.1f}', ha='center', va='bottom', fontsize=9)

    fig.tight_layout()
    fig.savefig(ARTIFACT_DIR / "fig4_iacs_monotonicity.png", dpi=150)
    plt.close(fig)
    return {"monotone": ok, "values_MPa": {c: p / 1e6 for c, p in values}}


def compute_results():
    """Compute all acceptance-criterion numbers."""
    creep = NortonCreepParams(B_ref=1e-16, Q=0.0, n=3.0, T_K=263.15)
    v = 0.01
    L_c = 0.5

    # Rigid limit
    p_rigid = rigid_limit_pressure(v, creep, L_c)

    # Solver result
    cfg = CreepCrushingConfig(
        v=v, k_s=1e10, A=0.1, L_c=L_c,
        creep=creep, t_end=50.0, n_steps=2000,
    )
    res = solve_creep_crushing(cfg)

    # Scaling exponent
    m = res.m_scaling

    # Relaxation time
    tau = res.tau

    # IACS monotonicity
    ok_iacs, _ = check_iacs_monotonicity()

    results = {
        "p_rigid_Pa": p_rigid,
        "p_rigid_kPa": p_rigid / 1e3,
        "p_sustained_Pa": res.p_sustained,
        "p_sustained_kPa": res.p_sustained / 1e3,
        "p_peak_Pa": res.p_peak,
        "tau_s": tau,
        "m_scaling": m,
        "m_theory": 1.0 / creep.n,
        "m_error_pct": abs(m - 1.0 / creep.n) / (1.0 / creep.n) * 100,
        "converged": res.converged,
        "iacs_monotone": ok_iacs,
        "n_exponent": creep.n,
        "B_eff": creep.B_eff,
        "v_imposed": v,
        "L_c": L_c,
    }
    return results


def main():
    print("Generating figures...")
    info1 = fig1_pressure_history()
    print(f"  fig1: p_rigid = {info1['p_rigid_kPa']:.1f} kPa")

    info2 = fig2_scaling_law()
    print(f"  fig2: m_theory = {info2['m_theory']:.4f}")

    info3 = fig3_relaxation()
    print(f"  fig3: tau0 = {info3['tau0_s']:.4f} s")

    info4 = fig4_iacs_monotonicity()
    print(f"  fig4: IACS monotone = {info4['monotone']}")

    results = compute_results()
    print(f"\nResults:")
    for k, v in results.items():
        print(f"  {k}: {v}")

    with open(ARTIFACT_DIR / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nArtifacts saved to {ARTIFACT_DIR}")


if __name__ == "__main__":
    main()
