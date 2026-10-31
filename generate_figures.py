#!/usr/bin/env python3
"""Generate figures for shaft torsional vibration analysis report."""

import numpy as np
import matplotlib.pyplot as plt
import json
import os
from src.tensorphysics.structural_dynamics import (
    UniformShaftAnalytical,
    ShaftTorsionAnalyzer,
    build_uniform_shaft_model,
    sdof_frequency_response,
)

# Reference parameters
G_REF = 79.3e9       # Pa
RHO_REF = 7850.0     # kg/m³
L_REF = 12.0         # m
D_REF = 0.5          # m
TARGET_FREQS = [66.216, 198.647, 331.078, 463.509]

os.makedirs("outputs/figures", exist_ok=True)

# Figure 1: Modal analysis - first 4 modes comparison
print("Generating Figure 1: Modal analysis...")
shaft = UniformShaftAnalytical(G=G_REF, rho=RHO_REF, L=L_REF)
analyzer = ShaftTorsionAnalyzer(
    G=G_REF, rho=RHO_REF, L=L_REF, D=D_REF,
    propeller_J=0.0, engine_J=0.0,
)
freqs_ana, freqs_num = analyzer.modal_analysis(N=50, n_modes=4)

fig, ax = plt.subplots(figsize=(10, 6))
modes = np.arange(1, 5)
width = 0.35
bars1 = ax.bar(modes - width/2, freqs_ana, width, label='Analytical', color='steelblue')
bars2 = ax.bar(modes + width/2, freqs_num, width, label='Numerical (N=50)', color='coral')

ax.set_xlabel('Mode Number', fontsize=12)
ax.set_ylabel('Natural Frequency [Hz]', fontsize=12)
ax.set_title('Natural Frequencies: Analytical vs Numerical', fontsize=14, fontweight='bold')
ax.set_xticks(modes)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add error annotations
for i, (fa, fn) in enumerate(zip(freqs_ana, freqs_num)):
    err = abs(fn - fa) / fa * 100
    ax.text(i + 1 + width/2, fn + 5, f'err={err:.3f}%', 
            ha='center', va='bottom', fontsize=9, color='darkred')

plt.tight_layout()
plt.savefig('outputs/figures/fig1_modal_analysis.png', dpi=150)
plt.close()

# Figure 2: Convergence study
print("Generating Figure 2: Convergence study...")
N_values = [5, 10, 15, 20, 30, 40, 50, 80, 100, 160]
freqs_ana_ref = shaft.frequencies(4)
max_errors = []

for N in N_values:
    model = build_uniform_shaft_model(shaft, N, propeller_J=0.0, engine_J=0.0)
    freqs_num, _ = model.solve_eigenvalue(n_modes=4)
    err = np.max(np.abs(freqs_num - freqs_ana_ref) / freqs_ana_ref)
    max_errors.append(err)

fig, ax = plt.subplots(figsize=(10, 6))
ax.loglog(N_values, max_errors, 'bo-', linewidth=2, markersize=8, label='Max error (4 modes)')

# Reference slope N^-2
N_ref = np.array([5, 160])
err_ref = max_errors[0] * (N_ref / N_values[0])**(-2)
ax.loglog(N_ref, err_ref, 'r--', linewidth=2, label='Reference slope N⁻²')

ax.set_xlabel('Number of Elements N', fontsize=12)
ax.set_ylabel('Maximum Relative Error', fontsize=12)
ax.set_title('Mesh Convergence: Error vs Number of Elements', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/fig2_convergence.png', dpi=150)
plt.close()

# Figure 3: Transient response (ice milling excitation)
print("Generating Figure 3: Transient response...")
from src.tensorphysics.structural_dynamics import IceMillingExcitation, NewmarkBetaSolver

exc = IceMillingExcitation(T_mean=50000.0, n_blades=5, RPM=120.0)
model = build_uniform_shaft_model(shaft, N=30, propeller_J=5000.0, engine_J=10000.0)
M_full, K_full = model.build_matrices()
free = [i for i in range(model.n_nodes) if i not in model.fixed_nodes]
M = np.diag(M_full)[free][:, free]
K = K_full[np.ix_(free, free)]

# Rayleigh damping
freqs_num, _ = model.solve_eigenvalue(n_modes=2)
omega1 = 2 * np.pi * freqs_num[0]
omega2 = 2 * np.pi * freqs_num[1]
zeta = 0.02
a0 = 2 * zeta * omega1 * omega2 / (omega1 + omega2)
a1 = 2 * zeta / (omega1 + omega2)
C = a0 * M + a1 * K

duration = 2.0
dt = 1e-4
t = np.arange(0, duration + dt, dt)
T_t = exc.torque(t)

F = np.zeros((len(t), len(free)))
F[:, -1] = T_t

solver = NewmarkBetaSolver(M, C, K)
result = solver.solve(t, F)
x = result["x"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Excitation torque
ax1.plot(t * 1000, T_t / 1000, 'b-', linewidth=1.5)
ax1.set_ylabel('Torque [kN·m]', fontsize=12)
ax1.set_title('Ice Milling Excitation (BPF = 10 Hz, 5 blades @ 120 RPM)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.axhline(y=exc.T_mean/1000, color='r', linestyle='--', label=f'Mean = {exc.T_mean/1000:.1f} kN·m')
ax1.legend(fontsize=10)

# Propeller angular displacement
ax2.plot(t * 1000, x[:, -1] * 180 / np.pi, 'r-', linewidth=1.5)
ax2.set_xlabel('Time [ms]', fontsize=12)
ax2.set_ylabel('Angular Displacement [deg]', fontsize=12)
ax2.set_title('Propeller Torsional Response', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/fig3_transient_response.png', dpi=150)
plt.close()

# Figure 4: Frequency response function
print("Generating Figure 4: Frequency response...")
fr = sdof_frequency_response(zeta=0.05, freq_range=(0.3, 2.0), n_freq=100)

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(fr["freqs"], fr["H_analytical"], 'b-', linewidth=2, label='Analytical |H(ω)|')
ax.plot(fr["freqs"], fr["H_numerical"], 'ro', markersize=6, label='Numerical (Newmark-β)')

ax.set_xlabel('Frequency [Hz]', fontsize=12)
ax.set_ylabel('|H(ω)| [m/N]', fontsize=12)
ax.set_title('SDOF Frequency Response Function (ζ = 0.05)', fontsize=14, fontweight='bold')
ax.set_yscale('log')
ax.legend(fontsize=11)
ax.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/fig4_frequency_response.png', dpi=150)
plt.close()

# Figure 5: Resonance Q-factor verification
print("Generating Figure 5: Resonance Q-factor...")
from src.tensorphysics.structural_dynamics import sdof_resonance_test

zeta_values = [0.02, 0.05, 0.1]
Q_analytical = []
Q_numerical = []

for zeta in zeta_values:
    res = sdof_resonance_test(zeta, n_cycles=100, pp_cycles=80)
    Q_analytical.append(res["Q_analytical"])
    Q_numerical.append(res["Q_numerical"])

fig, ax = plt.subplots(figsize=(10, 6))
x_pos = np.arange(len(zeta_values))
width = 0.35

bars1 = ax.bar(x_pos - width/2, Q_analytical, width, label='Analytical Q = 1/(2ζ)', color='steelblue')
bars2 = ax.bar(x_pos + width/2, Q_numerical, width, label='Numerical', color='coral')

ax.set_xlabel('Damping Ratio ζ', fontsize=12)
ax.set_ylabel('Resonance Amplification Factor Q', fontsize=12)
ax.set_title('SDOF Resonance Amplification: Q = 1/(2ζ)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels([f'ζ = {z}' for z in zeta_values])
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')

# Add error annotations
for i, (qa, qn) in enumerate(zip(Q_analytical, Q_numerical)):
    err = abs(qn - qa) / qa * 100
    ax.text(i + width/2, qn + 1, f'err={err:.2f}%', 
            ha='center', va='bottom', fontsize=9, color='darkred')

plt.tight_layout()
plt.savefig('outputs/figures/fig5_resonance_Q.png', dpi=150)
plt.close()

print("All figures generated successfully!")
print("Figures saved to outputs/figures/")
