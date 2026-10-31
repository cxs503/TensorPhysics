"""Generate publication-quality figures for propeller-ice interaction model."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorphysics.propeller_ice import (
    PropellerGeometry, IceProperties, OperatingCondition,
    ice_milling_force, torque_spectrum, run_propeller_ice_simulation,
)

os.makedirs('/root/tp-agent/work/m722/artifacts/m722', exist_ok=True)

# Run simulation
prop = PropellerGeometry()
ice = IceProperties(sigma_c=5.0e6)
op = OperatingCondition(rpm=120.0, Va=3.0, ice_concentration=0.3)
result = ice_milling_force(prop, ice, op, n_samples=5000, seed=42)
spec = torque_spectrum(result['torque_time_series'], result['dt'], op.rpm, prop.n_blades)

# ---- Figure 1: Force time series ----
fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
t = np.arange(len(result['force_time_series'])) * result['dt']

axes[0].plot(t, result['force_time_series'] / 1000, 'b-', linewidth=0.5, alpha=0.8)
axes[0].axhline(result['mean_force'] / 1000, color='r', linestyle='--', linewidth=1.5,
                label=f'Mean = {result["mean_force"]/1000:.1f} kN')
axes[0].set_ylabel('Ice Milling Force [kN]', fontsize=12)
axes[0].set_title('Propeller–Ice Milling Force Time Series\n'
                  f'(D={prop.D}m, {prop.n_blades}-blade, RPM={op.rpm}, '
                  f'ice conc.={op.ice_concentration}, σ_c={ice.sigma_c/1e6:.0f} MPa)',
                  fontsize=13)
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3)

axes[1].plot(t, result['torque_time_series'] / 1000, 'g-', linewidth=0.5, alpha=0.8)
axes[1].axhline(result['mean_torque'] / 1000, color='r', linestyle='--', linewidth=1.5,
                label=f'Mean = {result["mean_torque"]/1000:.1f} kN·m')
axes[1].set_ylabel('Ice Torque [kN·m]', fontsize=12)
axes[1].set_xlabel('Time [s]', fontsize=12)
axes[1].legend(fontsize=11)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig('/root/tp-agent/work/m722/artifacts/m722/fig1_force_torque_timeseries.png', dpi=150)
plt.close()

# ---- Figure 2: Torque PSD spectrum ----
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

freq = spec['freq']
psd = spec['psd']
# Plot in dB scale
psd_db = 10 * np.log10(np.maximum(psd, 1e-30))
ax.semilogx(freq[1:], psd_db[1:], 'b-', linewidth=0.8, alpha=0.8)

# Mark BPF and shaft frequency
ax.axvline(spec['bpf'], color='r', linestyle='--', linewidth=1.5,
           label=f'BPF = {spec["bpf"]:.1f} Hz')
ax.axvline(spec['shaft_freq'], color='orange', linestyle=':', linewidth=1.5,
           label=f'Shaft freq = {spec["shaft_freq"]:.1f} Hz')

# Mark peak
bpf_idx = np.argmin(np.abs(freq - spec['bpf']))
ax.plot(spec['bpf'], psd_db[bpf_idx], 'rv', markersize=12,
        label=f'Peak at BPF: {spec["peak_ratio_bpf"]:.1f}× mean')

ax.set_xlabel('Frequency [Hz]', fontsize=12)
ax.set_ylabel('PSD [dB(N²·m²/Hz)]', fontsize=12)
ax.set_title('Torque Fluctuation Power Spectral Density\n'
             f'(BPF peak ratio = {spec["peak_ratio_bpf"]:.1f}, '
             f'Cv = {spec["torque_fluctuation_coeff"]:.3f})',
             fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3, which='both')
ax.set_xlim([0.1, freq[-1]])

plt.tight_layout()
fig.savefig('/root/tp-agent/work/m722/artifacts/m722/fig2_torque_psd_spectrum.png', dpi=150)
plt.close()

# ---- Figure 3: Parameter sensitivity ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 3a: Force vs ice crushing strength
sigmas = np.linspace(2e6, 15e6, 20)
forces_sigma = []
for s in sigmas:
    r = ice_milling_force(prop, IceProperties(sigma_c=s), op, n_samples=1000, seed=42)
    forces_sigma.append(r['mean_force'] / 1000)
axes[0].plot(sigmas / 1e6, forces_sigma, 'b-o', linewidth=2, markersize=5)
axes[0].set_xlabel('Ice Crushing Strength [MPa]', fontsize=11)
axes[0].set_ylabel('Mean Milling Force [kN]', fontsize=11)
axes[0].set_title('(a) Force vs Ice Strength', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 3b: Force vs ice concentration
concs = np.linspace(0.05, 0.8, 20)
forces_conc = []
for c in concs:
    r = ice_milling_force(prop, ice, OperatingCondition(rpm=120, Va=3, ice_concentration=c),
                          n_samples=1000, seed=42)
    forces_conc.append(r['mean_force'] / 1000)
axes[1].plot(concs, forces_conc, 'r-s', linewidth=2, markersize=5)
axes[1].set_xlabel('Ice Concentration [-]', fontsize=11)
axes[1].set_ylabel('Mean Milling Force [kN]', fontsize=11)
axes[1].set_title('(b) Force vs Ice Concentration', fontsize=12)
axes[1].grid(True, alpha=0.3)

# 3c: Force vs RPM
rpms = np.linspace(40, 200, 20)
forces_rpm = []
for rpm in rpms:
    r = ice_milling_force(prop, ice, OperatingCondition(rpm=rpm, Va=3, ice_concentration=0.3),
                          n_samples=1000, seed=42)
    forces_rpm.append(r['mean_force'] / 1000)
axes[2].plot(rpms, forces_rpm, 'g-^', linewidth=2, markersize=5)
axes[2].set_xlabel('Shaft RPM [rev/min]', fontsize=11)
axes[2].set_ylabel('Mean Milling Force [kN]', fontsize=11)
axes[2].set_title('(c) Force vs Shaft Speed', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Propeller–Ice Interaction: Parameter Sensitivity Analysis', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig('/root/tp-agent/work/m722/artifacts/m722/fig3_parameter_sensitivity.png', dpi=150,
            bbox_inches='tight')
plt.close()

# ---- Figure 4: Validation comparison ----
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Experimental reference
exp_force = 150.0  # kN
exp_torque_std = 25.0  # kN·m
exp_bpf_peak = 8.0

# Model results
model_force = result['mean_force'] / 1000
model_torque_std = result['std_torque'] / 1000
model_bpf_peak = spec['peak_ratio_bpf']

categories = ['Mean Force\n[kN]', 'Torque Std\n[kN·m]', 'BPF Peak\nRatio']
exp_vals = [exp_force, exp_torque_std, exp_bpf_peak]
model_vals = [model_force, model_torque_std, model_bpf_peak]

x = np.arange(3)
width = 0.35
bars1 = axes[0].bar(x - width/2, exp_vals, width, label='Experiment (Ref.)', color='steelblue', alpha=0.8)
bars2 = axes[0].bar(x + width/2, model_vals, width, label='Model (m722)', color='coral', alpha=0.8)
axes[0].set_xticks(x)
axes[0].set_xticklabels(categories, fontsize=10)
axes[0].legend(fontsize=10)
axes[0].set_title('Model vs Experiment', fontsize=12)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar in bars1:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
for bar in bars2:
    axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                 f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

# Error bars
errors = [abs(m - e) / e * 100 for m, e in zip(model_vals, exp_vals)]
colors = ['green' if e < 20 else 'orange' if e < 25 else 'red' for e in errors]
axes[1].bar(categories, errors, color=colors, alpha=0.8)
axes[1].axhline(20, color='green', linestyle='--', linewidth=1, label='Force threshold (20%)')
axes[1].axhline(25, color='orange', linestyle='--', linewidth=1, label='Spectrum threshold (25%)')
axes[1].set_ylabel('Relative Error [%]', fontsize=11)
axes[1].set_title('Validation Errors', fontsize=12)
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3, axis='y')
for i, (cat, err) in enumerate(zip(categories, errors)):
    axes[1].text(i, err + 0.5, f'{err:.1f}%', ha='center', fontsize=10)

# Pass/Fail summary
criteria = ['Force Error\n< 20%', 'Torque Std\nError < 25%', 'BPF Peak\nError < 25%']
actual_errors = [errors[0], errors[1], errors[2]]
thresholds = [20, 25, 25]
pass_fail = ['PASS' if e < t else 'FAIL' for e, t in zip(actual_errors, thresholds)]
colors_pf = ['green' if p == 'PASS' else 'red' for p in pass_fail]
axes[2].bar(criteria, actual_errors, color=colors_pf, alpha=0.8)
for i, (cat, err, pf) in enumerate(zip(criteria, actual_errors, pass_fail)):
    axes[2].text(i, err + 0.3, f'{pf}\n({err:.1f}%)', ha='center', fontsize=10, fontweight='bold')
axes[2].set_ylabel('Error [%]', fontsize=11)
axes[2].set_title('Acceptance Criteria', fontsize=12)
axes[2].grid(True, alpha=0.3, axis='y')

plt.suptitle('Propeller–Ice Model Validation Against Ice-Tank Experiments', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig('/root/tp-agent/work/m722/artifacts/m722/fig4_validation.png', dpi=150,
            bbox_inches='tight')
plt.close()

print("All figures generated successfully.")
print(f"  fig1_force_torque_timeseries.png")
print(f"  fig2_torque_psd_spectrum.png")
print(f"  fig3_parameter_sensitivity.png")
print(f"  fig4_validation.png")
