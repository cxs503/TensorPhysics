"""Generate figures for P165W2-N08 maturity report."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

OUT = "/root/tp-agent/work/m9319/docs/figures"
os.makedirs(OUT, exist_ok=True)

# ---- Figure 1: Void fraction profile ----
from tensorphysics.domains.sph_aeration import compute_void_fraction_profile

y = np.linspace(0, 2.0, 100)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for Fr in [0.2, 0.3, 0.4, 0.5]:
    alpha = compute_void_fraction_profile(y, U=Fr*np.sqrt(9.81*10), L=10.0, Fr=Fr)
    axes[0].plot(alpha, y, label=f"Fr={Fr}")
axes[0].set_xlabel("Void fraction α")
axes[0].set_ylabel("Height above free surface y [m]")
axes[0].set_title("SPH Aeration: Void Fraction Profile")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ---- Figure 2: Sensitivity analysis ----
from tensorphysics.domains.multiphase_structure_coupling import sensitivity_analysis

alpha_range = np.linspace(0, 0.5, 50)
results = sensitivity_analysis(alpha_range, v_impact=5.0, G_dry=1000.0)

axes[1].plot(alpha_range, results["pressure_reduction_pct"], "b-", label="Pressure reduction")
axes[1].plot(alpha_range, results["G_reduction_pct"], "r--", label="G reduction")
axes[1].set_xlabel("Mean void fraction α")
axes[1].set_ylabel("Reduction [%]")
axes[1].set_title("Aeration Buffering Sensitivity Analysis")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/p165w2_n08_fig1_void_fraction_sensitivity.png", dpi=150)
plt.close()

# ---- Figure 3: Crack propagation ----
from tensorphysics.domains.phase_field_fracture import PhaseFieldFractureSolver

solver = PhaseFieldFractureSolver(E=210e9, G_c=2700.0, L=1.0, n_elements=30)
result = solver.solve(problem_type="sent", max_load=2e6, n_load_steps=20)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].plot(result.load_steps / 1e6, result.crack_length_history * 1000, "b-o", markersize=3)
axes[0].set_xlabel("Applied stress [MPa]")
axes[0].set_ylabel("Crack length [mm]")
axes[0].set_title("Phase-Field Fracture: Crack Growth Curve")
axes[0].grid(True, alpha=0.3)

axes[1].plot(result.load_steps / 1e6, result.G_history, "r-s", markersize=3)
axes[1].axhline(y=2700.0, color='k', linestyle='--', label=f"G_c = 2700 J/m²")
axes[1].set_xlabel("Applied stress [MPa]")
axes[1].set_ylabel("Energy release rate G [J/m²]")
axes[1].set_title("Energy Release Rate Evolution")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/p165w2_n08_fig2_fracture_propagation.png", dpi=150)
plt.close()

# ---- Figure 4: Interface consistency ----
from tensorphysics.domains.sph_aeration import SPHAerationSolver, SPHToResistanceInterface
from tensorphysics.domains.phase_field_fracture import FEMToStructuralInterface

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# SPH interface
sph_solver = SPHAerationSolver(L=5.0, U=1.5, n_particles=20)
sph_result = sph_solver.solve(duration=1.0)
interface = SPHToResistanceInterface()
alpha_hull = interface.transform(sph_result.void_field)

patches = np.arange(len(alpha_hull))
axes[0].bar(patches, alpha_hull, color='steelblue', alpha=0.7)
axes[0].set_xlabel("Hull patch index")
axes[0].set_ylabel("Mean void fraction α")
axes[0].set_title("SPH→Resistance: Void Fraction per Hull Patch")
axes[0].grid(True, alpha=0.3, axis='y')

# FEM interface
fem_solver = PhaseFieldFractureSolver(L=1.0, n_elements=30)
fem_result = fem_solver.solve(n_load_steps=15)
fem_interface = FEMToStructuralInterface()
crack_data = fem_interface.transform(fem_result)

steps = np.arange(len(crack_data["crack_length_history"]))
axes[1].plot(steps, crack_data["crack_length_history"] * 1000, "g-o", markersize=3)
axes[1].set_xlabel("Load step")
axes[1].set_ylabel("Crack length [mm]")
axes[1].set_title("FEM→Structural: Crack Length History")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/p165w2_n08_fig3_interface_consistency.png", dpi=150)
plt.close()

# ---- Figure 5: Impedance vs void fraction ----
from tensorphysics.domains.multiphase_structure_coupling import effective_impedance

alpha_scan = np.linspace(0, 0.5, 100)
Z_vals = np.array([effective_impedance(a)[0] for a in alpha_scan])
rho_vals = np.array([effective_impedance(a)[1] for a in alpha_scan])
c_vals = np.array([effective_impedance(a)[2] for a in alpha_scan])

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].plot(alpha_scan, Z_vals / 1e6, "b-")
axes[0].set_xlabel("Void fraction α")
axes[0].set_ylabel("Z_eff [MPa·s/m]")
axes[0].set_title("Effective Impedance")
axes[0].grid(True, alpha=0.3)

axes[1].plot(alpha_scan, rho_vals, "g-")
axes[1].set_xlabel("Void fraction α")
axes[1].set_ylabel("ρ_eff [kg/m³]")
axes[1].set_title("Effective Density")
axes[1].grid(True, alpha=0.3)

axes[2].plot(alpha_scan, c_vals, "r-")
axes[2].set_xlabel("Void fraction α")
axes[2].set_ylabel("c_eff [m/s]")
axes[2].set_title("Effective Sound Speed (Wood's eq.)")
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT}/p165w2_n08_fig4_impedance_properties.png", dpi=150)
plt.close()

print("All figures generated successfully.")
print(f"Output directory: {OUT}")
for f in sorted(os.listdir(OUT)):
    if "p165w2" in f:
        print(f"  {f}")
