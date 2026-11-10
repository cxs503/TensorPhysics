"""Run the C1..C6 acceptance for the torsional-chain node and emit artifacts.

Produces (all numbers programmatically injected, nothing hand-copied):

    artifacts/p348w3_n08_torsional_chain/acceptance.json
    artifacts/p348w3_n08_torsional_chain/fig_*.png

Usage::

    python run_torsional_chain_acceptance.py [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.torsional_chain import acceptance as acc  # noqa: E402
from tensorphysics.torsional_chain.theory import sdof_amplitude  # noqa: E402

OUT_DEFAULT = ROOT / "artifacts" / "p348w3_n08_torsional_chain"


def _sanitize(obj):
    """JSON-safe copy: non-finite floats become strings, numpy scalars become Python."""
    if isinstance(obj, dict):
        return {str(k): _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if math.isinf(value):
            return "inf" if value > 0 else "-inf"
        if math.isnan(value):
            return "nan"
        return value
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


def _figure_chain(c1, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [case["configuration"].replace("_", "\n") for case in c1["cases"]]
    sim = [case["simulated_hz"] for case in c1["cases"]]
    ref = [case["reference_hz"] for case in c1["cases"]]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    ax.bar(x - 0.2, sim, width=0.4, label="simulated (eigen-solve)")
    ax.bar(x + 0.2, ref, width=0.4, label="closed-form reference", alpha=0.85)
    for xi, case in zip(x, c1["cases"]):
        ax.text(xi, max(case["simulated_hz"], case["reference_hz"]) * 1.04,
                f"{100 * case['relative_error']:.2e}%", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_ylabel("natural frequency [Hz]")
    ax.set_title(f"C1 torsional natural frequencies, worst error "
                 f"{100 * c1['worst_relative_error']:.3f}% (gate 3%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_gear(c2, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorphysics.torsional_chain.gear import MeshStiffness, ParametricMeshOscillator

    mesh = MeshStiffness(4.0e7, 0.2, 21)
    omega_n = math.sqrt(4.0e7 / 40.0)
    damping = 2.0 * 0.02 * math.sqrt(4.0e7 * 40.0)
    osc = ParametricMeshOscillator(40.0, damping, mesh, 0.1 * omega_n / 21)
    mesh_omega = osc.mesh_frequency
    hist = osc.integrate(4.0 * 2.0 * math.pi / mesh_omega, (2.0 * math.pi / omega_n) / 200.0,
                         lambda t: 2.0e4 * math.cos(mesh_omega * t))
    window = hist["t"] >= 0.75 * hist["t"][-1]
    phase = (mesh_omega * hist["t"][window]) % (2.0 * math.pi)
    quasi = 2.0e4 * np.cos(mesh_omega * hist["t"][window]) / hist["stiffness"][window]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(phase, hist["phi"][window] * 1e3, "o", ms=3, label="RK4 response")
    axes[0].plot(phase, quasi * 1e3, "-", lw=1.0, label="quasi-static theory")
    axes[0].set_xlabel("pinion mesh phase [rad]")
    axes[0].set_ylabel("torsional displacement [mrad]")
    axes[0].set_title(f"C2a periodic mesh response (amplitude error "
                      f"{100 * c2['amplitude_relative_error']:.2f}%, gate 5%)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    osc_m = ParametricMeshOscillator(1.0, 2.0 * 0.005 * 2.0 * math.pi * 30.0,
                                     MeshStiffness((2.0 * math.pi * 30.0) ** 2, 0.2, 21),
                                     2.0 * (2.0 * math.pi * 30.0) / 21)
    hist_m = osc_m.integrate(0.6, (2.0 * math.pi / (2.0 * math.pi * 30.0)) / 200.0,
                             lambda t: 0.0, phi0=1.0e-6)
    env = np.abs(hist_m["phi"])
    theory = 1.0e-6 * np.exp(c2["mathieu_growth_theory"] * hist_m["t"])
    axes[1].semilogy(hist_m["t"], env, lw=1.2, label="|response|")
    axes[1].semilogy(hist_m["t"], theory, "--", lw=1.0, label="exp(sigma t), sigma theory")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("envelope [rad]")
    axes[1].set_title(f"C2b Mathieu growth at tooth passing = 2$\\omega_n$ "
                      f"(error {100 * c2['mathieu_growth_error']:.2f}%)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_motor(c3, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from tensorphysics.torsional_chain.motor import PMSM, simulate_motor

    motor = PMSM(0.05, 5.0e-3, 1.2, 1.15, 400.0)
    run = simulate_motor(motor, 0.5, 20.0, t_end=0.6, dt=1.0e-4)
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    ax.plot(run["t"], run["omega"], label="speed $\\omega$")
    ax.plot(run["t"], run["back_emf"], "--", label="back-EMF $k_e\\omega$ [V]")
    ax.plot(run["t"], run["current"], ":", label="current $i_q$ [A]")
    ax.set_xlabel("time [s]")
    ax.set_title("C3a back-EMF feedback transient (load 20 N m)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    loads = [row["load_torque_nm"] for row in c3["cases"]]
    sim = [row["omega_sim"] for row in c3["cases"]]
    ref = [row["omega_analytic"] for row in c3["cases"]]
    x = np.arange(len(loads))
    ax.bar(x - 0.2, sim, width=0.4, label="RK4 steady speed")
    ax.bar(x + 0.2, ref, width=0.4, label="analytic $\\omega_{ss}$", alpha=0.85)
    for xi, row in zip(x, c3["cases"]):
        ax.text(xi, row["omega_sim"] * 1.002,
                f"{100 * row['relative_error']:.1e}%", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f} N m" for v in loads])
    ax.set_ylim(min(ref) * 0.98, max(ref) * 1.01)
    ax.set_ylabel("steady speed [rad/s]")
    ax.set_title(f"C3b steady speed vs load, worst error "
                 f"{100 * c3['worst_relative_error']:.1e}% (gate 2%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_transfer(c4, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ref = acc.torsional_reference_chain()
    k, c, inertia = ref["k"], ref["c"], ref["inertia"]
    omega_n = ref["omega_n"]
    wp = ref["segment"].section_modulus
    freqs = np.linspace(0.3, 1.8, 400) * omega_n / (2.0 * math.pi)
    theory = np.array([sdof_amplitude(2.0 * math.pi * f, inertia, k, c) * k / wp
                       for f in freqs])
    sim_f = [row["frequency_hz"] for row in c4["frf_cases"]]
    sim_a = [row["stress_amplitude_pa"] for row in c4["frf_cases"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.loglog(freqs, theory, "-", label="closed form $k|H(\\omega)|/W_p$")
    ax.loglog(sim_f, sim_a, "o", ms=5, label="chain + RK4 + projection")
    for line in c4["ice_lines"]:
        ax.axvline(line["frequency_hz"], color="0.7", ls=":", lw=0.8)
    ax.axvline(c4["natural_frequency_hz"], color="C3", ls="--", lw=0.9,
               label=f"$f_n$ = {c4['natural_frequency_hz']:.2f} Hz")
    ax.set_xlabel("frequency [Hz]")
    ax.set_ylabel("shaft shear stress amplitude [Pa]")
    ax.set_title(f"C4 ice torque -> shaft stress transfer, worst line error "
                 f"{100 * c4['worst_frf_error']:.2e}%, peak error "
                 f"{100 * c4['peak_position_error']:.2f}% (gate 3%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_timestep(c4_unused, c5, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    ax = axes[0]
    dt_macro = [point["dt_macro"] for point in c5["convergence"]]
    residual = [point["work_residual"] for point in c5["convergence"]]
    ax.loglog(dt_macro, residual, "o-", label="interface work residual")
    ax.axhline(0.02, color="C3", ls="--", label="2 % gate")
    ax.set_xlabel("macro step $dt_{macro}$ [s]")
    ax.set_ylabel("relative work residual")
    ax.set_title(f"C5a residual vs macro step (observed order "
                 f"{c5['observed_order']:.2f})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    names = list(c5["stability_limits"])
    limits = [c5["stability_limits"][n] for n in names]
    sub = [c5["plans"]["subcycling"]["dt_solver"][n] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, limits, width=0.4, label="stability limit $dt_{max}$")
    ax.bar(x + 0.2, sub, width=0.4, label="sub-cycled $dt$", alpha=0.85)
    for xi, n in zip(x, names):
        ax.text(xi, limits[list(names).index(n)] * 1.15,
                f"N={c5['plans']['subcycling']['substeps'][n]}", ha="center", fontsize=7)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    ax.set_ylabel("time step [s]")
    ax.set_title(f"C5b coordinated sub-cycling, worst residual "
                 f"{100 * c5['worst_work_residual']:.2f}% (gate 2%)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _figure_iacs(c6, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    checks = c6["polar_class_checks"]
    severe = c6["severe_resonance_case"]
    labels = [row["polar_class"] for row in checks] + ["PC2 @ resonance\n(hypothetical)"]
    util = [row["stress_utilization"] for row in checks] + [severe["stress_utilization"]]
    colors = ["C0"] * len(checks) + ["C3"]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(labels, util, color=colors)
    ax.axhline(1.0, color="k", ls="--", lw=0.9, label="allowable (utilisation = 1)")
    for i, value in enumerate(util):
        ax.text(i, value * 1.02, f"{value:.2f}", ha="center", fontsize=8)
    ax.set_yscale("log")
    ax.set_ylabel("stress utilisation $\\tau_a / \\tau_{allow}$")
    ax.set_title(f"C6 IACS Polar Class screening (design life {c6['design_life_years']:.0f} yr)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    c1 = acc.c1_natural_frequencies()
    c2 = acc.c2_gear_parametric()
    c3 = acc.c3_motor_steady_state()
    c4 = acc.c4_ice_torque_transfer()
    c5 = acc.c5_time_step_coordination()
    c6 = acc.c6_iacs_check()

    payload = {
        "node": "p348w3-n08-physics-torsional-timestep",
        "repository": "TensorPhysics",
        "module": "tensorphysics.torsional_chain",
        "checks": [c1, c2, c3, c4, c5, c6],
        "all_pass": all(c["pass"] for c in (c1, c2, c3, c4, c5, c6)),
        "models": {
            "drivetrain": "propeller - shaft1 - wheel ==mesh== pinion - shaft2 - motor "
                          "(lumped inertias, torsional spring-dampers, RK4)",
            "gear": "k_m(theta) = k_m0 (1 + eps cos(Z1 theta)), optional backlash dead zone",
            "motor": "PMSM q-axis L di/dt = V - R i - k_e omega, T_e = k_t i, "
                     "speed feedback into back-EMF",
            "excitation": "ice-milling torque = mean + harmonic comb at blade passing "
                          "frequency with 1/n decay",
            "coordination": "synchronized / sub-cycling / super-cycling, "
                            "state-dependent stability limits, adaptive step with growth cap",
            "iacs": "Polar Class alternating shear stress limits (model parameters) + "
                    "Miner damage on an S-N curve with a knee",
        },
    }
    (out / "acceptance.json").write_text(json.dumps(_sanitize(payload), indent=2))

    for name, fn, args_ in (
        ("fig_c1_modal.png", _figure_chain, (c1,)),
        ("fig_c2_gear_parametric.png", _figure_gear, (c2,)),
        ("fig_c3_motor_feedback.png", _figure_motor, (c3,)),
        ("fig_c4_transfer.png", _figure_transfer, (c4,)),
        ("fig_c5_timestep.png", _figure_timestep, (c4, c5)),
        ("fig_c6_iacs.png", _figure_iacs, (c6,)),
    ):
        path = out / name
        fn(*args_, path)
        print(f"wrote {path}")

    print(json.dumps({c["criterion"]: {"pass": c["pass"]} for c in payload["checks"]}))
    print(f"all_pass = {payload['all_pass']}")
    print(f"artifacts -> {out}")
    return 0 if payload["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())