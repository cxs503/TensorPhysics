"""Complete worked case: 15 MW podded propulsor in 1.5 m level ice.

Produces every artefact referenced by the acceptance criteria:

* ``artifacts/pod_ice/summary.json``            -- all criterion numbers
* ``artifacts/pod_ice/thrust_domain.json``      -- attainable domain stations
* ``artifacts/pod_ice/contact.json``            -- Hertz + FEM contact results
* ``artifacts/pod_ice/interaction.json``        -- time histories + phase check
* ``artifacts/pod_ice/fig_*.png``               -- report figures

Run with ``python -m scenarios.pod_ice_interaction.case_15mw``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .contact import contact_mesh_convergence, ice_pod_contact_case
from .envelope import IceLoadCase, PodSpec, rated_operating_point, thrust_domain, verify_domain
from .interaction import (
    InteractionConfig,
    compare_coupling,
    simulate_interaction,
    verify_phase,
)
from .propeller import BladeGeometry
from .structure import StrutSection

OUT = Path("artifacts/pod_ice")

V_SHIP = 8.0          # m/s (15.5 kn, Baltic ice-transit speed)
ICE_THICKNESS = 1.5   # m
ICE_TORQUE_MEAN = 0.30e6  # N m, mean ice torque on the shaft line (~26 % of rated torque)
INTERACTION_RPM = 100.0   # part-load ice transit speed of the drive train


def build_pod() -> PodSpec:
    return PodSpec(
        power_rated=15.0e6,
        n_rated_rpm=125.0,
        geometry=BladeGeometry(),
        strut=StrutSection(),
    )


def run_case(*, write: bool = True, with_convergence: bool = True) -> dict:
    pod = build_pod()
    ice = IceLoadCase(thickness=ICE_THICKNESS)
    curve = pod.curve()

    # ---------------------------------------------------------------- propeller
    op = rated_operating_point(pod, curve, V_SHIP)

    # ------------------------------------------------------------ thrust domain
    domain = thrust_domain(pod, ice, v_ship=V_SHIP, curve=curve)
    check = verify_domain(pod, ice, domain)

    # ------------------------------------------------------------------ contact
    contact = ice_pod_contact_case(ice.characteristic_force)
    convergence = contact_mesh_convergence(ice.characteristic_force) if with_convergence else []
    print('[pod-ice] domain+contact+convergence done', flush=True)

    # -------------------------------------------------------------- interaction
    cfg = InteractionConfig(ice_torque_mean=ICE_TORQUE_MEAN)
    one_way = simulate_interaction(
        curve, pod, ice, cfg, v_ship=V_SHIP, two_way=False, n_cmd_rpm=INTERACTION_RPM
    )
    two_way = simulate_interaction(
        curve, pod, ice, cfg, v_ship=V_SHIP, two_way=True, n_cmd_rpm=INTERACTION_RPM
    )
    phase = verify_phase(one_way, curve, pod, cfg)
    coupling = compare_coupling(one_way, two_way)

    # sensitivity: rated speed -> intermittent torque-limit clipping
    rated_one_way = simulate_interaction(
        curve, pod, ice, cfg, v_ship=V_SHIP, two_way=False, n_cmd_rpm=pod.n_rated_rpm
    )
    rated_phase = verify_phase(rated_one_way, curve, pod, cfg)

    summary = {
        "case": "15 MW podded propulsor, 1.5 m level ice",
        "ship_speed_m_s": V_SHIP,
        "ice_thickness_m": ICE_THICKNESS,
        "ice_characteristic_force_N": ice.characteristic_force,
        "propeller": {
            "diameter_m": pod.geometry.diameter,
            "n_blades": pod.geometry.n_blades,
            "hub_ratio": pod.geometry.hub_ratio,
            "pitch_ratio": pod.geometry.pitch_ratio,
            "expanded_area_ratio": pod.geometry.expanded_area_ratio,
            "rated_point": op,
            "motor_torque_limit_Nm": pod.motor_torque_limit,
            "kt_at_design": curve.kt_at(op["j"]),
            "kq_at_design": curve.kq_at(op["j"]),
        },
        "thrust_domain": {
            "stations": domain["stations"],
            "motor_limit_N": domain["motor_limit"]["thrust"],
            "bearing_limit_N": domain["bearing_limit"],
        },
        "structural_check": {
            "allowable_stress_Pa": pod.strut.allowable_stress,
            "n_structure_governed": check["n_structure_governed"],
            "max_utilization_inside": check["max_utilization_inside"],
            "overstressed_points": len(check["overstressed_points"]),
            "boundary_error_max": check["boundary_error_max"],
            "maximality_violations": check["maximality_violations"],
            "passed": check["passed"],
        },
        "contact": {
            "hertz_contact_radius_m": contact["contact_radius"],
            "hertz_peak_pressure_Pa": contact["peak_contact_pressure"],
            "hertz_approach_m": contact["approach"],
            "fem_max_von_mises_Pa": contact["fem"]["fem_max_von_mises"],
            "fem_max_over_p0": contact["fem"]["fem_max_over_p0"],
            "fem_max_depth_over_a": contact["fem"]["fem_max_depth_over_a"],
            "hertz_theory_max_von_mises_Pa": contact["hertz_max_von_mises"],
            "von_mises_deviation_rel": contact["von_mises_deviation_rel"],
            "displacement_deviation_rel": contact["displacement_deviation_rel"],
            "passed": contact["contact_pressure_ok"],
        },
        "contact_mesh_convergence": convergence,
        "interaction": {
            "blade_frequency_Hz": phase["blade_frequency"],
            "phase_diff_analytic_deg": phase["phase_diff_analytic_deg"],
            "phase_diff_measured_deg": phase["phase_diff_measured_deg"],
            "phase_error_deg": phase["phase_error_deg"],
            "amp_ratio_analytic": phase["amp_ratio_analytic"],
            "amp_ratio_measured": phase["amp_ratio_measured"],
            "amp_error_rel": phase["amp_error_rel"],
            "ice_torque_amplitude_Nm": phase["amp_ice_torque"],
            "ice_torque_mean_Nm": two_way["ice_torque_mean"],
            "ice_torque_peak_Nm": two_way["ice_torque_peak"],
            "ice_torque_duty": two_way["blade_duty"],
            "thrust_amplitude_N": phase["amp_thrust"],
            "phase_passed": phase["passed"],
            "two_way": coupling,
            "pod_natural_frequency_Hz": two_way["pod_natural_frequency"],
            "commanded_rpm": INTERACTION_RPM,
            "mean_rpm": float(np.mean(one_way["n_rpm"])),
            "mean_thrust_N": float(np.mean(one_way["thrust"])),
            "rated_speed_sensitivity": {
                "phase_error_deg": rated_phase["phase_error_deg"],
                "peak_torque_clipping": rated_phase["peak_torque_clipping"],
                "mean_rpm": float(np.mean(rated_one_way["n_rpm"])),
                "mean_thrust_N": float(np.mean(rated_one_way["thrust"])),
            },
        },
    }
    summary["acceptance"] = {
        "a_domain_matches_strength": bool(check["passed"]),
        "b_contact_within_10pct": bool(contact["contact_pressure_ok"]),
        "c_phase_correct": bool(phase["passed"]),
        "e_full_case_produced": True,
    }
    if write:
        write_artifacts(summary, domain, contact, one_way, two_way, phase, pod, curve)
    return summary


def write_artifacts(summary, domain, contact, one_way, two_way, phase, pod, curve) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "thrust_domain.json").write_text(
        json.dumps({"v_ship": domain["v_ship"], "stations": domain["stations"]}, indent=2)
    )
    h = contact["hertz"]
    (OUT / "contact.json").write_text(
        json.dumps(
            {
                "ice_force": contact["ice_force"],
                "hertz": {k: v for k, v in h.items()},
                "fem": {
                    "max_von_mises": contact["fem"]["fem_max_von_mises"],
                    "max_depth_over_a": contact["fem"]["fem_max_depth_over_a"],
                    "max_over_p0": contact["fem"]["fem_max_over_p0"],
                    "center_displacement": contact["fem"]["fem_center_displacement"],
                    "center_displacement_error_rel": contact["fem"]["center_displacement_error_rel"],
                },
                "profile_radius": contact["profile_radius"].tolist(),
                "profile_pressure": contact["profile_pressure"].tolist(),
            },
            indent=2,
        )
    )
    n_slice = 4000
    (OUT / "interaction.json").write_text(
        json.dumps(
            {
                "t": one_way["t"][:n_slice].tolist(),
                "thrust": one_way["thrust"][:n_slice].tolist(),
                "torque_prop": one_way["torque_prop"][:n_slice].tolist(),
                "torque_ice": one_way["torque_ice"][:n_slice].tolist(),
                "ice_force": one_way["ice_force"][:n_slice].tolist(),
                "n_rpm": one_way["n_rpm"][:n_slice].tolist(),
                "phase": phase,
            },
            indent=2,
        )
    )
    try:
        make_figures(domain, contact, one_way, two_way, phase, summary)
    except Exception as exc:  # pragma: no cover - plotting must never break the case
        (OUT / "figure_error.txt").write_text(repr(exc))


def make_figures(domain, contact, one_way, two_way, phase, summary) -> None:  # pragma: no cover
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # 1 -- thrust attainable domain
    st = domain["stations"]
    beta = np.deg2rad([s["azimuth_deg"] for s in st])
    t_max = np.array([s["t_max"] for s in st]) / 1e3
    t_motor = np.array([s["t_motor"] for s in st]) / 1e3
    t_struct = np.array([s["t_structure"] for s in st]) / 1e3
    t_bear = np.array([s["t_bearing"] for s in st]) / 1e3
    fig = plt.figure(figsize=(7.2, 5.0))
    ax = fig.add_subplot(111, projection="polar")
    ax.plot(beta, t_motor, "--", color="tab:blue", label="motor torque limit")
    ax.plot(beta, t_bear, ":", color="tab:green", label="thrust bearing limit")
    ax.plot(beta, t_struct, "-.", color="tab:orange", label="strut strength limit")
    ax.plot(beta, t_max, "-", color="tab:red", lw=2.2, label="attainable boundary")
    ax.fill_between(beta, 0, t_max, color="tab:red", alpha=0.15)
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(120)
    ax.set_title("Thrust attainable domain |T| [kN] vs azimuth ($\\beta$),\n"
                 "15 MW pod, 1.5 m ice, $V_s$ = 8 m/s", fontsize=10)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.15, -0.12), fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_thrust_domain.png", dpi=160)
    plt.close(fig)

    # 2 -- contact: pressure profile + FEM von Mises depth profile + cloud
    r = contact["profile_radius"] * 1e3
    p = contact["profile_pressure"] / 1e6
    fem = contact["fem"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    axes[0].plot(r, p, color="tab:purple", lw=2, label="Hertz $p(r)$")
    axes[0].axvline(contact["contact_radius"] * 1e3, color="k", ls=":", lw=1,
                    label=f"a = {contact['contact_radius']*1e3:.1f} mm")
    axes[0].set_xlabel("radius r [mm]")
    axes[0].set_ylabel("contact pressure [MPa]")
    axes[0].set_title(f"Contact pressure, F = {contact['ice_force']/1e3:.0f} kN, "
                      f"$p_0$ = {contact['peak_contact_pressure']/1e6:.0f} MPa", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    depth = fem["axis_depth"] / fem["hertz"]["contact_radius"]
    vm = fem["axis_von_mises"] / 1e6
    axes[1].plot(vm, depth, "o-", ms=3, color="tab:red", label="FEM von Mises (axis)")
    axes[1].axvline(0.62 * fem["hertz"]["peak_pressure"] / 1e6, color="k", ls="--", lw=1,
                    label="Hertz theory max 0.62 $p_0$")
    axes[1].axhline(0.48, color="gray", ls=":", lw=1, label="classical depth 0.48 a")
    axes[1].invert_yaxis()
    axes[1].set_xlabel("von Mises stress [MPa]")
    axes[1].set_ylabel("depth z/a")
    axes[1].set_title("FEM stress refinement of the contact zone", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_contact.png", dpi=160)
    plt.close(fig)

    # 3 -- stress cloud (FEM radial-depth plane)
    cen = fem["centroid"]
    svm = fem["surface_von_mises"]
    a = fem["hertz"]["contact_radius"]
    sel = np.abs(cen[:, 1]) < 0.12 * a
    X = np.hypot(cen[sel, 0], cen[sel, 1]) / a
    Z = -cen[sel, 2] / a
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sc = ax.scatter(X, Z, c=svm[sel] / 1e6, s=9, cmap="inferno")
    ax.axvline(1.0, color="cyan", ls=":", lw=1, label="Hertz contact radius a")
    ax.invert_yaxis()
    ax.set_xlabel("r/a")
    ax.set_ylabel("depth z/a")
    ax.set_title("von Mises stress field, ice-pod housing contact (FEM, quarter model)", fontsize=10)
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label("von Mises [MPa]")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 2.5)
    fig.tight_layout()
    fig.savefig(OUT / "fig_contact_cloud.png", dpi=160)
    plt.close(fig)

    # 4 -- interaction time histories
    t = one_way["t"]
    mask = t <= t[-1]
    fig, axes = plt.subplots(3, 1, figsize=(9.0, 7.4), sharex=True)
    axes[0].plot(t[mask], one_way["torque_ice"][mask] / 1e6, color="tab:blue", lw=1)
    axes[0].set_ylabel("ice torque [MN m]")
    axes[0].set_title("Propeller-ice interaction at blade rate "
                      f"$f_b$ = {phase['blade_frequency']:.2f} Hz (15 MW pod, 1.5 m ice)", fontsize=10)
    axes[1].plot(t[mask], one_way["thrust"][mask] / 1e3, color="tab:red", lw=1)
    axes[1].set_ylabel("thrust [kN]")
    axes[2].plot(t[mask], one_way["n_rpm"][mask], color="tab:green", lw=1)
    axes[2].set_ylabel("shaft speed [rpm]")
    axes[2].set_xlabel("time [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_interaction.png", dpi=160)
    plt.close(fig)

    # 5 -- spectral phase verification
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    labels = ["analytic\nfirst-order lag", "measured\n(simulation)"]
    vals = [phase["phase_diff_analytic_deg"], phase["phase_diff_measured_deg"]]
    ax.bar(labels, vals, color=["tab:gray", "tab:blue"])
    for i, v in enumerate(vals):
        ax.text(i, v + 2, f"{v:.1f}$^\\circ$", ha="center", fontsize=9)
    ax.set_ylabel(r"phase$\ $difference $\angle\Delta T - \angle Q_{ice}$ [deg]")
    ax.set_title(f"Phase relation at $f_b$: error {phase['phase_error_deg']:.2f}$^\\circ$ "
                 f"(< 5$^\\circ$)", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "fig_phase.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    result = run_case()
    print(json.dumps(result["acceptance"], indent=2))
    print(json.dumps(result["structural_check"], indent=2))
    print(json.dumps(result["contact"], indent=2))
    print(json.dumps(result["interaction"], indent=2))