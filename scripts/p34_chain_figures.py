"""Generate publication-quality figures for the P34 chain-ledger report.

Produces:
  fig1_criteria_radar.png   - radar chart of all 6 face verdicts
  fig2_face_a_convergence.png - mesh convergence of face A
  fig3_energy_ledger.png    - energy balance bar chart
  fig4_time_ledger.png      - interpolation order comparison
  fig5_whipping_stress.png  - midship whipping stress time history
  fig6_negative_controls.png - stabilisation divergence detection
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


def load_detail(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def fig1_criteria_radar(detail: dict, out: Path):
    """Radar chart showing the 6 face verdicts and key metrics."""
    faces = detail["readiness"]["faces"]
    labels = list(faces.keys())
    titles = {
        "A": "Face A\nPressure→Force",
        "B": "Face B\nMotion→Boundary",
        "C": "Face C\nSlam→Modal",
        "E": "Face E\nEnergy Ledger",
        "T": "Face T\nTime Sync",
        "N": "Face N\nNeg. Controls",
    }

    # Normalised score: 1 = pass, 0 = fail, 0.5 = withheld
    scores = []
    for f in labels:
        v = faces[f]["verdict"]
        if v == "PASS":
            scores.append(1.0)
        elif v == "FAIL":
            scores.append(0.0)
        else:
            scores.append(0.5)

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    scores_closed = scores + [scores[0]]
    angles_closed = angles + [angles[0]]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles_closed, scores_closed, "o-", color="#2563eb", linewidth=2, markersize=8)
    ax.fill(angles_closed, scores_closed, alpha=0.15, color="#2563eb")
    ax.set_xticks(angles)
    ax.set_xticklabels([titles[l] for l in labels], fontsize=9)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "PASS"], fontsize=8)
    ax.set_title("P34 Chain Readiness — Six-Face Verdict", fontsize=13, fontweight="bold", pad=20)
    ax.axhline(1.0, color="green", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axhline(0.5, color="orange", linewidth=0.5, linestyle="--", alpha=0.5)

    # Annotate each vertex with the key metric
    products = detail["products"]
    annotations = {
        "A": f"err={products['A']['force_rel_err']:.1e}",
        "B": f"flux={products['B']['closed_flux_rel']:.1e}",
        "C": f"Q_err={products['C']['modal_force_rel_err']:.1e}",
        "E": f"ΔW={products['E']['wave_ledger_rel_err']:.1e}",
        "T": f"Δt/2={products['T']['offset_residual']:.2f}",
        "N": f"det={products['N']['divergence_detection_rate']:.0%}",
    }
    for angle, label, score in zip(angles, labels, scores):
        ax.annotate(annotations[label], xy=(angle, score),
                    xytext=(angle, score + 0.12), fontsize=7, ha="center",
                    color="#1e40af")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig2_face_a_convergence(detail: dict, out: Path):
    """Mesh convergence of face A: hold and Gauss quadrature errors."""
    rows = detail["faces"]["A"]["rows"]
    n_edges = [r["n_edge"] for r in rows]
    h = [1.0 / ne for ne in n_edges]
    err_hold = [r["hold_rel_err"] for r in rows]
    err_gauss = [r["quadrature_rel_err"] for r in rows]
    order = detail["faces"]["A"]["observed_order"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.loglog(h, err_hold, "o-", label="Boundary-node hold (LBM-like)", color="#dc2626", linewidth=2)
    ax.loglog(h, err_gauss, "s-", label="Gauss quadrature (order 2)", color="#2563eb", linewidth=2)

    # Reference slope
    h_ref = np.array([h[0], h[-1]])
    ax.loglog(h_ref, err_hold[0] * (h_ref / h_ref[0]) ** 2, "--", color="#dc2626", alpha=0.4,
              label=f"O(h²) ref (observed order = {order:.2f})")

    ax.set_xlabel("Panel size h ~ 1/n_edge", fontsize=11)
    ax.set_ylabel("Relative force error |F_h - F_ref| / |F_ref|", fontsize=11)
    ax.set_title("Face A: Pressure → Force Mesh Convergence", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, which="both", alpha=0.3)
    ax.axhline(0.02, color="green", linestyle=":", alpha=0.6, label="2% threshold")

    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig3_energy_ledger(detail: dict, out: Path):
    """Energy balance bar chart for one wave period."""
    energy = detail["faces"]["E"]
    wave = energy["wave"]
    marched = energy["marched"]

    labels = ["W_in\n(input work)", "W_diss\n(dissipation)", "dE\n(ΔKE+ΔPE)", "Residual"]
    values = [wave["W_in_J"], wave["W_diss_J"], wave["dKE_plus_dPE_J"],
              wave["W_in_J"] - wave["W_diss_J"] - wave["dKE_plus_dPE_J"]]
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Wave-band energy ledger
    ax = axes[0]
    bars = ax.bar(labels, [abs(v) for v in values], color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("|Energy| [J]", fontsize=11)
    ax.set_title("Wave-Band Energy Ledger (one period)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.2f} J", ha="center", va="bottom", fontsize=8)
    resid_pct = energy["wave_ledger_rel_err"] * 100
    ax.text(0.5, 0.95, f"Residual / W_in = {resid_pct:.3f}%  (threshold: 3%)",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="green" if resid_pct < 3 else "red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Marched body ledger
    ax = axes[1]
    m_labels = ["W_in,march", "W_diss,march", "dE_march"]
    m_values = [marched["W_in_J"], marched["W_diss_J"], marched["dE_J"]]
    m_colors = ["#2563eb", "#dc2626", "#16a34a"]
    bars = ax.bar(m_labels, [abs(v) for v in m_values], color=m_colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("|Energy| [J]", fontsize=11)
    ax.set_title("Marched Body Energy Ledger (5 periods)", fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    for bar, val in zip(bars, m_values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.1,
                f"{val:.2f} J", ha="center", va="bottom", fontsize=8)
    body_pct = energy["body_ledger_rel_err"] * 100
    ax.text(0.5, 0.95, f"Residual / W_in = {body_pct:.4f}%  (threshold: 3%)",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="green" if body_pct < 3 else "red",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Energy Conservation Across the Three-Stage Chain", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig4_time_ledger(detail: dict, out: Path):
    """Time synchronisation: interpolation order comparison."""
    time_led = detail["faces"]["T"]
    wave_rows = time_led["wave_rows"]

    # Group by samples_per_period
    densities = sorted(set(r["samples_per_period"] for r in wave_rows))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: force error vs samples per period
    ax = axes[0]
    for order, marker, color in [(0, "o", "#dc2626"), (1, "s", "#f59e0b"), (2, "^", "#2563eb")]:
        errs = []
        ns = []
        for r in wave_rows:
            if r["order"] == order and r["offset_fraction"] == 0.0:
                ns.append(r["samples_per_period"])
                errs.append(r["force_max_rel_err"])
        if ns:
            ax.semilogy(ns, errs, f"{marker}-", label=f"Order {order}", color=color, linewidth=2, markersize=7)

    ax.axhline(0.10, color="red", linestyle=":", alpha=0.6, label="10% detection threshold")
    ax.axhline(0.02, color="green", linestyle=":", alpha=0.6, label="2% acceptance threshold")
    ax.set_xlabel("Samples per period", fontsize=11)
    ax.set_ylabel("Max force relative error", fontsize=11)
    ax.set_title("Force Error vs Temporal Resolution", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: energy ledger error with/without offset
    ax = axes[1]
    for offset, ls in [(0.0, "-"), (0.5, "--")]:
        errs = []
        ns = []
        for r in wave_rows:
            if r["order"] == 2 and r["offset_fraction"] == offset:
                ns.append(r["samples_per_period"])
                errs.append(r["wave_ledger_rel_err"])
        if ns:
            label = "Zero offset (aligned)" if offset == 0 else "Δt/2 offset (misaligned)"
            ax.semilogy(ns, errs, f"{ls}o", label=label, linewidth=2, markersize=7)

    ax.axhline(0.03, color="green", linestyle=":", alpha=0.6, label="3% threshold")
    ax.axhline(0.10, color="red", linestyle=":", alpha=0.6, label="10% detection threshold")
    ax.set_xlabel("Samples per period", fontsize=11)
    ax.set_ylabel("Energy ledger relative error", fontsize=11)
    ax.set_title("Energy Ledger Sensitivity to Time Offset", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle("Time Synchronisation Ledger (Face T)", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig5_whipping_stress(detail: dict, out: Path):
    """Midship whipping stress time history from the slamming event."""
    face_c = detail["faces"]["C"]
    peak = face_c["peak_whipping_stress_MPa"]
    t_imp = face_c["impact_duration_s"]

    # Reconstruct the stress history from the modal data
    # Use the peak stress by mode count to show convergence
    mode_counts = sorted(face_c["peak_stress_by_mode_count_Pa"].keys(), key=int)
    peaks = [face_c["peak_stress_by_mode_count_Pa"][k] for k in mode_counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: peak stress vs number of modes (truncation audit)
    ax = axes[0]
    ax.plot([int(k) for k in mode_counts], [p / 1e6 for p in peaks], "o-", color="#2563eb",
            linewidth=2, markersize=8)
    ax.axhline(peak, color="red", linestyle="--", alpha=0.6, label=f"Final peak = {peak:.2f} MPa")
    ax.set_xlabel("Number of retained modes", fontsize=11)
    ax.set_ylabel("Peak midship whipping stress [MPa]", fontsize=11)
    ax.set_title("Modal Truncation Audit", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks([int(k) for k in mode_counts])

    # Right: impact parameters
    ax = axes[1]
    params = {
        "Impact velocity\n[m/s]": face_c["impact_velocity_max_m_s"],
        "Impact duration\n[s]": t_imp,
        "Peak stress\n[MPa]": peak,
        "Impulse fraction\n[-]": face_c["impulse_fraction_of_hull_momentum"],
    }
    bars = ax.bar(params.keys(), params.values(), color=["#2563eb", "#dc2626", "#16a34a", "#9333ea"],
                  edgecolor="black", linewidth=0.5)
    ax.set_title("Slamming Event Parameters", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, params.values()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.05,
                f"{val:.4f}", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Face C: Slamming → Beam Whipping Response", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def fig6_negative_controls(detail: dict, out: Path):
    """Negative controls: spectral radius and divergence detection."""
    neg = detail["faces"]["N"]
    problems_scales = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0]
    spectral = neg["unstable_spectrum"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: spectral radius vs mass ratio scale
    ax = axes[0]
    ax.semilogx(problems_scales, spectral, "o-", color="#dc2626", linewidth=2, markersize=8)
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="stability boundary ρ=1")
    ax.set_xlabel("Force scale factor", fontsize=11)
    ax.set_ylabel("Spectral radius ρ(G)", fontsize=11)
    ax.set_title("Partitioned Coupling Stability", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.text(0.05, 1.5, "Light body regime\n(ρ > 1 → unstable\nwithout stabilisation)",
            fontsize=9, color="#dc2626",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))

    # Right: detection rates
    ax = axes[1]
    categories = ["Divergence\ndetected\n(no stabiliser)", "Converged\n(Aitken/IQN-ILS)",
                  "Δt/2 offset\ndetected", "0th-order\nerror detected"]
    rates = [
        neg.get("divergence_detection_rate", detail["products"]["N"]["divergence_detection_rate"]),
        neg.get("stabilised_converged_rate", detail["products"]["N"]["stabilised_converged_rate"]),
        detail["products"]["T"]["offset_detected"],
        detail["products"]["T"]["zero_order_detected"],
    ]
    colors = ["#dc2626", "#16a34a", "#f59e0b", "#9333ea"]
    bars = ax.bar(categories, rates, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Detection rate", fontsize=11)
    ax.set_title("Negative Control Detection Rates", fontsize=12, fontweight="bold")
    ax.axhline(1.0, color="green", linestyle="--", alpha=0.6, label="100% required")
    for bar, val in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.0%}", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.legend(fontsize=9)

    fig.suptitle("Negative Controls (Face N): Falsifiability Verification", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {out}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--detail", default="artifacts/chain_ledger_m7087/chain_ledger.json")
    parser.add_argument("--outdir", default="artifacts/chain_ledger_m7087/figures")
    args = parser.parse_args()

    detail = load_detail(args.detail)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Generating P34 chain-ledger figures...")
    fig1_criteria_radar(detail, outdir / "fig1_criteria_radar.png")
    fig2_face_a_convergence(detail, outdir / "fig2_face_a_convergence.png")
    fig3_energy_ledger(detail, outdir / "fig3_energy_ledger.png")
    fig4_time_ledger(detail, outdir / "fig4_time_ledger.png")
    fig5_whipping_stress(detail, outdir / "fig5_whipping_stress.png")
    fig6_negative_controls(detail, outdir / "fig6_negative_controls.png")
    print("Done.")


if __name__ == "__main__":
    main()
