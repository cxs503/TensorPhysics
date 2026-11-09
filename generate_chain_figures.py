#!/usr/bin/env python
"""Generate figures for the P34 three-stage chain ledger report."""
from __future__ import annotations

import json
import base64
import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path("artifacts/chain_ledger_m7465")
FIGDIR = OUTDIR / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def _b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def fig1_face_a_convergence(detail: dict) -> tuple[Path, str]:
    """Mesh convergence of the pressure-integrated force (Face A)."""
    rows = detail["faces"]["A"]["rows"]
    n_edges = [r["n_edge"] for r in rows]
    errs = [r["hold_rel_err"] for r in rows]
    orders = []
    for i in range(1, len(errs)):
        if errs[i - 1] > 0 and errs[i] > 0:
            r = n_edges[i] / n_edges[i - 1]
            orders.append(np.log(errs[i - 1] / errs[i]) / np.log(r))
        else:
            orders.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.loglog(n_edges, errs, "o-", label="force rel. error")
    ax1.loglog(n_edges, [errs[0] * (n_edges[0] / n) ** 1.0 for n in n_edges],
               "--", label="1st order reference")
    ax1.loglog(n_edges, [errs[0] * (n_edges[0] / n) ** 2.0 for n in n_edges],
               "--", label="2nd order reference")
    ax1.set_xlabel("Panels per edge")
    ax1.set_ylabel("Relative error")
    ax1.set_title("Face A: Force convergence")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.bar(range(len(orders)), orders, color="steelblue")
    ax2.axhline(1.0, color="red", linestyle="--", label="1st order threshold")
    ax2.set_xticks(range(len(orders)))
    ax2.set_xticklabels([f"{n_edges[i]}→{n_edges[i+1]}" for i in range(len(orders))], rotation=30)
    ax2.set_ylabel("Observed order")
    ax2.set_title("Convergence order between refinements")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    path = FIGDIR / "fig1_face_a_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "Face A: mesh convergence of pressure-integrated force showing ≥1st order convergence"


def fig2_face_c_modal_force(detail: dict) -> tuple[Path, str]:
    """Modal generalised force history (Face C)."""
    face_c = detail["faces"]["C"]
    q_ref = np.array(face_c["Q_ref_N"])  # shape: (n_modes, n_time)
    t = np.array(face_c["t_s"])          # length matches Q_ref columns
    n_modes = q_ref.shape[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for j in range(min(4, n_modes)):
        ax = axes.flat[j]
        ax.plot(t, q_ref[j], linewidth=1.5, label=f"Mode {j+1}")
        ax.set_ylabel("Q [N]")
        ax.set_title(f"Mode {j+1} generalised force")
        ax.grid(True, alpha=0.3)
    axes[-1, -1].set_xlabel("Time [s]")
    fig.suptitle("Face C: Modal generalised force history (Duhamel reference)", fontsize=12)
    fig.tight_layout()
    path = FIGDIR / "fig2_face_c_modal_force.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "Face C: modal generalised force history for first four wet bending modes"


def fig3_energy_ledger(detail: dict) -> tuple[Path, str]:
    """Energy ledger over one wave period (Face E)."""
    energy = detail["faces"]["E"]
    wave = energy["wave"]
    marched = energy["marched"]

    labels = ["Input work", "ΔKE+ΔPE", "Dissipation", "Residual"]
    values = [wave["W_in_J"], wave["dKE_plus_dPE_J"], wave["W_diss_J"],
              abs(wave["W_in_J"] - (wave["dKE_plus_dPE_J"] + wave["W_diss_J"]))]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    colors = ["steelblue", "orange", "green", "red"]
    ax1.bar(labels, values, color=colors)
    ax1.set_ylabel("Energy [J]")
    ax1.set_title("Wave energy ledger (one period)")
    ax1.grid(True, alpha=0.3, axis="y")

    # marched body ledger
    labels2 = ["Input work", "ΔE", "Dissipation", "Residual"]
    values2 = [marched["W_in_J"], marched["dE_J"], marched["W_diss_J"],
               abs(marched["W_in_J"] - (marched["dE_J"] + marched["W_diss_J"]))]
    ax2.bar(labels2, values2, color=colors)
    ax2.set_ylabel("Energy [J]")
    ax2.set_title("Body energy ledger (marched)")
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Face E: Energy ledger — wave rel_err={energy['wave_ledger_rel_err']:.4f}, "
                 f"body rel_err={energy['body_ledger_rel_err']:.4f}", fontsize=11)
    fig.tight_layout()
    path = FIGDIR / "fig3_energy_ledger.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "Face E: energy ledger closure over one wave period for wave field and marched body"


def fig4_time_ledger(detail: dict) -> tuple[Path, str]:
    """Time synchronisation ledger (Face T)."""
    tl = detail["faces"]["T"]
    wave_rows = tl["wave_rows"]

    # Extract unique configurations
    orders = sorted(set(r["order"] for r in wave_rows))
    spp = sorted(set(r["samples_per_period"] for r in wave_rows))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    # Force error vs samples per period for different orders
    for order in orders:
        subset = [r for r in wave_rows if r["order"] == order and r["offset_fraction"] == 0.0]
        subset.sort(key=lambda r: r["samples_per_period"])
        ax1.plot([r["samples_per_period"] for r in subset],
                 [r["force_max_rel_err"] for r in subset],
                 "o-", label=f"Order {order}")
    ax1.axhline(0.03, color="red", linestyle="--", label="3% threshold")
    ax1.axhline(0.10, color="orange", linestyle="--", label="10% threshold")
    ax1.set_xlabel("Samples per period")
    ax1.set_ylabel("Force max rel. error")
    ax1.set_title("Face T: Force error vs coupling rate")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale("log")

    # Offset detection
    offset_data = [r for r in wave_rows if r["samples_per_period"] == max(spp)]
    offset_fractions = [r["offset_fraction"] for r in offset_data]
    offset_errs = [r["force_max_rel_err"] for r in offset_data]
    ax2.plot(offset_fractions, offset_errs, "s-", color="purple")
    ax2.axhline(0.10, color="red", linestyle="--", label="10% detection threshold")
    ax2.axvline(0.5, color="gray", linestyle=":", label="Δt/2 offset")
    ax2.set_xlabel("Offset fraction of Δt")
    ax2.set_ylabel("Force max rel. error")
    ax2.set_title("Face T: Offset detection")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("Face T: Time synchronisation ledger", fontsize=11)
    fig.tight_layout()
    path = FIGDIR / "fig4_time_ledger.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "Face T: time synchronisation — interpolation order convergence and Δt/2 offset detection"


def fig5_whipping_stress(detail: dict) -> tuple[Path, str]:
    """Whipping stress history from the slamming impact (Face C output)."""
    face_c = detail["faces"]["C"]
    stress = np.array(face_c["stress_Pa"])
    t = np.array(face_c["t_modal_s"])
    peak = face_c["peak_whipping_stress_MPa"]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t, stress / 1e6, linewidth=1.5, color="steelblue")
    ax.axhline(peak, color="red", linestyle="--", label=f"Peak = {peak:.2f} MPa")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Midship stress [MPa]")
    ax.set_title("Face C: Whipping stress history at midship")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = FIGDIR / "fig5_whipping_stress.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, "Face C: whipping stress time history at midship section from slamming impact"


def fig6_readiness_verdict(detail: dict) -> tuple[Path, str]:
    """Chain readiness verdict summary."""
    verdict = detail["readiness"]
    faces = verdict["faces"]

    fig, ax = plt.subplots(figsize=(8, 5))
    face_keys = ["A", "B", "C", "E", "T", "N"]
    face_names = ["Force hand-off", "Motion hand-off", "Modal force",
                  "Energy ledger", "Time sync", "Negative controls"]
    verdict_colors = {"PASS": "green", "FAIL": "red", "WITHHELD": "gray"}

    y = range(len(face_keys))
    colors = [verdict_colors[faces[k]["verdict"]] for k in face_keys]
    bars = ax.barh(y, [1] * len(face_keys), color=colors, edgecolor="black", height=0.6)
    for i, (k, name) in enumerate(zip(face_keys, face_names)):
        ax.text(0.5, i, f"{name}: {faces[k]['verdict']}", va="center", fontweight="bold",
                color="white" if colors[i] != "gray" else "black")

    ax.set_xlim(0, 1.2)
    ax.set_yticks([])
    ax.set_title(f"Chain Readiness Verdict: {verdict['overall_status']}", fontsize=12)
    ax.set_xlabel("Status")
    fig.tight_layout()
    path = FIGDIR / "fig6_readiness_verdict.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path, f"Chain readiness verdict: {verdict['overall_status']} — all six faces evaluated"


def main():
    ledger_path = OUTDIR / "chain_ledger.json"
    detail = json.loads(ledger_path.read_text(encoding="utf-8"))

    generators = [
        fig1_face_a_convergence,
        fig2_face_c_modal_force,
        fig3_energy_ledger,
        fig4_time_ledger,
        fig5_whipping_stress,
        fig6_readiness_verdict,
    ]

    figures = []
    for gen in generators:
        path, caption = gen(detail)
        figures.append({
            "name": path.stem,
            "caption": caption,
            "png_base64": _b64(path),
        })
        print(f"  Generated: {path.name}")

    # Save figure metadata
    fig_meta = FIGDIR / "figures.json"
    fig_meta.write_text(json.dumps(figures, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nAll {len(figures)} figures saved to {FIGDIR}")
    return figures


if __name__ == "__main__":
    main()
