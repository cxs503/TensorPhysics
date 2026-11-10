#!/usr/bin/env python3
"""Generate figures for P519-N02 cross-implementation attribution report.

Produces:
* fig1_component_breakdown.png – five-component breakdown for all four impls
* fig2_divergence_attribution.png – component-wise divergence (IA vs TP)
* fig3_calibration_residual.png – calibration effect
* fig4_sensitivity_heatmap.png – sensitivity matrix heatmap
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = HERE.parents[1]
for _p in (str(REPO / "src"),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.ice.cross_impl_attribution import (  # noqa: E402
    COMPONENT_NAMES,
    RefShip,
    attribute_divergence,
    calibrate_crushing_coefficient,
    decompose_five,
    impl_tp_level,
    run_full_attribution,
    sensitivity_matrix,
)

SHIP = RefShip()
H_REF = 1.0
V_REF = 3.0
OUTDIR = REPO / "artifacts" / "ice_cross_impl"
OUTDIR.mkdir(parents=True, exist_ok=True)

COMPONENT_LABELS = ["Crushing", "Bending", "Friction", "Momentum", "Buoyancy"]
COMPONENT_KEYS = ["R_crush_N", "R_bend_N", "R_friction_N", "R_momentum_N", "R_buoyancy_N"]
IMPL_NAMES = ["TP_level", "IA_lindqvist", "Lit_original", "Lit_corrected"]
IMPL_LABELS = ["TP (corrected)", "IA (legacy)", "Lit (original)", "Lit (corrected)"]
IMPL_COLORS = ["#2b7bba", "#c0392b", "#e67e22", "#27ae60"]


def _shrink(path: Path, colors: int = 96) -> Path:
    try:
        from PIL import Image
    except Exception:
        return path
    img = Image.open(path).convert("RGB")
    img = img.convert("P", palette=Image.ADAPTIVE, colors=colors)
    img.save(path, optimize=True)
    return path


def fig1_component_breakdown() -> Path:
    """Five-component breakdown for all four implementations."""
    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(COMPONENT_LABELS))
    width = 0.2

    for i, impl in enumerate(IMPL_NAMES):
        d = decompose_five(impl, H_REF, V_REF, SHIP)
        vals = [d[c] / 1e3 for c in COMPONENT_KEYS]  # kN
        bars = ax.bar(x + i * width, vals, width, label=IMPL_LABELS[i],
                      color=IMPL_COLORS[i], alpha=0.85)
        for bar, v in zip(bars, vals):
            if abs(v) > 10:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(COMPONENT_LABELS, fontsize=9)
    ax.set_ylabel("Component resistance [kN]")
    ax.set_title(f"Five-component breakdown: h={H_REF} m, V={V_REF} m/s, PC6 ship")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)

    path = OUTDIR / "fig1_component_breakdown.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _shrink(path)


def fig2_divergence_attribution() -> Path:
    """Component-wise divergence: IA_lindqvist vs TP_level."""
    attr = attribute_divergence("TP_level", "IA_lindqvist", H_REF, V_REF, SHIP)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))

    # Left: component differences
    diffs = [attr["component_diffs_N"][c] / 1e3 for c in COMPONENT_KEYS]
    colors = ["#c0392b" if d > 0 else "#2b7bba" for d in diffs]
    bars = ax1.bar(COMPONENT_LABELS, diffs, color=colors, alpha=0.85)
    for bar, d in zip(bars, diffs):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{d:.0f}", ha="center", va="bottom" if d > 0 else "top", fontsize=8)
    ax1.axhline(0, color="k", lw=0.5)
    ax1.set_ylabel("Component difference [kN] (IA − TP)")
    ax1.set_title("Component-wise divergence")
    ax1.tick_params(axis="x", rotation=15)
    ax1.grid(axis="y", alpha=0.3)

    # Right: fraction of total divergence
    fracs = [attr["component_fractions"][c] * 100 for c in COMPONENT_KEYS]
    colors2 = ["#c0392b" if f > 0 else "#2b7bba" for f in fracs]
    bars2 = ax2.bar(COMPONENT_LABELS, fracs, color=colors2, alpha=0.85)
    for bar, f in zip(bars2, fracs):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{f:.1f}%", ha="center", va="bottom" if f > 0 else "top", fontsize=8)
    ax2.axhline(0, color="k", lw=0.5)
    ax2.set_ylabel("Fraction of total divergence [%]")
    ax2.set_title("Attribution fractions")
    ax2.tick_params(axis="x", rotation=15)
    ax2.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Divergence attribution: IA_lindqvist vs TP_level "
                 f"(ratio={attr['ratio']:.2f}x)", fontsize=11, y=1.02)
    path = OUTDIR / "fig2_divergence_attribution.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _shrink(path)


def fig3_calibration_residual() -> Path:
    """Calibration effect: before and after crushing coefficient calibration."""
    cal = calibrate_crushing_coefficient("IA_lindqvist", H_REF, V_REF, SHIP)
    base = decompose_five("TP_level", H_REF, V_REF, SHIP)
    cand = decompose_five("IA_lindqvist", H_REF, V_REF, SHIP)

    fig, ax = plt.subplots(figsize=(7, 4))

    totals = [
        base["R_total_N"] / 1e3,
        cand["R_total_N"] / 1e3,
        cal["calibrated_total_N"] / 1e3,
    ]
    labels = ["TP_level\n(corrected)", "IA_lindqvist\n(before cal)", "IA_lindqvist\n(after cal)"]
    colors = ["#2b7bba", "#c0392b", "#27ae60"]

    bars = ax.bar(labels, totals, color=colors, alpha=0.85)
    for bar, t in zip(bars, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{t:.1f}", ha="center", va="bottom", fontsize=10)

    ax.axhline(base["R_total_N"] / 1e3, color="#2b7bba", ls="--", lw=1, alpha=0.5)
    ax.set_ylabel("Total resistance [kN]")
    ax.set_title(f"Crushing coefficient calibration (k={cal['k_crush_calibrated']:.4f})")
    ax.grid(axis="y", alpha=0.3)

    # Add annotation
    ax.text(0.95, 0.95, f"Remaining: {cal['remaining_pct']:.4f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))

    path = OUTDIR / "fig3_calibration_residual.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _shrink(path)


def fig4_sensitivity_heatmap() -> Path:
    """Sensitivity matrix heatmap."""
    sens = sensitivity_matrix(H_REF, V_REF, SHIP)
    params = ["sigma_c", "sigma_f", "mu", "flare_deg", "k_bend"]

    # Build matrix: rows = components, cols = parameters
    matrix = np.zeros((len(COMPONENT_KEYS), len(params)))
    for j, p in enumerate(params):
        for i, c in enumerate(COMPONENT_KEYS):
            matrix[i, j] = sens["sensitivity_matrix"][p][c]

    # Normalize each column for visualization
    col_max = np.max(np.abs(matrix), axis=0)
    col_max[col_max == 0] = 1
    matrix_norm = matrix / col_max

    fig, ax = plt.subplots(figsize=(7, 4))
    im = ax.imshow(matrix_norm, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xticks(range(len(params)))
    ax.set_xticklabels(params, rotation=20, ha="right", fontsize=9)
    ax.set_yticks(range(len(COMPONENT_KEYS)))
    ax.set_yticklabels(COMPONENT_LABELS, fontsize=9)
    ax.set_title("Sensitivity matrix (normalized): ∂R/∂p")
    fig.colorbar(im, ax=ax, label="Normalized sensitivity")

    # Add rank annotation
    rank = sens["identifiability"]["numerical_rank"]
    n = sens["identifiability"]["n_params"]
    ax.text(0.02, 0.98, f"Rank: {rank}/{n} (deficient)",
            transform=ax.transAxes, ha="left", va="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.7))

    path = OUTDIR / "fig4_sensitivity_heatmap.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return _shrink(path)


def encode_png(path: Path) -> str:
    """Encode PNG as base64 for report embedding."""
    return base64.b64encode(path.read_bytes()).decode("ascii")


def main():
    print("Generating figures...")
    paths = {
        "fig1": fig1_component_breakdown(),
        "fig2": fig2_divergence_attribution(),
        "fig3": fig3_calibration_residual(),
        "fig4": fig4_sensitivity_heatmap(),
    }
    for name, path in paths.items():
        print(f"  {name}: {path} ({path.stat().st_size} bytes)")

    # Also generate the full report
    report = run_full_attribution(H_REF, V_REF, SHIP)
    report_path = OUTDIR / "attribution_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nReport: {report_path}")

    # Build figures list for report API
    figures = []
    captions = {
        "fig1": "Five-component breakdown of level-ice resistance for four implementations "
                "(h=1 m, V=3 m/s, PC6 reference ship). TP_level and Lit_corrected are identical; "
                "IA_lindqvist shows inflated crushing and misplaced buoyancy term.",
        "fig2": "Component-wise divergence attribution: IA_lindqvist vs TP_level. "
                "Crushing accounts for 114.5% of total divergence (bending/buoyancy term "
                "goes in opposite direction, partially offsetting).",
        "fig3": "Crushing coefficient calibration effect. After calibrating k_crush, "
                "remaining unexplained difference drops to <0.001%.",
        "fig4": "Sensitivity matrix heatmap. Rank deficiency (4/5) confirms that σ_c and "
                "flare_deg are not independently identifiable from R_total alone "
                "(multiplicative coupling in R_crush).",
    }
    for name, path in paths.items():
        figures.append({
            "name": name,
            "caption": captions[name],
            "png_base64": encode_png(path),
        })

    # Write figures manifest
    manifest = {
        "figures": [{"name": f["name"], "caption": f["caption"], "path": str(paths[f["name"]])}
                    for f in figures],
    }
    manifest_path = OUTDIR / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Manifest: {manifest_path}")

    return figures


if __name__ == "__main__":
    main()
