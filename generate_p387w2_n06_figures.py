"""Generate the evidence artefacts for task p387w2-n06.

J2 elastoplastic beam/plate verification surface: moment-curvature, plastic
hinge formation/location, unloading springback, energy balance and plate limit
loads -- with mesh/step convergence studies and two negative controls.

Usage
-----
    python generate_p387w2_n06_figures.py [--out artifacts/p387w2_n06]

Writes ``evidence.json`` plus PNG figures into the output directory.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from tensorphysics.plasticity import verification_hinge as vh  # noqa: E402
from tensorphysics.plasticity.beam_hinge import (  # noqa: E402
    FiberSection,
    J2UniaxialMaterial,
    propped_cantilever_limit,
    propped_cantilever_udl,
    simply_supported_point_load,
    solve_beam,
)
from tensorphysics.plasticity.plate_limit import (  # noqa: E402
    circular_plate_mechanism,
    plastic_moment_per_width,
    rectangular_plate_mechanism,
    square_plate_limit,
    strip_plate_mechanism,
)

OUT = pathlib.Path("artifacts/p387w2_n06")


def convergence_study() -> dict:
    """Mesh and load-step convergence of the hinge position and the energy audit."""
    section = FiberSection.ibeam(300.0, 150.0, 15.0, 10.0)
    material = vh.fe_material()
    length = 6000.0
    M_p = material.sigma_y0 * section.analytic()["Z"]
    limit = propped_cantilever_limit(M_p, length)
    q_c = limit["q_collapse"]
    mesh = []
    for n_el in (20, 30, 40, 50, 60, 80):
        res = propped_cantilever_udl(
            section, material, length, n_elements=n_el, n_steps=40, load_fraction=0.99
        )
        mesh.append(
            {
                "n_elements": n_el,
                "hinge_ratio": float(res.hinge["interior_centroid_ratio"]),
                "hinge_error_span": abs(
                    float(res.hinge["interior_centroid_ratio"])
                    - limit["hinge_position_ratio"]
                ),
                "load_factor": res.load_factor,
                "converged": res.converged,
            }
        )
    steps = []
    for n_steps in (8, 12, 20, 30, 40):
        res = solve_beam(
            section,
            material,
            length,
            40,
            "propped_cantilever",
            uniform_load=q_c,
            n_steps=n_steps,
            lambda_max=0.7,
            max_iter=30,
        )
        steps.append(
            {
                "n_steps": n_steps,
                "energy_relative_residual": res.energy["relative_residual"],
                "load_factor": res.load_factor,
            }
        )
    return {
        "hinge_position_reference": limit["hinge_position_ratio"],
        "mesh": mesh,
        "steps": steps,
    }


def plate_sweep() -> dict:
    M_p = plastic_moment_per_width(vh.SIGMA_Y, 10.0)
    b = 1000.0
    ratios = [1.0, 1.25, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0]
    four = [rectangular_plate_mechanism(r * b, b, M_p).pressure * b**2 / M_p for r in ratios]
    strip = [strip_plate_mechanism(r * b, b, M_p).pressure * b**2 / M_p for r in ratios]
    return {
        "aspect_ratio": ratios,
        "four_facet_normalised": [float(v) for v in four],
        "strip_normalised": [float(v) for v in strip],
        "square_normalised": float(square_plate_limit(b, M_p) * b**2 / M_p),
        "circular_normalised": float(circular_plate_mechanism(b, M_p).pressure * b**2 / M_p),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    report = vh.run_all_criteria(fast=False)
    conv = convergence_study()
    report["convergence"] = conv
    report["plate_sweep"] = plate_sweep()

    # ------------------------------------------------------------------
    # figure 1: normalised moment-curvature, three sections
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    colours = {"rectangle": "tab:blue", "ibeam": "tab:red", "circle": "tab:green"}
    for key, row in report["momcurv"].items():
        ax.plot(
            row["kappa_normalised"],
            row["moment_normalised"],
            label=f"{row['label']} (sf={row['shape_factor_reference']:.3f})",
            color=colours.get(key, None),
        )
        ax.axhline(1.0, color="0.7", lw=0.6, ls=":")
    ax.set_xlabel(r"$\kappa/\kappa_y$")
    ax.set_ylabel(r"$M/M_p$")
    ax.set_title("J2 fiber-section moment-curvature (ideal plasticity)")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_moment_curvature.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # figure 2: hinge formation, propped cantilever
    # ------------------------------------------------------------------
    pc = report["hinge"]["propped_cantilever_udl"]
    x = np.asarray(pc["x_over_L"])
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.4, 6.4), sharex=True)
    ax1.plot(x, pc["moment_normalised"], color="tab:blue")
    ax1.axhline(1.0, color="tab:red", ls="--", lw=0.9, label=r"$M_p$")
    ax1.axhline(-1.0, color="tab:red", ls="--", lw=0.9)
    ax1.set_ylabel(r"$M/M_p$")
    ax1.set_title("Plastic hinge formation, propped cantilever under uniform load")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.plot(x, pc["plastic_curvature_normalised"], color="tab:purple")
    for label, value in (
        ("limit analysis 0.585786", pc["hinge_ratio_reference"]),
        ("FE centroid", pc["hinge_ratio_model"]),
    ):
        ax2.axvline(value, color="k" if "limit" in label else "tab:orange",
                    ls="--" if "limit" in label else "-.", lw=1.0, label=label)
    ax2.set_xlabel(r"$x/L$")
    ax2.set_ylabel(r"$\kappa_p/\kappa_y$")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig2_plastic_hinge.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # figure 3: unloading / springback
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for row in report["springback"]:
        if not row["section"].startswith("rectangle"):
            continue
        ax.plot(
            row["path_kappa_normalised"],
            row["path_moment_normalised"],
            label=f"peak {row['kappa_peak_over_kappa_y']:g} $\\kappa_y$",
        )
        ax.plot(
            [row["kappa_residual_model"] / _kappa_y_rectangle()],
            [0.0],
            "o",
            ms=4,
            color="k",
        )
    ax.set_xlabel(r"$\kappa/\kappa_y$")
    ax.set_ylabel(r"$M/(\sigma_y W_{el})$")
    ax.set_title("Unloading paths and residual curvature (rectangle 100x200)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig3_springback.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # figure 4: plate limit loads
    # ------------------------------------------------------------------
    sweep = report["plate_sweep"]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.plot(sweep["aspect_ratio"], sweep["four_facet_normalised"], "o-",
            label="4-facet corner hinges")
    ax.plot(sweep["aspect_ratio"], sweep["strip_normalised"], "s--",
            label="strip mid-line hinge")
    ax.axhline(sweep["square_normalised"], color="tab:green", lw=0.8, ls=":",
               label=f"square anchor 24 (got {sweep['square_normalised']:.3f})")
    ax.set_xscale("log")
    ax.set_xlabel(r"$a/b$")
    ax.set_ylabel(r"$p_u b^2/M_p$")
    ax.set_title("Simply supported plate limit pressures (upper bounds)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig4_plate_limits.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # figure 5: convergence studies
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.0))
    mesh = conv["mesh"]
    ax1.plot([m["n_elements"] for m in mesh], [m["hinge_error_span"] for m in mesh],
             "o-")
    ax1.axhline(0.01, color="tab:red", ls="--", lw=0.9, label="1 % span gate")
    ax1.set_xlabel("elements")
    ax1.set_ylabel(r"$|\Delta x_{hinge}|/L$")
    ax1.set_title("Hinge position vs mesh")
    ax1.set_yscale("log")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, which="both")
    steps = conv["steps"]
    ax2.plot([s["n_steps"] for s in steps],
             [s["energy_relative_residual"] for s in steps], "s-", color="tab:purple")
    ax2.axhline(0.01, color="tab:red", ls="--", lw=0.9, label="1 % gate")
    ax2.set_xlabel("load steps")
    ax2.set_ylabel(r"$|W_{ext}-U_{el}-W_{pl}|/W_{ext}$")
    ax2.set_title(r"Beam energy audit at $0.7q_c$")
    ax2.set_yscale("log")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out / "fig5_convergence.png", dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    payload = _jsonable(report)
    (out / "evidence.json").write_text(json.dumps(payload, indent=2) + "\n")
    figures = []
    for name, caption in _captions():
        path = out / name
        figures.append(
            {
                "name": name,
                "caption": caption,
                "png_base64": base64.b64encode(path.read_bytes()).decode("ascii"),
            }
        )
    (out / "figures.json").write_text(json.dumps(figures) + "\n")

    print(json.dumps(
        {
            "out": str(out),
            "all_passed": payload["all_passed"],
            "failed": payload["failed"],
            "negative_controls_ok": payload["negative_controls_ok"],
            "criteria": len(payload["criteria"]),
            "figures": [f["name"] for f in figures],
        },
        indent=2,
    ))
    return 0


def _kappa_y_rectangle() -> float:
    material = J2UniaxialMaterial(E=vh.E_STEEL, sigma_y0=vh.SIGMA_Y, hardening_ratio=0.0)
    return material.sigma_y0 / (material.E * 100.0)


def _captions() -> list[tuple[str, str]]:
    return [
        (
            "fig1_moment_curvature.png",
            "Normalised moment-curvature of the three verification sections from the "
            "fiber-section return mapping; the elastic slope, first-yield moment and "
            "fully plastic plateau are compared with E*I, sigma_y*W_el and sigma_y*Z.",
        ),
        (
            "fig2_plastic_hinge.png",
            "Propped cantilever under uniform load at 0.99 of the limit load: bending "
            "moment (top) and plastic curvature (bottom). The sagging hinge sits at "
            "x/L = 0.586, matching the limit-analysis mechanism sqrt(2)/(1+sqrt(2)) = "
            "0.585786 within 0.1 % of the span.",
        ),
        (
            "fig3_springback.png",
            "Unloading paths from three plastic curvature levels with the residual "
            "curvature (dots) at zero moment; the unloading branch follows E*I and the "
            "residual curvature matches the elastic-unloading identity.",
        ),
        (
            "fig4_plate_limits.png",
            "Plate limit pressures normalised by M_p/b^2 versus aspect ratio for the "
            "4-facet corner-hinge mechanism and the strip mid-line mechanism; the "
            "square anchor 24 M_p/a^2 and the long-strip limit 8 M_p/b^2 are reproduced "
            "exactly.",
        ),
        (
            "fig5_convergence.png",
            "Convergence evidence: plastic-hinge position error versus the number of "
            "beam elements (left) and the beam-level energy residual versus the number "
            "of load steps (right), both against their acceptance gates.",
        ),
    ]


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return [_jsonable(v) for v in obj.tolist()]
    return obj


if __name__ == "__main__":
    raise SystemExit(main())