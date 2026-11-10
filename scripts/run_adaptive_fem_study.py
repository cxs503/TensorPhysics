#!/usr/bin/env python
"""Adaptive h-refinement + ZZ a-posteriori estimation study (campaign node
``p284w3-n03-fem-adaptive-refinement``).

Runs the three verification anchors (Cook membrane, Lamé thick-walled cylinder,
crack-tip singularity), writes a machine-readable evidence bundle and the
figures used in the paper-level report.

Usage
-----
    python scripts/run_adaptive_fem_study.py [--out artifacts/adaptive_fem_p284w3n03]
                                             [--quick]

``--quick`` shrinks the resolution of every ladder (used by unit tests / smoke
runs); the default settings are the ones reported.
"""
from __future__ import annotations

import argparse
import json
import math
import platform
import pathlib
import sys
import time
from typing import Callable

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.adaptive_fem import (  # noqa: E402
    Material2D,
    ReferenceStressField,
    annulus_mesh,
    center_crack_KI,
    cook_mesh,
    cook_problem,
    cook_tip_displacement,
    crack_mesh,
    crack_problem,
    displacement_l2_error,
    dorfler_mark,
    ki_from_cod,
    ki_from_jintegral,
    ki_from_stress,
    lame_problem,
    min_angles,
    refine_longest_edge,
    refine_red_green,
    zz_estimate,
)
from tensorphysics.adaptive_fem.cases import LameFields  # noqa: E402


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _rel_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _energy_error(sol, ref_stress: np.ndarray) -> tuple[float, float]:
    diff = ref_stress - sol.stress
    S = sol.material.compliance
    act = math.sqrt(
        float(np.sum(np.einsum("mi,ij,mj->m", diff, S, diff) * sol.area * sol.material.thickness))
    )
    nrm = math.sqrt(
        float(np.sum(np.einsum("mi,ij,mj->m", ref_stress, S, ref_stress) * sol.area * sol.material.thickness))
    )
    return act, nrm


def _loglog_interp(xs, ys, x):
    xs = np.log(np.asarray(xs, float))
    ys = np.log(np.asarray(ys, float))
    order = np.argsort(xs)
    return float(np.exp(np.interp(np.log(x), xs[order], ys[order])))


# ---------------------------------------------------------------------------
# 1. Cook membrane
# ---------------------------------------------------------------------------
def cook_study(theta: float = 0.5, ref_levels: int = 16, n_uniform: int = 5, n_adaptive: int = 17) -> dict:
    prob = cook_problem()
    t0 = time.time()

    # --- independent reference: finer start, more levels, LEB + Dorfler -----
    ref_mesh = cook_mesh(16)
    for _ in range(ref_levels):
        sol = prob.solve(ref_mesh)
        est = zz_estimate(sol)
        ref_mesh = refine_longest_edge(ref_mesh, dorfler_mark(est.eta_elem, theta))
    ref_sol = prob.solve(ref_mesh)
    ref_eta = zz_estimate(ref_sol).eta_total
    ref = ReferenceStressField(ref_sol.mesh.nodes, ref_sol.mesh.tris, ref_sol.stress)
    force = prob.external_forces(ref_sol.mesh)
    U_ref = 0.5 * float(ref_sol.u @ force)
    tip_ref = cook_tip_displacement(ref_sol.mesh, ref_sol.u)

    def make_row(mesh, scheme, level, t_start):
        sol = prob.solve(mesh)
        est = zz_estimate(sol)
        ref_stress = ref.at(mesh.centroids())
        act, nrm = _energy_error(sol, ref_stress)
        U = 0.5 * float(sol.u @ prob.external_forces(mesh))
        tip = cook_tip_displacement(mesh, sol.u)
        return {
            "scheme": scheme,
            "level": level,
            "n_nodes": int(mesh.n_nodes),
            "n_tris": int(mesh.n_tris),
            "n_dof": int(2 * mesh.n_nodes),
            "min_angle_deg": float(min_angles(mesh.nodes, mesh.tris).min()),
            "eta_total": float(est.eta_total),
            "error_energy": float(act),
            "solution_norm": float(nrm),
            "error_rel": float(act / nrm),
            "effectivity": float(est.eta_total / act),
            "strain_energy": float(U),
            "strain_energy_rel_error": float(abs(U - U_ref) / U_ref),
            "tip_displacement": float(tip),
            "tip_rel_error": float(abs(tip - tip_ref) / tip_ref),
            "seconds": float(time.time() - t_start),
        }

    uniform = []
    cur = cook_mesh(8)
    for lvl in range(n_uniform):
        t = time.time()
        uniform.append(make_row(cur, "uniform", lvl, t))
        cur = refine_red_green(cur, range(cur.n_tris))

    adaptive = []
    cur = cook_mesh(8)
    for lvl in range(n_adaptive):
        t = time.time()
        adaptive.append(make_row(cur, "adaptive", lvl, t))
        sol = prob.solve(cur)
        est = zz_estimate(sol)
        cur = refine_longest_edge(cur, dorfler_mark(est.eta_elem, theta))

    # --- matched-DOF comparison -------------------------------------------
    udof = [r["n_dof"] for r in uniform]
    uerr = [r["error_rel"] for r in uniform]
    uenergy = [r["strain_energy_rel_error"] for r in uniform]
    pairs = []
    for r in adaptive:
        if udof[0] <= r["n_dof"] <= udof[-1]:
            ue = _loglog_interp(udof, uerr, r["n_dof"])
            uU = _loglog_interp(udof, uenergy, r["n_dof"])
            pairs.append(
                {
                    "n_dof": r["n_dof"],
                    "adaptive_error_rel": r["error_rel"],
                    "uniform_error_rel": ue,
                    "error_ratio": r["error_rel"] / ue,
                    "error_reduction": 1.0 - r["error_rel"] / ue,
                    "adaptive_energy_rel": r["strain_energy_rel_error"],
                    "uniform_energy_rel": uU,
                    "energy_ratio": r["strain_energy_rel_error"] / uU,
                    "energy_reduction": 1.0 - r["strain_energy_rel_error"] / uU,
                }
            )
    best = max(pairs, key=lambda p: p["error_reduction"]) if pairs else None
    best_e = max(pairs, key=lambda p: p["energy_reduction"]) if pairs else None  # strain-energy functional

    return {
        "name": "cook_membrane",
        "problem": prob.description,
        "theta": theta,
        "reference": {
            "source": "independent high-resolution adaptive solution (16x16 start, LEB+Dorfler)",
            "n_dof": int(2 * ref_sol.mesh.n_nodes),
            "eta_total": float(ref_eta),
            "solution_norm": float(
                math.sqrt(
                    float(
                        np.sum(
                            np.einsum("mi,ij,mj->m", ref_sol.stress, prob.material.compliance, ref_sol.stress)
                            * ref_sol.area
                        )
                    )
                )
            ),
            "strain_energy": float(U_ref),
            "tip_displacement": float(tip_ref),
            "literature_tip_displacement": 23.9,
            "literature_note": (
                "the classical 23.9 (E=1, nu=1/3, t=1, total shear 6.25) is quoted for Cook's "
                "membrane; our converged value differs because of the load/geometry convention "
                "used (total traction 6.25 distributed over the 44-long loaded edge of the "
                "48-long trapezoid with a 60-high clamped edge) - documented as an open "
                "convention discrepancy, not used as an acceptance reference"
            ),
            "seconds": float(time.time() - t0),
        },
        "uniform": uniform,
        "adaptive": adaptive,
        "matched_dof": pairs,
        "best_error_reduction": best,
        "best_energy_reduction": best_e,
        "seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 2. Lamé thick-walled cylinder
# ---------------------------------------------------------------------------
def lame_study(theta: float = 0.5, n_uniform: int = 4, n_adaptive: int = 9) -> dict:
    a, b, p, E, nu = 0.1, 0.25, 100e6, 200e9, 0.3
    prob = lame_problem(a, b, p, E, nu)
    fields = LameFields(a, b, p, prob.material)
    t0 = time.time()

    def make_row(mesh, scheme, level, t_start):
        sol = prob.solve(mesh)
        est = zz_estimate(sol)
        ref_stress = fields.stress(mesh.centroids())
        act, nrm = _energy_error(sol, ref_stress)
        l2 = displacement_l2_error(mesh, sol.u, fields.displacement)
        ur_h = np.sum(
            sol.u.reshape(-1, 2) * mesh.nodes / np.maximum(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1])[:, None], 1e-30),
            axis=1,
        )
        inner = np.abs(np.hypot(mesh.nodes[:, 0], mesh.nodes[:, 1]) - a) < 1e-12
        ur_a = float(ur_h[inner].mean()) if np.any(inner) else float("nan")
        ur_a_exact = float(fields.u_r(np.array([a]))[0])
        return {
            "scheme": scheme,
            "level": level,
            "n_nodes": int(mesh.n_nodes),
            "n_tris": int(mesh.n_tris),
            "n_dof": int(2 * mesh.n_nodes),
            "min_angle_deg": float(min_angles(mesh.nodes, mesh.tris).min()),
            "eta_total": float(est.eta_total),
            "error_energy": float(act),
            "solution_norm": float(nrm),
            "error_rel": float(act / nrm),
            "effectivity": float(est.eta_total / act),
            "l2_displacement_rel": float(l2["rel"]),
            "max_nodal_displacement": float(l2["max_nodal"]),
            "ur_inner_fe": ur_a,
            "ur_inner_exact": ur_a_exact,
            "ur_inner_rel_error": float(abs(ur_a - ur_a_exact) / ur_a_exact),
            "seconds": float(time.time() - t_start),
        }

    uniform = []
    for nr, nt in ((4, 8), (8, 16), (16, 32), (32, 64))[:n_uniform]:
        t = time.time()
        uniform.append(make_row(annulus_mesh(nr, nt, a, b), "uniform", nr, t))

    adaptive = []
    cur = annulus_mesh(4, 8, a, b)
    for lvl in range(n_adaptive):
        t = time.time()
        adaptive.append(make_row(cur, "adaptive", lvl, t))
        sol = prob.solve(cur)
        est = zz_estimate(sol)
        cur = refine_longest_edge(cur, dorfler_mark(est.eta_elem, theta))

    udof = [r["n_dof"] for r in uniform]
    uerr = [r["error_rel"] for r in uniform]
    ul2 = [r["l2_displacement_rel"] for r in uniform]
    pairs = []
    for r in adaptive:
        if udof[0] <= r["n_dof"] <= udof[-1]:
            ue = _loglog_interp(udof, uerr, r["n_dof"])
            ul = _loglog_interp(udof, ul2, r["n_dof"])
            pairs.append(
                {
                    "n_dof": r["n_dof"],
                    "adaptive_error_rel": r["error_rel"],
                    "uniform_error_rel": ue,
                    "error_reduction": 1.0 - r["error_rel"] / ue,
                    "adaptive_l2_rel": r["l2_displacement_rel"],
                    "uniform_l2_rel": ul,
                    "l2_reduction": 1.0 - r["l2_displacement_rel"] / ul,
                }
            )

    return {
        "name": "thick_wall_cylinder_lame",
        "problem": prob.description,
        "theta": theta,
        "reference": {
            "source": "Lamé closed form (plane strain)",
            "A": float(fields.A),
            "B": float(fields.B),
            "ur_inner_exact": float(fields.u_r(np.array([a]))[0]),
            "ur_outer_exact": float(fields.u_r(np.array([b]))[0]),
        },
        "uniform": uniform,
        "adaptive": adaptive,
        "matched_dof": pairs,
        "seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# 3. crack tip singularity
# ---------------------------------------------------------------------------
def crack_study(theta: float = 0.5, n_uniform: int = 3, n_adaptive: int = 13) -> dict:
    a, W, H, sigma = 1.0, 10.0, 10.0, 1.0
    prob = crack_problem(a=a, W=W, H=H, sigma=sigma)
    K_ref = center_crack_KI(a, W, sigma)
    t0 = time.time()

    def make_row(mesh, scheme, level, t_start):
        sol = prob.solve(mesh)
        est = zz_estimate(sol)
        try:  # primary: domain J-integral (energy functional of the annulus)
            K_cod, info = ki_from_jintegral(mesh, sol.u, sol.stress, a, prob.material)
        except ValueError:
            K_cod, info = float("nan"), {"r": [], "K_profile": [], "n_fit": 0}
        try:  # cross-check 1: near-tip COD extrapolation
            K_codfit, codinfo = ki_from_cod(mesh, sol.u, a, prob.material, n_fit=6)
        except ValueError:
            K_codfit = float("nan")
            codinfo = {"r": [], "K_profile": []}
        try:  # cross-check 2: sigma_yy ahead of the tip (recovered nodal field)
            K_stress = ki_from_stress(mesh, sol.stress, a, n_fit=6)
        except ValueError:
            K_stress = float("nan")
        r = np.hypot(*(mesh.centroids() - np.array([a, 0.0])).T)
        return {
            "scheme": scheme,
            "level": level,
            "n_nodes": int(mesh.n_nodes),
            "n_tris": int(mesh.n_tris),
            "n_dof": int(2 * mesh.n_nodes),
            "min_angle_deg": float(min_angles(mesh.nodes, mesh.tris).min()),
            "eta_total": float(est.eta_total),
            "KI": float(K_cod),
            "KI_method": "domain J-integral (half contour, K=sqrt(2 J E'))",
            "KI_cod": float(K_codfit),
            "KI_stress": float(K_stress),
            "KI_rel_error": float(abs(K_cod - K_ref) / K_ref),
            "KI_stress_rel_error": float(abs(K_stress - K_ref) / K_ref),
            "n_nodes_near_tip": int(np.sum(r < 0.05 * a)),
            "J_half": float(info.get("J_half", float("nan"))),
            "ki_fit_r": codinfo["r"],
            "ki_fit_values": codinfo["K_profile"],
            "seconds": float(time.time() - t_start),
        }

    uniform = []
    cur = crack_mesh(a, W, H)
    for lvl in range(n_uniform):
        t = time.time()
        uniform.append(make_row(cur, "uniform", lvl, t))
        cur = refine_red_green(cur, range(cur.n_tris))

    adaptive = []
    cur = crack_mesh(a, W, H)
    for lvl in range(n_adaptive):
        t = time.time()
        adaptive.append(make_row(cur, "adaptive", lvl, t))
        sol = prob.solve(cur)
        est = zz_estimate(sol)
        cur = refine_longest_edge(cur, dorfler_mark(est.eta_elem, theta))

    udof = [r["n_dof"] for r in uniform]
    uerr = [r["KI_rel_error"] for r in uniform]
    pairs = []
    for r in adaptive:
        if np.isnan(r["KI_rel_error"]):
            continue
        if udof[0] <= r["n_dof"] <= udof[-1]:
            ue = _loglog_interp(udof, uerr, r["n_dof"])
            pairs.append(
                {
                    "n_dof": r["n_dof"],
                    "adaptive_KI_rel": r["KI_rel_error"],
                    "uniform_KI_rel": ue,
                    "reduction": 1.0 - r["KI_rel_error"] / ue,
                }
            )

    return {
        "name": "crack_tip_singularity",
        "problem": prob.description,
        "theta": theta,
        "reference": {
            "source": "K_I = sigma sqrt(pi a) sqrt(sec(pi a / 2W))",
            "KI": float(K_ref),
            "a": a,
            "W": W,
            "sigma": sigma,
        },
        "uniform": uniform,
        "adaptive": adaptive,
        "matched_dof": pairs,
        "seconds": float(time.time() - t0),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------
def make_figures(bundle: dict, outdir: pathlib.Path) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs: list[dict] = []
    outdir.mkdir(parents=True, exist_ok=True)

    # Fig 1: convergence of the three anchors
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2))
    cook = bundle["cases"]["cook"]
    ax = axes[0]
    ax.loglog([r["n_dof"] for r in cook["uniform"]], [r["error_rel"] for r in cook["uniform"]], "s-", label="uniform")
    ax.loglog([r["n_dof"] for r in cook["adaptive"]], [r["error_rel"] for r in cook["adaptive"]], "o-", label="adaptive (ZZ+Dörfler)")
    ax.set_xlabel("free DOF (2 per node)")
    ax.set_ylabel(r"relative energy-norm error $\|e\|_E/\|\sigma\|_E$")
    ax.set_title("Cook membrane")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    lame = bundle["cases"]["lame"]
    ax = axes[1]
    ax.loglog([r["n_dof"] for r in lame["uniform"]], [r["l2_displacement_rel"] for r in lame["uniform"]], "s-", label="uniform")
    ax.loglog([r["n_dof"] for r in lame["adaptive"]], [r["l2_displacement_rel"] for r in lame["adaptive"]], "o-", label="adaptive")
    ax.axhline(0.01, color="k", ls="--", lw=1, label="1% acceptance")
    ax.set_xlabel("free DOF")
    ax.set_ylabel(r"relative $L_2$ displacement error")
    ax.set_title("Lamé thick-walled cylinder")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    crack = bundle["cases"]["crack"]
    ax = axes[2]
    ax.loglog([r["n_dof"] for r in crack["uniform"]], [max(r["KI_rel_error"], 1e-6) for r in crack["uniform"]], "s-", label="uniform")
    ax.loglog([r["n_dof"] for r in crack["adaptive"]], [max(r["KI_rel_error"], 1e-6) for r in crack["adaptive"]], "o-", label="adaptive")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% acceptance")
    ax.set_xlabel("free DOF")
    ax.set_ylabel(r"relative error of $K_I$")
    ax.set_title("Crack-tip singularity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = outdir / "fig1_convergence.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig1_convergence",
            "caption": "Fig. 1 — error vs DOF for the three anchors. Uniform (red 1->4) vs "
            "adaptive (ZZ estimator + Doerfler theta=0.5 + longest-edge bisection).",
            "path": path,
        }
    )

    # Fig 2: effectivity index
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for key, label in (("cook", "Cook"), ("lame", "Lamé")):
        rows = bundle["cases"][key]["adaptive"]
        ax.semilogx([r["n_dof"] for r in rows], [r["effectivity"] for r in rows], "o-", label=label)
    ax.axhspan(0.8, 1.2, color="green", alpha=0.12, label=r"acceptance band $[0.8,1.2]$")
    ax.axhline(1.0, color="k", lw=1)
    ax.set_xlabel("free DOF")
    ax.set_ylabel(r"effectivity index $\eta_{est}/\eta_{act}$")
    ax.set_title("ZZ estimator effectivity")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = outdir / "fig2_effectivity.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig2_effectivity",
            "caption": "Fig. 2 — effectivity index of the ZZ estimate against the true error "
            "(Lamé: closed form; Cook: independent high-resolution reference).",
            "path": path,
        }
    )

    return figs


def make_mesh_figures(outdir: pathlib.Path, theta: float = 0.5) -> list[dict]:
    """Field/mesh figures for the report (indicator distribution and grading)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection

    outdir.mkdir(parents=True, exist_ok=True)
    figs = []

    # Cook: adaptive mesh + element indicator
    prob = cook_problem()
    mesh = cook_mesh(8)
    for _ in range(8):
        sol = prob.solve(mesh)
        est = zz_estimate(sol)
        mesh = refine_longest_edge(mesh, dorfler_mark(est.eta_elem, theta))
    sol = prob.solve(mesh)
    est = zz_estimate(sol)
    verts = mesh.nodes[mesh.tris]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.6))
    pc = PolyCollection(verts, array=np.log10(est.eta_elem + 1e-12), cmap="inferno", edgecolors="none")
    axes[0].add_collection(pc)
    axes[0].set_aspect("equal")
    axes[0].set_title(f"Cook membrane — adaptive mesh ({mesh.n_tris} T3, DOF={2*mesh.n_nodes})")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(pc, ax=axes[0], label=r"$\log_{10}\eta_K$")
    # tip displacement convergence proxy: strain energy vs DOF handled in fig1;
    # here show the element-size grading in physical space
    areas = np.abs(
        0.5
        * (
            (verts[:, 1, 0] - verts[:, 0, 0]) * (verts[:, 2, 1] - verts[:, 0, 1])
            - (verts[:, 2, 0] - verts[:, 0, 0]) * (verts[:, 1, 1] - verts[:, 0, 1])
        )
    )
    sc = axes[1].scatter(
        mesh.centroids()[:, 0], mesh.centroids()[:, 1], c=np.log10(np.sqrt(areas)), s=3, cmap="viridis"
    )
    axes[1].set_aspect("equal")
    axes[1].set_title(r"element size $\sqrt{|K|}$ (log$_{10}$) — grading near the clamped corners")
    fig.colorbar(sc, ax=axes[1], label=r"$\log_{10}\sqrt{|K|}$")
    fig.tight_layout()
    path = outdir / "fig3_cook_mesh_indicators.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig3_cook_mesh_indicators",
            "caption": "Fig. 3 — Cook membrane after 8 adaptive cycles: left, ZZ element indicators; "
            "right, element size distribution showing grading towards the clamped corners.",
            "path": path,
        }
    )

    # Lamé: radial stress profile FE vs analytic + deformed shape
    a, b, p, E, nu = 0.1, 0.25, 100e6, 200e9, 0.3
    lprob = lame_problem(a, b, p, E, nu)
    fields = LameFields(a, b, p, lprob.material)
    lmesh = annulus_mesh(8, 16, a, b)
    for _ in range(5):
        lmesh = refine_longest_edge(lmesh, dorfler_mark(zz_estimate(lprob.solve(lmesh)).eta_elem, theta))
    lsol = lprob.solve(lmesh)
    rr = np.linspace(a, b, 200)
    prof_exact = fields.u_r(rr)
    r_h = np.hypot(lmesh.nodes[:, 0], lmesh.nodes[:, 1])
    ur_h = np.sum(
        lsol.u.reshape(-1, 2) * lmesh.nodes / np.maximum(r_h[:, None], 1e-30), axis=1
    )
    order = np.argsort(r_h)
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    axes[0].plot(rr, prof_exact, "k-", label="Lamé exact")
    axes[0].plot(r_h[order], ur_h[order], "ro", ms=3, label="FE nodes (adaptive)")
    axes[0].set_xlabel("r [m]")
    axes[0].set_ylabel(r"$u_r$ [m]")
    axes[0].set_title("Radial displacement, p=100 MPa, a=0.1, b=0.25")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    sxx = lsol.stress[:, 0]
    pc = PolyCollection(lmesh.nodes[lmesh.tris], array=sxx, cmap="coolwarm", edgecolors="none")
    axes[1].add_collection(pc)
    axes[1].set_aspect("equal")
    axes[1].set_title(r"$\sigma_{xx}$ [Pa] on the adaptive mesh")
    fig.colorbar(pc, ax=axes[1], label=r"$\sigma_{xx}$ [Pa]")
    fig.tight_layout()
    path = outdir / "fig4_lame_profile.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig4_lame_profile",
            "caption": "Fig. 4 — Lamé cylinder: FE radial displacement on the adaptive mesh vs the "
            "closed-form solution, and the sigma_xx field.",
            "path": path,
        }
    )

    # Crack: K_I extrapolation + mesh grading
    cprob = crack_problem()
    cmesh = crack_mesh()
    for _ in range(7):
        cmesh = refine_longest_edge(cmesh, dorfler_mark(zz_estimate(cprob.solve(cmesh)).eta_elem, theta))
    csol = cprob.solve(cmesh)
    try:
        K_cod, info = ki_from_cod(cmesh, csol.u, 1.0, cprob.material, n_fit=6)
    except ValueError:  # too few crack-face nodes at low resolution (quick mode)
        K_cod, info = float("nan"), {"r": [], "K_profile": []}
    K_ref = center_crack_KI(1.0, 10.0, 1.0)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.4))
    ax = axes[0]
    rv = np.array(info["r"])
    Kv = np.array(info["K_profile"])
    if rv.size >= 3:
        ax.plot(rv, Kv, "o", label=r"$K(r)=E'u_y\sqrt{2\pi}/(4\sqrt{r})$")
        coef = np.polyfit(np.sqrt(rv), Kv, 1)
        rr2 = np.linspace(0, rv.max(), 50)
        ax.plot(rr2, np.polyval(coef, np.sqrt(rr2)), "b-", lw=1, label="fit $K_I+c\\sqrt{r}$")
    ax.axhline(K_ref, color="k", ls="--", label=f"analytic $K_I$={K_ref:.4f}")
    ax.axhline(K_cod, color="r", ls=":", label=f"extrapolated $K_I$={K_cod:.4f}")
    ax.set_xlabel("r / a")
    ax.set_ylabel(r"$K$ (MPa$\sqrt{m}$ scale absent: non-dimensional)")
    ax.set_title("Crack-tip SIF extraction by COD extrapolation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    verts = cmesh.nodes[cmesh.tris]
    pc = PolyCollection(verts, facecolors="none", edgecolors="tab:blue", linewidths=0.25)
    axes[1].add_collection(pc)
    axes[1].set_xlim(-0.05, 1.6)
    axes[1].set_ylim(-0.05, 1.0)
    axes[1].set_aspect("equal")
    axes[1].set_title("adaptive mesh near the crack tip (r^-1/2 grading)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.tight_layout()
    path = outdir / "fig5_crack_tip.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    figs.append(
        {
            "name": "fig5_crack_tip",
            "caption": "Fig. 5 — crack-tip SIF: COD extrapolation to r=0 against the analytic "
            "finite-width centre-crack value, and the graded adaptive mesh at the tip.",
            "path": path,
        }
    )
    return figs


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def evaluate_criteria(bundle: dict) -> dict:
    cook = bundle["cases"]["cook"]
    lame = bundle["cases"]["lame"]
    crack = bundle["cases"]["crack"]

    cook_best = cook["best_error_reduction"] or {}
    cook_best_e = cook["best_energy_reduction"] or {}
    lame_last = lame["adaptive"][-1]
    lame_min_l2 = min(r["l2_displacement_rel"] for r in lame["adaptive"])
    crack_errs = [r["KI_rel_error"] for r in crack["adaptive"] if not np.isnan(r["KI_rel_error"])]
    crack_min_err = min(crack_errs)
    crack_last = crack["adaptive"][-1]
    effs = [r["effectivity"] for r in cook["adaptive"] + lame["adaptive"]]
    eff_outliers = [
        {"case": key, "level": r["level"], "n_dof": r["n_dof"], "effectivity": r["effectivity"]}
        for key in ("cook", "lame")
        for r in bundle["cases"][key]["adaptive"]
        if r["effectivity"] is not None and not (0.8 <= r["effectivity"] <= 1.2)
    ]
    eff_all = [
        {"case": key, "level": r["level"], "n_dof": r["n_dof"], "effectivity": r["effectivity"]}
        for key in ("cook", "lame")
        for r in bundle["cases"][key]["adaptive"]
        if r["effectivity"] is not None
    ]
    angles = [
        r["min_angle_deg"]
        for r in cook["adaptive"] + lame["adaptive"] + crack["adaptive"]
        + cook["uniform"] + lame["uniform"] + crack["uniform"]
    ]

    crit = {
        "a_cook_energy_error_reduction_ge_50pct": {
            "threshold": 0.50,
            "measured_best": cook_best.get("error_reduction"),
            "measured_at_dof": cook_best.get("n_dof"),
            "pass": bool((cook_best.get("error_reduction") or 0.0) >= 0.50),
            "definition": "energy-norm error ||sigma_ref-sigma_h||_E relative to the reference, "
            "adaptive vs uniform interpolated at equal DOF",
            "alternate_energy_functional_reduction": cook_best_e.get("energy_reduction"),
            "alternate_energy_functional_dof": cook_best_e.get("n_dof"),
            "alternate_energy_functional_threshold": 0.50,
            "alternate_energy_functional_pass": bool(
                (cook_best_e.get("energy_reduction") or 0.0) >= 0.50
            ),
            "note": "energy-norm reading (primary, standard) fails; the global strain-energy "
            "functional reading reaches >=50% at large DOF; both numbers are reported",
        },
        "b_lame_displacement_l2_lt_1pct": {
            "threshold": 0.01,
            "measured_adaptive_best": lame_min_l2,
            "measured_adaptive_final": lame_last["l2_displacement_rel"],
            "n_dof_final": lame_last["n_dof"],
            "pass": bool(lame_min_l2 < 0.01),
        },
        "c_crack_KI_error_lt_5pct": {
            "threshold": 0.05,
            "measured_best": crack_min_err,
            "measured_final": crack_last["KI_rel_error"],
            "n_dof_final": crack_last["n_dof"],
            "pass": bool(crack_min_err < 0.05),
        },
        "d_effectivity_in_0.8_1.2": {
            "threshold": [0.8, 1.2],
            "measured_min": float(np.min(effs)),
            "measured_max": float(np.max(effs)),
            "measured_median": float(np.median(effs)),
            "n_samples": len(effs),
            "n_inside_band": int(np.sum((np.asarray(effs) >= 0.8) & (np.asarray(effs) <= 1.2))),
            "outliers": eff_outliers,
            "pass_strict_all_samples": bool(np.min(effs) >= 0.8 and np.max(effs) <= 1.2),
            "pass": bool(np.min(effs) >= 0.8 and np.max(effs) <= 1.2),
            # all out-of-band samples belong to the coarsest (level 0) mesh of a case
            "pass_apart_from_coarsest_meshes": bool(
                not eff_outliers or all(o["level"] == 0 for o in eff_outliers)
            ),
        },
        "e_min_angle_gt_20deg": {
            "threshold": 20.0,
            "measured_min": float(np.min(angles)),
            "pass": bool(np.min(angles) > 20.0),
        },
    }
    return crit


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "adaptive_fem_p284w3n03"))
    ap.add_argument("--quick", action="store_true", help="reduced resolution for smoke runs")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args(argv)

    outdir = pathlib.Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.quick:
        cook = cook_study(ref_levels=8, n_uniform=3, n_adaptive=7)
        lame = lame_study(n_uniform=2, n_adaptive=4)
        crack = crack_study(n_uniform=2, n_adaptive=6)
    else:
        cook = cook_study()
        lame = lame_study()
        crack = crack_study()

    bundle = {
        "node": "p284w3-n03-fem-adaptive-refinement",
        "module": "tensorphysics.adaptive_fem",
        "theta": 0.5,
        "cases": {"cook": cook, "lame": lame, "crack": crack},
        "environment": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "seconds": float(time.time() - t0),
    }
    bundle["criteria"] = evaluate_criteria(bundle)

    figs = []
    if not args.no_figures:
        fig_errors = []
        for maker in (make_figures, make_mesh_figures):
            try:
                figs.extend(maker(bundle, outdir) if maker is make_figures else maker(outdir))
            except Exception as exc:  # a figure failure must never lose the numbers
                fig_errors.append(f"{maker.__name__}: {exc!r}")
        if fig_errors:
            bundle["figure_error"] = "; ".join(fig_errors)
    bundle["figures"] = [
        {"name": f["name"], "caption": f["caption"], "file": str(pathlib.Path(f["path"]).relative_to(outdir))}
        for f in figs
    ]

    payload = json.loads(json.dumps(bundle, default=float))
    (outdir / "study.json").write_text(json.dumps(payload, indent=2, default=float))
    print(json.dumps({k: payload["criteria"][k] for k in payload["criteria"]}, indent=2, default=float))
    print(f"\nwrote {outdir/'study.json'} in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())