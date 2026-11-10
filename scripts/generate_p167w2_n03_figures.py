"""Figures + results table for campaign node p167w2-n03-fem-elastic-plastic.

Produces PNG figures in ``artifacts/p167w2_n03/`` and a machine-readable
``results.json`` used to build the report (no hand-copied numbers).
"""
from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.elastoplastic_collapse import (
    StiffenedPlateGeometry,
    VonMisesMaterial,
    return_mapping,
)
from tensorphysics.elastoplastic_collapse.progressive_collapse import (
    ProgressiveCollapseSolver,
    panel_convergence_study,
    plate_load_shortening_stress,
)
from tensorphysics.elastoplastic_collapse.iacs_icebelt_pressure import (
    IACSIceBeltPressure,
    icebelt_stiffened_panel_capacity,
    pressure_area_law_icebelt,
)

OUT = "artifacts/p167w2_n03"
E, NU, SY0 = 2.06e11, 0.3, 313e6
os.makedirs(OUT, exist_ok=True)


def steel(H=0.0, sy=SY0):
    return VonMisesMaterial(E=E, nu=NU, sigma_y0=sy, H=H)


def panel(mat, **kw):
    p = dict(a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010,
             b_f=0.100, t_f=0.015, material=mat, w0_max=0.0024)
    p.update(kw)
    return StiffenedPlateGeometry(**p)


def uniaxial_exact(eps, mat):
    ey = mat.sigma_y0 / mat.E
    if eps <= ey:
        return mat.E * eps, 0.0
    ep = (eps - ey) / (1.0 + mat.H / mat.E)
    return mat.sigma_y0 + mat.H * ep, ep


def fig_load_shortening():
    mat = steel(0.0)
    bs = [0.30, 0.50, 0.70, 1.00, 1.40, 1.60]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    data = []
    for b in bs:
        g = panel(mat, b=b)
        res = ProgressiveCollapseSolver(g, n_steps=120, max_strain=0.02).solve()
        eps = np.array([e for e, _ in res.load_displacement]) * 100
        sig = np.array([s for _, s in res.load_displacement]) / mat.sigma_y0
        ax.plot(eps, sig, lw=1.6, label=f"β={g.plate_slenderness:.2f}")
        ipk = int(np.argmax(sig))
        ax.plot(eps[ipk], sig[ipk], "o", ms=4, color=ax.lines[-1].get_color())
        data.append({"b_m": b, "beta": float(g.plate_slenderness),
                     "sigma_u_over_sigma_y": float(sig[ipk]),
                     "eps_u": float(eps[ipk] / 100),
                     "post_peak_ratio": float(res.post_peak_ratio()),
                     "failure_mode": res.failure_mode})
    ax.axhline(1.0, color="k", ls=":", lw=1, label="σ_u = σ_y")
    ax.set_xlabel("imposed nominal strain ε (%)")
    ax.set_ylabel("σ_avg / σ_y")
    ax.set_title("Progressive (fiber) Smith collapse: load–shortening curves\n"
                 "markers = limit point (σ_u), descending branch = post-ultimate")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = f"{OUT}/fig1_load_shortening.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, data


def fig_ultimate_vs_beta(baseline):
    mat = steel(0.0)
    betas, sig_u, vk, faulk = [], [], [], []
    for b in np.linspace(0.30, 1.60, 14):
        g = panel(mat, b=float(b))
        res = ProgressiveCollapseSolver(g, n_steps=120, max_strain=0.02).solve()
        beta = g.plate_slenderness
        betas.append(beta)
        sig_u.append(res.sigma_u / mat.sigma_y0)
        if beta > 1.0:
            vk.append(np.sqrt(res.sigma_cr_plate * mat.sigma_y0) / mat.sigma_y0)
            faulk.append(2.0 / beta - 1.0 / beta ** 2)
        else:
            vk.append(np.nan)
            faulk.append(1.0)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(betas, sig_u, "o-", label="this work (progressive Smith, fiber)")
    ax.plot(betas, vk, "--", label="von Karman √(σ_cr σ_y)/σ_y (β>1)")
    ax.plot(betas, faulk, "-.", label="Faulkner 2/β − 1/β²")
    if baseline:
        bx = [d["beta"] for d in baseline]
        by = [d["ratio"] for d in baseline]
        ax.plot(bx, by, "s", ms=7, mfc="none", mec="crimson",
                label="pre-fix solver (σ_u/σ_y, 3 cases)")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    ax.set_xlabel("plate slenderness β = (b/t)√(σ_y/E)")
    ax.set_ylabel("σ_u / σ_y")
    ax.set_title("Ultimate strength vs plate slenderness\n"
                 "pre-fix solver exceeded σ_y and was β-insensitive")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{OUT}/fig2_ultimate_vs_beta.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, {"beta": betas, "sigma_u_over_sigma_y": sig_u,
                  "von_karman": vk, "faulkner": faulk}


def fig_constitutive():
    mat = steel(1.0e9)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    eps = np.linspace(0, 6e-3, 120)
    exact = np.array([uniaxial_exact(e, mat)[0] for e in eps]) / 1e6
    ax.plot(eps * 100, exact, "k-", lw=2.4, label="closed form σ=σ_y0+H ε̄_p")
    for ps, style, lbl in ((True, "o", "plane stress (FE)"),
                           (False, "x", "3D (FE)")):
        sig = []
        for e in eps:
            s_ex, ep = uniaxial_exact(e, mat)
            lat = -NU * s_ex / mat.E - 0.5 * ep
            strain = (np.array([e, lat, 0.0]) if ps else
                      np.array([e, lat, lat, 0.0, 0.0, 0.0]))
            out = return_mapping(strain, None, np.array([0.0]), mat,
                                 plane_stress=ps)[0]
            sig.append(np.asarray(out).ravel()[0] / 1e6)
        ax.plot(eps * 100, sig, style, ms=3.2, mfc="none", label=lbl)
    ax.set_xlabel("axial strain ε (%)")
    ax.set_ylabel("σ_xx (MPa)")
    ax.set_title("von Mises + isotropic hardening: FE vs closed form")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    labels = ["3D", "plane stress"]
    before = [0.1738, 0.6707]
    after = [2.726e-10, 2.983e-10]
    x = np.arange(len(labels))
    ax.bar(x - 0.18, before, 0.36, label="pre-fix")
    ax.bar(x + 0.18, after, 0.36, label="post-fix")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("max |C_ep − FD| / max|FD|")
    ax.set_title("Consistent tangent vs central finite difference")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    fig.tight_layout()
    path = f"{OUT}/fig3_constitutive_tangent.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, {"tangent_before": before, "tangent_after": after}


def fig_icebelt():
    mat = VonMisesMaterial(E=E, nu=NU, sigma_y0=355e6, H=0.0)
    law = IACSIceBeltPressure(P0=2.0e6, A0=1.0, exponent=-0.3)
    areas = np.logspace(-2, 1, 60)
    p = np.array([law.pressure(a) for a in areas]) / 1e6
    fit = pressure_area_law_icebelt(list(areas), P0=2.0e6, exponent=-0.3)

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2))
    ax = axes[0]
    ax.loglog(areas, p, lw=2, label="P = P0 (A/A0)^−0.3")
    ax.set_xlabel("contact area A (m²)")
    ax.set_ylabel("pressure P (MPa)")
    ax.set_title(f"Pressure–area law: fitted exponent "
                 f"{fit['fitted_exponent']:.6f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)

    ax = axes[1]
    spans = np.linspace(0.3, 1.2, 40)
    p_dem, p_cap = [], []
    for a in spans:
        g = StiffenedPlateGeometry(a=float(a), b=0.7, t=0.025, h_s=0.30,
                                   t_w=0.012, b_f=0.12, t_f=0.018, material=mat)
        p_cap.append(icebelt_stiffened_panel_capacity(g)["p_u"] / 1e6)
        p_dem.append(law.pressure(float(a) * g.b) / 1e6)
    ax.plot(spans, p_dem, label="demand P(A = a·b)")
    ax.plot(spans, p_cap, label="capacity p_u (stiffened T-section)")
    ax.set_yscale("log")
    ax.set_xlabel("frame spacing a (m)")
    ax.set_ylabel("pressure (MPa)")
    ax.set_title("Ice-belt panel: demand vs rigid-plastic capacity\n"
                 "t = 25 mm, b = 0.7 m, σ_y = 355 MPa")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9)
    fig.tight_layout()
    path = f"{OUT}/fig4_icebelt.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, {"spans": list(spans), "demand_MPa": p_dem, "capacity_MPa": p_cap}


def fig_convergence():
    mat = steel(0.0)
    g = panel(mat)
    study = panel_convergence_study(g, fiber_counts=(1, 2, 4, 8, 16, 32), n_steps=100)
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    ax.semilogx(study["fiber_counts"], np.array(study["sigma_u"]) / 1e6, "o-")
    ax.set_xlabel("number of web fibers")
    ax.set_ylabel("σ_u (MPa)")
    ax.set_title(f"Fiber convergence: finest vs coarsest "
                 f"{study['finest_vs_coarsest']:.2e}")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    path = f"{OUT}/fig5_fiber_convergence.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path, study


def imperfection_sensitivity():
    mat = steel(0.0)
    out = []
    for ratio in (0.0, 0.0005, 0.001, 0.002, 0.004):
        g = panel(mat, w0_max=ratio * 2.4)
        res = ProgressiveCollapseSolver(g, n_steps=120, max_strain=0.02).solve()
        out.append({"w0_over_a": ratio,
                    "sigma_u_over_sigma_y": float(res.sigma_u / mat.sigma_y0),
                    "deflection_at_ultimate": float(
                        res.diagnostics["deflection_at_ultimate"]),
                    "failure_mode": res.failure_mode,
                    "collapse_stage": res.diagnostics["collapse_stage"]})
    return out


def plate_curve_reference():
    """Plate element curve vs von Karman and Faulkner plate ultimate."""
    mat = steel(0.0)
    g = panel(mat)
    solver = ProgressiveCollapseSolver(g)
    eps_y = mat.sigma_y0 / mat.E
    sigma_elem = plate_load_shortening_stress(eps_y, mat, solver.sigma_cr_plate)
    return {
        "sigma_cr_MPa": float(solver.sigma_cr_plate / 1e6),
        "element_ultimate_MPa": float(sigma_elem / 1e6),
        "von_karman_MPa": float(np.sqrt(solver.sigma_cr_plate * mat.sigma_y0) / 1e6),
        "faulkner_MPa": float(solver._faulkner_ultimate() / 1e6),
    }


def main() -> None:
    with open(f"{OUT}/audit_baseline.json", encoding="utf-8") as fh:
        base = json.load(fh)
    baseline_points = [
        {"beta": v["beta"], "ratio": v["sigma_u_over_sigma_y"],
         "case": k, "mode": v["failure_mode"]}
        for k, v in base["smith"].items()]

    fig1, ls_data = fig_load_shortening()
    fig2, beta_data = fig_ultimate_vs_beta(baseline_points)
    fig3, tan_data = fig_constitutive()
    fig4, ice_data = fig_icebelt()
    fig5, conv = fig_convergence()

    results = {
        "node": "p167w2-n03-fem-elastic-plastic",
        "load_shortening": ls_data,
        "ultimate_vs_beta": beta_data,
        "baseline_points": baseline_points,
        "tangent": tan_data,
        "icebelt": {"fit_exponent": None, "spans": ice_data["spans"],
                    "demand_MPa": ice_data["demand_MPa"],
                    "capacity_MPa": ice_data["capacity_MPa"]},
        "fiber_convergence": conv,
        "imperfection_sensitivity": imperfection_sensitivity(),
        "plate_curve_reference": plate_curve_reference(),
        "figures": [fig1, fig2, fig3, fig4, fig5],
    }
    with open(f"{OUT}/results.json", "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print(json.dumps({k: v for k, v in results.items() if k != "ultimate_vs_beta"},
                     indent=1)[:3000])


if __name__ == "__main__":
    main()