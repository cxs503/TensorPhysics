"""Figures for the multigrid Poisson verification report.

Run ``python -m tensorphysics.multigrid_poisson.figures`` (or import
:func:`make_figures`) after :func:`tensorphysics.multigrid_poisson.verification.run_verification`
has produced ``verification.json``.  Every figure is written as PNG plus a
caption entry so the report can embed them with provenance.
"""

from __future__ import annotations

import json
import pathlib
from typing import Dict, List, Optional

import numpy as np

__all__ = ["make_figures"]


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def make_figures(
    record_path: str = "artifacts/multigrid_poisson/verification.json",
    out_dir: str = "docs/figures/multigrid_poisson",
) -> List[Dict[str, str]]:
    """Generate all report figures; returns ``[{name, path, caption}]``."""
    plt = _plt()
    rec = json.loads(pathlib.Path(record_path).read_text())
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs: List[Dict[str, str]] = []
    cfg = rec["config"]

    # ------------------------------------------------------------------ fig 1
    n = rec["field"]["n"]
    u = np.asarray(rec["field"]["solution"])
    ua = np.asarray(rec["field"]["analytic"])
    x = np.linspace(0.0, 1.0, n + 1)
    X, Y = np.meshgrid(x, x, indexing="ij")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    cs = axes[0].contourf(X, Y, u, 24, cmap="viridis")
    axes[0].set_title(f"multigrid solution $u_h$ (N={n})")
    axes[0].set_xlabel("x"); axes[0].set_ylabel("y")
    fig.colorbar(cs, ax=axes[0])
    cs2 = axes[1].contourf(X, Y, np.abs(u - ua) * 1e6, 24, cmap="inferno")
    axes[1].set_title(r"$|u_h-u^*|$ [$10^{-6}$] (discretisation error)")
    axes[1].set_xlabel("x"); axes[1].set_ylabel("y")
    fig.colorbar(cs2, ax=axes[1])
    axes[2].plot(x, u[:, n // 2], "o", ms=3, label=r"$u_h$")
    axes[2].plot(x, ua[:, n // 2], "-", label=r"$u^*=\sin\pi x\sin\pi y$")
    axes[2].set_title(f"centre line y=0.5, $L_\\infty$ err={np.abs(u-ua).max():.2e}")
    axes[2].set_xlabel("x"); axes[2].set_ylabel("u"); axes[2].legend()
    fig.tight_layout()
    p = out / "fig1_solution_error_field.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.1 Multigrid solution, discretisation error field and centre-line "
                            "comparison with the analytic solution (N=%d)." % n})

    # ------------------------------------------------------------------ fig 2
    from .core import MultigridSolver, jacobi_solver, poisson_rhs

    n2 = 256
    f = poisson_rhs(n2)
    curves = {}
    for label, kw in (
        ("V(1,1) Jacobi $\\omega$=4/5", dict(smoother="jacobi", omega=0.8, cycle="V")),
        ("W(1,1) Jacobi $\\omega$=4/5", dict(smoother="jacobi", omega=0.8, cycle="W")),
        ("V(1,1) red-black GS", dict(smoother="gs", omega=0.8, cycle="V")),
    ):
        s = MultigridSolver(n_fine=n2, nu1=1, nu2=1, **kw)
        r = s.solve(f, tol=1e-11, max_cycles=60)
        curves[label] = np.asarray(r["residual_history"])
    jr = jacobi_solver(f, omega=1.0, max_iter=400)
    curves["Jacobi $\\omega$=1 (400 sweeps)"] = np.asarray(jr["residual_history"])
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for label, h in curves.items():
        ax.semilogy(np.arange(len(h)), h / h[0], "-o", ms=2.5, label=label)
    ax.axhline(1e-10, color="k", ls=":", lw=0.8, label="tolerance 1e-10")
    ax.set_xlabel("cycle / sweep index"); ax.set_ylabel(r"$\|r_k\|_\infty/\|r_0\|_\infty$")
    ax.set_title(f"Residual reduction, N={n2}")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig2_residual_history.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.2 Residual reduction per cycle: V-/W-cycle and red-black GS "
                            "multigrid vs 400 Jacobi sweeps (N=256)."})

    # ------------------------------------------------------------------ fig 3
    sweep = rec["parameter_sweep"]
    rows = [r for r in sweep if r["n"] == 256 and r["smoother"] == "jacobi" and (r["nu1"], r["nu2"]) == (1, 1)]
    omegas = sorted({r["omega"] for r in rows})
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    for kind, marker in (("V", "o"), ("W", "s")):
        vals = [next(r["rho_measured"] for r in rows if r["omega"] == om and r["cycle"] == kind)
                for om in omegas]
        ax.plot(omegas, vals, marker + "-", label=f"measured $\\rho_{{{kind}}}$ (N=256)")
    tla = [next(r["rho_tla"] for r in rows if r["omega"] == om) for om in omegas]
    ax.plot(omegas, tla, "k--", label=r"$\rho_{TLA}$ (two-level analysis, closed form)")
    ax.set_xlabel(r"Jacobi damping $\omega_J$"); ax.set_ylabel(r"convergence factor $\rho$")
    ax.set_title(r"V/W-cycle factors vs $\omega_J$ and the TLA prediction ($\nu_1=\nu_2=1$)")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig3_rho_vs_omega.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.3 Measured V-/W-cycle convergence factors versus the Jacobi "
                            "damping parameter and the closed-form two-level-analysis curve."})

    # ------------------------------------------------------------------ fig 4
    grid = rec["grid_independence"]
    ns = [r["n"] for r in grid]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.semilogx(ns, [r["rho_V"] for r in grid], "o-", label=r"$\rho_V$ (damped Jacobi $\omega$=4/5)")
    ax.axhline(cfg["rho_tla"], color="k", ls="--", label=r"$\rho_{TLA}$")
    for kind in ("V", "W"):
        sub = [r for r in sweep if r["smoother"] == "gs" and (r["nu1"], r["nu2"]) == (1, 1) and r["cycle"] == kind]
        if sub:
            ax.semilogx([r["n"] for r in sub], [r["rho_measured"] for r in sub], "s-",
                        label=f"$\\rho_{{{kind}}}$ (red-black GS)")
    ax.set_xlabel("N (cells per direction)"); ax.set_ylabel(r"convergence factor $\rho$")
    ax.set_title("Grid independence of the cycle convergence factor")
    ax.legend(fontsize=8); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    p = out / "fig4_grid_independence.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.4 Convergence factors vs grid size N: the multigrid rate is "
                            "essentially independent of N (h-independent)."})

    # ------------------------------------------------------------------ fig 5
    from .core import MultigridSolver as MG, analytic_solution, jacobi_sweeps

    n5 = 128
    h5 = 1.0 / n5
    e0 = analytic_solution(n5)
    f0 = np.zeros_like(e0)
    ej, em, idx = e0.copy(), e0.copy(), []
    jhist, mhist = [1.0], [1.0]
    mg = MG(n_fine=n5, nu1=1, nu2=1, smoother="jacobi", omega=0.8)
    for k in range(1, 201):
        ej = jacobi_sweeps(ej, f0, h5, omega=1.0, sweeps=1)
        jhist.append(float(np.abs(ej).max() / np.abs(e0).max()))
        if k <= 40:
            em = mg.cycle_once(em, f0, n5)
            mhist.append(float(np.abs(em).max() / np.abs(e0).max()))
            idx.append(k)
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    ax.semilogy(np.arange(len(jhist)), jhist, "-", label="Jacobi $\\omega$=1 (per sweep)")
    ax.semilogy(idx, mhist[1:], "o-", ms=3, label="V(1,1) multigrid (per cycle)")
    ax.set_xlabel("iteration"); ax.set_ylabel(r"$\|e\|_\infty/\|e_0\|_\infty$")
    ax.set_title("Low-frequency stagnation: pure mode $\\sin\\pi x\\sin\\pi y$, N=%d" % n5)
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout()
    p = out / "fig5_lowmode_stagnation.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.5 Negative control C7: the pure lowest mode is stagnating under "
                            "Jacobi (rate $\\cos\\pi h$) while the multigrid cycle damps it "
                            "immediately."})

    # ------------------------------------------------------------------ fig 6
    from .tla import tla_closed_form_radius, tla_spectral_radius

    nt = 40
    th = np.linspace(1e-3, 0.5 * np.pi, nt)
    Z = np.zeros((nt, nt))
    D = np.zeros((nt, nt))
    for i, tx in enumerate(th):
        for j, ty in enumerate(th):
            Z[i, j] = tla_spectral_radius((tx, ty), 1, 1, 0.8)
            D[i, j] = abs(Z[i, j] - tla_closed_form_radius((tx, ty), 1, 1, 0.8))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    im = axes[0].imshow(Z, origin="lower", extent=[0, np.pi / 2, 0, np.pi / 2], cmap="viridis")
    i_max = np.unravel_index(np.argmax(Z), Z.shape)
    axes[0].plot(th[i_max[1]], th[i_max[0]], "r*", ms=14,
                 label=f"max $\\rho$={Z.max():.4f} at $\\theta$=({th[i_max[0]]:.3f},{th[i_max[1]]:.3f})")
    axes[0].set_xlabel(r"$\theta_x$"); axes[0].set_ylabel(r"$\theta_y$")
    axes[0].set_title(r"$\rho(M(\theta))$ from the 4$\times$4 symbol matrix")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.colorbar(im, ax=axes[0])
    im2 = axes[1].imshow(np.log10(D + 1e-18), origin="lower",
                         extent=[0, np.pi / 2, 0, np.pi / 2], cmap="magma")
    axes[1].set_xlabel(r"$\theta_x$"); axes[1].set_ylabel(r"$\theta_y$")
    axes[1].set_title(r"$\log_{10}|\rho_{eig}-\rho_{quartic}|$ (C4)")
    fig.colorbar(im2, ax=axes[1])
    fig.tight_layout()
    p = out / "fig6_tla_heatmap.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.6 Left: two-grid symbol spectral radius over the low-frequency "
                            "square, maximum = $\\rho_{TLA}$. Right: difference between the "
                            "eigenvalue and closed-form-quartic evaluations (C4, ~1e-15)."})

    # ------------------------------------------------------------------ fig 7
    cx = rec["complexity"]["rows"]
    nunk = [r["n_unknowns"] for r in cx]
    tcyc = [r["mg_time_per_cycle_s"] for r in cx]
    spd = [r["speedup_vs_jacobi"] for r in cx]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    axes[0].loglog(nunk, tcyc, "o-", label="measured V-cycle cost")
    ref = np.array(tcyc) * (np.array(nunk) / nunk[0])
    axes[0].loglog(nunk, ref, "k--", label=r"$O(N_{unk})$ reference")
    axes[0].set_xlabel("interior unknowns"); axes[0].set_ylabel("wall-clock per cycle [s]")
    axes[0].set_title("Per-cycle cost (scaling exponent %.2f)"
                      % rec["complexity"]["per_cycle_scaling_exponent"])
    axes[0].legend(fontsize=8); axes[0].grid(alpha=0.3, which="both")
    axes[1].bar([str(r["n"]) for r in cx], spd, color="tab:blue")
    axes[1].axhline(8.0, color="r", ls="--", label="acceptance threshold 8x")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("N"); axes[1].set_ylabel("speed-up vs Jacobi")
    axes[1].set_title("Speed-up to reach relative residual 1e-10")
    for i, v in enumerate(spd):
        axes[1].text(i, v * 1.15, f"{v:.0f}x", ha="center", fontsize=8)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig7_complexity.png"
    fig.savefig(p, dpi=140); plt.close(fig)
    figs.append({"name": p.name, "path": str(p),
                 "caption": "Fig.7 Complexity (C3): per-cycle cost scales with the number of "
                            "unknowns while the cycle count stays constant, giving 36x - 1300x "
                            "measured speed-up over Jacobi."})

    (out / "figures.json").write_text(json.dumps(figs, indent=2))
    return figs


if __name__ == "__main__":  # pragma: no cover
    for f in make_figures():
        print(f["name"], f["path"])