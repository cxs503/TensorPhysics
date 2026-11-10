"""Recompute the p270w1-N10 cross-solver conservation audit and its figures.

Usage
-----
``python scripts/recompute_cross_solver_conservation.py``
    run the full audit (FEM -> DEM -> LBM + multirate), write
    ``artifacts/cross_solver_conservation/audit.json`` and the four figures.

``python scripts/recompute_cross_solver_conservation.py --check``
    re-gate the stored audit against the frozen criteria without re-running the
    physics (exit code 0 iff every criterion passes).

``python scripts/recompute_cross_solver_conservation.py --figures-only``
    regenerate the figures from the stored audit (no physics).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from tensorphysics.cross_solver_conservation import audit as audit_mod  # noqa: E402
from tensorphysics.cross_solver_conservation import criteria as crit  # noqa: E402
from tensorphysics.cross_solver_conservation import fem_beam as fb  # noqa: E402

OUT = pathlib.Path("artifacts/cross_solver_conservation")


def _figures(audit: dict, out: pathlib.Path) -> list:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    case = audit["case"]

    # --- figure 1: FEM bending stress field + failure-face traction ----------
    section = fb.BeamSection(E=case["E"], b=case["b"], h=case["h"])
    sol = fb.solve_cantilever(
        section, case["L"], audit["stage_fem"]["P_applied"], n_elem=200
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    xs = np.linspace(0, case["L"], 200)
    ys = np.linspace(-case["h"] / 2, case["h"] / 2, 41)
    XX, YY = np.meshgrid(xs, ys)
    sigma = sol.stress(XX, YY)
    cs = axes[0].contourf(XX * 1e3, YY * 1e3, sigma / 1e6, levels=21, cmap="RdBu_r")
    fig.colorbar(cs, ax=axes[0], label=r"$\sigma_{xx}$ [MPa]")
    axes[0].set_xlabel("x [mm]")
    axes[0].set_ylabel("y [mm]")
    axes[0].set_title("FEM bending stress field (ice plate strip)")

    face = audit["stage_fem"]["failure_face"]
    axes[1].plot(np.array(face["traction"]) / 1e6, np.array(face["y_fibres"]) * 1e3, "o-")
    axes[1].axvline(case["sigma_f"] / 1e6, ls="--", c="k", label=r"$\sigma_f$")
    axes[1].axvline(-case["sigma_f"] / 1e6, ls="--", c="k")
    axes[1].set_xlabel(r"traction $\sigma_{xx}(y)$ [MPa]")
    axes[1].set_ylabel("y [mm]")
    axes[1].set_title("failure-face traction profile (handoff to DEM)")
    axes[1].legend(fontsize=8)

    axes[2].plot(np.array(case["lbm"]["m_ice"]), [0] * len(case["lbm"]["m_ice"]))
    axes[2].remove()
    axes[2] = fig.add_subplot(1, 3, 3)
    g_anal = np.array(
        [
            audit["stage_fem"]["G_lefm"] * (1 - x / case["L"]) ** 2
            for x in np.linspace(0, case["L"], 200)
        ]
    )
    axes[2].plot(np.linspace(0, case["L"], 200) * 1e3, g_anal, label=r"$G(x)$ LEFM")
    axes[2].axhline(case["G_f"], ls="--", c="r", label=r"$G_f$")
    axes[2].set_xlabel("crack position x [mm]")
    axes[2].set_ylabel(r"$G$ [J/m$^2$]")
    axes[2].set_title("energy-release rate vs fracture energy")
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig1_fem_stress_and_failure_face.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # --- figure 2: DEM energy ledger + crack length --------------------------
    runs = audit["stage_dem"]["runs"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ref = max(runs, key=lambda r: r["n_bonds"])
    h = ref["history"]
    t_us = np.asarray(h["t"]) * 1e6
    axes[0].plot(t_us, np.asarray(h["ke"]) + np.asarray(h["u_el"]), label="KE + elastic")
    axes[0].plot(t_us, np.asarray(h["e_diss"]), label="cohesive dissipation")
    axes[0].plot(t_us, np.asarray(h["total"]) - np.asarray(h["e_damp"]) / 2, alpha=0.0)
    axes[0].plot(t_us, np.asarray(h["total"]), "k--", lw=1, label="ledger total")
    axes[0].axhline(ref["U_ref"], c="r", ls=":", label=r"$U_{FEM}$")
    axes[0].set_xlabel("t [us]")
    axes[0].set_ylabel("energy [J]")
    axes[0].set_title(f"DEM energy ledger (n_bonds={ref['n_bonds']})")
    axes[0].legend(fontsize=8)

    nb = [r["n_bonds"] for r in runs]
    crack = [r["crack_length"] * 1e3 for r in runs]
    axes[1].plot(nb, crack, "o-", label="DEM dynamic crack length")
    axes[1].axhline(audit["stage_fem"]["a_arrest_lefm_static"] * 1e3, ls="--", c="r",
                    label="LEFM static arrest")
    axes[1].set_xlabel("DEM bonds")
    axes[1].set_ylabel("crack length [mm]")
    axes[1].set_xscale("log")
    axes[1].set_title("crack length vs DEM resolution")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig2_dem_energy_ledger.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # --- figure 3: LBM profile + momentum closure ---------------------------
    steady = audit["stage_lbm"]["steady"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    y = (np.arange(len(steady["profile"])) + 0.5)
    axes[0].plot(np.array(steady["profile"]), y, "o", ms=3, label="LBM")
    axes[0].plot(np.array(steady["u_analytic"]), y, "-", label="analytic Poiseuille")
    axes[0].set_xlabel("u [lattice]")
    axes[0].set_ylabel("y [lattice]")
    axes[0].set_title(f"channel profile (rel. err {steady['profile_rel_error']:.2e})")
    axes[0].legend(fontsize=8)

    coup = audit["stage_lbm"]["couplings"]
    m = [c["m_ice"] for c in coup]
    axes[1].semilogx(m, [c["closure_ice_vs_ladd"] * 100 for c in coup], "o-",
                     label="ice momentum vs Ladd impulse")
    axes[1].semilogx(m, [c["closure_shear_vs_ladd"] * 100 for c in coup], "s--",
                     label="shear integral vs Ladd")
    axes[1].axhline(2.0, c="r", ls=":", label="2 % gate")
    axes[1].set_xlabel("ice fragment mass [lattice]")
    axes[1].set_ylabel("closure error [%]")
    axes[1].set_title("DEM->LBM momentum closure")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig3_lbm_momentum_closure.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)

    # --- figure 4: multirate error vs exchange rate -------------------------
    sweep = audit["stage_multirate"]["analytic_sweep"]["results"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    for scheme in ("linear", "cubic"):
        pts = sorted(
            [(r["rate_hz"], r["l2_rel"], r["work_rel"]) for r in sweep if r["scheme"] == scheme]
        )
        axes[0].loglog([p[0] for p in pts], [p[1] for p in pts], "o-", label=scheme)
        axes[1].loglog([p[0] for p in pts], [p[2] for p in pts], "o-", label=scheme)
    axes[0].axhline(0.01, c="r", ls=":", label="1 % gate")
    axes[0].set_xlabel("exchange rate [Hz]")
    axes[0].set_ylabel("relative L2 error of v(t)")
    axes[0].set_title("field reconstruction error vs exchange rate")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.01, c="r", ls=":", label="1 % gate")
    axes[1].set_xlabel("exchange rate [Hz]")
    axes[1].set_ylabel(r"relative error of $\int \tau v\,dt$")
    axes[1].set_title("exchanged work error vs exchange rate")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p = out / "fig4_multirate_exchange_error.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    made.append(p)
    return made


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="re-gate the stored audit")
    ap.add_argument("--figures-only", action="store_true")
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args(argv)

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "audit.json"

    if args.check:
        if not path.exists():
            print(f"missing {path}; run without --check first")
            return 2
        stored = json.loads(path.read_text())
        if stored.get("criteria_digest") != crit.CRITERIA_DIGEST:
            print("criteria digest changed: stored audit was produced by other criteria")
            return 2
        outcomes = crit.evaluate(stored["metrics"])
        print(crit.format_table(outcomes))
        ok = all(o["status"] == "PASS" for o in outcomes)
        print("PASS" if ok else "FAIL")
        return 0 if ok else 1

    if args.figures_only:
        if not path.exists():
            print(f"missing {path}; run without --figures-only first")
            return 2
        stored = json.loads(path.read_text())
        for p in _figures(stored, out):
            print(f"wrote {p}")
        return 0

    audit = audit_mod.run_audit(write=True, out_dir=str(out))
    print(crit.format_table(audit["criteria"]))
    print(f"verdict: {audit['verdict']}  ({audit['wall_clock_s']:.1f} s)")
    for p in _figures(audit, out):
        print(f"wrote {p}")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())