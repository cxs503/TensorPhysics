"""P33-W2-N06: 3-D isoparametric solid element kernel -- verification driver.

Runs the six acceptance benchmarks of the战役 node and writes

* ``artifacts/p33_w2_n06/results.json`` -- every number cited by the report,
* ``artifacts/p33_w2_n06/fig_*.png`` -- the report figures.

Usage
-----
    PYTHONPATH=src python scripts/p33_w2_n06_solid_fem.py [--out DIR] [--quick]
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from tensorphysics.solid_fem import (  # noqa: E402
    J2Material,
    PlasticState,
    gauss_stresses_plastic,
    hourglass_energy_ratio,
    hourglass_rank,
    isotropic_elasticity,
    lame_limit_pressure,
    lame_radial_displacement,
    lame_stresses,
    numerical_tangent,
    patch_test,
    thick_cylinder,
    uniaxial_plasticity,
)
from tensorphysics.solid_fem.verification import (  # noqa: E402
    cantilever_tip_deflection,
    cylinder_limit_pressure,
    timoshenko_tip_deflection,
)

E_STEEL = 210.0e9
NU_STEEL = 0.3
SIGMA_Y = 250.0e6
A_CYL, B_CYL, P_CYL = 1.0, 2.0, 50.0e6


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    return obj


# ---------------------------------------------------------------------------
# 1. Patch tests
# ---------------------------------------------------------------------------


def run_patch_tests() -> dict:
    out = []
    for element, dims in (("hex8", (2, 2, 2)), ("hex20", (3, 3, 3))):
        for distort in (0.0, 0.25):
            res = patch_test(element=element, dims=dims, distort=distort, seed=11)
            out.append(
                {
                    "element": element,
                    "dims": list(dims),
                    "distort": distort,
                    "n_nodes": res["n_nodes"],
                    "n_interior_nodes": res["n_interior_nodes"],
                    "displacement_error": res["displacement_error"],
                    "stress_error": res["stress_error"],
                }
            )
    worst_u = max(r["displacement_error"] for r in out)
    worst_s = max(r["stress_error"] for r in out)
    return {
        "cases": out,
        "worst_displacement_error": worst_u,
        "worst_stress_error": worst_s,
        "criterion": "< 1e-10",
        "pass": bool(worst_u < 1e-10 and worst_s < 1e-10),
    }


# ---------------------------------------------------------------------------
# 2. Volumetric locking
# ---------------------------------------------------------------------------


def run_locking() -> dict:
    cases = []
    for nu in (0.3, 0.4999):
        for formulation in ("full", "bbar", "sri"):
            for element in ("hex8", "hex20"):
                form = formulation
                if element == "hex20" and form == "bbar":
                    # mean dilatation is unstable for the 20-node brick
                    form = "bbar_stab"
                setup = thick_cylinder(
                    a=A_CYL, b=B_CYL, p=P_CYL, E=E_STEEL, nu=nu, n_radial=4, n_circ=8,
                    element=element, formulation=form,
                )
                fe = setup.gauss_field_errors()
                ur = setup.inner_displacement_error()
                cases.append(
                    {
                        "element": element,
                        "nu": nu,
                        "formulation": form,
                        "eps_v_rel_error": fe["eps_v_rel_error"],
                        "eps_v_pointwise_max_rel_error": fe["eps_v_pointwise_max_rel_error"],
                        "eps_v_fe_mean": fe["eps_v_fe_mean"],
                        "eps_v_exact": fe["eps_v_exact"],
                        "sigma_r_max_rel_error": fe["sigma_r_max_rel_error"],
                        "sigma_theta_max_rel_error": fe["sigma_theta_max_rel_error"],
                        "inner_displacement_rel_error": ur["rel_error"],
                    }
                )
    lock = [c for c in cases
            if c["nu"] == 0.4999 and c["formulation"] in ("bbar", "bbar_stab")]
    worst = max(c["eps_v_rel_error"] for c in lock)
    full = {(c["element"]): c for c in cases if c["nu"] == 0.4999 and c["formulation"] == "full"}
    return {
        "cases": cases,
        "nearly_incompressible_bbar_worst_eps_v_rel_error": worst,
        "nearly_incompressible_full_eps_v_rel_error": {
            k: v["eps_v_rel_error"] for k, v in full.items()
        },
        "criterion": "eps_v relative error < 1% at nu = 0.4999",
        "pass": bool(worst < 0.01),
    }


# ---------------------------------------------------------------------------
# 3. Hourglass control
# ---------------------------------------------------------------------------


def run_hourglass() -> dict:
    ranks = {str(a): hourglass_rank(alpha_hg=a) for a in (0.0, 0.1, 1.0)}
    C = isotropic_elasticity(E_STEEL, NU_STEEL)
    G = E_STEEL / (2.0 * (1.0 + NU_STEEL))

    sweep = []
    for alpha in (0.0, 0.01, 0.03, 0.1, 0.3, 1.0):
        setup = thick_cylinder(
            a=A_CYL, b=B_CYL, p=P_CYL, E=E_STEEL, nu=NU_STEEL, n_radial=4, n_circ=8,
            element="hex8", formulation="full", rule="1pt", hourglass=alpha,
        )
        e_hg, e_tot = hourglass_energy_ratio(
            setup.mesh, setup.u, C, hourglass=alpha, shear_modulus=G, rule="1pt"
        )
        fe = setup.gauss_field_errors()
        ur = setup.inner_displacement_error()
        sweep.append(
            {
                "alpha_hg": alpha,
                "hourglass_energy": float(e_hg),
                "total_energy": float(e_tot),
                "hourglass_fraction": float(e_hg / e_tot),
                "inner_displacement_rel_error": ur["rel_error"],
                "sigma_theta_max_rel_error": fe["sigma_theta_max_rel_error"],
            }
        )

    # distort the 1-point mesh solution with a bending-dominated problem to show
    # what the spurious modes do when they are actually excited
    bending = []
    for kwargs in (
        {"element": "hex20", "formulation": "bbar_stab"},
        {"element": "hex20", "formulation": "bbar_stab", "n_thick": 2},
        {"element": "hex8", "formulation": "full", "rule": "1pt", "hourglass": 0.0},
        {"element": "hex8", "formulation": "full", "rule": "1pt", "hourglass": 0.1},
        {"element": "hex8", "formulation": "full", "rule": "1pt", "hourglass": 1.0},
        {"element": "hex8", "formulation": "full", "rule": "2x2x2"},
    ):
        res = cantilever_tip_deflection(
            n_length=8, n_thick=kwargs.pop("n_thick", 1), n_width=1, **kwargs
        )
        res.pop("u", None)
        res.pop("mesh", None)
        bending.append(res)

    smooth = [s for s in sweep if s["alpha_hg"] > 0.0]
    best = min(smooth, key=lambda s: s["hourglass_fraction"])
    return {
        "rank_test": ranks,
        "sweep_smooth_cylinder": sweep,
        "bending_cases": bending,
        "selected_alpha_hg": best["alpha_hg"],
        "selected_hourglass_fraction": best["hourglass_fraction"],
        "criterion": "hourglass energy / total strain energy < 5% (well-posed smooth field)",
        "pass": bool(best["hourglass_fraction"] < 0.05),
    }


# ---------------------------------------------------------------------------
# 4. J2 plasticity
# ---------------------------------------------------------------------------


def run_plasticity(quick: bool = False) -> dict:
    tangent_checks = []
    mat = J2Material(E=E_STEEL, nu=NU_STEEL, sigma_y0=SIGMA_Y, hardening=20e9)
    for name, eps in (
        ("multiaxial", np.array([1.5e-3, -4e-4, -6e-4, 3e-4, -2e-4, 1e-4])),
        ("uniaxial", np.array([2.0e-3, -7e-4, -7e-4, 0.0, 0.0, 0.0])),
        ("pure_shear", np.array([0.0, 0.0, 0.0, 4.0e-3, 0.0, 0.0])),
        ("elastic", np.array([1.0e-6, 0.0, 0.0, 0.0, 0.0, 0.0])),
    ):
        state = PlasticState()
        _sig, _st, C_ep, plastic = mat.return_map(eps, state)
        C_num = numerical_tangent(mat, eps, state)
        tangent_checks.append(
            {
                "case": name,
                "plastic": bool(plastic),
                "max_rel_error": float(
                    np.max(np.abs(C_ep - C_num)) / np.max(np.abs(C_num))
                ),
                "symmetry_error": float(np.max(np.abs(C_ep - C_ep.T))
                                        / np.max(np.abs(C_ep))),
            }
        )

    uni = uniaxial_plasticity(n_steps=40, max_strain=0.003, dims=(2, 2, 2))
    uni.pop("iterations", None)
    analytic_curve = []
    from tensorphysics.solid_fem import uniaxial_tension_analytic

    eps_grid = np.linspace(0.0, 0.003, 121)
    for e in eps_grid:
        analytic_curve.append(
            {"eps_xx": float(e),
             "sigma_xx": float(uniaxial_tension_analytic(np.array([e]), E_STEEL, SIGMA_Y, 20e9)[0])}
        )

    limit = cylinder_limit_pressure(
        n_radial=4, n_circ=8, n_steps=8 if quick else 30,
        sigma_y0=SIGMA_Y, hardening=0.0, formulation="bbar",
    )
    return {
        "consistent_tangent": tangent_checks,
        "uniaxial": uni,
        "uniaxial_analytic_curve": analytic_curve,
        "limit_pressure": limit,
        "criterion": "uniaxial stress error < 1% ; limit pressure within 5%",
        "pass": bool(uni["rel_error"] < 0.01 and (limit["rel_error"] or 1.0) < 0.05),
    }


# ---------------------------------------------------------------------------
# 5. Lamé thick cylinder (elastic anchor) + mesh convergence
# ---------------------------------------------------------------------------


def run_lame() -> dict:
    series = []
    for n_radial, n_circ in ((2, 8), (4, 8), (6, 12), (8, 16)):
        setup = thick_cylinder(
            a=A_CYL, b=B_CYL, p=P_CYL, E=E_STEEL, nu=NU_STEEL,
            n_radial=n_radial, n_circ=n_circ, element="hex8", formulation="bbar",
        )
        fe = setup.gauss_field_errors()
        ur = setup.inner_displacement_error()
        series.append(
            {
                "element": "hex8",
                "n_radial": n_radial,
                "n_circ": n_circ,
                "sigma_r_max_rel_error": fe["sigma_r_max_rel_error"],
                "sigma_theta_max_rel_error": fe["sigma_theta_max_rel_error"],
                "inner_displacement_rel_error": ur["rel_error"],
                "eps_v_rel_error": fe["eps_v_rel_error"],
            }
        )
    hex20 = thick_cylinder(
        a=A_CYL, b=B_CYL, p=P_CYL, E=E_STEEL, nu=NU_STEEL,
        n_radial=4, n_circ=8, element="hex20", formulation="bbar_stab",
    )
    fe20 = hex20.gauss_field_errors()
    ur20 = hex20.inner_displacement_error()

    # per-Gauss-point field for the figure
    C = isotropic_elasticity(E_STEEL, NU_STEEL)
    from tensorphysics.solid_fem.element import element_kinematics
    from tensorphysics.solid_fem.shape import shape as _shape

    gp = []
    for e in range(hex20.mesh.n_elem):
        coords = hex20.mesh.element_coords(e)
        kin = element_kinematics("hex20", coords, None, "bbar_stab")
        u_e = hex20.u[hex20.mesh.element_dofs(e)]
        for g in range(kin.integration_points):
            N, _ = _shape("hex20", *kin.points[g])
            x = N @ coords
            r = float(np.hypot(x[0], x[1]))
            er = np.array([x[0] / r, x[1] / r])
            sig = C @ (kin.B[g] @ u_e)
            sr = er @ np.array([[sig[0], sig[3]], [sig[3], sig[1]]]) @ er
            et = np.array([-er[1], er[0]])
            st = et @ np.array([[sig[0], sig[3]], [sig[3], sig[1]]]) @ et
            gp.append({"r_over_a": r / A_CYL, "sigma_r_over_p": sr / P_CYL,
                       "sigma_theta_over_p": st / P_CYL})
    gp.sort(key=lambda d: d["r_over_a"])

    r_line = np.linspace(A_CYL, B_CYL, 200)
    sr_line, st_line = lame_stresses(r_line, A_CYL, B_CYL, P_CYL)

    return {
        "hex8_refinement": series,
        "hex20_4x8": {
            "sigma_r_max_rel_error": fe20["sigma_r_max_rel_error"],
            "sigma_theta_max_rel_error": fe20["sigma_theta_max_rel_error"],
            "inner_displacement_rel_error": ur20["rel_error"],
            "eps_v_rel_error": fe20["eps_v_rel_error"],
        },
        "gauss_points": gp,
        "analytic_curve": {
            "r_over_a": (r_line / A_CYL).tolist(),
            "sigma_r_over_p": (sr_line / P_CYL).tolist(),
            "sigma_theta_over_p": (st_line / P_CYL).tolist(),
        },
        "stage_a": {
            "ur_inner_fe": ur20["ur_fe_mean"],
            "ur_inner_exact": lame_radial_displacement(A_CYL, A_CYL, B_CYL, P_CYL, E_STEEL, NU_STEEL),
        },
        "criterion": "Lame radial / hoop stress within 2% (Hex20, 4x8 quarter annulus)",
        "pass": bool(
            fe20["sigma_r_max_rel_error"] < 0.02 and fe20["sigma_theta_max_rel_error"] < 0.02
        ),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(res: dict, out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    files = []
    plt.rcParams.update({"font.size": 9, "figure.dpi": 130})

    # --- fig 1: Lame stress distribution -------------------------------
    lame = res["lame"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    r = np.array(lame["analytic_curve"]["r_over_a"])
    ax[0].plot(r, lame["analytic_curve"]["sigma_r_over_p"], "k-", label="Lame analytic $\\sigma_r/p$")
    ax[0].plot(r, lame["analytic_curve"]["sigma_theta_over_p"], "k--", label="Lame analytic $\\sigma_\\theta/p$")
    gp = lame["gauss_points"]
    ax[0].plot([g["r_over_a"] for g in gp], [g["sigma_r_over_p"] for g in gp], "o", ms=3, mfc="none",
               label="Hex20 FE $\\sigma_r$")
    ax[0].plot([g["r_over_a"] for g in gp], [g["sigma_theta_over_p"] for g in gp], "s", ms=3, mfc="none",
               label="Hex20 FE $\\sigma_\\theta$")
    ax[0].set_xlabel("$r/a$")
    ax[0].set_ylabel("stress / p")
    ax[0].set_title("Lame cylinder, p = 50 MPa, plane strain")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    ser = lame["hex8_refinement"]
    h = [1.0 / s["n_radial"] for s in ser]
    ax[1].loglog(h, [s["sigma_theta_max_rel_error"] for s in ser], "o-", label="Hex8 $\\sigma_\\theta$")
    ax[1].loglog(h, [s["sigma_r_max_rel_error"] for s in ser], "s-", label="Hex8 $\\sigma_r$")
    ax[1].axhline(0.02, color="r", ls=":", label="2 % gate")
    ax[1].set_xlabel("element size through the wall, $h/(b-a)$")
    ax[1].set_ylabel("max relative error")
    ax[1].set_title("Hex8 h-convergence (B-bar)")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fn = out / "fig1_lame_stress.png"
    fig.savefig(fn)
    plt.close(fig)
    files.append(str(fn))

    # --- fig 2: locking -------------------------------------------------
    lock = res["locking"]["cases"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    labels, vals_full, vals_bbar = [], [], []
    for element in ("hex8", "hex20"):
        for nu in (0.3, 0.4999):
            labels.append(f"{element}\n$\\nu$={nu}")
            f = next(c for c in lock if c["element"] == element and c["nu"] == nu
                     and c["formulation"] == "full")
            bform = "bbar" if element == "hex8" else "bbar_stab"
            b = next(c for c in lock if c["element"] == element and c["nu"] == nu
                     and c["formulation"] == bform)
            vals_full.append(f["eps_v_rel_error"])
            vals_bbar.append(b["eps_v_rel_error"])
    x = np.arange(len(labels))
    ax[0].bar(x - 0.2, vals_full, 0.4, label="full integration")
    ax[0].bar(x + 0.2, vals_bbar, 0.4, label="B-bar")
    ax[0].set_yscale("log")
    ax[0].axhline(0.01, color="r", ls=":", label="1 % gate")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels, fontsize=7)
    ax[0].set_ylabel("volumetric strain relative error")
    ax[0].set_title("Volumetric locking (thick cylinder, p = 50 MPa)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, axis="y")

    ur_full, ur_bbar = [], []
    for element in ("hex8", "hex20"):
        for nu in (0.3, 0.4999):
            f = next(c for c in lock if c["element"] == element and c["nu"] == nu
                     and c["formulation"] == "full")
            bform = "bbar" if element == "hex8" else "bbar_stab"
            b = next(c for c in lock if c["element"] == element and c["nu"] == nu
                     and c["formulation"] == bform)
            ur_full.append(f["inner_displacement_rel_error"])
            ur_bbar.append(b["inner_displacement_rel_error"])
    ax[1].bar(x - 0.2, ur_full, 0.4, label="full integration")
    ax[1].bar(x + 0.2, ur_bbar, 0.4, label="B-bar")
    ax[1].set_yscale("log")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=7)
    ax[1].set_ylabel("inner radial displacement relative error")
    ax[1].set_title("Displacement error against the Lame solution")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fn = out / "fig2_locking.png"
    fig.savefig(fn)
    plt.close(fig)
    files.append(str(fn))

    # --- fig 3: hourglass ----------------------------------------------
    hg = res["hourglass"]
    sw = [s for s in hg["sweep_smooth_cylinder"] if s["alpha_hg"] > 0.0]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    ax[0].semilogx([s["alpha_hg"] for s in sw], [s["hourglass_fraction"] for s in sw], "o-")
    ax[0].axhline(0.05, color="r", ls=":", label="5 % gate")
    ax[0].set_xlabel("hourglass coefficient $\\alpha_{hg}$")
    ax[0].set_ylabel("hourglass energy / total strain energy")
    ax[0].set_title("Stabilisation energy, smooth cylinder field (Hex8, 1 pt)")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3, which="both")
    ax[0].annotate("unstabilised: rank deficient\n(18 zero-energy modes/element)", xy=(0.02, 0.02),
                   xytext=(0.011, 0.55), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=0.7))

    ax[1].semilogx([s["alpha_hg"] for s in sw],
                   [100.0 * s["inner_displacement_rel_error"] for s in sw], "o-")
    ax[1].set_xlabel("hourglass coefficient $\\alpha_{hg}$")
    ax[1].set_ylabel("inner displacement error [%]")
    ax[1].set_title("Accuracy vs stabilisation strength")
    ax[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fn = out / "fig3_hourglass.png"
    fig.savefig(fn)
    plt.close(fig)
    files.append(str(fn))

    # --- fig 4: J2 plasticity ------------------------------------------
    pl = res["plasticity"]
    fig, ax = plt.subplots(1, 2, figsize=(9.0, 3.6))
    curve = pl["uniaxial"]["curve"]
    ana = pl["uniaxial_analytic_curve"]
    ax[0].plot([c["eps_xx"] for c in ana], [c["sigma_xx"] / 1e6 for c in ana], "k-",
               label="analytic (linear hardening)")
    ax[0].plot([c["eps_xx"] for c in curve], [c["sigma_xx"] / 1e6 for c in curve], "o",
               ms=4, mfc="none", label="3-D FE, Hex8")
    ax[0].axvline(pl["uniaxial"]["eps_yield"], color="grey", ls=":", label="$\\varepsilon_y$")
    ax[0].set_xlabel("$\\varepsilon_{xx}$")
    ax[0].set_ylabel("$\\sigma_{xx}$ [MPa]")
    ax[0].set_title("Uniaxial tension, J2 + linear hardening")
    ax[0].legend(fontsize=7)
    ax[0].grid(alpha=0.3)

    lp = pl["limit_pressure"]["load_curve"]
    ax[1].plot([c["ur_inner"] / A_CYL * 100 for c in lp],
               [c["pressure_over_p_limit"] for c in lp], "o-", ms=3, label="FE (displacement stepping)")
    ax[1].axhline(1.0, color="r", ls="--", label="$p_L$ analytic $=\\frac{2\sigma_y}{\sqrt{3}}\ln(b/a)$")
    ax[1].set_xlabel("inner radial displacement $u_r(a)/a$ [%]")
    ax[1].set_ylabel("$p / p_L$")
    ax[1].set_title("Plane-strain cylinder: plastic collapse")
    ax[1].legend(fontsize=7)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    fn = out / "fig4_plasticity.png"
    fig.savefig(fn)
    plt.close(fig)
    files.append(str(fn))

    # --- fig 5: Newton convergence (consistent tangent) -----------------
    fn = out / "fig5_newton.png"
    traces = res["plasticity"]["limit_pressure"].get("residual_traces", [])
    if traces:
        fig, ax = plt.subplots(figsize=(4.6, 3.4))
        for i, tr in enumerate(traces):
            ax.semilogy(range(1, len(tr) + 1), tr, "o-", ms=3, lw=0.9,
                        label=f"step {i + 1}")
        ax.set_xlabel("Newton iteration")
        ax.set_ylabel("relative residual")
        ax.set_title("Quadratic convergence (consistent tangent)")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(fn)
        plt.close(fig)
        files.append(str(fn))
    return files


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "artifacts" / "p33_w2_n06"))
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    results = {
        "node": "P33-W2-N06",
        "title": "3-D isoparametric solid element kernel: patch test / locking / hourglass / J2 / Lame",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "material": {"E_Pa": E_STEEL, "nu": NU_STEEL, "sigma_y0_Pa": SIGMA_Y},
        "cylinder": {"a_m": A_CYL, "b_m": B_CYL, "p_Pa": P_CYL},
    }
    print("[1/5] patch tests")
    results["patch"] = run_patch_tests()
    print("[2/5] volumetric locking")
    results["locking"] = run_locking()
    print("[3/5] hourglass control")
    results["hourglass"] = run_hourglass()
    print("[4/5] Lame cylinder")
    results["lame"] = run_lame()
    print("[5/5] J2 plasticity")
    results["plasticity"] = run_plasticity(quick=args.quick)

    results["criteria"] = {
        "a_patch_test": {
            "value": results["patch"]["worst_displacement_error"],
            "gate": "< 1e-10",
            "pass": results["patch"]["pass"],
        },
        "b_volumetric_locking": {
            "value": results["locking"]["nearly_incompressible_bbar_worst_eps_v_rel_error"],
            "gate": "< 0.01",
            "pass": results["locking"]["pass"],
        },
        "c_hourglass": {
            "value": results["hourglass"]["selected_hourglass_fraction"],
            "gate": "< 0.05",
            "pass": results["hourglass"]["pass"],
        },
        "d_j2_uniaxial": {
            "value": results["plasticity"]["uniaxial"]["rel_error"],
            "gate": "< 0.01",
            "pass": bool(results["plasticity"]["uniaxial"]["rel_error"] < 0.01),
        },
        "e_lame": {
            "value": {
                "sigma_r": results["lame"]["hex20_4x8"]["sigma_r_max_rel_error"],
                "sigma_theta": results["lame"]["hex20_4x8"]["sigma_theta_max_rel_error"],
            },
            "gate": "< 0.02",
            "pass": results["lame"]["pass"],
        },
        "supplementary_limit_pressure": {
            "value": results["plasticity"]["limit_pressure"]["rel_error"],
            "gate": "< 0.05",
            "pass": bool((results["plasticity"]["limit_pressure"]["rel_error"] or 1.0) < 0.05),
        },
    }
    results["runtime_seconds"] = time.time() - t0
    results["figures"] = make_figures(results, out)

    payload = _jsonable(results)
    (out / "results.json").write_text(json.dumps(payload, indent=2))
    all_pass = all(v["pass"] for v in payload["criteria"].values())
    print(json.dumps({k: v for k, v in payload["criteria"].items()}, indent=2))
    print("ALL PASS:", all_pass, " runtime %.1f s" % payload["runtime_seconds"])
    print("written:", out / "results.json")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())