#!/usr/bin/env python
"""m8870 R1 audit: accuracy + verification integrity of `geometric_nonlinear`.

Produces (regenerable, zero hand-copying):

    artifacts/m8870_r1/elastica_quadrature.json
    artifacts/m8870_r1/elastica_energy.json
    artifacts/m8870_r1/plate_law_vs_fem.json
    artifacts/m8870_r1/plate_energy.json
    artifacts/m8870_r1/plate_convergence.json
    artifacts/m8870_r1/arch_limit_load.json
    artifacts/m8870_r1/summary.json
    artifacts/m8870_r1/fig_*.png

Run:  python docs/m8870_r1_audit.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scipy import integrate  # noqa: E402
from scipy.integrate import solve_bvp  # noqa: E402

from tensorphysics.geometric_nonlinear.elastica import ElasticaExact  # noqa: E402
from tensorphysics.geometric_nonlinear.elastica_fem import ElasticaFEMSolver  # noqa: E402
from tensorphysics.geometric_nonlinear.shallow_arch import (  # noqa: E402
    ShallowArchModel,
    VonMisesTruss,
)
from tensorphysics.geometric_nonlinear.verification import (  # noqa: E402
    convergence_order_plate,
    full_verification_plate,
    verify_energy_conservation_elastica,
    verify_energy_conservation_plate,
)
from tensorphysics.geometric_nonlinear.von_karman_plate import (  # noqa: E402
    VonKarmanPlateExact,
    VonKarmanPlateFEM,
    load_path_work,
    plate_cubic_coefficient,
)

OUT = ROOT / "artifacts" / "m8870_r1"
OUT.mkdir(parents=True, exist_ok=True)


def dump(name: str, payload) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, default=float) + "\n")


# ---------------------------------------------------------------------------
# 0. legacy (pre-R1) quadrature, reproduced verbatim to quantify the defect
# ---------------------------------------------------------------------------

def legacy_elastica(theta_B: float, L: float = 1.0, EI: float = 1.0):
    """Pre-R1 quadrature: theta = theta_upper t^2 with 100 Gauss points.

    This substitution does *not* remove the 1/sqrt(1-t) endpoint singularity,
    so both the elliptic integrals and the derived load carry a ~1.2-1.5%
    systematic bias.
    """
    n_quad = 100
    xi, wi = np.polynomial.legendre.leggauss(n_quad)
    t_pts = 0.5 * (xi + 1)
    w_pts = 0.5 * wi

    def val(t, kind):
        theta = theta_B * t**2
        dtheta = 2 * theta_B * t
        diff = np.sin(theta_B) - np.sin(theta)
        if diff <= 1e-15:
            return 0.0
        trig = 1.0 if kind == 0 else (np.cos(theta) if kind == 1 else np.sin(theta))
        return trig * dtheta / np.sqrt(diff)

    I_L = float(np.sum(w_pts * [val(t, 0) for t in t_pts]))
    I_y = float(np.sum(w_pts * [val(t, 2) for t in t_pts]))
    P = EI * I_L**2 / (2.0 * L**2)
    delta = L * I_y / I_L
    return P, delta


# ---------------------------------------------------------------------------
# 1. independent references for the elastica
# ---------------------------------------------------------------------------

def ref_quadrature(theta_B: float, L: float = 1.0, EI: float = 1.0):
    """Adaptive quadrature with theta = theta_B (1 - xi^2), xi in [0, 1]."""
    def integrand(kind):
        def f(x):
            theta = theta_B * (1.0 - x**2)
            diff = 2.0 * np.cos(0.5 * (theta_B + theta)) * np.sin(0.5 * (theta_B - theta))
            diff = max(diff, 1e-300)
            trig = 1.0 if kind == 0 else (np.cos(theta) if kind == 1 else np.sin(theta))
            return 2.0 * theta_B * x * trig / np.sqrt(diff)

        return integrate.quad(f, 0.0, 1.0, limit=400, epsabs=1e-14, epsrel=1e-13)[0]

    I_L, I_y = integrand(0), integrand(2)
    P = EI * I_L**2 / (2.0 * L**2)
    delta = L * I_y / I_L
    return P, delta


def ref_bvp(theta_B: float, L: float = 1.0, EI: float = 1.0):
    """Direct BVP solve of EI theta'' + P cos theta = 0, P unknown."""
    s = np.linspace(0.0, L, 201)

    def ode(sx, y, p):
        return np.vstack((y[1], -p[0] / EI * np.cos(y[0])))

    def bc(ya, yb, p):
        return np.array([ya[0], yb[1], yb[0] - theta_B])

    u0 = theta_B * (2.0 * s / L - (s / L) ** 2) * 0.9
    sol = solve_bvp(ode, bc, s, np.vstack((u0, np.gradient(u0, s))),
                    p=np.array([3.0 * EI * theta_B / L**2]), tol=1e-10)
    P = float(sol.p[0])
    delta = float(integrate.trapezoid(np.sin(sol.y[0]), sol.x))
    return P, delta, int(sol.status)


# ---------------------------------------------------------------------------
# audits
# ---------------------------------------------------------------------------

def audit_elastica_quadrature():
    exact = ElasticaExact(L=1.0, EI=1.0)
    thetas = [0.02, 0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

    rows = []
    for tb in thetas:
        sol = exact.compute_from_theta_B(tb)
        P_ref, d_ref = ref_quadrature(tb)
        P_leg, d_leg = legacy_elastica(tb)
        row = {
            "theta_B": tb,
            "P_module": sol.P,
            "P_reference_quad": P_ref,
            "P_legacy": P_leg,
            "error_module_pct": abs(sol.P - P_ref) / P_ref * 100,
            "error_legacy_pct": abs(P_leg - P_ref) / P_ref * 100,
            "delta_module": sol.y_tip,
            "delta_reference_quad": d_ref,
            "delta_legacy": d_leg,
            "error_delta_module_pct": abs(sol.y_tip - d_ref) / d_ref * 100,
            "error_delta_legacy_pct": abs(d_leg - d_ref) / d_ref * 100,
        }
        if tb <= 1.5:
            P_bvp, d_bvp, status = ref_bvp(tb)
            row.update({
                "P_reference_bvp": P_bvp,
                "delta_reference_bvp": d_bvp,
                "bvp_status": status,
                "error_module_vs_bvp_pct": abs(sol.P - P_bvp) / P_bvp * 100,
            })
        rows.append(row)

    dump("elastica_quadrature.json", {"rows": rows})

    # figure: load-deflection + error
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

    tb_grid = np.linspace(0.02, 1.5, 60)
    P_mod, d_mod = [], []
    P_leg, d_leg = [], []
    P_ref, d_ref = [], []
    for tb in tb_grid:
        s = exact.compute_from_theta_B(tb)
        P_mod.append(s.P)
        d_mod.append(s.y_tip)
        p, d = legacy_elastica(tb)
        P_leg.append(p)
        d_leg.append(d)
        p, d = ref_quadrature(tb)
        P_ref.append(p)
        d_ref.append(d)

    ax[0].plot(np.array(d_ref) * 1.0, P_ref, "k-", lw=2, label="independent reference (adaptive quadrature)")
    ax[0].plot(d_mod, P_mod, "r--", lw=1.4, label="ElasticaExact after fix")
    ax[0].plot(d_leg, P_leg, "b:", lw=1.6, label="ElasticaExact before fix (t^2 substitution)")
    ax[0].set_xlabel(r"tip deflection  $\delta/L$")
    ax[0].set_ylabel(r"normalised load  $PL^2/EI$")
    ax[0].set_title("Cantilever elastica: load-deflection")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    err_mod = np.abs(np.array(P_mod) - np.array(P_ref)) / np.array(P_ref) * 100
    err_leg = np.abs(np.array(P_leg) - np.array(P_ref)) / np.array(P_ref) * 100
    ax[1].semilogy(tb_grid, np.maximum(err_leg, 1e-14), "b-o", ms=3, label="before fix")
    ax[1].semilogy(tb_grid, np.maximum(err_mod, 1e-14), "r-s", ms=3, label="after fix")
    ax[1].set_xlabel(r"free-end angle  $\theta_B$  [rad]")
    ax[1].set_ylabel("relative error of $P$  [%]")
    ax[1].set_title("Quadrature error of the elastica load")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3, which="both")

    fig.tight_layout()
    fig.savefig(OUT / "fig_elastica_quadrature.png", dpi=130)
    plt.close(fig)

    return rows


def audit_elastica_energy():
    res = verify_energy_conservation_elastica()
    rows = []
    for i, tb in enumerate(res["theta_B"]):
        rows.append({
            "theta_B": float(tb),
            "P": float(res["P"][i]),
            "delta": float(res["delta"][i]),
            "U_strain": float(res["U_strain"][i]),
            "W_deform": float(res["W_deform"][i]),
            "error_pct": float(res["error_pct"][i]),
            "P_times_delta": float(res["P"][i] * res["delta"][i]),
            "U_over_Pdelta": float(res["U_strain"][i] / (res["P"][i] * res["delta"][i])),
            "dU_ddelta": float(res["dU_ddelta"][i]),
            "generalized_force_error_pct": float(res["generalized_force_error_pct"][i]),
        })
    dump("elastica_energy.json", {"rows": rows})

    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(res["delta"], res["U_strain"], "o-", label=r"$U$ (bending energy)")
    ax.plot(res["delta"], res["W_deform"], "x--", label=r"$\int_0^\delta P\,d\delta'$")
    ax.plot(res["delta"], res["P"] * res["delta"], "s:", color="gray",
            label=r"$P\delta$ (wrong criterion)")
    ax.set_xlabel(r"tip deflection $\delta$")
    ax.set_ylabel("energy")
    ax.set_title("Elastica: work of deformation vs energy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_elastica_energy.png", dpi=130)
    plt.close(fig)
    return rows


def audit_plate_law_vs_fem():
    a, h, E, nu = 1.0, 0.01, 210e9, 0.3
    ref = VonKarmanPlateExact(a=a, h=h, E=E, nu=nu)

    targets = np.array([0.5, 1.0, 2.0, 3.0])
    _, _, P_vals = ref.compute_load_deflection_curve(targets)

    rows = []
    for n_elem in (16, 32):
        fem = VonKarmanPlateFEM(a=a, h=h, E=E, nu=nu, n_elem=n_elem)
        for wh, P in zip(targets, P_vals):
            res = fem.solve(float(P), n_steps=10, max_iter=60, tol=1e-10)
            w0h_fe = res["w0"] / h
            Pn = P * a**2 / (16 * np.pi * ref.D * h)
            rows.append({
                "n_elem": n_elem,
                "target_w0_over_h": float(wh),
                "P": float(P),
                "P_normalized": float(Pn),
                "fem_w0_over_h": float(w0h_fe),
                "error_pct": float(abs(w0h_fe - wh) / wh * 100),
                "implied_c": float((Pn - w0h_fe) / w0h_fe**3) if w0h_fe > 0.2 else float("nan"),
                "converged": bool(res["converged"]),
            })

    mesh_change = max(
        abs(r1["fem_w0_over_h"] - r2["fem_w0_over_h"]) / r1["fem_w0_over_h"] * 100
        for r1, r2 in zip([x for x in rows if x["n_elem"] == 16],
                          [x for x in rows if x["n_elem"] == 32])
    )

    dump("plate_law_vs_fem.json", {
        "cubic_coefficient": plate_cubic_coefficient(nu),
        "rows": rows,
        "max_mesh_change_pct": mesh_change,
    })

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    wh = np.linspace(0.1, 4.0, 200)
    _, Pn_law, _ = ref.compute_load_deflection_curve(wh)
    ax[0].plot(wh, Pn_law, "k-", lw=2,
               label=f"closed form  $\\bar P=\\bar w+{plate_cubic_coefficient(nu):.3f}\\bar w^3$")
    for n_elem, marker in ((16, "o"), (32, "s")):
        pts = [r for r in rows if r["n_elem"] == n_elem]
        ax[0].plot([p["fem_w0_over_h"] for p in pts], [p["P_normalized"] for p in pts],
                   marker, ms=6, mfc="none", label=f"von Karman FE, n_elem={n_elem}")
    ax[0].set_xlabel(r"$w_0/h$")
    ax[0].set_ylabel(r"$Pa^2/(16\pi D h)$")
    ax[0].set_title("Clamped plate, centre point load")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    for n_elem, marker in ((16, "o"), (32, "s")):
        pts = [r for r in rows if r["n_elem"] == n_elem]
        ax[1].plot([p["target_w0_over_h"] for p in pts], [p["implied_c"] for p in pts],
                   marker + "-", label=f"n_elem={n_elem}")
    ax[1].axhline(plate_cubic_coefficient(nu), color="k", ls="--", lw=1,
                  label="law coefficient c(0.3)")
    ax[1].set_xlabel(r"$w_0/h$")
    ax[1].set_ylabel("effective cubic coefficient c")
    ax[1].set_title("Constant-c cubic fit vs FE")
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_plate_law_vs_fem.png", dpi=130)
    plt.close(fig)
    return rows, mesh_change


def audit_plate_energy():
    res = verify_energy_conservation_plate(
        w0_h_values=np.array([0.5, 1.0, 2.0, 3.0]), n_elem=16, n_steps=12
    )
    rows = []
    for i, wh in enumerate(res["w0_h"]):
        rows.append({
            "w0_over_h": float(wh),
            "w0": float(res["w0"][i]),
            "P": float(res["P"][i]),
            "W_ext": float(res["w_ext"][i]),
            "U_total": float(res["u_total"][i]),
            "U_bending": float(res["u_bending"][i]),
            "U_membrane": float(res["u_membrane"][i]),
            "error_pct": float(res["error_pct"][i]),
            "converged": bool(res["converged"][i]),
        })
    dump("plate_energy.json", {"rows": rows})

    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    x = res["w0_h"]
    ax[0].plot(x, res["w_ext"], "ko-", label=r"$W_{ext}=\int P\,dw_0$")
    ax[0].plot(x, res["u_total"], "r^--", label=r"$U_{FE}$ total")
    ax[0].plot(x, res["u_bending"], "b:", label="bending")
    ax[0].plot(x, res["u_membrane"], "g:", label="membrane")
    ax[0].set_xlabel(r"$w_0/h$")
    ax[0].set_ylabel("energy [J]")
    ax[0].set_title("Plate energy balance")
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    ax[1].plot(x, res["error_pct"], "ko-")
    ax[1].axhline(0.0, color="gray", lw=0.8)
    ax[1].set_xlabel(r"$w_0/h$")
    ax[1].set_ylabel(r"$|W_{ext}-U_{FE}|/U_{FE}$  [%]")
    ax[1].set_title("Energy-balance residual (non-zero by construction)")
    ax[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(OUT / "fig_plate_energy.png", dpi=130)
    plt.close(fig)
    return rows


def audit_plate_convergence():
    conv = convergence_order_plate(n_elem_list=[8, 16, 32], n_steps=6)
    full = full_verification_plate(n_elem=16, n_steps=8)
    payload = {
        "convergence": {
            "n_elem": conv["n_elem"],
            "h_values": conv["h_values"].tolist(),
            "w0": conv["w0"].tolist(),
            "differences": conv["differences"].tolist(),
            "convergence_order": conv["convergence_order"],
            "P_target": conv["P_target"],
        },
        "closed_form_vs_fem": {
            "w0_h_check": full["w0_h_check"].tolist(),
            "w0_h_fem": full["w0_h_fem"].tolist(),
            "error_pct": full["P_error_pct"].tolist(),
            "all_pass": full["all_pass"],
        },
    }
    dump("plate_convergence.json", payload)
    return payload


def audit_arch_limit_load():
    L, EA = 10.0, 1e9
    rows = []
    for lam in (0.1, 0.15, 0.2, 0.25, 0.3):
        H = lam * L
        truss = VonMisesTruss(L=L, H=H, EA=EA)
        res = truss.find_limit_points(n_pts=4000)

        # closed-form limit load: maximise q(h) = 2 EA/L0 * h * (L0/L1(h) - 1)
        L0 = np.hypot(L / 2, H)
        hs = np.linspace(1e-9, H, 4_000_001)
        L1 = np.hypot(L / 2, hs)
        q = 2 * EA / L0 * hs * (L0 / L1 - 1.0)
        q_analytic = float(q.max())
        h_analytic = float(hs[int(np.argmax(q))])

        rows.append({
            "rise_span": lam,
            "H": H,
            "q_limit_numeric": float(res["q_limit_upper"]),
            "q_limit_analytic": q_analytic,
            "w_at_limit_numeric": float(res["w_at_upper"]),
            "h_at_limit_analytic": h_analytic,
            "error_pct": abs(res["q_limit_upper"] - q_analytic) / q_analytic * 100,
            "has_negative_stiffness": "unstable" in res["stability"],
        })

    arch = ShallowArchModel(L=L, H=1.0, EI=1e6, EA=EA, n_elem=10)
    payload = {
        "rows": rows,
        "arch_model_analytical_limit_load": float(arch.analytical_limit_load()),
    }
    dump("arch_limit_load.json", payload)

    truss = VonMisesTruss(L=L, H=1.0, EA=EA)
    v = np.linspace(-0.2, 1.6, 400)
    q_path, p_path = truss.equilibrium_path(v)

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(np.array(p_path), np.array(q_path), "k-", lw=1.8, label="von Mises truss path")
    ax.axhline(rows[3]["q_limit_analytic"], color="r", ls="--", lw=1,
               label="closed-form limit load (H/L=0.25)")
    ax.set_xlabel("apex displacement")
    ax.set_ylabel("load")
    ax.set_title("Shallow arch snap-through")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig_arch_limit_load.png", dpi=130)
    plt.close(fig)
    return payload


def main() -> int:
    print("== elastica quadrature ==")
    quad_rows = audit_elastica_quadrature()
    for r in quad_rows:
        print(f"  theta_B={r['theta_B']:.2f}  P={r['P_module']:.8f}  "
              f"err_module={r['error_module_pct']:.2e}%  err_legacy={r['error_legacy_pct']:.4f}%")

    print("== elastica energy ==")
    energy_rows = audit_elastica_energy()
    for r in energy_rows:
        print(f"  theta_B={r['theta_B']:.2f}  |W-U|/U={r['error_pct']:.2e}%  "
              f"U/(P delta)={r['U_over_Pdelta']:.3f}  dU/ddelta err={r['generalized_force_error_pct']:.1e}%")

    print("== plate: closed form vs FE ==")
    plate_rows, mesh_change = audit_plate_law_vs_fem()
    for r in plate_rows:
        print(f"  n_elem={r['n_elem']:3d}  w0/h target={r['target_w0_over_h']:.1f}  "
              f"FE={r['fem_w0_over_h']:.4f}  err={r['error_pct']:.2f}%  c_implied={r['implied_c']:.3f}")
    print(f"  mesh sensitivity (16 vs 32): {mesh_change:.3f}%")

    print("== plate energy ==")
    pe = audit_plate_energy()
    for r in pe:
        print(f"  w0/h={r['w0_over_h']:.1f}  W_ext={r['W_ext']:.3f}  U={r['U_total']:.3f}  "
              f"residual={r['error_pct']:.3f}%")

    print("== plate convergence ==")
    conv = audit_plate_convergence()
    print(f"  order={conv['convergence']['convergence_order']:.2f}  "
          f"closed-form errors={np.round(conv['closed_form_vs_fem']['error_pct'], 3)}")

    print("== arch ==")
    arch = audit_arch_limit_load()
    for r in arch["rows"]:
        print(f"  H/L={r['rise_span']:.2f}  numeric={r['q_limit_numeric']:.6e}  "
              f"analytic={r['q_limit_analytic']:.6e}  err={r['error_pct']:.2e}%")

    summary = {
        "mission": "m8870 R1 (TensorPhysics#367 reopen)",
        "elastica_quadrature": {
            "max_error_after_fix_pct": max(r["error_module_pct"] for r in quad_rows),
            "max_error_before_fix_pct": max(r["error_legacy_pct"] for r in quad_rows),
            "max_delta_error_before_fix_pct": max(r["error_delta_legacy_pct"] for r in quad_rows),
        },
        "elastica_energy": {
            "max_work_vs_energy_pct": max(r["error_pct"] for r in energy_rows),
            "max_generalized_force_error_pct": max(r["generalized_force_error_pct"] for r in energy_rows),
            "u_over_pdelta_range": [
                min(r["U_over_Pdelta"] for r in energy_rows),
                max(r["U_over_Pdelta"] for r in energy_rows),
            ],
        },
        "plate_law_vs_fem": {
            "max_error_pct": max(r["error_pct"] for r in plate_rows),
            "mesh_sensitivity_pct": mesh_change,
            "cubic_coefficient": plate_cubic_coefficient(0.3),
            "implied_c_range": [
                min(r["implied_c"] for r in plate_rows),
                max(r["implied_c"] for r in plate_rows),
            ],
        },
        "plate_energy": {
            "max_residual_pct": max(r["error_pct"] for r in pe),
        },
        "plate_convergence_order": conv["convergence"]["convergence_order"],
        "arch_limit_load": {
            "max_error_pct": max(r["error_pct"] for r in arch["rows"]),
        },
    }
    dump("summary.json", summary)
    print("\n== summary ==")
    print(json.dumps(summary, indent=2, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())