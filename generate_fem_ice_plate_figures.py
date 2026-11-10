"""Generate the figures of the FEM ice-plate cohesive/plasticity node.

Figures (written to ``artifacts/fem_ice_plate/``):

1. ``fig1_hetenyi_plate.png``      -- plate deflection and radial moment vs the
   Hetenyi/Kelvin closed form (C1).
2. ``fig2_uniaxial_hardening.png`` -- hardening curves from the 3-D return
   mapping vs the closed-form law, plus the plastic tangent (C2).
3. ``fig3_cylinder_stress.png``    -- radial/hoop/axial stress and plastic zone
   of the pressurised cylinder vs the classical autofrettage solution (C3).
4. ``fig4_ice_plate_load.png``     -- load-deflection of the ring-cracked ice
   plate for three fracture energies, with the first-crack markers (C4).
5. ``fig5_energy_unload.png``      -- energy audit bars and the elastic
   unloading path (C5, C7).
6. ``fig6_controls.png``           -- cohesive-bar response and the three
   negative controls (C6).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorphysics.elastic_foundation.plates import (
    winkler_plate_patch_load,
    winkler_plate_patch_moment_r,
)
from tensorphysics.fem_ice_plate.cohesive_zone import BilinearTSL, CohesiveHinge
from tensorphysics.fem_ice_plate.coupled import solve_bar_with_interface
from tensorphysics.fem_ice_plate.foundation_plate import (
    plate_flexural_rigidity,
    solve_axisym_plate_foundation,
)
from tensorphysics.fem_ice_plate.j2_plasticity import (
    J2Material,
    thick_cylinder_fe,
    thick_cylinder_plastic_reference,
    uniaxial_reference,
    uniaxial_tension_curve,
)

OUT = Path("artifacts/fem_ice_plate")
OUT.mkdir(parents=True, exist_ok=True)


def fig1():
    E, h, nu, k = 30.0e9, 1.0, 0.3, 1.0e4
    D = plate_flexural_rigidity(E, h, nu)
    l0 = (D / k) ** 0.25
    a, P = 0.1 * l0, 1.0e5
    w0 = float(winkler_plate_patch_load(np.array([0.0]), P, a, D, k)[0])
    res = solve_axisym_plate_foundation(
        D, nu, k, patch_radius=a, n_el=100, grading=1.6, w_center_steps=[w0]
    )
    Q = res.load_factor[-1]
    m = (res.r > 0.05 * l0) & (res.r < 6 * l0)
    w_ana = winkler_plate_patch_load(res.r[m], Q, a, D, k)
    s = (res.r_elem > 0.2 * l0) & (res.r_elem < 4 * l0)
    M_ana = winkler_plate_patch_moment_r(res.r_elem[s], Q, a, D, k, nu)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(res.r[m] / l0, res.w[m] * 1e3, "o", ms=2.5, label="FE (Hermite axisym.)")
    ax[0].plot(res.r[m] / l0, w_ana * 1e3, "-", lw=1.2, label="Hetenyi/Kelvin closed form")
    ax[0].set_xlabel(r"$r/l$"), ax[0].set_ylabel("deflection [mm]")
    ax[0].set_xlim(0, 4), ax[0].legend(), ax[0].set_title("(a) deflection, intact limit")
    ax[1].plot(res.r_elem[s] / l0, res.M_r_elem[s] / 1e3, "o", ms=2.5, label="FE")
    ax[1].plot(res.r_elem[s] / l0, M_ana / 1e3, "-", lw=1.2, label="closed form")
    ax[1].set_xlabel(r"$r/l$"), ax[1].set_ylabel(r"$M_r$ [kN m/m]")
    ax[1].legend(), ax[1].set_title("(b) radial moment")
    fig.suptitle("C1 -- axisymmetric Kirchhoff plate on Winkler foundation vs Hetenyi anchor")
    fig.tight_layout()
    fig.savefig(OUT / "fig1_hetenyi_plate.png", dpi=160)
    plt.close(fig)


def fig2():
    b = 25.0
    Qv = (510e6 - 355e6) / (1.0 - np.exp(-b * 0.15))
    voce = J2Material(E=206e9, nu=0.3, sigma_y0=355e6, hardening="voce", Q=Qv, b=b)
    swift = J2Material(E=193e9, nu=0.3, hardening="swift", K=1400e6, n=0.40, eps0=0.005)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mat, lab in ((voce, "Voce (structural steel)"), (swift, "Swift (stainless)")):
        num = uniaxial_tension_curve(mat, eps_max=0.1, n_steps=400)
        ref = uniaxial_reference(mat, num["eps"])
        ax[0].plot(num["eps"] * 100, num["sigma"] / 1e6, "o", ms=2, label=f"{lab} FE/return map")
        ax[0].plot(ref["eps"] * 100, ref["sigma"] / 1e6, "-", lw=1.0, label=f"{lab} closed form")
    ax[0].set_xlabel("axial strain [%]"), ax[0].set_ylabel(r"$\sigma$ [MPa]")
    ax[0].legend(fontsize=7), ax[0].set_title("(a) hardening curves")
    eps = np.linspace(0, 0.1, 200)
    sy = np.array([voce.yield_stress(e) for e in eps])
    Hp = np.array([voce.hardening_modulus(e) for e in eps])
    ax[1].plot(eps * 100, voce.E * Hp / (voce.E + Hp) / 1e9, label=r"$E_{ep}=EH'/(E+H')$ (Voce)")
    ax[1].set_xlabel("accumulated plastic strain [%]"), ax[1].set_ylabel("tangent modulus [GPa]")
    ax[1].legend(), ax[1].set_title("(b) elastoplastic tangent")
    fig.suptitle("C2 -- uniaxial hardening from the implicit 3-D J2 return mapping")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_uniaxial_hardening.png", dpi=160)
    plt.close(fig)


def fig3():
    a, b, sy, E, nu = 0.05, 0.10, 250e6, 200e9, 0.3
    mat = J2Material(E=E, nu=nu, sigma_y0=sy, hardening="perfect")
    p_target = 170e6
    res = thick_cylinder_fe(mat, a, b, p_target, n_el=80, n_inc=24, tol=1e-8)
    ref = thick_cylinder_plastic_reference(a, b, p_target, sy, E=E, nu=nu)
    fld = ref["fields"](res.r_gauss)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for i, key, lab in ((0, "sigma_r", r"$\sigma_r$"), (1, "sigma_theta", r"$\sigma_\theta$"),
                        (2, "sigma_z", r"$\sigma_z$")):
        ax[0].plot(res.r_gauss / a, res.sigma_gauss[:, i] / 1e6, ".", ms=3, label=f"{lab} FE")
        ax[0].plot(res.r_gauss / a, fld[key] / 1e6, "-", lw=1.0, label=f"{lab} classical")
    ax[0].axvline(ref["c"] / a, color="k", ls="--", lw=0.8)
    ax[0].set_xlabel(r"$r/a$"), ax[0].set_ylabel("stress [MPa]"), ax[0].legend(fontsize=7)
    ax[0].set_title(f"(a) p = {p_target/1e6:.0f} MPa, $r_p/a$ = {ref['c']/a:.2f}")
    ax[1].plot(res.r_gauss / a, res.ep_gauss * 100, ".-", ms=3)
    ax[1].set_xlabel(r"$r/a$"), ax[1].set_ylabel(r"equivalent plastic strain [%]")
    ax[1].set_title("(b) plastic strain distribution")
    fig.suptitle("C3 -- plane-strain thick cylinder: radial-return FE vs autofrettage solution")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_cylinder_stress.png", dpi=160)
    plt.close(fig)


def fig4():
    E, h, nu, k, st = 5.0e9, 1.0, 0.3, 9810.0, 0.7e6
    D = plate_flexural_rigidity(E, h, nu)
    l0 = (D / k) ** 0.25
    ac = 0.3 * l0
    M_peak = st * h**2 / 6.0
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    steps = np.linspace(0.0, 0.35, 51)[1:]
    for Gf, col in ((5.0, "C0"), (20.0, "C1"), (80.0, "C2")):
        hinge = CohesiveHinge.from_fracture_energy(M_peak, Gf, h, phi_0_frac=1e-5)
        res = solve_axisym_plate_foundation(
            D, nu, k, patch_radius=0.2 * l0, n_el=80, grading=1.6,
            crack_radius=ac, hinge=hinge, w_center_steps=steps, tol=1e-9,
        )
        ax[0].plot(res.w_center * 1e3, res.total_load / 1e6, "-", color=col,
                   label=rf"$G_f$ = {Gf:.0f} J/m$^2$")
        ax[1].plot(np.abs(res.phi_history), np.clip(res.total_load / 1e6, 0, None), ".",
                   ms=2, color=col)
    ax[0].set_xlabel("centre deflection [mm]"), ax[0].set_ylabel("load [MN]")
    ax[0].legend(fontsize=8), ax[0].set_title("(a) ice-plate response with a ring crack")
    ax[0].axhline(M_peak / 0.127 / 1e6, color="k", ls=":", lw=0.8,
                  label="elastic flexural criterion")
    ax[1].set_xscale("log"), ax[1].set_xlabel(r"hinge rotation $\varphi$ [rad]")
    ax[1].set_ylabel("load [MN]"), ax[1].set_title("(b) softening branch (log scale)")
    fig.suptitle("C4 -- cohesive ring crack in an ice sheet on elastic foundation")
    fig.tight_layout()
    fig.savefig(OUT / "fig4_ice_plate_load.png", dpi=160)
    plt.close(fig)


def fig5():
    results = json.loads((OUT / "verification_results.json").read_text())
    bl = results["C7"]["coupled_bar"]
    labels = ["W_ext", "U_elastic", "W_plastic", "W_fracture"]
    vals = [bl["W_ext"], bl["U_elastic"], bl["W_plastic"], bl["W_fracture"]]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(labels, np.array(vals) / 1e5, color=["C0", "C1", "C2", "C3"])
    ax[0].set_ylabel(r"energy [$\times 10^5$ J/m$^2$]")
    ax[0].set_title(f"(a) coupled bar audit, closure {bl['residual_rel']*100:.3f} %")
    # unloading path of the cylinder
    a, b, pmax, E, nu, sy = 0.05, 0.10, 170e6, 200e9, 0.3, 250e6
    path = list(np.linspace(0, pmax, 24)) + list(np.linspace(pmax, 0, 25)[1:])
    res = thick_cylinder_fe(
        J2Material(E=E, nu=nu, sigma_y0=sy, hardening="perfect"),
        a, b, pmax, n_el=60, p_path=path, tol=1e-8,
    )
    ax[1].plot(res.p_history / 1e6, res.u_bore_history * 1e3, "o-", ms=3)
    ax[1].set_xlabel("internal pressure [MPa]"), ax[1].set_ylabel(r"$u_r(a)$ [mm]")
    ax[1].set_title("(b) loading / elastic unloading path")
    fig.suptitle("C5/C7 -- elastic unloading and three-way energy conservation")
    fig.tight_layout()
    fig.savefig(OUT / "fig5_energy_unload.png", dpi=160)
    plt.close(fig)


def fig6():
    mat = J2Material(E=200e9, nu=0.3, sigma_y0=300e6, hardening="linear", H=2e9)
    elastic = J2Material(E=200e9, nu=0.3, sigma_y0=1e15)
    base = BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=2e-4)
    brittle = BilinearTSL(sigma_max=420e6, delta_f=1e-7, delta_0=1e-8)
    rigid = BilinearTSL(sigma_max=420e6, delta_f=2e-3, delta_0=1e-7)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for mat_i, tsl, lab in (
        (mat, base, "reference (plastic bulk + cohesive)"),
        (elastic, base, r"$\sigma_y\to\infty$: elastic bulk"),
        (mat, brittle, r"$G_f\to 0$: no softening"),
        (mat, rigid, r"$K\to\infty$: rigid interface"),
    ):
        r = solve_bar_with_interface(mat_i, tsl, L=0.1, n_steps=300)
        ax[0].plot(r.delta * 1e3, r.force / 1e6, "-", lw=1.1, label=lab)
        ax[1].plot(r.delta_interface * 1e3, r.delta * 1e3, "--", lw=1.0, label=lab)
    ax[0].set_xlabel("end displacement [mm]"), ax[0].set_ylabel("force [MN]")
    ax[0].set_ylim(0, 470), ax[0].legend(fontsize=7)
    ax[0].set_title("(a) cohesive bar response")
    ax[1].set_xlabel("interface opening [mm]"), ax[1].set_ylabel("end displacement [mm]")
    ax[1].set_title("(b) opening vs end displacement")
    fig.suptitle("C6 -- negative controls of the cohesive law")
    fig.tight_layout()
    fig.savefig(OUT / "fig6_controls.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    print("figures written to", OUT)