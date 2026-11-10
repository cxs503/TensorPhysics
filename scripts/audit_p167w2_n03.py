"""Audit script for campaign node p167w2-n03-fem-elastic-plastic (TensorFEM surface).

Runs quantitative probes of the elastoplastic_collapse verification surface and
dumps a JSON record.  The same script is used to capture the *baseline* (pre-fix)
and the *post-fix* state, so that the improvement is traceable to numbers.

Usage
-----
    PYTHONPATH=src python scripts/audit_p167w2_n03.py --tag baseline
    PYTHONPATH=src python scripts/audit_p167w2_n03.py --tag postfix
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from tensorphysics.elastoplastic_collapse import (
    VonMisesMaterial,
    return_mapping,
    StiffenedPlateGeometry,
    SmithCollapseSolver,
)
from tensorphysics.elastoplastic_collapse.iacs_icebelt_pressure import (
    IACSIceBeltPressure,
    icebelt_design_ratio,
    icebelt_panel_plastic_capacity,
    icebelt_stiffened_panel_capacity,
    plastic_section_modulus,
    pressure_area_law_icebelt,
)
from tensorphysics.elastoplastic_collapse.progressive_collapse import (
    ProgressiveCollapseSolver,
    panel_convergence_study,
)


E = 2.06e11
NU = 0.3
SY0 = 3.13e8
H = 1.0e9


def uniaxial_exact(eps: float, E: float, sy0: float, H: float) -> tuple[float, float]:
    """Exact 1D elastoplastic solution (isotropic linear hardening).

    Returns (sigma, eps_p) with sigma = sy0 + H eps_p and eps = sigma/E + eps_p.
    """
    eps_y = sy0 / E
    if eps <= eps_y:
        return E * eps, 0.0
    eps_p = (eps - eps_y) / (1.0 + H / E)
    return sy0 + H * eps_p, eps_p


def probe_uniaxial(mat: VonMisesMaterial, plane_stress: bool) -> dict:
    """Strain-driven uniaxial tension from the virgin state (single increment).

    Total strain is imposed together with the *analytic* lateral strain
    ``-νσ/E - ε_p/2``; the return mapping started from α=0 must reproduce the
    closed-form solution σ = σ_y0 + H ε_p and α = ε_p.
    """
    strains = np.linspace(1.0e-4, 6.0e-3, 60)
    errs, alpha_errs, spur = [], [], []
    for eps in strains:
        sigma_ex, eps_p = uniaxial_exact(eps, mat.E, mat.sigma_y0, mat.H)
        eps_lat = -mat.nu * sigma_ex / mat.E - 0.5 * eps_p
        if plane_stress:
            strain = np.array([eps, eps_lat, 0.0])
            sig, _C, _ep, alpha = return_mapping(
                strain, None, np.array([0.0]), mat, plane_stress=True)
            sig = np.asarray(sig).ravel()
            spur.append(abs(sig[1]) / sigma_ex)
        else:
            strain = np.array([eps, eps_lat, eps_lat, 0.0, 0.0, 0.0])
            sig, _C, _ep, alpha = return_mapping(
                strain, None, np.array([0.0]), mat, plane_stress=False)
            sig = np.asarray(sig).ravel()
            spur.append(max(abs(sig[1]), abs(sig[2])) / sigma_ex)
        errs.append(abs(sig[0] - sigma_ex) / sigma_ex)
        alpha_errs.append(abs(float(np.ravel(alpha)[0]) - eps_p) / max(eps_p, 1e-6))
    return {
        "max_stress_rel_error": float(max(errs)),
        "max_alpha_rel_error": float(max(alpha_errs)),
        "max_spurious_transverse_stress_rel": float(max(spur)),
        "n_points": len(strains),
    }


def fd_tangent(mat: VonMisesMaterial, plane_stress: bool) -> dict:
    """Finite-difference consistency check of the algorithmic tangent."""
    if plane_stress:
        strain = np.array([3.2e-3, -1.0e-3, 6.0e-4])
        alpha = np.array([0.0])
        eps_p = np.zeros(3)
        n = 3
    else:
        strain = np.array([3.2e-3, 1.0e-3, -0.5e-3, 4.0e-4, 0.0, 0.0])
        alpha = np.array([0.0])
        eps_p = np.zeros(6)
        n = 6

    s0, C, _ep, _a = return_mapping(
        strain, eps_p, alpha, mat, plane_stress=plane_stress)
    C = np.asarray(C).reshape(n, n)

    num = np.zeros((n, n))
    h = 1e-9
    for j in range(n):
        d = np.zeros(n)
        d[j] = h
        sp, _, _, _ = return_mapping(
            strain + d, eps_p, alpha, mat, plane_stress=plane_stress)
        sm, _, _, _ = return_mapping(
            strain - d, eps_p, alpha, mat, plane_stress=plane_stress)
        num[:, j] = (np.asarray(sp).ravel() - np.asarray(sm).ravel()) / (2 * h)

    scale = np.max(np.abs(num))
    asym = np.max(np.abs(C - C.T)) / scale
    rel = np.max(np.abs(C - num)) / scale
    return {
        "max_rel_deviation_from_fd": float(rel),
        "max_rel_asymmetry": float(asym),
        "fd_scale_GPa": float(scale / 1e9),
    }


def _panel(material, b_ratio_target=None, **kw):
    return StiffenedPlateGeometry(
        a=kw.pop("a", 2.4), b=kw.pop("b", 0.7), t=kw.pop("t", 0.012),
        h_s=kw.pop("h_s", 0.200), t_w=kw.pop("t_w", 0.010),
        b_f=kw.pop("b_f", 0.100), t_f=kw.pop("t_f", 0.015),
        material=material, w0_max=kw.pop("w0_max", 0.002),
        **kw)


def probe_smith(mat: VonMisesMaterial) -> dict:
    mat_pp = VonMisesMaterial(E=mat.E, nu=mat.nu, sigma_y0=mat.sigma_y0, H=0.0)
    out = {}
    for name, geom in {
        "stocky": _panel(mat_pp, b=0.30, t=0.014, h_s=0.25, t_w=0.012,
                         b_f=0.120, t_f=0.016, w0_max=0.0),
        "reference": _panel(mat_pp),
        "slender": _panel(mat_pp, b=1.20, t=0.010),
    }.items():
        solver = SmithCollapseSolver(geom, n_steps=200, max_strain=0.02)
        res = solver.solve()
        s = np.array([v for _, v in res.load_displacement])
        post = float(s[-1] / s.max()) if s.max() > 0 else float("nan")
        out[name] = {
            "sigma_u_MPa": float(res.sigma_u / 1e6),
            "sigma_u_over_sigma_y": float(res.sigma_u / mat.sigma_y0),
            "tail_over_peak": post,
            "sigma_cr_plate_MPa": float(solver.sigma_cr_plate / 1e6),
            "beta": float(geom.plate_slenderness),
            "lambda_col": float(geom.column_slenderness),
            "failure_mode": res.failure_mode,
            "has_limit_point": bool(np.argmax(s) < len(s) - 1),
        }
    return out


def probe_icebelt() -> dict:
    law = pressure_area_law_icebelt([0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0],
                                    P0=5.0e6, exponent=-0.3)
    L, t, sy = 0.8, 0.025, 355e6
    m_p = sy * t**2 / 4.0
    strip = icebelt_panel_plastic_capacity(L, t, sy, 0.0)
    strip_fixed = icebelt_panel_plastic_capacity(L, t, sy, 1.0)
    mat_ice = VonMisesMaterial(E=E, nu=NU, sigma_y0=sy, H=0.0)
    geom_ice = StiffenedPlateGeometry(
        a=0.6, b=0.7, t=0.025, h_s=0.30, t_w=0.012,
        b_f=0.12, t_f=0.018, material=mat_ice)
    stiff = icebelt_stiffened_panel_capacity(geom_ice)
    design = icebelt_design_ratio(
        IACSIceBeltPressure(P0=2.0e6, A0=1.0, exponent=-0.3),
        geom_ice.a, geom_ice.b, geom_ice.t, sy, geometry=geom_ice)
    return {
        "fitted_exponent": float(law["fitted_exponent"]),
        "exponent_error": float(law["exponent_error"]),
        "verified": bool(law["verified"]),
        "P_at_0.1m2_MPa": float(law["pressures"][0] / 1e6),
        "P_at_10m2_MPa": float(law["pressures"][-1] / 1e6),
        "strip_p_u_MPa": float(strip["p_u"] / 1e6),
        "strip_p_u_analytic_MPa": float(8.0 * m_p / L**2 / 1e6),
        "strip_rel_error": float(abs(strip["p_u"] - 8.0 * m_p / L**2)
                                 / (8.0 * m_p / L**2)),
        "clamped_over_simply_supported": float(strip_fixed["p_u"] / strip["p_u"]),
        "stiffened_Z_p_m3": float(stiff["Z_p"]),
        "stiffened_p_u_MPa": float(stiff["p_u"] / 1e6),
        "design_patch_area_m2": float(design["patch_area"]),
        "design_demand_MPa": float(design["pressure_demand"] / 1e6),
        "design_utilization": float(design["utilization"]),
    }


def probe_progressive(material: VonMisesMaterial) -> dict:
    """Progressive (fiber) Smith collapse: sweep over plate slenderness."""
    mat_pp = VonMisesMaterial(E=material.E, nu=material.nu,
                              sigma_y0=material.sigma_y0, H=0.0)
    out = {}
    for b in (0.30, 0.50, 0.70, 1.00, 1.40, 1.60):
        geom = StiffenedPlateGeometry(
            a=2.4, b=b, t=0.012, h_s=0.200, t_w=0.010,
            b_f=0.100, t_f=0.015, material=mat_pp, w0_max=0.0024)
        solver = ProgressiveCollapseSolver(geom, n_steps=120, max_strain=0.02)
        res = solver.solve()
        stresses = [v for _, v in res.load_displacement]
        peak_idx = int(np.argmax(stresses))
        out[f"b={b:.2f}"] = {
            "beta": float(geom.plate_slenderness),
            "lambda_col": float(geom.column_slenderness),
            "sigma_u_MPa": float(res.sigma_u / 1e6),
            "sigma_u_over_sigma_y": float(res.sigma_u / mat_pp.sigma_y0),
            "von_karman_plate_MPa": float(
                np.sqrt(solver.sigma_cr_plate * mat_pp.sigma_y0) / 1e6),
            "faulkner_plate_MPa": float(
                res.diagnostics["faulkner_plate_ultimate"] / 1e6),
            "vf_over_vk": float(res.sigma_u
                                / np.sqrt(solver.sigma_cr_plate * mat_pp.sigma_y0)),
            "vf_over_faulkner": float(
                res.sigma_u / res.diagnostics["faulkner_plate_ultimate"]),
            "post_peak_ratio": float(res.post_peak_ratio()),
            "peak_is_interior": bool(peak_idx < len(stresses) - 1),
            "failure_mode": res.failure_mode,
            "n_steps_resolved": len(res.load_displacement),
            "reached_max_strain": bool(res.converged),
        }
    geom_conv = StiffenedPlateGeometry(
        a=2.4, b=0.7, t=0.012, h_s=0.200, t_w=0.010,
        b_f=0.100, t_f=0.015, material=mat_pp, w0_max=0.0024)
    conv = panel_convergence_study(geom_conv, fiber_counts=(2, 4, 8, 16, 32),
                                   n_steps=100)
    out["fiber_convergence"] = {
        "fiber_counts": conv["fiber_counts"],
        "sigma_u_MPa": [v / 1e6 for v in conv["sigma_u"]],
        "finest_vs_coarsest": float(conv["finest_vs_coarsest"]),
        "converged": bool(conv["converged"]),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run")
    ap.add_argument("--outdir", default="artifacts/p167w2_n03")
    args = ap.parse_args()

    mat = VonMisesMaterial(E=E, nu=NU, sigma_y0=SY0, H=H)
    rec = {
        "tag": args.tag,
        "material": {"E": E, "nu": NU, "sigma_y0": SY0, "H": H},
        "uniaxial_3d": probe_uniaxial(mat, plane_stress=False),
        "uniaxial_plane_stress": probe_uniaxial(mat, plane_stress=True),
        "tangent_3d": fd_tangent(mat, plane_stress=False),
        "tangent_plane_stress": fd_tangent(mat, plane_stress=True),
        "smith_legacy": probe_smith(mat),
        "progressive": probe_progressive(mat),
        "icebelt": probe_icebelt(),
    }
    os.makedirs(args.outdir, exist_ok=True)
    path = os.path.join(args.outdir, f"audit_{args.tag}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2, sort_keys=True)
    print(json.dumps(rec, indent=2, sort_keys=True))
    print(f"\n[written] {path}")


if __name__ == "__main__":
    main()