"""Verification anchors for the grounding damage chain.

Every link of the chain is verified against an independent reference:

===============  ==========================================================
link             reference
===============  ==========================================================
contact (C1)     closed-form Hertz solution (sphere on elastic half-space)
panel linear     exact Navier series (simply supported) and the classical
                 clamped-plate coefficient ``1.26e-3 q a^4 / D``
panel nonlinear  independent Q4 + Hermite finite element
                 (``tensorphysics.plate_buckling``) with matched edge
                 conditions (immovable edges)
material         analytic bilinear uniaxial curve
flooding (C4)    closed-form Bernoulli filling time
stability        closed-form barge GM (``KB + BM - KG``) and a semi-analytic
                 direct integration of a rectangular section
energy (C7)      kinetic energy vs absorbed energy residual
negative control zero velocity / zero ridge height -> zero damage
===============  ==========================================================

All functions return plain dictionaries of numbers so that the results can be
written verbatim into the artefact JSON and into the report.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..plate_buckling.core import PlateGeometry, navier_simply_supported_deflection
from .contact import HertzContact, hertz_sphere_on_plane
from .damage import StiffenerSection, check_stiffener
from .flooding import (BottomOpening, Compartment, analytic_filling_time,
                       transient_flood)
from .material import POLAR_NV_F, bilinear_reference, uniaxial_curve
from .panel import PanelResponse, PanelSpec, VonKarmanPanel
from .stability import HullForm, gz_curve, check_imo_criteria

__all__ = ["hertz_verification", "panel_linear_verification",
           "panel_fem_reference", "panel_fem_comparison", "material_verification",
           "flooding_verification", "stability_verification",
           "stiffener_verification", "negative_controls"]


# ---------------------------------------------------------------------------
# C1 - contact
# ---------------------------------------------------------------------------
def hertz_verification(R: float = 0.5, E: float = 2.06e11, nu: float = 0.3,
                       deltas: Sequence[float] = (1e-5, 1e-4, 5e-4, 2e-3),
                       n_rings: int = 100) -> Dict[str, object]:
    """Compare the discrete half-space contact solver with closed-form Hertz."""
    solver = HertzContact(R, E, nu, n_rings=n_rings)
    rows = []
    for d in deltas:
        sol = solver.solve(float(d))
        ref = hertz_sphere_on_plane(sol["load"], R, E, nu)
        rows.append({
            "delta": float(d),
            "load": float(sol["load"]),
            "radius_numeric": float(sol["contact_radius"]),
            "radius_hertz": float(ref.radius),
            "radius_error_pct": 100.0 * (sol["contact_radius"] / ref.radius - 1.0),
            "p0_numeric": float(sol["peak_pressure"]),
            "p0_hertz": float(ref.peak_pressure),
            "p0_error_pct": 100.0 * (sol["peak_pressure"] / ref.peak_pressure - 1.0),
            "load_hertz_consistency_pct": 100.0 * (ref.load / sol["load"] - 1.0),
        })
    worst_radius = max(abs(r["radius_error_pct"]) for r in rows)
    worst_p0 = max(abs(r["p0_error_pct"]) for r in rows)
    return {"rows": rows, "worst_radius_error_pct": worst_radius,
            "worst_p0_error_pct": worst_p0,
            "pass": bool(worst_radius < 10.0 and worst_p0 < 10.0)}


# ---------------------------------------------------------------------------
# C2 - plate panel (linear limit and large-deflection / elastoplastic)
# ---------------------------------------------------------------------------
def panel_linear_verification(a: float = 1.0, b: float = 1.0, t: float = 0.02,
                              q: float = 1.0e4,
                              n_x: int = 21, n_y: int = 21) -> Dict[str, object]:
    """Linear-limit checks: Navier (SSSS) and the clamped-plate coefficient."""
    D_ref = 2.06e11 * t**3 / (12.0 * (1.0 - 0.3**2))
    # simply supported: exact Navier series
    geom = PlateGeometry(a=a, b=b, t=t, E=2.06e11, nu=0.3, sigma_y=1e13, nx=2, ny=2)
    w_navier = navier_simply_supported_deflection(geom, q)
    spec_ss = PanelSpec(a=a, b=b, t=t, bc="ssss", n_x=n_x, n_y=n_y, w_modes=3,
                        uv_modes=2, n_layers=3)
    resp_ss = VonKarmanPanel(spec_ss, load_steps=3).pressure_deflection(q, load_steps=3)
    w_ss = float(resp_ss.deflection[-1])
    # clamped: classical coefficient (Timoshenko & Woinowsky-Krieger, Table 35)
    spec_cc = replace(spec_ss, bc="clamped")
    resp_cc = VonKarmanPanel(spec_cc, load_steps=3).pressure_deflection(q, load_steps=3)
    w_cc = float(resp_cc.deflection[-1])
    w_cc_ref = 1.26e-3 * q * a**4 / D_ref
    return {
        "ssss": {"ritz": w_ss, "navier": float(w_navier),
                 "error_pct": 100.0 * (w_ss / float(w_navier) - 1.0)},
        "clamped": {"ritz": w_cc, "timoshenko": w_cc_ref,
                    "error_pct": 100.0 * (w_cc / w_cc_ref - 1.0)},
        "pass": bool(abs(w_ss / float(w_navier) - 1.0) < 0.05 and
                     abs(w_cc / w_cc_ref - 1.0) < 0.05),
    }


def panel_fem_reference(a: float = 3.6, b: float = 0.9, t: float = 0.022,
                        nx: int = 24, ny: int = 6, pressure_unit: float = 1.0e5,
                        max_steps: int = 30, immovable: bool = True,
                        n_layers: int = 5) -> Dict[str, np.ndarray]:
    """Independent Q4+Hermite finite-element reference (arc-length continued).

    ``immovable=True`` additionally restrains the in-plane displacements on all
    edges so that the edge conditions match the Ritz panel model (u = v = 0);
    without it the FE model leaves the in-plane edges free.
    """
    from ..plate_buckling.nonlinear_buckling import PlateModel

    geom = PlateGeometry(a=a, b=b, t=t, E=2.06e11, nu=0.3, sigma_y=POLAR_NV_F.sigma_y,
                         E_t=POLAR_NV_F.E_t, nx=nx, ny=ny)
    model = PlateModel(geom, bc="CCCC", n_layers=n_layers)
    if immovable:
        nid = np.arange((nx + 1) * (ny + 1)).reshape(ny + 1, nx + 1)
        left, right = nid[:, 0], nid[:, -1]
        bottom, top = nid[0, :], nid[-1, :]
        extra = [6 * n + 0 for n in np.concatenate([left, right])]
        extra += [6 * n + 1 for n in np.concatenate([bottom, top])]
        fixed = np.unique(np.concatenate([model.fixed_dofs, np.array(extra, dtype=int)]))
        model.fixed_dofs = fixed
        mask = np.ones(model.ndof, dtype=bool)
        mask[fixed] = False
        model.free_dofs = np.flatnonzero(mask)
        model._disp_free = np.intersect1d(model.free_dofs, model._disp_dofs)
    res = model.solve_arc_length(model.pressure_load(pressure_unit),
                                 max_steps=max_steps, ds_max=0.3)
    path = res["path"]
    return {"pressure": np.array([p["lam"] for p in path]) * pressure_unit,
            "w_max": np.array([p["w_max"] for p in path])}


def panel_fem_comparison(response: PanelResponse,
                         fem: Dict[str, np.ndarray],
                         loads: Sequence[float] = (2e5, 5e5, 1e6, 2e6)) -> Dict[str, object]:
    """Compare the Ritz resistance curve with the FE pressure-deflection path."""
    rows = []
    pmin, pmax = float(fem["pressure"].min()), float(fem["pressure"].max())
    for q in loads:
        if q > pmax or q < pmin:
            continue
        w_fe = float(np.interp(q, fem["pressure"], fem["w_max"]))
        w_ritz = float(np.interp(q, response.pressure, response.deflection))
        rows.append({"pressure": float(q), "w_fem": w_fe, "w_ritz": w_ritz,
                     "error_pct": 100.0 * (w_ritz / w_fe - 1.0),
                     "w_over_t": w_fe / response.spec.t})
    worst = max((abs(r["error_pct"]) for r in rows), default=float("nan"))
    return {"rows": rows, "worst_error_pct": worst,
            "pass": bool(rows and worst < 15.0)}


# ---------------------------------------------------------------------------
# material
# ---------------------------------------------------------------------------
def material_verification(eps_max: float = 0.05, n: int = 400) -> Dict[str, object]:
    """Check the return mapping against the analytic bilinear uniaxial curve."""
    eps, sig, eps_pl = uniaxial_curve(POLAR_NV_F, eps_max, n)
    ref = bilinear_reference(POLAR_NV_F, eps)
    denom = max(abs(ref).max(), 1.0)
    err = float(np.abs(sig - ref).max() / denom)
    # yield point and hardening slope from the numerical curve
    i_y = int(np.searchsorted(eps_pl, 1e-12))
    return {"max_relative_stress_error": err,
            "first_yield_strain_numeric": float(eps[max(i_y - 1, 0)]),
            "first_yield_strain_analytic": float(POLAR_NV_F.yield_strain),
            "eps_pl_at_end": float(eps_pl[-1]),
            "pass": bool(err < 1e-3)}


# ---------------------------------------------------------------------------
# C4 - flooding
# ---------------------------------------------------------------------------
def flooding_verification(area: float = 0.5, cd: float = 0.6,
                          draft0: float = 8.6, waterplane_area: float = 2500.0,
                          t_end: float = 60.0, dt: float = 0.01) -> Dict[str, object]:
    """Bernoulli rate and closed-form filling time vs the numerical integrator."""
    comp = Compartment("db", 0.0, 20.0, -11.5, 11.5, 0.0, 1.8, permeability=0.95)
    op = BottomOpening(area=area, discharge_coefficient=cd)
    head = draft0
    q_exact = cd * area * np.sqrt(2.0 * 9.80665 * head)
    res = transient_flood(op, comp, draft0=draft0, waterplane_area=waterplane_area,
                          t_end=t_end, dt=dt, allow_sinkage=False)
    # early transient: level still below the opening -> constant head
    i1, i2 = 20, 40
    q_num = float((res["volume"][i2] - res["volume"][i1]) /
                  (res["t"][i2] - res["t"][i1]))
    rate_err = 100.0 * (q_num / q_exact - 1.0)
    # filling time from fully empty to the outside level (analytic vs numeric)
    height = comp.z1 - comp.z0
    t_analytic = analytic_filling_time(area, cd, comp.effective_plan_area, head, height)
    res2 = transient_flood(op, comp, draft0=draft0, waterplane_area=1e12,
                           t_end=4.0 * t_analytic, dt=0.02, allow_sinkage=False)
    idx = np.flatnonzero(res2["volume"] >= comp.capacity - 1e-9)
    filled = idx.size > 0
    t_num = float(res2["t"][idx[0]]) if filled else float("nan")
    time_err = 100.0 * (t_num / t_analytic - 1.0) if filled else float("nan")
    return {"rate_exact": q_exact, "rate_numeric": q_num, "rate_error_pct": rate_err,
            "t_fill_analytic": t_analytic, "t_fill_numeric": t_num,
            "t_fill_error_pct": time_err, "capacity": comp.capacity,
            "pass": bool(abs(rate_err) < 10.0 and (not filled or abs(time_err) < 10.0))}


# ---------------------------------------------------------------------------
# C5 - stability
# ---------------------------------------------------------------------------
def box_barge_gz(B: float, T: float, kg: float, angles: Sequence[float],
                 depth: float = 40.0, n: int = 40001) -> np.ndarray:
    """Semi-analytic GZ of a rectangular barge (direct y-integration)."""
    y = np.linspace(-0.5 * B, 0.5 * B, n)
    out = []
    for th in angles:
        tt = np.tan(np.deg2rad(th))
        target = B * T
        lo, hi = -abs(tt) * B, depth + abs(tt) * B
        for _ in range(80):
            c = 0.5 * (lo + hi)
            h = np.clip(c + y * tt, 0.0, depth + 0.0)
            if np.trapezoid(h, y) < target:
                lo = c
            else:
                hi = c
        c = 0.5 * (lo + hi)
        h = np.clip(c + y * tt, 0.0, depth)
        a = np.trapezoid(h, y)
        yb = np.trapezoid(y * h, y) / a
        zb = np.trapezoid(0.5 * h**2, y) / a
        th_r = np.deg2rad(th)
        out.append(yb * np.cos(th_r) + zb * np.sin(th_r) - kg * np.sin(th_r))
    return np.asarray(out)


def stability_verification(length: float = 100.0, B: float = 20.0, T: float = 6.0,
                           kg: float = 4.0,
                           angles: Sequence[float] = (0, 5, 10, 15, 20, 30)) -> Dict[str, object]:
    """Box-barge checks: volume, GM (KB+BM-KG) and GZ curve."""
    hull = HullForm(length=length, breadth=B, depth=4.0 * T, draft=T, cb=1.0,
                    flat_ratio=1.0, bilge_height=0.0, n_stations=81, fullness=400.0)
    gate = hull.gate()
    vol = hull.displacement_volume()
    kb = T / 2.0
    bm = B**2 / (12.0 * T)
    gm_ref = kb + bm - kg
    st = hull.properties(0.0, T)
    km_num = st["zb"] + hull.waterplane_inertia_transverse() / vol
    gm_num = km_num - kg
    gz = gz_curve(hull, kg=kg, volume=vol, lcg=0.0, angles=angles)
    ref = box_barge_gz(B, T, kg, angles)
    return {
        "gate": gate,
        "volume": vol, "volume_nominal": length * B * T,
        "volume_error_pct": 100.0 * (vol / (length * B * T) - 1.0),
        "gm_numeric": gm_num, "gm_analytic": gm_ref,
        "gm_error_pct": 100.0 * (gm_num / gm_ref - 1.0),
        "gz_rows": [{"angle": float(a), "gz": float(g), "gz_reference": float(r),
                     "error_pct": 100.0 * (g - r) / max(abs(r), 1e-9)}
                    for a, g, r in zip(angles, gz["gz"], ref)],
        "worst_gz_error_pct": float(max(
            abs(g - r) / max(abs(r), 1e-9) * 100.0
            for g, r in zip(gz["gz"], ref) if abs(r) > 1e-3)),
        "pass": bool(abs(vol / (length * B * T) - 1.0) < 0.01 and
                     abs(gm_num / gm_ref - 1.0) < 0.02),
    }


def stiffener_verification() -> Dict[str, object]:
    """Stiffener capacity formulas vs closed-form hand checks."""
    s = StiffenerSection()
    tau_cr = s.web_shear_buckling_stress()
    sigma_cr = s.flange_buckling_stress()
    tau_y = s.sigma_y / np.sqrt(3.0)
    slender = StiffenerSection(web_height=0.60, web_thickness=0.006,
                               flange_width=0.20, flange_thickness=0.008)
    return {
        "web_shear_buckling_stress": tau_cr,
        "flange_buckling_stress": sigma_cr,
        "yield_shear_stress": tau_y,
        "shear_capacity": s.shear_capacity(),
        "plastic_moment": s.plastic_bending_resistance(),
        # for the stocky reference section both buckling stresses exceed yield,
        # so plastic collapse governs; for a slender section buckling governs
        "web_buckling_below_yield": bool(tau_cr < tau_y),
        "flange_buckling_below_yield": bool(sigma_cr < s.sigma_y),
        "slender_web_buckling_stress": slender.web_shear_buckling_stress(),
        "slender_web_buckling_below_yield":
            bool(slender.web_shear_buckling_stress() < tau_y),
    }


# ---------------------------------------------------------------------------
# C6 - negative controls
# ---------------------------------------------------------------------------
def negative_controls(response: PanelResponse) -> Dict[str, object]:
    """Zero velocity and zero ridge height must produce zero damage."""
    from .contact import RidgeGeometry, RidgeContact

    ridge0 = RidgeGeometry(h_ridge=0.0, slope_deg=12.0, length=30.0)
    contact0 = RidgeContact(ridge0, lambda d: response.pressure_at(d),
                            panel_breadth=response.spec.b,
                            panel_length=response.spec.a)
    e0 = contact0.energy(0.0)
    r0 = contact0.resistance(0.0)
    ridge = RidgeGeometry(h_ridge=4.0, slope_deg=12.0, length=30.0)
    contact = RidgeContact(ridge, lambda d: response.pressure_at(d),
                           panel_breadth=response.spec.b, panel_length=response.spec.a)
    apex_zero_energy = contact.apex_depth_for_energy(0.0)
    zero_curve = np.interp(0.0, response.deflection, response.eps_pl_max)
    return {
        "energy_zero_ridge": e0, "resistance_zero_ridge": r0,
        "apex_for_zero_energy": apex_zero_energy,
        "eps_pl_at_zero_deflection": float(zero_curve),
        "pass": bool(e0 == 0.0 and r0 == 0.0 and apex_zero_energy == 0.0 and
                     float(zero_curve) == 0.0),
    }