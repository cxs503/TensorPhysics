"""Diff-report: consolidated criteria JSON + publication figures.

The report is *generated*, never hand-written: every number in
``criteria.json`` comes from the per-case JSONs produced by
:mod:`tensorphysics.lbm_bem_cross.crossval`, and every figure is drawn from the
same files (no digit typing anywhere).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

__all__ = ["build_report", "figure_cp", "figure_cd", "figure_far_field",
           "figure_field", "figure_wake_band"]

CASES = ("re100", "re1000", "re10000")


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _load(art: Path, name: str) -> dict:
    p = art / f"cross_{name}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def figure_cp(art: Path, name: str = "re100") -> Path | None:
    """Cp(theta): LBM wall pressure vs BEM+BL vs inviscid potential."""
    data = _load(art, name)
    if not data or not data.get("lbm_solution_valid"):
        return None
    plt = _mpl()
    th = np.asarray(data["theta_wall_deg"])
    cp_lbm = np.asarray(data["cp_lbm"])
    cp_bem = np.asarray(data["cp_bem"])
    cp_pot = np.asarray(data.get("cp_potential", []))
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=140)
    ax.plot(th, cp_lbm, "o-", ms=3, lw=1.4, label="LBM (Navier-Stokes, real solver)")
    ax.plot(th, cp_bem, "-", lw=1.6, label="BEM + integral BL (Thwaites/VPI)")
    if cp_pot.size:
        ax.plot(data["theta_wall_deg"], cp_pot, "--", lw=1.0, label="BEM potential (inviscid)")
    ax.axvspan(0, 30, color="0.85", alpha=0.5, label="criterion window 0+-30 deg")
    ax.axvspan(330, 360, color="0.85", alpha=0.5)
    ax.set_xlabel(r"$\theta$ from front stagnation point [deg]")
    ax.set_ylabel(r"$C_p$")
    ax.set_title(f"Surface pressure, cylinder $Re$ = {data['re']:.0f}")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = art / f"fig_cp_{name}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_cd(art: Path) -> Path | None:
    """Cd: LBM vs BEM+BL vs literature anchor, with the wake-closure band."""
    plt = _mpl()
    cases = [c for c in CASES if (art / f"cross_{c}.json").exists()]
    if not cases:
        return None
    fig, ax = plt.subplots(figsize=(6.6, 4.2), dpi=140)
    x = np.arange(len(cases))
    w = 0.26
    cd_lbm, cd_bem, cd_ref, band_lo, band_hi = [], [], [], [], []
    for c in cases:
        d = _load(art, c)
        cd_lbm.append(d["cd"].get("cd_lbm") if d.get("lbm_solution_valid") else np.nan)
        cd_bem.append(d["cd"]["cd_bem_bl"])
        cd_ref.append(d["cd"]["cd_reference"])
        band = d.get("vpi_band", {})
        if band.get("cd"):
            band_lo.append(min(band["cd"]))
            band_hi.append(max(band["cd"]))
        else:
            band_lo.append(np.nan)
            band_hi.append(np.nan)
    cd_lbm = np.asarray(cd_lbm, float)
    ax.bar(x - w, cd_lbm, w, label="LBM (Navier-Stokes)", color="tab:blue")
    ax.bar(x, cd_bem, w, label="BEM + boundary layer", color="tab:orange")
    ax.bar(x + w, cd_ref, w, label="literature anchor", color="0.6")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Re={_load(art, c)['re']:.0f}" for c in cases])
    ax.set_ylabel(r"$C_D$")
    ax.set_title("Cylinder drag: two solvers and the reference anchor")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    out = art / "fig_cd.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_wake_band(art: Path) -> Path | None:
    """Sensitivity of the BEM+BL drag to the wake-pressure closure."""
    plt = _mpl()
    cases = [c for c in CASES if (art / f"cross_{c}.json").exists()]
    if not cases:
        return None
    fig, ax = plt.subplots(figsize=(6.2, 4.0), dpi=140)
    for c in cases:
        d = _load(art, c)
        band = d.get("vpi_band", {})
        if not band.get("cd"):
            continue
        ax.plot(band["cp_wake"], band["cd"], "o-", ms=4,
                label=f"Re={d['re']:.0f} (cp_sep={d['vpi']['cp_wake']:.2f})")
        ax.axhline(d["cd"]["cd_reference"], ls=":", lw=1.0,
                   color=ax.lines[-1].get_color())
    ax.set_xlabel(r"wake pressure $C_{p,\mathrm{wake}}$")
    ax.set_ylabel(r"$C_D$ (BEM + BL)")
    ax.set_title("Wake-closure sensitivity (dotted: literature anchor)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    out = art / "fig_wake_band.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_far_field(art: Path, name: str = "re100") -> Path | None:
    """Far-field disturbance decay: |u'|/U_inf versus r/D (log-log)."""
    data = _load(art, name)
    if not data or not data.get("lbm_solution_valid"):
        return None
    ff = data["far_field"]
    bem = json.loads((art / f"bem_{name}.json").read_text())
    npz = np.load(art / f"lbm_{name}.npz")
    from .crossval import bilinear_sample
    cx0, cy0 = float(npz["cx0"]), float(npz["cy0"])
    d_lat = float(npz["diameter"])
    u = npz["u"].astype(float)
    v = npz["v"].astype(float)
    uin = float(npz["u_in"])
    xs = cx0 + np.asarray(bem["far_field_samples"]["x"]) * d_lat
    ys = cy0 + np.asarray(bem["far_field_samples"]["y"]) * d_lat
    ul = bilinear_sample(u, xs, ys)
    vl = bilinear_sample(v, xs, ys)
    up = np.asarray(bem["far_field_samples"]["u_potential"]) * uin
    vp = np.asarray(bem["far_field_samples"]["v_potential"]) * uin
    r = np.hypot(xs - cx0, ys - cy0) / d_lat
    amp_l = np.hypot(ul - uin, vl) / uin
    amp_p = np.hypot(up - uin, vp) / uin

    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), dpi=140)
    ax = axes[0]
    ax.loglog(r, amp_p, "s", ms=3, alpha=0.6, label="BEM potential")
    ax.loglog(r, amp_l, "o", ms=3, alpha=0.6, label="LBM")
    ax.set_xlabel(r"$r/D$")
    ax.set_ylabel(r"$|u'|/U_\infty$")
    ax.set_title(f"decay exponents: LBM {ff['decay_exponent_lbm']:.2f}, "
                 f"potential {ff['decay_exponent_potential']:.2f}")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    ax = axes[1]
    dev = np.hypot(ul - up, vl - vp) / uin
    sc = ax.scatter(r, np.degrees(np.arctan2(ys - cy0, xs - cx0)), c=dev,
                    s=8, cmap="viridis", vmin=0.0, vmax=max(0.05, float(dev.max())))
    fig.colorbar(sc, ax=ax, label=r"$|u-u_{pot}|/U_\infty$")
    ax.set_xlabel(r"$r/D$")
    ax.set_ylabel("polar angle [deg] (0 = downstream)")
    ax.set_title(f"deviation, L2 = {ff['l2_deviation']:.3f}  (tol 0.05)")
    out = art / f"fig_far_field_{name}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def figure_field(art: Path, name: str = "re100") -> Path | None:
    """LBM velocity field with the body, the domain and the far-field sample ring."""
    p = art / f"lbm_{name}.npz"
    if not p.exists():
        return None
    npz = np.load(p)
    u = npz["u"].astype(float)
    v = npz["v"].astype(float)
    if not np.isfinite(u).all():
        return None
    cx0, cy0, d_lat = float(npz["cx0"]), float(npz["cy0"]), float(npz["diameter"])
    uin = float(npz["u_in"])
    nx = u.shape[1]
    x = np.arange(nx) / d_lat - cx0 / d_lat
    y = np.arange(u.shape[0]) / d_lat - cy0 / d_lat
    plt = _mpl()
    fig, axes = plt.subplots(2, 1, figsize=(7.4, 6.4), dpi=140, sharex=True)
    ax = axes[0]
    im = ax.pcolormesh(x, y, np.hypot(u, v) / uin, cmap="viridis", shading="auto")
    fig.colorbar(im, ax=ax, label=r"$|U|/U_\infty$")
    for rr in (10.0, 11.5):
        circ = plt.Circle((0, 0), rr, fill=False, ls=":", lw=1.0, color="w")
        ax.add_patch(circ)
    ax.set_aspect("equal")
    ax.set_title(f"LBM velocity magnitude, Re={float(npz['re']):.0f}, D={d_lat:.0f} nodes "
                 "(dotted: far-field sample radii)")
    ax.set_ylabel(r"$y/D$")
    ax = axes[1]
    vort = np.zeros_like(u)
    vort[1:-1, 1:-1] = (v[2:, 1:-1] - v[:-2, 1:-1]) - (u[1:-1, 2:] - u[1:-1, :-2])
    lim = np.percentile(np.abs(vort), 99.0)
    im = ax.pcolormesh(x, y, vort, cmap="RdBu_r", vmin=-lim, vmax=lim, shading="auto")
    fig.colorbar(im, ax=ax, label=r"$\omega_z$ [lattice]")
    ax.set_aspect("equal")
    ax.set_title("vorticity (central differences)")
    ax.set_xlabel(r"$x/D$ (flow left to right)")
    ax.set_ylabel(r"$y/D$")
    out = art / f"fig_field_{name}.png"
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def build_report(art: str | Path) -> dict:
    """Build figures + the consolidated criteria JSON."""
    art = Path(art)
    art.mkdir(parents=True, exist_ok=True)
    figs = {}
    for fn, key in ((figure_cp, "cp"), (figure_cd, "cd"), (figure_far_field, "far_field"),
                    (figure_field, "field"), (figure_wake_band, "wake_band")):
        try:
            out = fn(art)
        except Exception as exc:  # a missing artifact must not break the report
            figs[key] = {"error": str(exc)}
            continue
        figs[key] = None if out is None else str(out.relative_to(art))

    cases = {}
    for name in CASES:
        d = _load(art, name)
        if not d:
            cases[name] = {"status": "missing"}
            continue
        cases[name] = {
            "re": d["re"],
            "scenario": d.get("scenario"),
            "lbm_solution_valid": d.get("lbm_solution_valid"),
            "lbm_status": d.get("lbm_status", "ok"),
            "cd": d["cd"],
            "cp_front": d.get("cp_front"),
            "cp_full_surface": d.get("cp_full_surface"),
            "far_field": d.get("far_field"),
            "separation": d.get("separation"),
            "vpi": d.get("vpi"),
            "vpi_band": d.get("vpi_band"),
            "criteria": d["criteria"],
        }
    crit = {
        "campaign_node": "p84w3-n10-lbm-bem-cross-validation",
        "repo": "TensorPhysics",
        "module": "src/tensorphysics/lbm_bem_cross",
        "figures": figs,
        "cases": cases,
    }
    # aggregate verdict per criterion
    agg = {}
    for key in ("1a_cd_deviation", "1b_cd_vs_literature", "2_cp_front", "3_far_field",
                "4_geometry_contract"):
        vals = [c.get("criteria", {}).get(key, {}) for c in cases.values()]
        agg[key] = {
            "passed_all_adjudicable": bool(all(v.get("passed", False) for v in vals if v)),
            "n_adjudicable": int(sum(1 for v in vals if v.get("adjudicable", True))),
            "detail": {k: v for k, v in zip(cases, vals)},
        }
    crit["aggregate"] = agg
    (art / "criteria.json").write_text(json.dumps(crit, indent=1))
    return {"figures": figs, "aggregate": agg,
            "cases": {k: v.get("lbm_status", v.get("status")) for k, v in cases.items()}}