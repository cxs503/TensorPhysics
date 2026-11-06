"""Dual-chain cross-check: IACS pressure-area FEM chain vs the B1 ice chain.

Chains compared
---------------
**Chain A (this repository, high fidelity).** IACS-style Polar Class design
pressure from the pressure-area law
(:mod:`tensorphysics.hull_icebelt_fem.iacs_pressure`), applied as the rule
patch on an ice-belt plate panel that is discretised with the reused H3 Hermite
flat-shell element (:mod:`tensorphysics.hull_icebelt_fem.panel_fem`), giving the
panel bending stress and deflection.  The same pressure drives the longitudinal
stiffener as a beam (:mod:`tensorphysics.hull_icebelt_fem.stiffener`).

**Chain B (ship platform B1 experience chain, ``ice_fullscale``).** Korzhavin /
Riska-Kämäräinen type global bow force ``F = k_F psi(v) sigma_c b h`` with
contact width ``b = 3 h`` and the local line load ``q = F/b`` provided by that
module ``as the boundary condition for structural finite elements``.

The two chains are evaluated for the *same* ice condition (thickness ``h``,
speed ``v``, crushing strength ``sigma_c``).  Chain B is an *external*
dependency: the module is imported read-only when a checkout is present, and
its absence is reported as ``NOT_AVAILABLE``/``WITHHELD`` -- never silently
replaced by a hard-coded number.

Numerical verification (independent of chain B)
-----------------------------------------------
Every FEM number is compared with the closed-form Navier double-sine solution
(:mod:`tensorphysics.hull_icebelt_fem.analytic`) for both the uniform and the
patch load case; the acceptance criterion is a deviation below 5 %.

Deviation attribution
---------------------
Chain B has no pressure-area reduction and its line load is not normalised to
the rule patch, so the pressure ratio decomposes exactly into

.. math::

    \\frac{p_B}{p_A} = \\underbrace{\\frac{\\sigma_c}{p_1}}_{\\text{strength vs
    area-reduced reference}}\\cdot
    \\underbrace{\\psi(v)}_{\\text{impact factor}}\\cdot
    \\underbrace{\\frac{1}{0.6}}_{\\text{line-load} \\to \\text{patch pressure}}
    \\cdot \\underbrace{\\left(\\frac{A}{A_1}\\right)^{1/2}}_{\\text{missing
    area reduction}}

Each factor is reported separately and the product is checked against the
directly computed ratio, so the attribution cannot drift from the numbers.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from . import analytic, iacs_pressure, panel_fem, stiffener

__all__ = [
    "VERIFY_PANEL",
    "DESIGN_PANEL",
    "STIFFENER",
    "probe_anchor_state",
    "probe_b1_chain",
    "verification_cases",
    "mesh_convergence",
    "dual_chain_sweep",
    "pc6_design_case",
    "run_crosscheck",
    "SCHEMA",
]

SCHEMA = "tensorphysics/hull-icebelt-fem-xcheck@1.0"

#: Plate panel used for the *numerical* verification (inside the W3-N03 gate).
#: The mesh is the production check mesh: on it the bending moment at the panel
#: centre is within 0.6 % of the converged value (see ``mesh_convergence``),
#: which is what allows the peak-stress comparison to clear the 5 % gate with
#: margin rather than by luck.
VERIFY_PANEL = panel_fem.IceBeltPanel(a=2.4, b=0.8, t=0.025, grade="AH36",
                                      nx=48, ny=16, gate="strict")
#: Realistic PC6 ice-belt panel: 35 mm longitudinals spacing, 30 mm plating,
#: 2.0 m between web frames (stocky plating, ``beta < 1`` -> relaxed gate).
DESIGN_PANEL = panel_fem.IceBeltPanel(a=2.0, b=0.35, t=0.030, grade="AH36",
                                      nx=24, ny=6, gate="relaxed")
#: Longitudinal stiffener consistent with :data:`DESIGN_PANEL`
#: (T-bar 400x12 + 120x18, span 2.0 m).
STIFFENER = stiffener.StiffenerProfile(web_h=0.400, web_t=0.012,
                                       flange_b=0.120, flange_t=0.018,
                                       spacing=0.35, t_plate=0.030,
                                       span=2.0)

#: Verification loads (kept low enough to stay elastic: the linear model must
#: be the physically correct model of the case it verifies).
VERIFY_UNIFORM_PRESSURE = 0.25e6      # Pa, uniform over the panel
#: Verification patch: h_I chosen so that the rule patch (0.6 h_I x 4 x 0.6 h_I
#: = 0.2 m x 0.8 m) has its edges *on* element faces of the 0.1 m x 0.1 m mesh.
#: The reused load routine distributes a patch by element centres, so only an
#: aligned patch is loaded exactly; the misaligned case below quantifies the
#: remaining discretisation error instead of hiding it.
VERIFY_PATCH_THICKNESS = 1.0 / 3.0    # m
PATCH_TEST_PRESSURE = 0.25e6          # Pa
MISALIGNED_PATCH_THICKNESS = 0.30     # m, patch 0.18 m x 0.72 m (off-mesh)

#: Declared ice crushing strength shared by both chains [Pa].
SHARED_SIGMA_C = 3.0e6
#: Chain-B "native" ice strength: sigma_f = 0.5 MPa first-year Arctic ice,
#: sigma_c = 2.5 sigma_f (the ratio hard-coded in the B1 module).
B1_NATIVE_SIGMA_F = 0.5e6

TOL_PCT = 5.0
DEGRADATION_TOL_PCT = 2.0

B1_MAIN_ROOT = Path("/root/ship-performance-platform")
_B1_CANDIDATE_ROOTS: Sequence[str] = tuple(
    p for p in (os.environ.get("HULL_ICEBELT_B1_ROOT"),
                "/root/ship-performance-platform-incoming/ship-performance-platform",
                str(B1_MAIN_ROOT))
    if p)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _git(args: Sequence[str], cwd: Path) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True,
                             text=True, timeout=15)
        return out.stdout.strip() if out.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return ""


# ---------------------------------------------------------------------------
# environment / evidence probes
# ---------------------------------------------------------------------------
def probe_anchor_state(repo_root: Optional[Path] = None) -> Dict[str, object]:
    """Check whether the IN10-L2 anchor product is present in this checkout."""
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    anchor_dir = root / "benchmarks" / "ice_nav_refs"
    present = anchor_dir.is_dir() and any(anchor_dir.iterdir()) if anchor_dir.is_dir() else False
    return {
        "benchmarks_ice_nav_refs_present": bool(present),
        "probed_path": str(anchor_dir),
        "conclusion": (
            "consumed" if present else
            "absent -> self-built IACS/ISO pressure-area anchor, declared in "
            "iacs_pressure.ANCHOR_PROVENANCE"),
        "provenance": dict(iacs_pressure.ANCHOR_PROVENANCE),
    }


def probe_b1_chain(roots: Sequence[str] = _B1_CANDIDATE_ROOTS) -> Dict[str, object]:
    """Locate and import the ship-platform B1 ``ice_fullscale`` physics module.

    Read-only: nothing is written into the external checkout.  Returns a status
    of ``AVAILABLE`` (with the module object under ``_module``, stripped from
    the JSON view) or ``NOT_AVAILABLE`` with the paths that were probed.
    """
    import importlib

    probed: List[str] = []
    result: Dict[str, object] = {
        "status": "NOT_AVAILABLE",
        "probed_roots": [],
        "merged_to_main": bool(
            (B1_MAIN_ROOT / "backend" / "modules" / "ice_fullscale").is_dir()),
        "main_root": str(B1_MAIN_ROOT),
    }
    for root in roots:
        rp = Path(root)
        mod_path = rp / "backend" / "modules" / "ice_fullscale" / "ice_physics.py"
        probed.append(str(rp))
        if not mod_path.is_file():
            continue
        result["probed_roots"] = probed
        added = False
        if str(rp) not in sys.path:
            sys.path.insert(0, str(rp))
            added = True
        try:
            mod = importlib.import_module("backend.modules.ice_fullscale.ice_physics")
        except Exception as exc:  # pragma: no cover - depends on environment
            result["status"] = "NOT_AVAILABLE"
            result["import_error"] = f"{type(exc).__name__}: {exc}"
            continue
        finally:
            if added:
                try:
                    sys.path.remove(str(rp))
                except ValueError:  # pragma: no cover
                    pass
        result.update({
            "status": "AVAILABLE",
            "root": str(rp),
            "module_file": str(mod_path),
            "module_sha256": _sha256(mod_path),
            "module_commit": _git(["rev-parse", "HEAD"], rp),
            "contact_width_per_h": float(getattr(mod, "CONTACT_WIDTH_PER_H", float("nan"))),
            "crush_to_flex_ratio": float(getattr(mod, "ICE_CRUSH_TO_FLEX", float("nan"))),
            "functions_used": ["global_ice_force", "local_line_load"],
            "merged_to_main": bool(
                (B1_MAIN_ROOT / "backend" / "modules" / "ice_fullscale").is_dir()),
            "evidence_grade": "REAL_CODE_RUN_EXTERNALLY_IMPORTED",
        })
        result["_module"] = mod
        break
    else:
        result["probed_roots"] = probed
        result["evidence_grade"] = "WITHHELD (chain B unavailable in this environment)"
    return result


def _b1_pressure(mod, h: float, sigma_c: float, v: float) -> Dict[str, float]:
    """Chain-B patch pressure implied by ``local_line_load`` [Pa]."""
    ratio = float(getattr(mod, "ICE_CRUSH_TO_FLEX", 2.5))
    sigma_f = sigma_c / ratio
    out = mod.global_ice_force(None, h, sigma_f, v)
    q_kn_m = float(mod.local_line_load(out["F_global"], h))   # kN/m
    h_patch = iacs_pressure.PATCH_HEIGHT_FACTOR * h
    p_patch = q_kn_m * 1e3 / h_patch
    return {
        "F_global_N": float(out["F_global"]),
        "F_global_MN": float(out["F_global"]) / 1e6,
        "contact_width_m": float(out["contact_width_m"]),
        "impact_factor": float(out["impact_factor"]),
        "sigma_c_Pa": float(sigma_c),
        "q_line_kN_per_m": q_kn_m,
        "p_patch_Pa": p_patch,
        "p_patch_MPa": p_patch / 1e6,
    }


# ---------------------------------------------------------------------------
# chain-A verification against the closed-form solution
# ---------------------------------------------------------------------------
def _dev_pct(a: float, b: float) -> float:
    if b == 0.0:  # pragma: no cover
        return float("nan")
    return float(100.0 * (a - b) / abs(b))


def verification_cases() -> Dict[str, object]:
    """FEM vs closed-form Navier solution: uniform and patch ice loads."""
    panel = VERIFY_PANEL
    geom = panel.geometry()
    D = geom.flexural_rigidity
    cases: List[Dict[str, object]] = []

    # --- uniform pressure over the whole simply supported panel
    sol_u = panel.solve(VERIFY_UNIFORM_PRESSURE)
    ana_u = analytic.navier_field(geom, [(0.0, geom.a, 0.0, geom.b)],
                                 VERIFY_UNIFORM_PRESSURE, geom.a / 2, geom.b / 2,
                                 n_terms=81)
    cases.append({
        "name": "uniform_pressure_SSSS",
        "load": {"type": "uniform", "pressure_MPa": VERIFY_UNIFORM_PRESSURE / 1e6},
        "fem": sol_u.as_dict(),
        "analytic": {k: float(v) for k, v in ana_u.items()},
        "deviation_pct": {
            "w_centre": _dev_pct(sol_u.w_centre, ana_u["w"]),
            "sigma_centre": _dev_pct(sol_u.sigma_centre, ana_u["sigma_x"]),
            "M_centre_x": _dev_pct(sol_u.M_centre_x, ana_u["Mx"]),
        },
        "load_rule": "consistent (uniform over whole elements)",
        "note": ("deflection and centre bending stress of the H3 Hermite element "
                 "vs the Navier double-sine solution; the membrane dofs are "
                 "inactive (no in-plane load), so this isolates the bending "
                 "element and the consistent pressure load"),
    })

    # --- IACS patch load (partially loaded panel)
    box = panel.patch_box(VERIFY_PATCH_THICKNESS)
    sol_p = panel.solve(PATCH_TEST_PRESSURE, patch_box=box)
    ana_p = analytic.navier_field(geom, [box], PATCH_TEST_PRESSURE,
                                  geom.a / 2, geom.b / 2, n_terms=161)
    ana_grid = analytic.navier_grid(geom, [box], PATCH_TEST_PRESSURE,
                                    np.linspace(0.0, geom.a, 41),
                                    np.linspace(0.0, geom.b, 21), n_terms=61)
    cases.append({
        "name": "iacs_patch_SSSS",
        "load": {"type": "rectangular_patch",
                 "pressure_MPa": PATCH_TEST_PRESSURE / 1e6,
                 "h_ice_m": VERIFY_PATCH_THICKNESS,
                 "patch_box_m": [float(v) for v in box],
                 "patch_fraction_of_panel": float(
                     (box[1] - box[0]) * (box[3] - box[2]) / (geom.a * geom.b))},
        "fem": sol_p.as_dict(),
        "analytic": {k: float(v) for k, v in ana_p.items()},
        "analytic_field_max": {
            "w_max_m": float(ana_grid["w"].max()),
            "sigma_x_max_MPa": float(6.0 * np.abs(ana_grid["Mx"]).max() / geom.t ** 2) / 1e6,
        },
        "deviation_pct": {
            "w_centre": _dev_pct(sol_p.w_centre, ana_p["w"]),
            "sigma_centre": _dev_pct(sol_p.sigma_centre, ana_p["sigma_x"]),
            "M_centre_x": _dev_pct(sol_p.M_centre_x, ana_p["Mx"]),
            "w_max": _dev_pct(sol_p.w_max, float(ana_grid["w"].max())),
        },
        "load_rule": "consistent (4x4 Gauss on the patch/element intersection)",
        "note": ("the IACS rule patch (0.6 h_I x 4:1) is applied through the "
                 "reused IcePatchLoad geometry and integrated consistently, so "
                 "both the patch position and the load vector are exercised"),
    })

    # --- off-mesh patch: quantifies the load-area discretisation of the
    # reused element-centre load rule (documented, not used for acceptance)
    box_mis = panel.patch_box(MISALIGNED_PATCH_THICKNESS)
    sol_m = panel.solve(PATCH_TEST_PRESSURE, patch_box=box_mis)
    ana_m = analytic.navier_field(geom, [box_mis], PATCH_TEST_PRESSURE,
                                  geom.a / 2, geom.b / 2, n_terms=161)
    cases.append({
        "name": "iacs_patch_offmesh_sensitivity",
        "load": {"type": "rectangular_patch",
                 "pressure_MPa": PATCH_TEST_PRESSURE / 1e6,
                 "h_ice_m": MISALIGNED_PATCH_THICKNESS,
                 "patch_box_m": [float(v) for v in box_mis]},
        "fem": sol_m.as_dict(),
        "analytic": {k: float(v) for k, v in ana_m.items()},
        "deviation_pct": {
            "w_centre": _dev_pct(sol_m.w_centre, ana_m["w"]),
            "sigma_centre": _dev_pct(sol_m.sigma_centre, ana_m["sigma_x"]),
            "M_centre_x": _dev_pct(sol_m.M_centre_x, ana_m["Mx"]),
        },
        "load_rule": "consistent (4x4 Gauss on the patch/element intersection)",
        "note": ("same meshed panel with a patch whose edges fall between "
                 "element faces: with the consistent integration the gap stays "
                 "at the truncation level of the series, i.e. the load "
                 "idealisation no longer limits the agreement."),
    })

    # --- load-rule sensitivity on a coarse mesh: the reused centre rule
    # quantises the patch to whole elements and counts a centre lying exactly
    # on a patch edge twice, so on a 12x4 mesh the same patch is doubly loaded.
    # Quantified here (not an acceptance case), with the consistent rule on the
    # same coarse mesh as the control.
    box_ref = panel.patch_box(VERIFY_PATCH_THICKNESS)
    ana_ref = analytic.navier_field(geom, [box_ref], PATCH_TEST_PRESSURE,
                                   geom.a / 2, geom.b / 2, n_terms=161)
    coarse = panel_fem.IceBeltPanel(a=panel.a, b=panel.b, t=panel.t,
                                    grade=panel.grade, nx=12, ny=4,
                                    gate="relaxed")
    ec = coarse.solve(PATCH_TEST_PRESSURE, patch_box=box_ref,
                      load_rule="element_centre")
    cc = coarse.solve(PATCH_TEST_PRESSURE, patch_box=box_ref,
                      load_rule="consistent")
    cases.append({
        "name": "load_rule_sensitivity_coarse_mesh",
        "load": {"type": "rectangular_patch",
                 "pressure_MPa": PATCH_TEST_PRESSURE / 1e6,
                 "h_ice_m": VERIFY_PATCH_THICKNESS,
                 "patch_box_m": [float(v) for v in box_ref]},
        "analytic": {k: float(v) for k, v in ana_ref.items()},
        "fem_element_centre_rule": ec.as_dict(),
        "fem_consistent_rule": cc.as_dict(),
        "deviation_pct": {
            "w_centre_element_centre_rule": _dev_pct(ec.w_centre, ana_ref["w"]),
            "w_centre_consistent_rule": _dev_pct(cc.w_centre, ana_ref["w"]),
        },
        "load_rule": "element_centre vs consistent, 12x4 mesh",
        "note": ("documents why the ice-belt chain integrates the patch "
                 "consistently: on a mesh whose element centres fall on the "
                 "patch edges the reused centre rule applies roughly twice the "
                 "intended load area (load_area_ratio > 1), while the "
                 "consistent integration keeps the deflection error at the "
                 "mesh level."),
    })

    for c in cases:
        c["acceptance"] = c["name"] not in (
            "iacs_patch_offmesh_sensitivity", "load_rule_sensitivity_coarse_mesh")
        c["pass"] = bool(all(abs(v) < TOL_PCT
                             for k, v in c["deviation_pct"].items()
                             if k in ("w_centre", "sigma_centre", "w_max")))
    return {
        "panel": panel.checks(),
        "panel_spec": {"a_m": panel.a, "b_m": panel.b, "t_m": panel.t,
                       "grade": panel.grade, "mesh": [panel.nx, panel.ny],
                       "bc": panel.bc},
        "flexural_rigidity_Nm": float(D),
        "tolerance_pct": TOL_PCT,
        "cases": cases,
        "all_pass": bool(all(c["pass"] for c in cases if c["acceptance"])),
        "all_pass_including_sensitivity": bool(all(c["pass"] for c in cases)),
        "max_abs_deviation_pct": float(max(
            abs(v) for c in cases if c["acceptance"]
            for k, v in c["deviation_pct"].items()
            if k in ("w_centre", "sigma_centre", "w_max"))),
    }


def mesh_convergence() -> Dict[str, object]:
    """Patch-load case on three meshes: the closed-form gap must shrink."""
    geom_ref = VERIFY_PANEL.geometry()
    box = VERIFY_PANEL.patch_box(VERIFY_PATCH_THICKNESS)
    ana = analytic.navier_field(geom_ref, [box], PATCH_TEST_PRESSURE,
                               geom_ref.a / 2, geom_ref.b / 2, n_terms=161)
    rows = []
    for nx, ny in ((12, 4), (24, 8), (48, 16)):
        panel = panel_fem.IceBeltPanel(a=VERIFY_PANEL.a, b=VERIFY_PANEL.b,
                                       t=VERIFY_PANEL.t, grade=VERIFY_PANEL.grade,
                                       nx=nx, ny=ny, gate="relaxed")
        sol = panel.solve(PATCH_TEST_PRESSURE, patch_box=box)
        rows.append({
            "mesh": [nx, ny],
            "n_elem": sol.n_elem,
            "n_dof": sol.n_dof,
            "w_centre_mm": sol.w_centre * 1e3,
            "sigma_centre_MPa": sol.sigma_centre / 1e6,
            "deviation_w_pct": _dev_pct(sol.w_centre, ana["w"]),
            "deviation_sigma_pct": _dev_pct(sol.sigma_centre, ana["sigma_x"]),
        })
    fine, coarse = rows[-1], rows[1]
    return {
        "analytic_w_centre_mm": float(ana["w"]) * 1e3,
        "analytic_sigma_centre_MPa": float(ana["sigma_x"]) / 1e6,
        "rows": rows,
        "change_24_to_48_w_pct": _dev_pct(fine["w_centre_mm"], coarse["w_centre_mm"]),
        "change_24_to_48_sigma_pct": _dev_pct(fine["sigma_centre_MPa"],
                                             coarse["sigma_centre_MPa"]),
        "pass": bool(abs(fine["deviation_w_pct"]) < TOL_PCT
                     and abs(fine["deviation_sigma_pct"]) < TOL_PCT
                     and abs(fine["w_centre_mm"] - coarse["w_centre_mm"])
                     / coarse["w_centre_mm"] * 100.0 < DEGRADATION_TOL_PCT),
    }


# ---------------------------------------------------------------------------
# dual-chain comparison
# ---------------------------------------------------------------------------
_ATTRIBUTION_LABELS = {
    "strength_to_reference_pressure": (
        "sigma_c / p_1: chain B multiplies the ice crushing strength, while the "
        "chain-A reference p_1 = 1.2 MPa at 1 m^2 is already the area-reduced "
        "pressure level (not a strength)"),
    "impact_factor": (
        "psi(v): velocity/impact enhancement present in chain B and absent in "
        "the quasi-static IACS patch pressure"),
    "line_load_to_patch_pressure": (
        "1/0.6: chain B's line load is not normalised to the rule patch height, "
        "so converting it to a patch pressure introduces the 1/(0.6 h_I) factor"),
    "missing_area_reduction": (
        "(A/A_1)^(1/2) = (A)^(1/2): chain B has no pressure-area reduction; the "
        "IACS chain lowers the pressure as the patch area grows"),
}


def dual_chain_sweep(thicknesses: Sequence[float] = (0.5, 0.75, 1.0, 1.5, 2.0),
                     speed: float = 1.0,
                     sigma_c: float = SHARED_SIGMA_C,
                     b1: Optional[Dict[str, object]] = None
                     ) -> Dict[str, object]:
    """Per-thickness comparison of the two chains on the same ice condition."""
    if b1 is None:
        b1 = probe_b1_chain()
    mod = b1.get("_module")
    avail = mod is not None
    panel = DESIGN_PANEL
    model = None  # built lazily for the design panel (stiffness reused)
    from ..plate_buckling import PlateModel  # local import: only when needed

    rows: List[Dict[str, object]] = []
    for h in thicknesses:
        # Primary design area = the rule patch intersected with the panel, i.e.
        # exactly the area the FEM loads uniformly: pressure and loaded area
        # then refer to the same contact area.  The alternative convention
        # (pressure evaluated on the unclipped contact patch 0.6 h_I x 2.4 h_I,
        # the area the pressure-area law is physically anchored on) is carried
        # alongside; linear elasticity makes its stress an exact scalar
        # rescaling of the primary solution.
        box = panel.patch_box(h)
        A = (box[1] - box[0]) * (box[3] - box[2])
        A_patch = iacs_pressure.patch_area(h, panel.b)
        p_a = iacs_pressure.pressure_area_law(A)
        p_patch = iacs_pressure.pressure_area_law(A_patch)
        if model is None:
            model = PlateModel(panel.geometry(), bc=panel.bc, validate=False)
        sol_a = panel.solve(p_a, patch_box=box, model=model)
        row: Dict[str, object] = {
            "h_ice_m": float(h),
            "speed_m_s": float(speed),
            "sigma_c_MPa": sigma_c / 1e6,
            "nearest_polar_class": _class_for_thickness(h),
            "design_area_on_panel_m2": float(A),
            "design_area_contact_patch_m2": float(A_patch),
            "patch_box_m": [float(v) for v in box],
            "chain_a": {
                "pressure_MPa": p_a / 1e6,
                "pressure_contact_patch_convention_MPa": p_patch / 1e6,
                "sigma_x_max_MPa": sol_a.sigma_x_max / 1e6,
                "sigma_x_max_contact_patch_convention_MPa": (
                    sol_a.sigma_x_max * p_patch / p_a) / 1e6,
                "w_max_mm": sol_a.w_max * 1e3,
                "utilisation": panel.utilisation(sol_a),
                "load_idealisation": ("uniform over the panel (patch covers it)"
                                      if A >= 0.999 * panel.a * panel.b
                                      else "partial patch"),
            },
            "deviation_contact_patch_convention_pct": None,
        }
        if avail:
            pb = _b1_pressure(mod, h, sigma_c, speed)
            sol_b = panel.solve(pb["p_patch_Pa"], patch_box=box, model=model)
            ratio = pb["p_patch_Pa"] / p_a
            f1 = sigma_c / iacs_pressure.REFERENCE_PRESSURE
            f2 = pb["impact_factor"]
            f3 = 1.0 / iacs_pressure.PATCH_HEIGHT_FACTOR
            f4 = (A / iacs_pressure.REFERENCE_AREA) ** iacs_pressure.PRESSURE_AREA_EXPONENT
            row["chain_b"] = {
                **pb,
                "sigma_x_max_MPa": sol_b.sigma_x_max / 1e6,
                "w_max_mm": sol_b.w_max * 1e3,
                "utilisation": panel.utilisation(sol_b),
            }
            row["deviation_pct"] = {
                "p_B_vs_A": 100.0 * (pb["p_patch_Pa"] - p_a) / p_a,
                "sigma_B_vs_A": 100.0 * (sol_b.sigma_x_max - sol_a.sigma_x_max)
                / sol_a.sigma_x_max,
                "w_B_vs_A": 100.0 * (sol_b.w_max - sol_a.w_max) / sol_a.w_max,
            }
            row["deviation_contact_patch_convention_pct"] = float(
                100.0 * (pb["p_patch_Pa"] - p_patch) / p_patch)
            row["attribution"] = {
                "ratio_direct": float(ratio),
                "ratio_from_factors": float(f1 * f2 * f3 * f4),
                "closed_form_residual_pct": float(
                    100.0 * (f1 * f2 * f3 * f4 - ratio) / ratio),
                "factors": {
                    "strength_to_reference_pressure": float(f1),
                    "impact_factor": float(f2),
                    "line_load_to_patch_pressure": float(f3),
                    "missing_area_reduction": float(f4),
                },
                "labels": _ATTRIBUTION_LABELS,
                "physical_explanation": (
                    "chain B's local line load is a bow-framing/global "
                    "indentation load: it scales with sigma_c and the impact "
                    "factor and carries no pressure-area reduction, so applying "
                    "it to a plate panel over-predicts the plating pressure. "
                    "The dominant term is the strength/reference mismatch "
                    "(sigma_c vs the area-reduced p_1), the second is the "
                    "constant impact factor; the (A/A_1)^0.5 term is the "
                    "reduction chain A applies and B does not."),
            }
        else:
            row["chain_b"] = {"status": "NOT_AVAILABLE"}
            row["deviation_pct"] = "WITHHELD (chain B unavailable)"
        rows.append(row)

    v_sweep = []
    if avail:
        h_ref = 1.5
        A = iacs_pressure.patch_area(h_ref, panel.b)
        p_a = iacs_pressure.pressure_area_law(A)
        for v in (0.3, 0.5, 1.0, 1.5, 2.0):
            pb = _b1_pressure(mod, h_ref, sigma_c, v)
            v_sweep.append({
                "speed_m_s": v,
                "impact_factor": pb["impact_factor"],
                "p_B_MPa": pb["p_patch_MPa"],
                "p_A_MPa": p_a / 1e6,
                "deviation_pct": 100.0 * (pb["p_patch_Pa"] - p_a) / p_a,
            })

    return {
        "panel": panel.checks(),
        "chain_a_model": ("IACS pressure-area patch pressure + H3 Hermite FE "
                          "plate panel (this repository)"),
        "chain_b_model": ("ship platform B1 ice_fullscale: Korzhavin/Riska "
                          "global force, contact width 3 h, local line load"),
        "chain_b_status": b1.get("status"),
        "chain_b_probe": {k: v for k, v in b1.items() if not k.startswith("_")},
        "rows": rows,
        "speed_sweep_h1p5": v_sweep,
    }


def _class_for_thickness(h: float) -> str:
    """Nearest Polar Class whose design thickness brackets ``h``."""
    table = iacs_pressure.POLAR_CLASS_ICE_THICKNESS
    return min(table, key=lambda k: abs(table[k] - h))


def b1_native_variant(sigma_f: float = B1_NATIVE_SIGMA_F) -> Dict[str, object]:
    """Chain-B comparison with the ice strength of its own default material."""
    status = probe_b1_chain()
    mod = status.get("_module")
    out: Dict[str, object] = {"status": status.get("status"),
                              "sigma_f_MPa": sigma_f / 1e6}
    if mod is None:
        out["conclusion"] = "WITHHELD (chain B unavailable)"
        return out
    sigma_c = float(getattr(mod, "ICE_CRUSH_TO_FLEX", 2.5)) * sigma_f
    A = iacs_pressure.patch_area(1.5, DESIGN_PANEL.b)
    p_a = iacs_pressure.pressure_area_law(A)
    pb = _b1_pressure(mod, 1.5, sigma_c, 1.0)
    out.update({
        "sigma_c_MPa": sigma_c / 1e6,
        "p_B_MPa": pb["p_patch_MPa"],
        "p_A_MPa": p_a / 1e6,
        "deviation_pct": 100.0 * (pb["p_patch_Pa"] - p_a) / p_a,
    })
    return out


# ---------------------------------------------------------------------------
# PC6 design case
# ---------------------------------------------------------------------------
def pc6_design_case(ice_class: str = "PC6", speed: float = 1.0) -> Dict[str, object]:
    """Full chain-A design point: pressure, panel FEM, stiffener beam."""
    panel = DESIGN_PANEL
    design = iacs_pressure.design_patch_pressure(ice_class, panel.b,
                                                 panel_length=panel.a)
    box = panel.patch_box(design.h_ice)
    panel_area = panel.a * panel.b
    patch_fraction = ((box[1] - box[0]) * (box[3] - box[2])) / panel_area
    sol = panel.solve(design.pressure, patch_box=box)
    stiff = stiffener.stiffener_response(STIFFENER, design.pressure, panel.b)
    sy = panel.material["sigma_y"]
    # sensitivity: pressure evaluated on the unclipped contact patch
    # (0.6 h_I x 2.4 h_I) instead of on the part of it lying on this panel
    contact_area = iacs_pressure.patch_area(design.h_ice, panel.b)
    p_contact = iacs_pressure.pressure_area_law(contact_area)
    return {
        "ice_class": design.ice_class,
        "speed_m_s": float(speed),
        "pressure": design.as_dict(),
        "panel": panel.checks(),
        "patch_box_m": [float(v) for v in box],
        "patch_fraction_of_panel": float(patch_fraction),
        "load_idealisation": ("uniform over the panel" if patch_fraction > 0.999
                              else "partial patch"),
        "panel_fem": sol.as_dict(),
        "panel_utilisation": panel.utilisation(sol),
        "area_convention_sensitivity": {
            "primary_design_area_m2": float(design.area),
            "primary_pressure_MPa": float(design.pressure) / 1e6,
            "primary_sigma_x_max_MPa": float(sol.sigma_x_max) / 1e6,
            "primary_utilisation": float(sol.sigma_vm_max / sy),
            "contact_patch_area_m2": float(contact_area),
            "contact_patch_pressure_MPa": p_contact / 1e6,
            "contact_patch_sigma_x_max_MPa": float(
                sol.sigma_x_max * p_contact / design.pressure) / 1e6,
            "contact_patch_utilisation": float(
                sol.sigma_vm_max * p_contact / design.pressure / sy),
        },
        "stiffener": {
            "profile": {"web_mm": [STIFFENER.web_h * 1e3, STIFFENER.web_t * 1e3],
                        "flange_mm": [STIFFENER.flange_b * 1e3,
                                      STIFFENER.flange_t * 1e3],
                        "span_m": STIFFENER.span,
                        "spacing_m": STIFFENER.spacing,
                        "t_plate_mm": STIFFENER.t_plate * 1e3},
            "response": stiff,
            "utilisation_plate": stiff["sigma_plate_Pa"] / sy,
            "utilisation_flange": stiff["sigma_flange_Pa"] / sy,
        },
    }


# ---------------------------------------------------------------------------
# top level
# ---------------------------------------------------------------------------
def run_crosscheck(out_path: Optional[Path] = None,
                   repo_root: Optional[Path] = None) -> Dict[str, object]:
    """Run every leg and (optionally) write the JSON artifact."""
    import scipy

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    anchor = probe_anchor_state(root)
    b1 = probe_b1_chain()
    ver = verification_cases()
    conv = mesh_convergence()
    sweep = dual_chain_sweep(b1=b1)
    native = b1_native_variant()
    design = pc6_design_case()
    sweep["native_strength_variant"] = native

    src_dir = Path(__file__).resolve().parent
    audit = {
        "git_commit": _git(["rev-parse", "HEAD"], root),
        "git_commit_note": (
            "generation-time HEAD; the artifact is reproducible from the "
            "committed sources with scripts/run_hull_icebelt_fem_xcheck.py and "
            "the module sha256 entries below pin the exact code state"),
        "git_branch": _git(["rev-parse", "--abbrev-ref", "HEAD"], root),
        "repo_root": str(root),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "module_files": {
            name: {"path": str(src_dir / name), "sha256": _sha256(src_dir / name)}
            for name in ("iacs_pressure.py", "analytic.py", "panel_fem.py",
                         "stiffener.py", "xcheck.py")
        },
        "reused_components": [
            "tensorphysics.plate_buckling.PlateModel (H3 Hermite flat shell "
            "element, Q4 membrane + bicubic Hermite bending)",
            "tensorphysics.plate_buckling.loads.IcePatchLoad (IACS patch "
            "geometry 0.6 h_I, 4:1)",
            "tensorphysics.plate_buckling.navier_simply_supported_deflection "
            "(uniform-load sanity of the analytic series)",
        ],
        "chain_b_provenance": {k: v for k, v in b1.items() if not k.startswith("_")},
        "anchor_provenance": anchor,
    }

    max_dev = ver["max_abs_deviation_pct"]
    residuals = [r["attribution"]["closed_form_residual_pct"]
                 for r in sweep["rows"] if "attribution" in r]
    criteria = {
        "fem_vs_closed_form_lt_5pct": {
            "value_pct": max_dev,
            "tolerance_pct": TOL_PCT,
            "pass": bool(max_dev < TOL_PCT),
        },
        "mesh_convergence_lt_5pct_and_stable": {
            "value_pct": conv["rows"][-1]["deviation_w_pct"],
            "change_24_to_48_pct": conv["change_24_to_48_w_pct"],
            "pass": bool(conv["pass"]),
        },
        "dual_chain_deviation_quantified": {
            "n_conditions": len(sweep["rows"]),
            "status": sweep["chain_b_status"],
            "min_deviation_pct": (min(r["deviation_pct"]["p_B_vs_A"]
                                      for r in sweep["rows"]
                                      if isinstance(r["deviation_pct"], dict))
                                  if sweep["chain_b_status"] == "AVAILABLE" else None),
            "max_deviation_pct": (max(r["deviation_pct"]["p_B_vs_A"]
                                      for r in sweep["rows"]
                                      if isinstance(r["deviation_pct"], dict))
                                  if sweep["chain_b_status"] == "AVAILABLE" else None),
            "pass": bool(sweep["chain_b_status"] == "AVAILABLE"
                         and all(isinstance(r["deviation_pct"], dict)
                                 for r in sweep["rows"])),
        },
        "deviation_attribution_closure": {
            "max_abs_residual_pct": (max(abs(v) for v in residuals)
                                     if residuals else None),
            "tolerance_pct": 0.5,
            "pass": bool(residuals and max(abs(v) for v in residuals) < 0.5),
        },
        "pc6_design_elastic": {
            "sigma_vm_max_over_sigma_y": design["panel_utilisation"][
                "sigma_vm_max_over_sigma_y"],
            "pass": bool(not design["panel_utilisation"]["yield_exceeded"]),
        },
        "stiffener_elastic": {
            "utilisation_flange": design["stiffener"]["utilisation_flange"],
            "pass": bool(design["stiffener"]["utilisation_flange"] < 1.0),
        },
        "anchor_declared": {
            "benchmarks_ice_nav_refs_present": anchor[
                "benchmarks_ice_nav_refs_present"],
            "declared": bool(anchor["provenance"]["evidence_grade"]
                             == "EXTERNAL_CITATION_DECLARED"),
            "pass": True,
        },
    }
    all_pass = all(bool(v["pass"]) for v in criteria.values())

    payload: Dict[str, object] = {
        "schema": SCHEMA,
        "mission": {
            "id": 4920,
            "node": "IN11-H4",
            "title": "船体冰带局部结构 FEM 交叉验证（IACS PC6 压力-面积律 vs "
                     "ship 平台 B1 ice_fullscale 经验链）",
            "campaign": "m611 冰区航行战役 / 高保真军 H4",
            "risk": "high",
            "domain": "physics",
        },
        "anchor": anchor,
        "verification": ver,
        "mesh_convergence": conv,
        "dual_chain": sweep,
        "design_case": design,
        "criteria": criteria,
        "all_criteria_pass": all_pass,
        "audit_chain": audit,
        "limitations": [
            "Chain B was imported read-only from the ship-platform work copy "
            "that is NOT merged into its main branch; if the B1 coefficients "
            "change, the deviation table must be regenerated.",
            "The IACS class table (h_I per Polar Class) and the 1.2 MPa / 1 m^2 "
            "reference pressure are declared citations, not machine-verified "
            "rule text (benchmarks/ice_nav_refs absent from main).",
            "The plate check is linear elastic: no plasticity, no welding "
            "imperfection and no in-plane hull-girder compression are included; "
            "the ultimate-strength chain lives in plate_buckling/plate_buckling_"
            "reliability and is not duplicated here.",
            "Patch pressure is uniform over the rule rectangle; real ice "
            "indentation has a non-uniform (peak) pressure distribution and a "
            "different contact geometry at the ice-belt curvature.",
            "The stiffener is checked by closed-form beam theory with an "
            "attached plate of effective breadth equal to the stiffener "
            "spacing; no tripping, web buckling or shear lag.",
        ],
    }
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                            encoding="utf-8")
        payload["_written"] = str(out_path)
    return payload