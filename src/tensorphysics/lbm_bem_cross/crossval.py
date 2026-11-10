"""Shared scenarios, scenario runners and cross-solver metrics.

The framework has two halves that must agree on *everything* except their
physics model:

* geometry — :class:`~tensorphysics.lbm_bem_cross.geometry.SurfaceGeometry`,
  loaded through the dual-format (NURBS/STL) contract;
* output — surface ``Cp(theta)``, ``C_D``, ``C_L`` and the disturbance velocity
  field on a common polar sample set.

:class:`CrossValidation` bundles those two halves for one Reynolds number, and
:func:`evaluate_criteria` applies the machine-checkable acceptance thresholds.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .bem import PanelMethod2D, analytic_cylinder_cp
from .boundary_layer import drag_band, viscous_inviscid_iteration
from .geometry import SurfaceGeometry, geometry_contract_check, load_geometry, nurbs_circle
from .lbm import lbm_reference_drag

__all__ = [
    "Scenario",
    "SCENARIOS",
    "shipped_geometry",
    "run_case",
    "summarize_case",
    "evaluate_criteria",
    "polar_samples",
    "bilinear_sample",
    "far_field_metrics",
    "CP_FRONT_DEV_TOL",
    "FARFIELD_TOL",
    "CD_TOL",
]

#: machine acceptance thresholds (from the campaign node metadata)
CD_TOL = {100.0: 0.10, 1000.0: 0.15, 10000.0: 0.20}
CP_FRONT_DEV_TOL = 0.05      # |dCp| / Cp_stagnation over theta = 0 +- 30 deg
FARFIELD_TOL = 0.05          # |u - u_pot| / U_inf for r > 10 D (outside the wake)


@dataclass(frozen=True)
class Scenario:
    """One shared physical scenario (cylinder in cross flow)."""

    name: str
    re: float
    diameter: int              # LBM lattice diameter
    u_in: float
    up_d: float
    down_d: float
    half_width_d: float
    total_steps: int
    les_cs: float = 0.0
    n_panels: int = 128

    @property
    def cd_tolerance(self) -> float:
        return CD_TOL[self.re]


#: shared scenarios.  Domain = 32 D x 24 D (12 D upstream, 20 D downstream,
#: +-12 D laterally) so that the far-field sample set r in [10, 11.5] D lies
#: strictly inside the lattice; the lattice diameter is the resolution knob.
SCENARIOS: dict[str, Scenario] = {
    "re100": Scenario("re100", 100.0, 16, 0.08, 12.0, 20.0, 12.0, 12000),
    # Re >= 1000: a small Smagorinsky sub-grid term is required for stability at
    # the lattice viscosity reached at these diameters (tau - 1/2 ~ 4e-3); the
    # effective (sub-grid) Reynolds number is therefore lower than nominal — it
    # is reported, not hidden (see docs/LBM_BEM_CROSS_VALIDATION.md).
    "re1000": Scenario("re1000", 1000.0, 20, 0.08, 12.0, 20.0, 12.0, 9000, les_cs=0.05),
    "re10000": Scenario("re10000", 10000.0, 20, 0.08, 12.0, 20.0, 12.0, 9000, les_cs=0.15),
}


def shipped_geometry(n_panels: int = 128, workdir: Path | None = None) -> SurfaceGeometry:
    """The shared body geometry (NURBS representation, D = 1)."""
    return load_geometry("nurbs", 1.0, n_panels, workdir)


# ---------------------------------------------------------------------------
# common output sample set
# ---------------------------------------------------------------------------

def polar_samples(diameter: float = 1.0, r_min_d: float = 10.0, r_max_d: float = 11.5,
                  n_r: int = 5, n_phi: int = 96, wake_half_angle_deg: float = 30.0,
                  centre: tuple[float, float] = (0.0, 0.0)) -> tuple[np.ndarray, np.ndarray, dict]:
    """Polar sample set for the far-field comparison, excluding the wake sector.

    The wake sector (``|phi| < 30 deg`` behind the body) is excluded *a priori*:
    inside it the potential model is invalid by construction (that is precisely
    the assumption being measured, not a numerical detail).
    """
    cx, cy = centre
    phis = np.linspace(-np.pi, np.pi, n_phi, endpoint=False)
    keep = np.abs(phis) > math.radians(wake_half_angle_deg)
    phis = phis[keep]
    rs = np.linspace(r_min_d, r_max_d, n_r) * diameter
    R, P = np.meshgrid(rs, phis)
    xs = cx + R * np.cos(P)
    ys = cy + R * np.sin(P)
    meta = {"n_points": int(xs.size), "r_min_over_D": r_min_d, "r_max_over_D": r_max_d,
            "wake_half_angle_deg": wake_half_angle_deg,
            "phi_excluded": [float(math.degrees(p)) for p in phis[:0]]}
    return xs, ys, meta


def bilinear_sample(field: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear interpolation of a nodal field (``field[iy, ix]``) at physical points."""
    ny, nx = field.shape
    x0 = int(np.floor(xs.min())) if False else 0
    ix = np.clip(xs, 0, nx - 1.001)
    iy = np.clip(ys, 0, ny - 1.001)
    ix0 = np.floor(ix).astype(int)
    iy0 = np.floor(iy).astype(int)
    fx = ix - ix0
    fy = iy - iy0
    f00 = field[iy0, ix0]
    f10 = field[iy0, np.minimum(ix0 + 1, nx - 1)]
    f01 = field[np.minimum(iy0 + 1, ny - 1), ix0]
    f11 = field[np.minimum(iy0 + 1, ny - 1), np.minimum(ix0 + 1, nx - 1)]
    _ = x0
    return (f00 * (1 - fx) * (1 - fy) + f10 * fx * (1 - fy)
            + f01 * (1 - fx) * fy + f11 * fx * fy)


def far_field_metrics(u_lbm: np.ndarray, v_lbm: np.ndarray, xs: np.ndarray, ys: np.ndarray,
                      u_pot: np.ndarray, v_pot: np.ndarray, u_inf: float) -> dict:
    """Deviation of the LBM disturbance velocity from the potential solution."""
    du = u_lbm - u_pot
    dv = v_lbm - v_pot
    mag = np.hypot(du, dv) / u_inf
    return {
        "n_points": int(mag.size),
        "l2_deviation": float(np.sqrt(np.mean(mag ** 2))),
        "max_deviation": float(np.max(mag)),
        "mean_deviation": float(np.mean(mag)),
    }


# ---------------------------------------------------------------------------
# BEM side of a case
# ---------------------------------------------------------------------------

def run_case(name: str, art_dir: str | Path) -> dict:
    """Solve the BEM + integral-BL half of a scenario and persist it as JSON."""
    art = Path(art_dir)
    art.mkdir(parents=True, exist_ok=True)
    sc = SCENARIOS[name]
    geom = shipped_geometry(sc.n_panels, art / "geometry")
    contract = geometry_contract_check(1.0, sc.n_panels, art / "geometry")

    pm = PanelMethod2D(geom, u_inf=1.0)
    sol = pm.solve()
    vpi = viscous_inviscid_iteration(geom, sc.re)
    band = drag_band(geom, sc.re)

    # far-field sample set and the potential solution on it
    xs, ys, meta = polar_samples(diameter=geom.hydraulic_diameter)
    uf, vf = pm.velocity_field(xs, ys)
    phi = pm.potential_field(xs, ys)

    out = {
        "case": name,
        "re": sc.re,
        "diameter_lattice": sc.diameter,
        "geometry": geom.to_dict(),
        "geometry_contract": contract,
        "bem_potential": sol.to_dict(),
        "bem_potential_cd": pm.drag_coefficient_pressure(),
        "vpi": vpi.to_dict(),
        "vpi_band": band,
        "theta_grid": [float(t) for t in vpi.theta_grid],
        "cp_bem_bl": [float(c) for c in vpi.cp],
        "cp_potential": [float(c) for c in sol.cp_at(vpi.theta_grid)],
        "cp_analytic": [float(c) for c in analytic_cylinder_cp(vpi.theta_grid)],
        "cd_bem_bl": float(vpi.cd),
        "cd_reference_literature": float(lbm_reference_drag(sc.re)),
        "far_field_samples": {
            **meta,
            "x": [float(v) for v in xs.ravel()],
            "y": [float(v) for v in ys.ravel()],
            "u_potential": [float(v) for v in np.asarray(uf).ravel()],
            "v_potential": [float(v) for v in np.asarray(vf).ravel()],
            "phi_potential": [float(v) for v in np.asarray(phi).ravel()],
        },
    }
    (art / f"bem_{name}.json").write_text(json.dumps(out, indent=1))
    return {"case": name, "cd_bem_bl": out["cd_bem_bl"], "vpi": out["vpi"],
            "contract_passed": contract["passed"]}


# ---------------------------------------------------------------------------
# cross-solver comparison
# ---------------------------------------------------------------------------

def _cp_front_deviation(theta: np.ndarray, cp_a: np.ndarray, cp_b: np.ndarray,
                        half_width_deg: float = 30.0) -> dict:
    """Front-region (theta = 0 +- 30 deg) Cp deviation, normalised by Cp_stag.

    The deviation is reported as ``max|dCp| / Cp_stag`` — for a pressure field
    that passes through zero (``Cp(30 deg) ~ 0`` in potential flow) a *relative*
    error per point is meaningless, and using the stagnation value as the scale
    is the standard convention (it is the dynamic-head scale of the problem).
    """
    th = np.asarray(theta)
    d = np.abs(np.mod(th + np.pi, 2.0 * np.pi) - np.pi)
    mask = d <= math.radians(half_width_deg)
    dcp = np.abs(np.asarray(cp_a)[mask] - np.asarray(cp_b)[mask])
    scale = max(float(np.max(np.abs(np.asarray(cp_a)[mask]))), 1e-6)
    return {
        "window_deg": half_width_deg,
        "n_samples": int(mask.sum()),
        "max_abs_dcp": float(np.max(dcp)),
        "cp_stagnation_bem": float(np.asarray(cp_a)[np.argmin(d)]),
        "max_rel_deviation": float(np.max(dcp) / scale),
        "l2_rel_deviation": float(np.sqrt(np.mean(dcp ** 2)) / scale),
    }


def _decay_exponent(r: np.ndarray, amp: np.ndarray) -> float:
    """Slope of ``log amp`` versus ``log r`` (2-D potential disturbance: -2)."""
    mask = amp > 1e-9
    if mask.sum() < 3:
        return float("nan")
    p = np.polyfit(np.log(r[mask]), np.log(amp[mask]), 1)
    return float(p[0])


def summarize_case(name: str, art_dir: str | Path) -> dict:
    """Cross-solver metrics + acceptance criteria for one scenario."""
    art = Path(art_dir)
    sc = SCENARIOS[name]
    bem = json.loads((art / f"bem_{name}.json").read_text())
    lbm_npz_path = art / f"lbm_{name}.npz"
    lbm_json_path = art / f"lbm_{name}_summary.json"
    if not lbm_npz_path.exists():
        summ = json.loads(lbm_json_path.read_text()) if lbm_json_path.exists() else {}
        status = "lbm_diverged" if summ.get("solution_valid") is False else "lbm_artifact_missing"
        cd = {"cd_lbm": None, "cd_bem_bl": float(bem["cd_bem_bl"]),
              "cd_reference": float(bem["cd_reference_literature"]),
              "cd_bem_vs_reference": float((bem["cd_bem_bl"] - bem["cd_reference_literature"])
                                           / bem["cd_reference_literature"]),
              "cd_tolerance": sc.cd_tolerance}
        out = {"case": name, "re": sc.re, "scenario": sc.__dict__,
               "lbm_solution_valid": False, "lbm_status": status,
               "lbm_summary": summ, "cd": cd, "vpi": bem["vpi"], "vpi_band": bem["vpi_band"],
               "geometry_contract": bem["geometry_contract"], "cp_bem_bl": bem["cp_bem_bl"],
               "criteria": evaluate_criteria(
                   {"cd_lbm_vs_bem": float("nan"), "cd_bem_vs_reference":
                    cd["cd_bem_vs_reference"], "cd_lbm_vs_reference": float("nan"),
                    "cd_tolerance": sc.cd_tolerance},
                   {"max_rel_deviation": float("nan")},
                   {"l2_deviation": float("nan"), "max_deviation": float("nan")},
                   bem["geometry_contract"], lbm_valid=False)}
        (art / f"cross_{name}.json").write_text(json.dumps(out, indent=1))
        return {"case": name, "status": status, "criteria": out["criteria"], "cd": cd}
    npz = np.load(lbm_npz_path)
    lbm_summary = json.loads(lbm_json_path.read_text()) if lbm_json_path.exists() else {}

    u = npz["u"].astype(np.float64)
    v = npz["v"].astype(np.float64)
    lbm_valid = bool(np.isfinite(u).all() and np.isfinite(v).all())
    if not lbm_valid:
        cd_hist = np.asarray(lbm_summary.get("cd_history") or [np.nan], dtype=float)
        finite = cd_hist[np.isfinite(cd_hist)]
        out = {
            "case": name, "re": sc.re, "scenario": sc.__dict__,
            "lbm_solution_valid": False,
            "lbm_cd_history": [None if not np.isfinite(c) else float(c) for c in cd_hist],
            "lbm_last_finite_cd": float(finite[-1]) if finite.size else None,
            "cd": {"cd_lbm": None, "cd_bem_bl": float(bem["cd_bem_bl"]),
                   "cd_reference": float(bem["cd_reference_literature"]),
                   "cd_bem_vs_reference": float((bem["cd_bem_bl"] - bem["cd_reference_literature"])
                                                / bem["cd_reference_literature"]),
                   "cd_tolerance": sc.cd_tolerance},
            "vpi": bem["vpi"], "vpi_band": bem["vpi_band"],
            "geometry_contract": bem["geometry_contract"],
            "cp_bem_bl": bem["cp_bem_bl"],
            "criteria": evaluate_criteria(
                {"cd_lbm_vs_bem": float("nan"), "cd_lbm": None,
                 "cd_bem_vs_reference": (bem["cd_bem_bl"] - bem["cd_reference_literature"])
                 / bem["cd_reference_literature"],
                 "cd_lbm_vs_reference": float("nan"), "tolerance": sc.cd_tolerance,
                 "cd_tolerance": sc.cd_tolerance},
                {"max_rel_deviation": float("nan")},
                {"l2_deviation": float("nan"), "max_deviation": float("nan")},
                bem["geometry_contract"], lbm_valid=False),
        }
        (art / f"cross_{name}.json").write_text(json.dumps(out, indent=1))
        return {"case": name, "criteria": out["criteria"], "cd": out["cd"],
                "status": "lbm_diverged", "cp_bem_bl_head": bem["cp_bem_bl"][:6]}
    cx0 = float(npz["cx0"])
    cy0 = float(npz["cy0"])
    u_inf = float(npz["u_in"])
    nu, nx = u.shape

    # --- surface pressure comparison -------------------------------------
    theta_lbm = npz["theta_wall"]
    cp_lbm = npz["cp_wall"]
    cp_bem = np.interp(theta_lbm, bem["theta_grid"], bem["cp_bem_bl"])
    cp_front = _cp_front_deviation(theta_lbm, cp_bem, cp_lbm)
    # full-surface comparison statistics (informative, not a criterion)
    dcp_full = np.abs(cp_lbm - cp_bem)
    cp_full = {
        "max_abs_dcp": float(np.max(dcp_full)),
        "rms_abs_dcp": float(np.sqrt(np.mean(dcp_full ** 2))),
        "cp_min_lbm": float(np.min(cp_lbm)),
        "cp_min_bem": float(np.min(cp_bem)),
        "theta_cpmin_lbm_deg": float(math.degrees(theta_lbm[int(np.argmin(cp_lbm))])),
        "theta_cpmin_bem_deg": float(math.degrees(theta_lbm[int(np.argmin(cp_bem))])),
    }

    # --- far-field comparison -------------------------------------------
    ff = bem["far_field_samples"]
    d_lat = float(npz["diameter"])      # physical sample coords (in diameters) -> lattice nodes
    xs = cx0 + np.asarray(ff["x"]) * d_lat
    ys = cy0 + np.asarray(ff["y"]) * d_lat
    u_lbm = bilinear_sample(u, xs, ys)
    v_lbm = bilinear_sample(v, xs, ys)
    u_pot = np.asarray(ff["u_potential"]) * u_inf
    v_pot = np.asarray(ff["v_potential"]) * u_inf
    far = far_field_metrics(u_lbm, v_lbm, xs, ys, u_pot, v_pot, u_inf)
    # decay rate of the disturbance (2-D potential: |u'| ~ r^-2)
    r = np.hypot(xs - cx0, ys - cy0)
    amp_lbm = np.hypot(u_lbm - u_inf, v_lbm) / u_inf
    amp_pot = np.hypot(u_pot - u_inf, v_pot) / u_inf
    far["decay_exponent_lbm"] = _decay_exponent(r, amp_lbm)
    far["decay_exponent_potential"] = _decay_exponent(r, amp_pot)

    # --- integral quantities --------------------------------------------
    hist = lbm_summary.get("cd_history")
    if hist:
        cd_hist = np.asarray(hist, dtype=float)
        finite = cd_hist[np.isfinite(cd_hist)]
        half = finite[finite.size // 2:] if finite.size else finite
        cd_lbm = float(np.mean(half)) if half.size else float("nan")
    else:  # summaries written before the history field existed
        cd_hist = np.asarray([np.nan])
        cd_lbm = float(lbm_summary.get("cd_mean_last_half", float("nan")))
    cd_bem = float(bem["cd_bem_bl"])
    cd_ref = float(bem["cd_reference_literature"])
    cd = {
        "cd_lbm": cd_lbm,
        "cd_bem_bl": cd_bem,
        "cd_reference": cd_ref,
        "cd_lbm_vs_reference": (cd_lbm - cd_ref) / cd_ref,
        "cd_bem_vs_reference": (cd_bem - cd_ref) / cd_ref,
        "cd_lbm_vs_bem": (cd_lbm - cd_bem) / cd_bem,
        "cd_tolerance": sc.cd_tolerance,
    }
    cl = {"cl_rms_lbm": lbm_summary.get("cl_rms_last_half"),
          "strouhal_lbm": lbm_summary.get("strouhal")}
    sep = {"sep_angle_lbm_deg": lbm_summary.get("sep_angle_deg"),
           "sep_angle_bem_deg": bem["vpi"]["sep_angle_deg"]}

    criteria = evaluate_criteria(cd, cp_front, far, bem["geometry_contract"],
                                 lbm_valid=lbm_valid)
    out = {
        "case": name, "re": sc.re, "scenario": sc.__dict__,
        "lbm_solution_valid": lbm_valid,
        "lbm_cd_history": [None if not np.isfinite(c) else float(c) for c in cd_hist],
        "cd": cd, "cl": cl, "separation": sep,
        "cp_front": cp_front, "cp_full_surface": cp_full,
        "far_field": far,
        "geometry_contract": bem["geometry_contract"],
        "vpi": bem["vpi"], "vpi_band": bem["vpi_band"],
        "criteria": criteria,
        "theta_wall_deg": [float(math.degrees(t)) for t in theta_lbm],
        "cp_lbm": [float(c) for c in cp_lbm],
        "cp_bem": [float(c) for c in cp_bem],
    }
    (art / f"cross_{name}.json").write_text(json.dumps(out, indent=1))
    return {"case": name, "criteria": criteria, "cd": cd, "cp_front": cp_front,
            "far_field": far, "separation": sep}


def evaluate_criteria(cd: dict, cp_front: dict, far: dict, contract: dict,
                      lbm_valid: bool = True) -> dict:
    """Machine-checkable acceptance criteria of the campaign node.

    ``lbm_valid = False`` marks the physical criteria as *not adjudicable* (the
    LBM solution diverged) instead of scoring them as passes or failures — an
    honest "no evidence" verdict, never masked.
    """
    out = {"lbm_solution_valid": bool(lbm_valid)}
    if not lbm_valid:
        reason = ("LBM solution diverged: the lattice viscosity required by this "
                  "Reynolds number at the affordable resolution is below the "
                  "stability limit of the regularised-BGK + half-way bounce-back "
                  "combination. Physics criteria are therefore not adjudicable "
                  "(see docs/LBM_BEM_CROSS_VALIDATION.md, section 'Limitations').")
        out["1a_cd_deviation"] = {"adjudicable": False, "passed": False, "reason": reason}
        out["1b_cd_vs_literature"] = {"adjudicable": False, "passed": False,
                                      "bem_value": abs(cd["cd_bem_vs_reference"]),
                                      "bem_passed": bool(abs(cd["cd_bem_vs_reference"]) <= 0.15)}
        out["2_cp_front"] = {"adjudicable": False, "passed": False, "reason": reason}
        out["3_far_field"] = {"adjudicable": False, "passed": False, "reason": reason}
        out["4_geometry_contract"] = {
            "max_vertex_deviation": contract["max_vertex_deviation"],
            "max_normal_deviation": contract["max_normal_deviation"],
            "threshold": 1e-9,
            "passed": bool(contract["passed"]),
        }
        out["all_physical_passed"] = False
        return out
    out["1a_cd_deviation"] = {
        "value": abs(cd["cd_lbm_vs_bem"]),
        "threshold": cd["cd_tolerance"],
        "passed": bool(abs(cd["cd_lbm_vs_bem"]) <= cd["cd_tolerance"]),
    }
    out["1b_cd_vs_literature"] = {
        "lbm_value": abs(cd["cd_lbm_vs_reference"]),
        "lbm_threshold": 0.15,
        "lbm_passed": bool(abs(cd["cd_lbm_vs_reference"]) <= 0.15),
        "bem_value": abs(cd["cd_bem_vs_reference"]),
        "bem_passed": bool(abs(cd["cd_bem_vs_reference"]) <= 0.15),
    }
    out["2_cp_front"] = {
        "value": cp_front["max_rel_deviation"],
        "threshold": CP_FRONT_DEV_TOL,
        "passed": bool(cp_front["max_rel_deviation"] <= CP_FRONT_DEV_TOL),
    }
    out["3_far_field"] = {
        "value": far["l2_deviation"],
        "threshold": FARFIELD_TOL,
        "passed": bool(far["l2_deviation"] <= FARFIELD_TOL),
        "max_value": far["max_deviation"],
    }
    out["4_geometry_contract"] = {
        "max_vertex_deviation": contract["max_vertex_deviation"],
        "max_normal_deviation": contract["max_normal_deviation"],
        "threshold": 1e-9,
        "passed": bool(contract["passed"]),
    }
    out["all_physical_passed"] = bool(
        out["1a_cd_deviation"]["passed"] and out["2_cp_front"]["passed"]
        and out["3_far_field"]["passed"] and out["4_geometry_contract"]["passed"])
    return out


class CrossValidation:
    """Convenience wrapper combining both solvers for one scenario (in-memory)."""

    def __init__(self, name: str, n_panels: int = 128) -> None:
        self.scenario = SCENARIOS[name]
        self.geom = nurbs_circle(1.0, n_panels)
        self.panel = PanelMethod2D(self.geom, u_inf=1.0)

    def bem_side(self) -> dict:
        sol = self.panel.solve()
        vpi = viscous_inviscid_iteration(self.geom, self.scenario.re)
        return {"bem": sol, "vpi": vpi}