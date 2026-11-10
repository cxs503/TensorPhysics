"""End-to-end acceptance cases for the FEM -> LBM conservative scalar remap (p399w2-n06).

Three physics chains, each with an independent reference:

``case_manufactured``  P1 FEM solution of ``-div(k grad T) = f`` for the manufactured
                       field ``T = sin(pi x) sin(pi y)`` (zero on the boundary), remapped
                       onto a 64x64 LBM lattice.  References: the analytic cell
                       averages (Gauss quadrature) and the analytic volume integral
                       ``4/pi^2``.
``case_consistency``   the patch test of the operators: a *linear* source field must be
                       reproduced exactly by every stage (source integrals, overlap
                       weighting, reconstruction).
``case_refinement``    source-mesh refinement at fixed target lattice -> observed order
                       of accuracy of the map.
``case_front``         steep thermal front / hot spot remapped onto a *coarser* lattice:
                       the case where a plain limited/unlimited reconstruction breaks
                       either conservation or the no-new-extrema property.  This is the
                       negative control that justifies the limiting + redistribution
                       stage.
``case_coupled``       FEM heat-conduction solution (Rayleigh-Benard conduction state)
                       -> mapped onto the thermal-LBM lattice -> used as the initial
                       condition of the natural-convection solver.  Energy is audited
                       through the exchange (integral identity), during transport
                       (insulated box, exact) and at the isothermal plates (wall budget
                       closure); the quiescent Nusselt number must be 1.

Every case returns plain floats/ints so the payload can be dumped to JSON verbatim
(the report generator copies the numbers from that file, never by hand).
"""
from __future__ import annotations

import numpy as np

from .lattice import CartesianLattice, build_overlap
from .mesh import (TriMesh, make_perturbed_tri_mesh, manufactured_linear,
                   manufactured_sine)
from .remap import (conservation_report, exact_cell_average, l2_error_vs_exact,
                    map_scalar_field, monotonicity_report)
from .thermal_lbm import _CS2, conductive_flux, run_thermal_lbm

__all__ = ["run_acceptance", "case_manufactured", "case_consistency",
           "case_refinement", "case_front", "case_coupled"]

NODE_ID = "p399w2-n06-cross-solver-conservative-map"
MODES = ("average", "linear", "bounded")


def _f(x) -> float:
    return float(x)


# ---------------------------------------------------------------------------
# case A: manufactured Poisson solution, smooth field
# ---------------------------------------------------------------------------
def case_manufactured(n_mesh: int = 32, n_lattice: int = 64, seed: int = 3) -> dict:
    mesh = make_perturbed_tri_mesh(n_mesh, jitter=0.30, seed=seed)
    geom = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in mesh.validate(expected_area=1.0).items()}
    sol = manufactured_sine(mesh)
    lattice = CartesianLattice(nx=n_lattice, ny=n_lattice)
    table = build_overlap(mesh, lattice)
    ref = exact_cell_average(lattice, sol["exact"])

    nodal_err = float(np.sqrt(np.mean((sol["phi"] - sol["exact_nodal"]) ** 2)))
    nodal_rel = nodal_err / float(np.sqrt(np.mean(sol["exact_nodal"] ** 2)))
    out: dict = {
        "geometry": geom,
        "fem": {
            "n_elements": mesh.n_elements,
            "solve_residual_norm": _f(sol["solve_residual_norm"]),
            "total_integral": _f(sol["total_integral"]),
            "exact_integral": _f(4.0 / np.pi ** 2),
            "integral_rel_error": _f(abs(sol["total_integral"] - 4.0 / np.pi ** 2) / (4.0 / np.pi ** 2)),
            "nodal_l2_abs": nodal_err,
            "nodal_l2_rel": nodal_rel,
        },
        "lattice": {"nx": lattice.nx, "ny": lattice.ny,
                    "min_coverage_fraction": _f(table.coverage_fraction().min()),
                    "n_elements_total": n_mesh * n_mesh * 2},
        "modes": {},
    }
    for mode in MODES:
        res = map_scalar_field(mesh, sol["phi"], table, mode=mode)
        cr = conservation_report(res)
        mr = monotonicity_report(mesh, sol["phi"], res)
        er = l2_error_vs_exact(lattice, res.phi, ref)
        out["modes"][mode] = {
            "conservation": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in cr.items()},
            "monotonicity": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in mr.items()},
            "accuracy": {k: _f(v) for k, v in er.items()},
            "repair": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                       for k, v in (res.repair.get("repair") or {}).items()},
            "n_limited_cells": res.repair.get("n_limited_cells"),
            "phi": res.phi,
        }
    return out


# ---------------------------------------------------------------------------
# case A2: patch test -- linear field must pass through untouched
# ---------------------------------------------------------------------------
def case_consistency(n_mesh: int = 16, n_lattice: int = 32, seed: int = 5) -> dict:
    mesh = make_perturbed_tri_mesh(n_mesh, jitter=0.30, seed=seed)
    lattice = CartesianLattice(nx=n_lattice, ny=n_lattice)
    table = build_overlap(mesh, lattice)
    a, b, c = 0.7, -1.3, 2.1
    phi = a + b * mesh.nodes[:, 0] + c * mesh.nodes[:, 1]
    ref = exact_cell_average(lattice, lambda xy: a + b * xy[:, 0] + c * xy[:, 1])
    out = {}
    for mode in MODES:
        res = map_scalar_field(mesh, phi, table, mode=mode)
        er = l2_error_vs_exact(lattice, res.phi, ref)
        out[mode] = {"linf_abs": _f(er["linf_abs"]), "l2_rel": _f(er["l2_rel"]),
                     "conservation_rel_defect": _f(conservation_report(res)["rel_defect"])}
    return out


# ---------------------------------------------------------------------------
# case A3: refinement study
# ---------------------------------------------------------------------------
def case_refinement(levels=(16, 24, 32, 48), n_lattice: int = 64, seed: int = 9) -> dict:
    lattice = CartesianLattice(nx=n_lattice, ny=n_lattice)
    rows = []
    for n in levels:
        mesh = make_perturbed_tri_mesh(n, jitter=0.30, seed=seed + n)
        sol = manufactured_sine(mesh)
        table = build_overlap(mesh, lattice)
        ref = exact_cell_average(lattice, sol["exact"])
        row = {"n_mesh": n, "h": 1.0 / n,
               "fem_nodal_l2_rel": _f(np.sqrt(np.mean((sol["phi"] - sol["exact_nodal"]) ** 2))
                                      / np.sqrt(np.mean(sol["exact_nodal"] ** 2)))}
        for mode in ("average", "bounded"):
            res = map_scalar_field(mesh, sol["phi"], table, mode=mode)
            row[f"map_l2_rel_{mode}"] = _f(l2_error_vs_exact(lattice, res.phi, ref)["l2_rel"])
        rows.append(row)
    orders = {}
    hs = np.array([r["h"] for r in rows])
    for key in ("fem_nodal_l2_rel", "map_l2_rel_average", "map_l2_rel_bounded"):
        errs = np.array([r[key] for r in rows])
        slope, _ = np.polyfit(np.log(hs), np.log(errs), 1)
        orders[key] = _f(slope)
    return {"rows": rows, "observed_order": orders}


# ---------------------------------------------------------------------------
# case B: steep field remapped onto a coarser lattice (negative control)
# ---------------------------------------------------------------------------
def case_front(n_mesh: int = 48, n_lattice: int = 24, seed: int = 11) -> dict:
    mesh = make_perturbed_tri_mesh(n_mesh, jitter=0.30, seed=seed)
    lattice = CartesianLattice(nx=n_lattice, ny=n_lattice)
    table = build_overlap(mesh, lattice)
    fields = {
        "hot_spot": lambda xy: np.exp(-((xy[:, 0] - 0.5) ** 2 + (xy[:, 1] - 0.5) ** 2) / 0.03 ** 2),
        "thermal_front": lambda xy: 0.5 * (1.0 + np.tanh((xy[:, 1] - 0.5) / 0.01)),
    }
    out: dict = {"mesh": {"n_elements": mesh.n_elements, "n_lattice": n_lattice}, "fields": {}}
    for name, fn in fields.items():
        phi = fn(mesh.nodes)
        ref = exact_cell_average(lattice, fn)
        entry = {"source_nodal_min": _f(phi.min()), "source_nodal_max": _f(phi.max()),
                 "exact_integral": _f(float(ref.sum() * lattice.cell_area())), "modes": {}}
        for mode in MODES:
            res = map_scalar_field(mesh, phi, table, mode=mode)
            cr = conservation_report(res)
            mr = monotonicity_report(mesh, phi, res)
            er = l2_error_vs_exact(lattice, res.phi, ref)
            entry["modes"][mode] = {
                "conservation_rel_defect": _f(cr["rel_defect"]),
                "overshoot": _f(mr["overshoot"]),
                "undershoot": _f(mr["undershoot"]),
                "overshoot_rel": _f(mr["overshoot_rel"]),
                "n_cells_out_of_range": int(mr["n_cells_above_nodal_max"] + mr["n_cells_below_nodal_min"]),
                "l2_rel": _f(er["l2_rel"]),
                "n_limited_cells": res.repair.get("n_limited_cells"),
                "repair": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                           for k, v in (res.repair.get("repair") or {}).items()},
                "phi": res.phi,
            }
        out["fields"][name] = entry
    return out


# ---------------------------------------------------------------------------
# case C: FEM heat conduction -> mapped LBM initial condition -> natural convection
# ---------------------------------------------------------------------------
def case_coupled(n_mesh_x: int = 64, n_mesh_y: int = 32, seed: int = 13,
                 steps_insulated: int = 400, steps_conduction: int = 1200,
                 steps_convection: int = 8000, Ra_convection: float = 2.0e4,
                 nu_lbm: float = 0.05) -> dict:
    # --- source: FEM conduction state on the LBM box [0,2] x [0,1] -------------
    mesh = make_perturbed_tri_mesh(n_mesh_x, n_mesh_y, jitter=0.25, seed=seed,
                                   domain=(0.0, 0.0, 2.0, 1.0))
    geom = {k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
            for k, v in mesh.validate(expected_area=2.0).items()}
    sol = manufactured_linear(mesh, t_bottom=1.0, t_top=0.0)
    lattice = CartesianLattice(nx=n_mesh_x + 1, ny=n_mesh_y + 1,
                               length_x=2.0, length_y=1.0)
    table = build_overlap(mesh, lattice)
    res = map_scalar_field(mesh, sol["phi"], table, mode="bounded")
    cr = conservation_report(res)
    mr = monotonicity_report(mesh, sol["phi"], res)
    exact_mean = exact_cell_average(lattice, sol["exact"])
    T0 = lattice.field_to_grid(res.phi)
    er = l2_error_vs_exact(lattice, res.phi, exact_mean)

    # --- LBM runs -------------------------------------------------------------
    # The isothermal plates are driven with the *mapped* wall-node temperatures, so
    # that the exchanged field is the exact discrete steady state of the LBM (a
    # half-cell offset of the plate temperature would otherwise show up as an O(dy)
    # error and mask the exchange accuracy).
    twist_b = T0[0].copy()
    twist_t = T0[-1].copy()
    adiab = run_thermal_lbm(T0, steps=steps_insulated, Ra=0.0, hot_wall=None,
                            cold_wall=None, nu=nu_lbm)
    cond = run_thermal_lbm(T0, steps=steps_conduction, Ra=0.0,
                           hot_wall=twist_b, cold_wall=twist_t, nu=nu_lbm)
    conv = run_thermal_lbm(T0, steps=steps_convection, Ra=Ra_convection,
                           hot_wall=twist_b, cold_wall=twist_t,
                           noise_amplitude=1.0e-3, nu=nu_lbm)

    height = cond.metadata["height"]
    D = _CS2 * (cond.tau_g - 0.5)
    q_cond = conductive_flux(D, 1.0, float(height), float(lattice.nx))

    def _rel(a, b):
        return _f(abs(a - b) / max(abs(b), 1e-300))

    condensed_profile = cond.T.mean(axis=1)
    out = {
        "geometry": geom,
        "fem": {
            "n_elements": mesh.n_elements,
            "solve_residual_norm": _f(sol["solve_residual_norm"]),
            "total_integral": _f(sol["total_integral"]),
            "exact_integral": _f(sol["exact_integral"]),
            "rel_error_integral": _rel(sol["total_integral"], sol["exact_integral"]),
        },
        "exchange": {
            "conservation": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in cr.items()},
            "monotonicity": {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                             for k, v in mr.items()},
            "l2_rel_vs_analytic": _f(er["l2_rel"]),
            "linf_abs_vs_analytic": _f(er["linf_abs"]),
            "fem_vs_target_integral_rel": _rel(res.target_integral, sol["total_integral"]),
        },
        "lbm": {
            "grid": {"nx": lattice.nx, "ny": lattice.ny, "height": height},
            "diffusivity": _f(D),
            "tau_f": _f(cond.tau_f),
            "tau_g": _f(cond.tau_g),
            "insulated": {
                "steps": steps_insulated,
                "energy_drift_rel": _f(abs(adiab.energy_history[-1] - adiab.energy_history[0])
                                       / adiab.energy_history[0]),
                "wall_budget": _f(adiab.wall_budget[-1]),
                "balance_mismatch": _f(adiab.budget_mismatch),
                "mass_mismatch": _f(adiab.mass_mismatch),
            },
            "conduction": {
                "steps": steps_conduction,
                "profile_max_dev_from_initial": _f(np.max(np.abs(cond.T - T0))),
                "nusselt": _f(np.nanmean(cond.nusselt_history[-max(50, steps_conduction // 10):])),
                "balance_mismatch": _f(cond.budget_mismatch),
                "profile": condensed_profile,
            },
            "convection": {
                "steps": steps_convection,
                "rayleigh": Ra_convection,
                "nusselt": _f(np.nanmean(conv.nusselt_history[-max(200, steps_convection // 10):])),
                "umax": _f(np.abs(conv.u).max()),
                "vmax": _f(np.abs(conv.v).max()),
                "t_min": _f(conv.T.min()),
                "t_max": _f(conv.T.max()),
                "balance_mismatch": _f(conv.budget_mismatch),
                "energy_drift_rel": _f(abs(conv.energy_history[-1] - conv.energy_history[0])
                                       / conv.energy_history[0]),
            },
        },
        "T0": T0,
        "T_conv": conv.T,
        "u_conv": conv.u,
        "v_conv": conv.v,
        "energy_insulated": adiab.energy_history,
        "nusselt_history": conv.nusselt_history,
        "profile_conv": conv.T.mean(axis=1),
    }
    return out


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------
def _criterion(cid: str, description: str, value, threshold, ok: bool,
               source: str) -> dict:
    return {"id": cid, "description": description,
            "value": (float(value) if isinstance(value, (int, float, np.floating)) else value),
            "threshold": (float(threshold) if isinstance(threshold, (int, float, np.floating)) else threshold),
            "pass": bool(ok), "source": source}


def run_acceptance(quick: bool = False, seed: int = 20261110) -> dict:
    """Run every case and evaluate the frozen acceptance criteria."""
    n_mesh = 20 if quick else 32
    n_lat = 32 if quick else 64
    a = case_manufactured(n_mesh=n_mesh, n_lattice=n_lat, seed=3)
    a2 = case_consistency(n_mesh=12 if quick else 16, n_lattice=n_lat, seed=5)
    a3 = case_refinement(levels=(12, 16, 24) if quick else (16, 24, 32, 48),
                         n_lattice=n_lat, seed=9)
    b = case_front(n_mesh=32 if quick else 48, n_lattice=16 if quick else 24, seed=11)
    c = case_coupled(
        n_mesh_x=32 if quick else 64, n_mesh_y=16 if quick else 32, seed=13,
        steps_insulated=200 if quick else 400,
        steps_conduction=300 if quick else 1200,
        steps_convection=1500 if quick else 8000,
        Ra_convection=2.0e4,
    )

    m_avg, m_lin, m_bnd = (a["modes"][m] for m in MODES)
    f_hot, f_front = b["fields"]["hot_spot"], b["fields"]["thermal_front"]

    criteria = [
        _criterion("C1_geometry_gate",
                   "FEM triangular meshes pass the geometry gate: area closure to 1e-12, "
                   "no inverted element, positive areas (mesh must be verified before any field claim)",
                   a["geometry"]["area_error_rel"], 1e-12,
                   a["geometry"]["n_inverted"] == 0 and a["geometry"]["area_error_rel"] < 1e-12,
                   "cases.A_manufactured.geometry"),
        _criterion("C2_fem_solve_residual",
                   "P1 heat-conduction linear system residual norm",
                   a["fem"]["solve_residual_norm"], 1e-8,
                   a["fem"]["solve_residual_norm"] < 1e-8,
                   "cases.A_manufactured.fem"),
        _criterion("C3_fem_integral_error",
                   "FEM total integral int T dA against the analytic 4/pi^2 (source-side sanity)",
                   a["fem"]["integral_rel_error"], 0.03,
                   a["fem"]["integral_rel_error"] < 0.03,
                   "cases.A_manufactured.fem"),
        _criterion("C4_remap_conservation_average",
                   "global conservation |int_T phi dA - sum_j |C_j| phi_j| / int, first-order overlap operator",
                   m_avg["conservation"]["rel_defect"], 1e-2,
                   m_avg["conservation"]["rel_defect"] < 1e-2,
                   "cases.A_manufactured.modes.average.conservation"),
        _criterion("C5_remap_conservation_bounded",
                   "global conservation of the second-order limited + redistributed operator",
                   m_bnd["conservation"]["rel_defect"], 1e-2,
                   m_bnd["conservation"]["rel_defect"] < 1e-2,
                   "cases.A_manufactured.modes.bounded.conservation"),
        _criterion("C6_monotonicity_bounded",
                   "no new extrema: max(min_target - min_source, max_target - max_source) over "
                   "cells for the bounded operator (0 = strictly inside the source range)",
                   m_bnd["monotonicity"]["overshoot_rel"], 1e-12,
                   m_bnd["monotonicity"]["n_cells_above_nodal_max"] == 0
                   and m_bnd["monotonicity"]["n_cells_below_nodal_min"] == 0,
                   "cases.A_manufactured.modes.bounded.monotonicity"),
        _criterion("C7_monotonicity_average",
                   "no new extrema of the first-order overlap operator",
                   m_avg["monotonicity"]["overshoot_rel"], 1e-12,
                   m_avg["monotonicity"]["n_cells_above_nodal_max"] == 0
                   and m_avg["monotonicity"]["n_cells_below_nodal_min"] == 0,
                   "cases.A_manufactured.modes.average.monotonicity"),
        _criterion("C8_negative_control_unlimited",
                   "negative control: the un-limited linear reconstruction MUST create new "
                   "extrema (otherwise the limiter would be untested)",
                   m_lin["monotonicity"]["n_cells_above_nodal_max"]
                   + m_lin["monotonicity"]["n_cells_below_nodal_min"], 1.0,
                   (m_lin["monotonicity"]["n_cells_above_nodal_max"]
                    + m_lin["monotonicity"]["n_cells_below_nodal_min"]) > 0,
                   "cases.A_manufactured.modes.linear.monotonicity"),
        _criterion("C9_consistency_linear_field",
                   "patch test: a linear source field is reproduced by every operator stage "
                   "(max abs deviation over the lattice)",
                   max(a2[m]["linf_abs"] for m in MODES), 1e-12,
                   max(a2[m]["linf_abs"] for m in MODES) < 1e-12,
                   "cases.A2_consistency"),
        _criterion("C10_accuracy_smooth",
                   "mapped field vs analytic cell averages, relative L2 (bounded operator, "
                   "mesh 32 / lattice 64)",
                   m_bnd["accuracy"]["l2_rel"], 0.03,
                   m_bnd["accuracy"]["l2_rel"] < 0.03,
                   "cases.A_manufactured.modes.bounded.accuracy"),
        _criterion("C11_refinement_order",
                   "observed convergence order of the map under source refinement at fixed "
                   "lattice (slope of log L2 vs log h)",
                   a3["observed_order"]["map_l2_rel_bounded"], 1.8,
                   a3["observed_order"]["map_l2_rel_bounded"] >= 1.8,
                   "cases.A3_refinement.observed_order"),
        _criterion("C12_front_conservation_bounded",
                   "steep thermal front onto a coarser lattice: conservation of the bounded "
                   "operator (<1% required)",
                   max(f_front["modes"]["bounded"]["conservation_rel_defect"],
                       f_hot["modes"]["bounded"]["conservation_rel_defect"]), 1e-2,
                   max(f_front["modes"]["bounded"]["conservation_rel_defect"],
                       f_hot["modes"]["bounded"]["conservation_rel_defect"]) < 1e-2,
                   "cases.B_front.fields.*.modes.bounded"),
        _criterion("C13_front_monotonicity_bounded",
                   "steep front / hot spot: zero out-of-range cells for the bounded operator",
                   max(f_front["modes"]["bounded"]["n_cells_out_of_range"],
                       f_hot["modes"]["bounded"]["n_cells_out_of_range"]), 0.0,
                   max(f_front["modes"]["bounded"]["n_cells_out_of_range"],
                       f_hot["modes"]["bounded"]["n_cells_out_of_range"]) == 0,
                   "cases.B_front.fields.*.modes.bounded"),
        _criterion("C14_negative_control_front",
                   "negative control: un-limited reconstruction on the steep front violates "
                   "conservation by more than the 1% budget (proves the repair stage is needed)",
                   f_hot["modes"]["linear"]["conservation_rel_defect"], 1e-2,
                   f_hot["modes"]["linear"]["conservation_rel_defect"] > 1e-2,
                   "cases.B_front.fields.hot_spot.modes.linear"),
        _criterion("C15_coupled_exchange_integral",
                   "FEM integral vs mapped lattice integral (coupled initial condition)",
                   c["exchange"]["fem_vs_target_integral_rel"], 1e-12,
                   c["exchange"]["fem_vs_target_integral_rel"] < 1e-12,
                   "cases.C_coupled.exchange"),
        _criterion("C16_coupled_energy_drift_insulated",
                   "relative drift of sum(T) over the LBM run in an insulated box (transport "
                   "must neither create nor destroy the mapped energy)",
                   c["lbm"]["insulated"]["energy_drift_rel"], 1e-10,
                   c["lbm"]["insulated"]["energy_drift_rel"] < 1e-10,
                   "cases.C_coupled.lbm.insulated"),
        _criterion("C17_coupled_energy_balance",
                   "closure of the energy balance: measured d(sum T) vs accumulated wall budget "
                   "over the whole coupled run (insulated + isothermal + convective)",
                   max(c["lbm"]["insulated"]["balance_mismatch"],
                       c["lbm"]["conduction"]["balance_mismatch"],
                       c["lbm"]["convection"]["balance_mismatch"]), 1e-6,
                   max(c["lbm"]["insulated"]["balance_mismatch"],
                       c["lbm"]["conduction"]["balance_mismatch"],
                       c["lbm"]["convection"]["balance_mismatch"]) < 1e-6,
                   "cases.C_coupled.lbm"),
        _criterion("C18_conduction_steady_state",
                   "the mapped FEM field is the exact discrete steady state of the thermal LBM "
                   "(max |T_lbm - T_mapped| after the conduction run)",
                   c["lbm"]["conduction"]["profile_max_dev_from_initial"], 1e-10,
                   c["lbm"]["conduction"]["profile_max_dev_from_initial"] < 1e-10,
                   "cases.C_coupled.lbm.conduction"),
        _criterion("C19_nusselt_quiescent",
                   "Nusselt number of the quiescent conduction state must be 1 (analytic)",
                   abs(c["lbm"]["conduction"]["nusselt"] - 1.0), 1e-6,
                   abs(c["lbm"]["conduction"]["nusselt"] - 1.0) < 1e-6,
                   "cases.C_coupled.lbm.conduction.nusselt"),
        _criterion("C20_convection_bounded",
                   "convective run at Ra=2e4/Pr=0.71 (mapped conduction field as initial "
                   "condition): temperature stays inside the plate temperatures [0,1] and the "
                   "flow is alive (max|u| > 0.01)",
                   c["lbm"]["convection"]["umax"], 0.01,
                   c["lbm"]["convection"]["umax"] > 0.01
                   and -1e-9 <= c["lbm"]["convection"]["t_min"]
                   and c["lbm"]["convection"]["t_max"] <= 1.0 + 1e-9,
                   "cases.C_coupled.lbm.convection"),
    ]

    payload = {
        "node": NODE_ID,
        "repo": "TensorPhysics",
        "module": "tensorphysics.fem_lbm_conservative_map",
        "quick": bool(quick),
        "criteria": criteria,
        "all_pass": all(x["pass"] for x in criteria),
        "n_pass": sum(1 for x in criteria if x["pass"]),
        "n_criteria": len(criteria),
        "cases": {"A_manufactured": a, "A2_consistency": a2,
                  "A3_refinement": a3, "B_front": b, "C_coupled": c},
    }
    return payload


def strip_arrays(payload: dict) -> dict:
    """JSON-safe copy: drop the raw fields, keep every number."""
    import copy

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
        if isinstance(obj, list):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        return obj

    return clean(copy.deepcopy(payload))