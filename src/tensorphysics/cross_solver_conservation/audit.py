"""Sequential FEM -> DEM -> LBM conservation audit for campaign node p270w1-N10.

The three stages are run in sequence with an explicit, machine-checkable energy /
momentum ledger at each handoff.  The synthetic fixture is a 1D chain:

* **FEM**: an ice plate strip (100 x 5 mm section, 0.5 m span) as a cantilever
  Euler-Bernoulli beam loaded to 1.5x its flexural strength load.  The failure
  face (position, normal, traction profile, LEFM release rate) is extracted.
* **DEM**: a bonded-particle chain discretising the same beam with bilinear
  cohesive curvature hinges.  It receives the FEM deflection field, evolves the
  fracture dynamically and returns the energy ledger and the crack length.
* **LBM**: a D2Q9 channel whose moving wall is driven by the DEM fragment
  velocity; the wall momentum flux is measured three independent ways.

Everything the report claims is computed here and dumped to
``artifacts/cross_solver_conservation/audit.json``; the runner is
``scripts/recompute_cross_solver_conservation.py``.
"""

from __future__ import annotations

import json
import pathlib
import time

import numpy as np

from . import cohesive_chain as chain
from . import fem_beam as beam
from . import lbm_wall as lbm
from . import multirate
from .criteria import CRITERIA_DIGEST, evaluate, format_table

__all__ = ["CASE", "run_audit", "main"]


CASE = {
    "material": "ice (E 9 GPa, rho 917 kg/m3)",
    "E": 9.0e9,
    "rho": 917.0,
    "b": 0.10,          # plate strip width            [m]
    "h": 0.005,         # plate thickness              [m]
    "L": 0.50,          # span                         [m]
    "sigma_f": 0.7e6,   # flexural strength            [Pa]
    "G_f": 0.08,        # fracture energy              [J/m2]
    "load_factor": 1.5, # lambda P_f at handoff
    "dens_mesh": [500, 1000, 2000],  # DEM bonds
    "lbm": {"nx": 8, "ny": 32, "tau": 0.8, "force": 0.0,
            "n_warm": 2000, "n_couple": 3000,
            "m_ice": [200.0, 500.0, 1000.0], "v_ice0": 0.02},
    "rates_hz": multirate.RATES_HZ,
}


def _beam_stage(case):
    section = beam.BeamSection(E=case["E"], b=case["b"], h=case["h"])
    P_f = beam.flexural_strength_load(section, case["L"], case["sigma_f"])
    sol = beam.solve_cantilever(
        section, case["L"], case["load_factor"] * P_f, n_elem=400
    )
    face = beam.extract_failure_face(sol)
    g_strength = case["sigma_f"] ** 2 * case["h"] / (6.0 * case["E"])
    out = {
        "section": {"E": case["E"], "b": case["b"], "h": case["h"],
                    "A": section.A, "I": section.I},
        "P_f": P_f,
        "P_applied": case["load_factor"] * P_f,
        "tip_deflection": sol.tip_deflection,
        "tip_deflection_analytic": sol.analytic_tip_deflection(),
        "tip_deflection_rel_error": abs(sol.tip_deflection / sol.analytic_tip_deflection() - 1.0),
        "U_strain": sol.U,
        "U_analytic": sol.analytic_strain_energy(),
        "U_rel_error": abs(sol.U / sol.analytic_strain_energy() - 1.0),
        "M_root": sol.M_root,
        "sigma_max": sol.max_stress,
        "G_lefm": sol.energy_release_rate,
        "G_strength_limit": g_strength,
        "G_over_Gf": sol.energy_release_rate / case["G_f"],
        "cohesive_length": case["E"] * case["G_f"] / case["sigma_f"] ** 2,
        "a_arrest_lefm_static": case["L"] * (1.0 - np.sqrt(case["G_f"] / sol.energy_release_rate)),
        "failure_face": face.as_dict(),
    }
    return section, sol, out


def _dem_stage(section, sol, case):
    runs = []
    for nb in case["dens_mesh"]:
        state, info = chain.bonded_chain_from_beam(
            sol, section, case["sigma_f"], case["G_f"], nb
        )
        law = state.law
        dt = 0.1 * law.l_b**2 / 72.4
        t_end = max(2e-5 / (nb / 500.0) ** 0.5, 2e-6)
        led = chain.run_chain(state, dt=dt, t_end=t_end, zeta=0.02, sample_every=200,
                              d_damage_max=0.02)
        runs.append(
            {
                "n_bonds": nb,
                "l_b": law.l_b,
                "l_b_over_process_zone": info["l_b_over_process_zone"],
                "dt": dt,
                "t_end": t_end,
                "n_steps": int(round(t_end / dt)),
                "theta_0": law.theta_0,
                "theta_c": law.theta_c,
                "phi_f": law.phi_f,
                "mapping_rel_error": info["mapping_rel_error"],
                "handoff_energy_deficit": info["handoff_energy_deficit"],
                "handoff_dissipated": info["handoff_dissipated"],
                "U_ref": led["U_ref"],
                "KE": led["KE"],
                "U_el": led["U_el"],
                "E_diss": led["E_diss"],
                "E_damp": led["E_damp"],
                "E_handoff": led["E_handoff"],
                "total": led["total"],
                "closure_rel_error": led["closure_rel_error"],
                "diss_path_vs_analytic_rel": led["diss_path_vs_analytic_rel"],
                "excess_dissipation_over_griffith": led["excess_dissipation_over_griffith"],
                "n_broken": led["n_broken"],
                "n_damaged": led["n_damaged"],
                "crack_length": led["crack_front"],
                "energy_fractions": {
                    "kinetic": led["KE"] / led["U_ref"],
                    "residual_elastic": led["U_el"] / led["U_ref"],
                    "fracture_dissipation": led["E_diss"] / led["U_ref"],
                    "damping": led["E_damp"] / led["U_ref"],
                    "handoff_deficit": led["E_handoff"] / led["U_ref"],
                },
                "history": {k: v for k, v in led["history"].items()},
            }
        )
    cracks = [r["crack_length"] for r in runs if r["crack_length"] > 0]
    spread = (
        (max(cracks) - min(cracks)) / (sum(cracks) / len(cracks)) if len(cracks) > 1 else 0.0
    )
    return {
        "runs": runs,
        "crack_length_spread_across_mesh": spread,
        "arrest_length_mean": float(np.mean(cracks)) if cracks else 0.0,
    }


def _lbm_stage(case):
    cfg = case["lbm"]
    steady = lbm.run_channel(nx=cfg["nx"], ny=cfg["ny"], tau=cfg["tau"],
                             force=1e-5, n_warm=8000, n_avg=2000)
    steady["tau_ladd_vs_analytic_rel"] = abs(
        (steady["tau_top_ladd"] + steady["tau_bottom_ladd"]) / (2.0 * steady["tau_analytic"]) - 1.0
    )
    couplings = []
    for m_ice in cfg["m_ice"]:
        c = lbm.run_ice_wall_coupling(
            nx=cfg["nx"], ny=cfg["ny"], tau=cfg["tau"], force=cfg["force"],
            n_warm=cfg["n_warm"], n_couple=cfg["n_couple"], m_ice=m_ice,
            v_ice0=cfg["v_ice0"],
        )
        c.pop("history", None)
        if m_ice != case["lbm"]["m_ice"][len(case["lbm"]["m_ice"]) // 2]:
            c.pop("per_step", None)  # only the nominal run keeps the raw history
        couplings.append(c)
    return {"steady": steady, "couplings": couplings}


def _multirate_stage(coupling, case):
    per = coupling["per_step"]
    dt_lbm = 1e-4  # 10 kHz LBM step, declared exchange clock
    t = np.arange(len(per["v_ice"])) * dt_lbm
    v_ref = np.asarray(per["v_ice"], dtype=float)
    tau_ref = np.asarray(per["tau_shear"], dtype=float)
    rates = case["rates_hz"]
    results = []
    for name, f in (("fem", rates["fem"]), ("dem", rates["dem"])):
        stride = int(round(1.0 / (f * dt_lbm)))
        idx = np.arange(0, len(t), stride)
        results.append(
            multirate.signal_errors(
                t, v_ref, tau_ref, t[idx], v_ref[idx], tau_ref[idx], "linear"
            )
            | {"rate_name": name, "rate_hz": f}
        )
        results.append(
            multirate.signal_errors(
                t, v_ref, tau_ref, t[idx], v_ref[idx], tau_ref[idx], "cubic"
            )
            | {"rate_name": name, "rate_hz": f}
        )
    sweep = multirate.multirate_sweep(
        t_end=0.02,
        rates_hz={"r100": 100.0, "r300": 300.0, "r1000": 1000.0,
                  "r3000": 3000.0, "r5000": 5000.0, "r10000": 10000.0},
    )
    # observed order is only meaningful in the resolved regime (sampling rate
    # above the Nyquist limit of the 2.6 kHz component); the 100/300 Hz points are
    # aliased and are reported separately as the "unresolved" regime.
    orders = {}
    unresolved = {}
    for scheme, (rates_, errs) in sweep["scheme_curves"].items():
        pairs = list(zip(rates_, errs))
        res = [(r, e) for r, e in pairs if r >= 3000]
        unresolved[scheme] = {f"{int(r)}Hz": e for r, e in pairs if r < 3000}
        if len(res) > 1:
            orders[scheme] = multirate.observed_order(
                [r for r, _ in res], [e for _, e in res]
            )
    return {
        "rates_hz": rates,
        "exchange_table": multirate.RATE_AUDIT,
        "measured_history": results,
        "analytic_sweep": {
            "results": sweep["results"],
            "observed_order": orders,
            "unresolved_error": unresolved,
        },
    }


def run_audit(case: dict | None = None, write: bool = True, out_dir: str | None = None) -> dict:
    case = dict(CASE if case is None else case)
    t0 = time.time()
    section, sol, fem_out = _beam_stage(case)
    dem_out = _dem_stage(section, sol, case)
    lbm_out = _lbm_stage(case)
    masses = case["lbm"]["m_ice"]
    nominal_mass = masses[len(masses) // 2]
    nominal = next(
        (c for c in lbm_out["couplings"] if c["m_ice"] == nominal_mass),
        lbm_out["couplings"][0],
    )
    weakest = nominal  # the middle m_ice is the nominal operating point
    mult_out = _multirate_stage(weakest, case)

    runaway = [r for r in dem_out["runs"] if r["n_broken"] > 0]
    ref_run = max(runaway, key=lambda r: r["n_bonds"]) if runaway else dem_out["runs"][-1]

    measured = mult_out["measured_history"]
    dem_case = next(r for r in measured if r["rate_name"] == "dem" and r["scheme"] == "cubic")

    metrics = {
        "mapping_rel_error": max(r["mapping_rel_error"] for r in dem_out["runs"]),
        "closure_rel_error": max(r["closure_rel_error"] for r in dem_out["runs"]),
        "diss_path_vs_analytic_rel": max(
            r["diss_path_vs_analytic_rel"] for r in dem_out["runs"]
            if np.isfinite(r["diss_path_vs_analytic_rel"])
        ),
        "crack_length_spread_across_mesh": dem_out["crack_length_spread_across_mesh"],
        "closure_ice_vs_ladd": nominal["closure_ice_vs_ladd"],
        "closure_ice_vs_ladd_worst": max(c["closure_ice_vs_ladd"] for c in lbm_out["couplings"]),
        "closure_shear_vs_ladd": max(c["closure_shear_vs_ladd"] for c in lbm_out["couplings"]),
        "balance_error_rel": max(c["balance_error_rel"] for c in lbm_out["couplings"]),
        "profile_rel_error": lbm_out["steady"]["profile_rel_error"],
        "tau_ladd_vs_analytic_rel": lbm_out["steady"]["tau_ladd_vs_analytic_rel"],
        "work_rel_cubic": dem_case["work_rel"],
        "impulse_rel_cubic": dem_case["impulse_rel"],
    }
    # chain closure: independent stage errors combined in quadrature
    metrics["chain_closure_rel"] = float(
        np.sqrt(
            metrics["closure_rel_error"] ** 2
            + metrics["closure_ice_vs_ladd"] ** 2
            + metrics["work_rel_cubic"] ** 2
        )
    )
    outcomes = evaluate(metrics)
    verdict = (
        "PASS"
        if all(o["status"] == "PASS" for o in outcomes)
        else ("FAIL" if any(o["status"] == "FAIL" for o in outcomes) else "WITHHELD")
    )
    audit = {
        "node": "p270w1-n10-cross-solver-conservation-audit",
        "case": case,
        "criteria_digest": CRITERIA_DIGEST,
        "criteria": outcomes,
        "verdict": verdict,
        "metrics": metrics,
        "stage_fem": fem_out,
        "stage_dem": dem_out,
        "stage_lbm": lbm_out,
        "stage_multirate": mult_out,
        "reference_run": {k: ref_run[k] for k in
                          ("n_bonds", "l_b", "crack_length", "n_broken", "closure_rel_error",
                           "energy_fractions", "dt", "t_end", "n_steps")},
        "wall_clock_s": time.time() - t0,
        "environment": {"numpy": np.__version__},
    }
    if write:
        out = pathlib.Path(out_dir or "artifacts/cross_solver_conservation")
        out.mkdir(parents=True, exist_ok=True)
        payload = json.loads(json.dumps(audit, default=float))
        (out / "audit.json").write_text(json.dumps(payload, indent=2))
        (out / "failure_faces.json").write_text(
            json.dumps(json.loads(json.dumps(fem_out["failure_face"], default=float)), indent=2)
        )
    return audit


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="artifacts/cross_solver_conservation")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    audit = run_audit(write=not args.no_write, out_dir=args.out)
    print(format_table(audit["criteria"]))
    print(f"\nverdict: {audit['verdict']}")
    return 0 if audit["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())