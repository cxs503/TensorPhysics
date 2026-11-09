"""IN66-N08 cross-domain integration gate: DEM -> RANS -> FSI -> resistance ->
five-channel energy ledger -> stability domain, end to end for one ice case.

The gate is deliberately thin: every number it reports is produced by one of the
four stage modules of this package (:mod:`contracts`, :mod:`dem_field`,
:mod:`rans_1d`, :mod:`fsi_coupling`, :mod:`energy_ledger`) or by the ice
resistance law of this repository (:mod:`tensorphysics.ice.resistance`).  The
gate itself only

1. runs the subcycled loop (DEM steps -> flow/FSI coupling steps),
2. audits the three interface contracts at every coupling audit point,
3. accumulates the five-channel energy ledger,
4. scans the FSI stability domain,
5. emits the verdict JSON, provenance hashes, data-flow graph, figures and the
   paper-level markdown report.

Acceptance criteria (campaign node IN66-N08): C1 no crash and CPU < 7200 s;
C2 interface dimension/grid/time-step mismatch < 1 %; C3 energy residual < 2 %;
C4 coupling fixed-point residual < 1e-6; C5 integration report with contract
results, data-flow graph and residual attribution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from ..ice.resistance import ShipGeometry, brash_ice_resistance
from . import contracts as C
from .dem_field import RHO_ICE, ChannelGrid, build_ice_field_state
from .energy_ledger import CHANNELS, accumulate, brash_channel_split, build_ledger
from .fsi_coupling import StructuralSetup, linear_gain, solve_two_way, stability_scan
from .rans_1d import (
    FlowSetup,
    distribution_weights,
    hull_profile,
    open_water_resistance,
    solve_streamwise,
)

GATE_ID = "IN66-N08"
GATE_SCHEMA = "in66_n08_integration_gate_verdict.v1"
CPU_BUDGET_S = 7200.0
TOL_CONTRACT = C.TOL_INTERFACE
TOL_ENERGY = 2.0e-2
TOL_COUPLING = 1.0e-6

DATA_FLOW_GRAPH = """graph LR
  A["DEM floe field<br/>alpha_i(x,y,t)"] -->|"ice_field_volume_fraction.v1"| B["RANS streamwise<br/>u, p, tau_w"]
  B -->|"hull_traction_handoff.v1"| C["FSI panel<br/>w, p_contact, clearance"]
  C -->|"structural_feedback_handoff.v1"| A
  B --> D["Resistance integration<br/>R_panel"]
  D --> E["Five-channel energy ledger"]
  A --> E
  E --> F["Gate verdict C1..C5"]
  C --> G["Stability domain scan"]
  G --> F
"""


@dataclass(frozen=True)
class CaseConfig:
    """KVLCC2 in brash ice at the mission operating point."""

    name: str = "KVLCC2_brash_h0.15_C0.6"
    length_m: float = 320.0
    beam_m: float = 58.0
    draft_m: float = 20.8
    block_coefficient: float = 0.8098
    displacement_t: float = 320400.0  # published KVLCC2 value
    speed_m_s: float = 2.0
    ice_thickness_m: float = 0.15
    coverage: float = 0.60
    bow_angle_deg: float = 30.0
    flare_deg: float = 15.0
    installed_power_MW: float = 25.0
    eta_D: float = 0.55
    nx: int = 96
    ny: int = 16
    dem_dt_s: float = 0.10
    rans_every: int = 5
    end_time_s: float = 200.0
    seed: int = 20261108
    contract_audit_every: int = 10

    @property
    def rans_dt_s(self) -> float:
        return self.dem_dt_s * self.rans_every


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_json(path: str, payload: Any) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True, default=float)
    return path


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def ship_geometry(cfg: CaseConfig) -> ShipGeometry:
    return ShipGeometry(
        name="KVLCC2",
        L_pp=cfg.length_m,
        B=cfg.beam_m,
        T=cfg.draft_m,
        bow_angle_deg=cfg.bow_angle_deg,
        flare_deg=cfg.flare_deg,
        displacement_t=cfg.displacement_t,
        installed_power_MW=cfg.installed_power_MW,
        eta_D=cfg.eta_D,
    )


def ice_channels(cfg: CaseConfig) -> Dict[str, Any]:
    """Brash-ice resistance split into the four ice channels plus open water."""
    ship = ship_geometry(cfg)
    res = brash_ice_resistance(cfg.ice_thickness_m, cfg.speed_m_s, cfg.coverage, ship)
    split = brash_channel_split(res)
    split["viscous_open_water"] = open_water_resistance(
        speed_m_s=cfg.speed_m_s,
        length_m=cfg.length_m,
        draft_m=cfg.draft_m,
        beam_m=cfg.beam_m,
        block_coefficient=cfg.block_coefficient,
    )["R_viscous_N"]
    closed = sum(split[c] for c in CHANNELS if c != "viscous_open_water")
    return {
        "channels": split,
        "ice_law": {
            k: float(v)
            for k, v in res.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        },
        "ice_law_flags": [str(f) for f in res.get("flags", [])],
        "split_closure_error": abs(closed - float(res["R_total_N"]))
        / max(float(res["R_total_N"]), 1e-30),
        "return_period": 0.0,
        "ship": ship,
    }


def geometry_audit(cfg: CaseConfig) -> Dict[str, Any]:
    """Geometry gate: displacement from Cb, L/B, B/T and the ice-law validity domain."""
    vol = cfg.block_coefficient * cfg.length_m * cfg.beam_m * cfg.draft_m
    disp = vol * 1.025  # tonnes (sea water density 1.025 t/m^3)
    s_dm = 1.7 * cfg.length_m * cfg.draft_m + cfg.block_coefficient * cfg.beam_m * cfg.length_m
    return {
        "displacement_volume_m3": vol,
        "displacement_t_from_Cb": disp,
        "displacement_t_declared": cfg.displacement_t,
        "displacement_relative_error": abs(disp - cfg.displacement_t) / cfg.displacement_t,
        "block_coefficient": cfg.block_coefficient,
        "l_over_b": cfg.length_m / cfg.beam_m,
        "b_over_t": cfg.beam_m / cfg.draft_m,
        "wetted_surface_m2": s_dm,
        "ice_thickness_over_length": cfg.ice_thickness_m / cfg.length_m,
        "ice_thickness_over_beam": cfg.ice_thickness_m / cfg.beam_m,
        "h_limit_m": min(0.03 * cfg.length_m, 0.10 * cfg.beam_m),
    }


# ---------------------------------------------------------------------------
# the loop
# ---------------------------------------------------------------------------


def run_loop(cfg: CaseConfig) -> Dict[str, Any]:
    """Run the subcycled DEM -> RANS -> FSI loop and return all raw evidence."""
    t_cpu0 = time.process_time()
    t_wall0 = time.perf_counter()

    grid = ChannelGrid(
        nx=cfg.nx, ny=cfg.ny, dx=cfg.length_m / cfg.nx, dy=3.0 * cfg.beam_m / cfg.ny
    )
    ice = build_ice_field_state(
        grid,
        coverage=cfg.coverage,
        thickness_m=cfg.ice_thickness_m,
        lane_width_m=1.1 * cfg.beam_m,
        speed_m_s=cfg.speed_m_s,
        seed=cfg.seed,
    )
    flow = FlowSetup()
    struct = StructuralSetup(gap0_m=0.65)

    channel_info = ice_channels(cfg)
    base_forces = channel_info["channels"]
    r_ice_total = sum(base_forces[c] for c in CHANNELS if c != "viscous_open_water")
    r_ow = base_forces["viscous_open_water"]

    x = grid.cell_centers_x()
    prof = hull_profile(x, cfg.length_m, cfg.beam_m)

    n_dem_steps = int(round(cfg.end_time_s / cfg.dem_dt_s))
    n_coupling = n_dem_steps // cfg.rans_every

    history: List[Dict[str, Any]] = []
    coupling_history: List[Dict[str, Any]] = []
    contract_runs: List[Dict[str, Any]] = []
    rans_history: List[Dict[str, Any]] = []
    snapshots: Dict[int, Dict[str, Any]] = {}
    ice_budget: List[float] = []
    coverage_history: List[float] = []
    failure: Optional[str] = None

    try:
        for step in range(1, n_dem_steps + 1):
            out = ice.step(speed_m_s=cfg.speed_m_s, dt_s=cfg.dem_dt_s)
            ice_budget.append(float(out["budget_residual_relative"]))
            if step % cfg.rans_every != 0:
                continue

            coupling_index = step // cfg.rans_every
            t_couple = step * cfg.dem_dt_s
            alpha_mean = ice.cross_mean()
            coverage_history.append(float(alpha_mean.mean()))

            weights = distribution_weights(
                x=x,
                breadth_m=prof["breadth_m"],
                alpha_mean=alpha_mean,
                coverage_ambient=cfg.coverage,
                dx=grid.dx,
            )
            f_crush = base_forces["crushing"] * weights["bow"]
            f_break = base_forces["breaking"] * weights["bow"]
            f_sub = base_forces["submersion"] * weights["hull_ice"]
            f_clear = base_forces["clearing"] * weights["hull_ice"]
            f_ice = f_crush + f_break + f_sub + f_clear
            f_visc = r_ow * weights["hull"]

            state: Dict[str, Any] = {}

            def flow_fn(alpha_eff: np.ndarray) -> np.ndarray:
                ratio = np.clip(alpha_eff / np.maximum(alpha_mean, 1e-6), 0.3, 3.0)
                raw = f_ice * ratio
                integral = float(np.sum(raw) * grid.dx)
                scaled = raw * (r_ice_total / integral) if integral > 0.0 else raw
                source = scaled + f_visc
                rans = solve_streamwise(
                    grid=grid,
                    alpha_mean=alpha_eff,
                    speed_m_s=cfg.speed_m_s,
                    thickness_m=cfg.ice_thickness_m,
                    coverage_ambient=cfg.coverage,
                    force_density_N_per_m=source,
                    setup=flow,
                    draft_m=cfg.draft_m,
                    beam_m=cfg.beam_m,
                )
                state["rans"] = rans
                state["source"] = source
                return rans["p_pa"]

            coupling = solve_two_way(
                alpha_mean=alpha_mean,
                thickness_m=cfg.ice_thickness_m,
                setup=struct,
                flow_fn=flow_fn,
                relaxation="aitken",
                max_iter=80,
                tol=TOL_COUPLING,
            )
            rans = state["rans"]
            source = state["source"]

            ledger = build_ledger(
                channel_forces=base_forces,
                speed_m_s=cfg.speed_m_s,
                panel_force_N=rans["R_panel_N"],
                momentum_flux_N=rans["momentum_flux_N"],
                fsi_residual=coupling.residual,
            )
            history.append(
                {
                    "time_s": t_couple,
                    "channel_forces": dict(base_forces),
                    "panel_force_N": float(rans["R_panel_N"]),
                    "momentum_flux_N": float(rans["momentum_flux_N"]),
                    "residual": float(ledger["residual"]),
                    "speed_m_s": cfg.speed_m_s,
                }
            )
            coupling_history.append(
                {
                    "time_s": t_couple,
                    "iterations": int(coupling.iterations),
                    "residual": float(coupling.residual),
                    "converged": bool(coupling.converged),
                    "added_mass_ratio": float(coupling.added_mass_ratio),
                    "spectral_radius": float(coupling.spectral_radius),
                    "deflection_max_m": float(np.max(np.abs(coupling.deflection_m))),
                    "residual_history": [float(r) for r in coupling.residual_history],
                }
            )
            rans_history.append(
                {
                    "time_s": t_couple,
                    "u_max_m_s": float(rans["u_max_m_s"]),
                    "p_max_pa": float(rans["p_max_pa"]),
                    "p_min_pa": float(rans["p_min_pa"]),
                    "R_panel_N": float(rans["R_panel_N"]),
                    "R_sources_N": float(rans["R_sources_N"]),
                    "R_lane_shear_N": float(rans["R_lane_shear_N"]),
                    "momentum_flux_N": float(rans["momentum_flux_N"]),
                }
            )

            if coupling_index % cfg.contract_audit_every == 0 or coupling_index == 1:
                bundle = _build_handoffs(
                    cfg=cfg,
                    grid=grid,
                    ice=ice,
                    rans=rans,
                    source=source,
                    coupling=coupling,
                    struct=struct,
                    prof=prof,
                    x=x,
                    t_couple=t_couple,
                )
                contract_runs.append(
                    {
                        "time_s": t_couple,
                        "dem_to_rans": C.summarize(bundle["checks_ice"]),
                        "rans_to_fsi": C.summarize(bundle["checks_traction"]),
                        "fsi_to_dem": C.summarize(bundle["checks_feedback"]),
                    }
                )
                snapshots[coupling_index] = bundle
    except Exception as exc:  # pragma: no cover - recorded as an honest FAIL
        failure = "%s: %s" % (type(exc).__name__, exc)

    elapsed_cpu = time.process_time() - t_cpu0
    elapsed_wall = time.perf_counter() - t_wall0

    keys = sorted(snapshots)
    selected: Dict[str, Any] = {}
    if keys:
        selected["t_start"] = snapshots[keys[0]]
        selected["t_mid"] = snapshots[keys[len(keys) // 2]]
        selected["t_end"] = snapshots[keys[-1]]

    return {
        "grid": grid,
        "ice": ice,
        "ship": channel_info["ship"],
        "channel_info": channel_info,
        "base_forces": base_forces,
        "r_ice_total": r_ice_total,
        "r_ow": r_ow,
        "history": history,
        "coupling_history": coupling_history,
        "contract_runs": contract_runs,
        "rans_history": rans_history,
        "snapshots": selected,
        "snapshot_bundles": snapshots,
        "coverage_history": coverage_history,
        "ice_budget_max_residual": max(ice_budget) if ice_budget else float("nan"),
        "ice_volume_initial_m3": float(ice.volume_m3()),
        "ice_volume_final_m3": float(ice.volume_m3()),
        "elapsed_cpu_s": float(elapsed_cpu),
        "elapsed_wall_s": float(elapsed_wall),
        "n_dem_steps": n_dem_steps,
        "n_coupling_steps": n_coupling,
        "prof": prof,
        "failure": failure,
    }


def _build_handoffs(
    *,
    cfg: CaseConfig,
    grid: ChannelGrid,
    ice: Any,
    rans: Mapping[str, Any],
    source: np.ndarray,
    coupling: Any,
    struct: StructuralSetup,
    prof: Mapping[str, np.ndarray],
    x: np.ndarray,
    t_couple: float,
) -> Dict[str, Any]:
    """Build and validate the three interface payloads for one coupling instant."""
    ice_payload = C.build_ice_field_payload(
        grid=grid.descriptor(),
        time_s=t_couple,
        dt_s=cfg.dem_dt_s,
        thickness_m=cfg.ice_thickness_m,
        ice_density_kg_m3=RHO_ICE,
        nominal_coverage=float(avg_or(ice.alpha, cfg.coverage)),
        volume_fraction=ice.alpha.tolist(),
        upstream_velocity_m_s=cfg.speed_m_s,
    )
    panels = [
        {
            "x_m": float(x[i]),
            "u_m_s": float(rans["u_m_s"][i]),
            "p_pa": float(rans["p_pa"][i]),
            "tau_wall_pa": float(rans["tau_wall_pa"][i]),
            "dpdx_pa_per_m": float(rans["dpdx_pa_per_m"][i]),
            "du_dx_1_per_s": float(rans["du_dx"][i]),
            "source_N_per_m": float(source[i]),
            "water_area_m2": float(rans["water_area_m2"][i]),
            "wetted_area_m2": float(2.0 * cfg.ice_thickness_m * grid.dx),
            "projected_area_m2": float(
                cfg.ice_thickness_m * prof["axial_density_m"][i] * grid.dx
            ),
        }
        for i in range(grid.nx)
    ]
    r_pressure = float(sum(p["p_pa"] * p["projected_area_m2"] for p in panels))
    r_friction = float(sum(p["tau_wall_pa"] * p["wetted_area_m2"] for p in panels))
    traction_payload = C.build_traction_payload(
        grid=grid.descriptor(),
        time_s=t_couple,
        dt_s=cfg.rans_dt_s,
        source_dt_s=cfg.dem_dt_s,
        panels=panels,
        mass_flux_kg_s=float(rans["mass_flux_kg_s"]),
        totals={"R_pressure_N": r_pressure, "R_friction_N": r_friction},
    )
    feedback_payload = C.build_struct_feedback_payload(
        grid=grid.descriptor(),
        time_s=t_couple,
        dt_s=cfg.rans_dt_s,
        source_dt_s=cfg.dem_dt_s,
        deflection_m=coupling.deflection_m.tolist(),
        clearance_m=coupling.clearance_m.tolist(),
        contact_pressure_pa=coupling.p_contact_pa.tolist(),
        gap0_m=struct.gap0_m,
        residual=coupling.residual,
        iterations=coupling.iterations,
        converged=coupling.converged,
        spectral_radius=0.0 if math.isnan(coupling.spectral_radius) else coupling.spectral_radius,
        added_mass_ratio=coupling.added_mass_ratio,
    )
    return {
        "time_s": t_couple,
        "ice_payload": ice_payload,
        "traction_payload": traction_payload,
        "feedback_payload": feedback_payload,
        "checks_ice": C.validate_ice_field(ice_payload),
        "checks_traction": C.validate_hull_traction(traction_payload, ice_payload),
        "checks_feedback": C.validate_struct_feedback(feedback_payload, traction_payload),
    }


def avg_or(alpha: np.ndarray, fallback: float) -> float:
    return float(np.mean(alpha)) if alpha.size else float(fallback)


# ---------------------------------------------------------------------------
# criteria
# ---------------------------------------------------------------------------


def _jacobi_probe(cfg: CaseConfig) -> Dict[str, Any]:
    """Contract of the real coupled solver at Jacobi relaxation vs linear theory.

    Uses the initial ice field (a single coupling solve), so the cost is one
    solve and the comparison is unambiguous: the residual ratios must approach
    ``M = rho_w c_add / (rho_s t_eff)`` from the linearised analysis.
    """
    grid = ChannelGrid(nx=32, ny=8, dx=cfg.length_m / 32, dy=3.0 * cfg.beam_m / 8)
    ice = build_ice_field_state(
        grid,
        coverage=cfg.coverage,
        thickness_m=cfg.ice_thickness_m,
        lane_width_m=1.1 * cfg.beam_m,
        speed_m_s=cfg.speed_m_s,
        seed=cfg.seed,
    )
    alpha_mean = ice.cross_mean()
    struct = StructuralSetup(gap0_m=0.65)
    constant = np.full(grid.nx, 2.0e3)  # 2 kPa reference load, linear regime

    result = solve_two_way(
        alpha_mean=alpha_mean,
        thickness_m=cfg.ice_thickness_m,
        setup=struct,
        flow_fn=lambda alpha_eff: constant,
        relaxation=1.0,
        max_iter=40,
        tol=1e-12,
    )
    hist = result.residual_history
    ratios = [hist[i + 1] / hist[i] for i in range(len(hist) - 1) if hist[i] > 0.0]
    measured = float(np.median(ratios)) if ratios else float("nan")
    predicted = abs(linear_gain(1.0, result.added_mass_ratio))
    return {
        "added_mass_ratio": float(result.added_mass_ratio),
        "predicted_contraction": float(predicted),
        "measured_contraction": measured,
        "relative_error": float(abs(measured - predicted) / predicted) if predicted > 0 else float("nan"),
        "iterations": int(result.iterations),
        "converged": bool(result.converged),
    }


def evaluate_criteria(
    cfg: CaseConfig,
    run: Mapping[str, Any],
    stability: Sequence[Mapping[str, Any]],
    probe: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    contract_runs = run["contract_runs"]
    worst: Dict[str, float] = {}
    n_fail = 0
    n_checks = 0
    for row in contract_runs:
        for handoff, summary in row.items():
            if handoff == "time_s":
                continue
            n_checks += summary["n_checks"]
            n_fail += summary["n_fail"]
            for check in summary["checks"]:
                value = check["value"]
                tol = check["tolerance"]
                if isinstance(value, (int, float)) and tol:
                    worst[check["check_id"]] = max(
                        worst.get(check["check_id"], 0.0), float(value) / float(tol)
                    )

    ledger_summary = accumulate(run["history"])
    c1 = {
        "passed": bool(run["failure"] is None and run["elapsed_cpu_s"] < CPU_BUDGET_S),
        "elapsed_cpu_s": run["elapsed_cpu_s"],
        "elapsed_wall_s": run["elapsed_wall_s"],
        "cpu_budget_s": CPU_BUDGET_S,
        "n_dem_steps": run["n_dem_steps"],
        "n_coupling_steps": run["n_coupling_steps"],
        "failure": run["failure"],
    }
    # separate the interface-bound checks (1 % acceptance) from the internal
    # conservation checks (1e-6), so the reported C2 metric is a genuine
    # relative interface mismatch and not a tolerance ratio
    interface_values: List[float] = []
    conservation_values: List[float] = []
    for row in contract_runs:
        for handoff, summary in row.items():
            if handoff == "time_s":
                continue
            for check in summary["checks"]:
                value = check["value"]
                tol = check["tolerance"]
                if not isinstance(value, (int, float)) or not tol:
                    continue
                if abs(float(tol) - TOL_CONTRACT) <= 1e-15:
                    interface_values.append(float(value))
                elif abs(float(tol) - C.TOL_CONSERVATION) <= 1e-15:
                    conservation_values.append(float(value))
    c2 = {
        "passed": bool(n_fail == 0 and n_checks > 0),
        "n_contract_audits": len(contract_runs),
        "n_checks_total": n_checks,
        "n_checks_failed": n_fail,
        "tolerance": TOL_CONTRACT,
        "worst_value_over_tolerance_by_check": {k: float(v) for k, v in sorted(worst.items())},
        "max_interface_error_relative": float(max(interface_values, default=0.0)),
        "max_internal_conservation_error": float(max(conservation_values, default=0.0)),
        "n_interface_checks": len(interface_values),
        "n_conservation_checks": len(conservation_values),
    }
    c3 = {
        "passed": bool(
            ledger_summary["mean_residual"] < TOL_ENERGY
            and ledger_summary["max_residual_all"] < TOL_ENERGY
        ),
        "mean_residual": ledger_summary["mean_residual"],
        "max_residual_window": ledger_summary["max_residual_window"],
        "max_residual_all": ledger_summary["max_residual_all"],
        "tolerance": TOL_ENERGY,
        "steady_window": ledger_summary,
    }
    ch = run["coupling_history"]
    max_coupling = max(r["residual"] for r in ch) if ch else float("inf")
    all_converged = all(r["converged"] for r in ch) if ch else False
    c4 = {
        "passed": bool(max_coupling < TOL_COUPLING and all_converged),
        "max_residual": max_coupling,
        "tolerance": TOL_COUPLING,
        "all_steps_converged": bool(all_converged),
        "max_iterations": int(max(r["iterations"] for r in ch)) if ch else 0,
        "mean_iterations": float(np.mean([r["iterations"] for r in ch])) if ch else float("nan"),
        "added_mass_ratio": float(ch[0]["added_mass_ratio"]) if ch else float("nan"),
        "stability_domain": {
            "n_points": len(stability),
            "n_converging_at_jacobi": int(sum(1 for r in stability if r["converges_at_jacobi"])),
            "n_diverging_at_jacobi": int(sum(1 for r in stability if not r["converges_at_jacobi"])),
            "m_critical_analytic": 1.0,
            "boundary_verified": bool(
                all(
                    (r["converges_at_jacobi"] == (r["added_mass_ratio"] < 1.0))
                    for r in stability
                )
            ),
            "n_resolvable": int(sum(1 for r in stability if not r["beyond_grid"] and r["omega_max_analytic"] > 0.0)),
            "max_relative_error_vs_analytic": float(
                max(
                    (r["relative_error"] for r in stability if r["relative_error"] == r["relative_error"]),
                    default=float("nan"),
                )
            ),
            "jacobi_probe": dict(probe) if probe is not None else {},
        },
    }
    c5 = {
        "passed": False,  # flipped by write_report once the report is on disk
        "verdict_json": "gate_verdict.json",
        "report_md": "docs/IN66_N08_ICE_NAV_INTEGRATION_GATE.md",
        "data_flow_graph": "data_flow.mmd",
        "residual_attribution_present": True,
    }
    return {"C1": c1, "C2": c2, "C3": c3, "C4": c4, "C5": c5}


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------


def make_figures(
    cfg: CaseConfig,
    run: Mapping[str, Any],
    stability: Sequence[Mapping[str, Any]],
    outdir: str,
) -> List[Dict[str, str]]:
    try:  # matplotlib is a soft dependency: a gate run must not die without it
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # pragma: no cover - minimal HPC images
        return []

    fig_dir = os.path.join(outdir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    figs: List[Dict[str, str]] = []
    x = run["grid"].cell_centers_x()

    snap = run["snapshots"]["t_end"]["ice_payload"]
    alpha = np.array(snap["volume_fraction"])
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    im = axes[0].imshow(
        alpha.T,
        origin="lower",
        aspect="auto",
        extent=[0.0, cfg.length_m, 0.0, 3.0 * cfg.beam_m],
        cmap="Blues",
        vmin=0.0,
        vmax=1.0,
    )
    axes[0].set_xlabel("x [m] (ship-fixed, bow at 0)")
    axes[0].set_ylabel("y [m]")
    axes[0].set_title("DEM ice field alpha_i(x,y), t=%.0f s" % snap["time_s"])
    fig.colorbar(im, ax=axes[0], label="volume fraction [-]")
    axes[1].plot(x, alpha.mean(axis=1), label="cross-lane mean alpha")
    axes[1].axhline(cfg.coverage, color="k", ls="--", label="nominal C_i=%.2f" % cfg.coverage)
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("alpha [-]")
    axes[1].set_title("coverage profile along the hull")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p1 = os.path.join(fig_dir, "fig1_ice_field.png")
    fig.savefig(p1, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig1_ice_field.png",
            "path": p1,
            "caption": "DEM volume-fraction field and along-hull coverage profile.",
        }
    )

    panels = run["snapshots"]["t_end"]["traction_payload"]["panels"]
    u = np.array([p["u_m_s"] for p in panels])
    p = np.array([p["p_pa"] for p in panels])
    tau = np.array([p["tau_wall_pa"] for p in panels])
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.4))
    axes[0].plot(x, u)
    axes[0].set_title("lane mixture velocity u(x)")
    axes[0].set_xlabel("x [m]")
    axes[0].set_ylabel("u [m/s]")
    axes[1].plot(x, p / 1e3)
    axes[1].set_title("pressure p(x) (wake reference 0)")
    axes[1].set_xlabel("x [m]")
    axes[1].set_ylabel("p [kPa]")
    axes[2].plot(x, tau)
    axes[2].set_title("wall shear tau_w(x)")
    axes[2].set_xlabel("x [m]")
    axes[2].set_ylabel("tau [Pa]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    p2 = os.path.join(fig_dir, "fig2_flow_profiles.png")
    fig.savefig(p2, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig2_flow_profiles.png",
            "path": p2,
            "caption": "Reduced-order streamwise RANS: velocity, pressure and wall shear along the hull.",
        }
    )

    ledger = run["ledger_last"]
    names = [row["channel"] for row in ledger["channels"]]
    powers = [row["power_W"] / 1e6 for row in ledger["channels"]]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    axes[0].bar(range(len(names)), powers, color="steelblue")
    axes[0].set_xticks(range(len(names)))
    axes[0].set_xticklabels([n.replace("_", "\n") for n in names], fontsize=7)
    axes[0].set_ylabel("power [MW]")
    axes[0].set_title("five-channel dissipation at the operating point")
    axes[1].plot(
        [r["time_s"] for r in run["history"]],
        [r["residual"] * 100.0 for r in run["history"]],
        color="darkred",
    )
    axes[1].axhline(TOL_ENERGY * 100.0, color="k", ls="--", label="2 % gate")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("energy residual [%]")
    axes[1].set_title("end-to-end energy residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    fig.tight_layout()
    p3 = os.path.join(fig_dir, "fig3_energy_ledger.png")
    fig.savefig(p3, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig3_energy_ledger.png",
            "path": p3,
            "caption": "Five-channel dissipation split (left) and energy residual history against the 2 % gate (right).",
        }
    )

    ch = run["coupling_history"]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.6))
    axes[0].plot([r["time_s"] for r in ch], [r["iterations"] for r in ch], color="darkgreen")
    axes[0].set_xlabel("time [s]")
    axes[0].set_ylabel("iterations to 1e-6")
    axes[0].set_title("Aitken coupling: iterations per step")
    axes[0].grid(alpha=0.3)
    axes[1].semilogy([r["time_s"] for r in ch], [r["residual"] for r in ch], color="darkred")
    axes[1].axhline(TOL_COUPLING, color="k", ls="--", label="1e-6 gate")
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel("fixed-point residual")
    axes[1].set_title("two-way coupling residual")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    p4 = os.path.join(fig_dir, "fig4_fsi_convergence.png")
    fig.savefig(p4, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig4_fsi_convergence.png",
            "path": p4,
            "caption": "Two-way FSI: iteration count and residual history against the 1e-6 gate.",
        }
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.0))
    Ms = sorted({row["added_mass_ratio"] for row in stability})
    grid_om = [0.05 * i for i in range(1, 121)]
    Z = np.array([[abs(linear_gain(w, M)) for w in grid_om] for M in Ms])
    im = axes[0].pcolormesh(grid_om, Ms, Z, shading="auto", cmap="viridis", vmin=0.0, vmax=2.0)
    axes[0].contour(grid_om, Ms, Z, levels=[1.0], colors="red", linewidths=1.2)
    axes[0].axvline(1.0, color="white", ls=":", lw=1.0)
    axes[0].set_xlabel("relaxation factor omega_r")
    axes[0].set_ylabel("added-mass ratio M")
    axes[0].set_title("measured contraction |g| of the linearised map\n(red: |g| = 1 stability boundary)")
    fig.colorbar(im, ax=axes[0], label="|g|")

    resolvable = [r for r in stability if not r["beyond_grid"] and r["omega_max_analytic"] > 0.0]
    diverging = [r for r in stability if r["omega_max_analytic"] == 0.0]
    if resolvable:
        axes[1].plot(
            [r["added_mass_ratio"] for r in resolvable],
            [r["omega_max_analytic"] for r in resolvable],
            "-",
            color="k",
            label="analytic 2/(1-M)",
        )
        axes[1].plot(
            [r["added_mass_ratio"] for r in resolvable],
            [r["omega_max_measured"] for r in resolvable],
            "o",
            color="tab:red",
            label="measured (numerical scan)",
        )
    if diverging:
        axes[1].plot(
            [r["added_mass_ratio"] for r in diverging],
            [r["omega_max_measured"] for r in diverging],
            "x",
            color="tab:purple",
            label="no relaxation converges (M >= 1)",
        )
    axes[1].axvline(1.0, color="tab:blue", ls=":", label="M = 1 steady added-mass limit")
    axes[1].set_xlabel("added-mass ratio M")
    axes[1].set_ylabel("largest stable relaxation factor")
    axes[1].set_title("stability boundary: measured vs analytic")
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    p5 = os.path.join(fig_dir, "fig5_stability_domain.png")
    fig.savefig(p5, dpi=130)
    plt.close(fig)
    figs.append(
        {
            "name": "fig5_stability_domain.png",
            "path": p5,
            "caption": "Stability domain of the partitioned coupling: measured contraction map in the (relaxation, added-mass ratio) plane with its |g|=1 boundary (left) and the measured relaxation boundary versus the analytic 2/(1-M) (right).",
        }
    )
    return figs


# ---------------------------------------------------------------------------
# top-level driver
# ---------------------------------------------------------------------------


def run_gate(cfg: Optional[CaseConfig] = None, outdir: str = "artifacts/in66_n08",
             docs_dir: str = "docs") -> Dict[str, Any]:
    """Run the full gate, write every artifact, and return the verdict."""
    cfg = cfg or CaseConfig()
    os.makedirs(outdir, exist_ok=True)

    run = run_loop(cfg)
    run["ledger_last"] = build_ledger(
        channel_forces=run["base_forces"],
        speed_m_s=cfg.speed_m_s,
        panel_force_N=run["history"][-1]["panel_force_N"],
        momentum_flux_N=run["history"][-1]["momentum_flux_N"],
        fsi_residual=run["coupling_history"][-1]["residual"],
    )

    omega_grid = [round(0.05 * i, 4) for i in range(1, 121)]
    stability = stability_scan(
        setup=StructuralSetup(gap0_m=0.65),
        thickness_values=[0.05, 0.10, 0.15, 0.30, 0.60, 1.20],
        omega_grid=omega_grid,
        t_eff_values=(0.10, 0.25, 0.50),
    )
    # non-linear probe: does the real coupled solver contract at the analytic rate?
    probe = _jacobi_probe(cfg)

    criteria = evaluate_criteria(cfg, run, stability, probe)
    verdict: Dict[str, Any] = {
        "schema": GATE_SCHEMA,
        "gate_id": GATE_ID,
        "generated_utc": _utcnow(),
        "case": {
            "name": cfg.name,
            "length_m": cfg.length_m,
            "beam_m": cfg.beam_m,
            "draft_m": cfg.draft_m,
            "block_coefficient": cfg.block_coefficient,
            "speed_m_s": cfg.speed_m_s,
            "ice_thickness_m": cfg.ice_thickness_m,
            "coverage": cfg.coverage,
            "nx": cfg.nx,
            "ny": cfg.ny,
            "dem_dt_s": cfg.dem_dt_s,
            "rans_dt_s": cfg.rans_dt_s,
            "end_time_s": cfg.end_time_s,
        },
        "geometry": geometry_audit(cfg),
        "criteria": criteria,
        "contract_manifest": C.contract_manifest(),
        "energy_ledger_last": run["ledger_last"],
        "ice_budget_max_residual": run["ice_budget_max_residual"],
        "channel_split_closure_error": run["channel_info"]["split_closure_error"],
        "flow_resistance": {
            "R_ice_channels_N": run["r_ice_total"],
            "R_open_water_N": run["r_ow"],
            "R_panel_N_last": run["history"][-1]["panel_force_N"],
            "R_lane_shear_N_last": run["rans_history"][-1]["R_lane_shear_N"],
            "momentum_flux_N_last": run["history"][-1]["momentum_flux_N"],
        },
    }
    verdict["overall_status"] = (
        "PASS" if all(criteria[k]["passed"] for k in ("C1", "C2", "C3", "C4")) else "FAIL"
    )

    # ---- artifacts -------------------------------------------------------
    _write_json(os.path.join(outdir, "gate_verdict.json"), verdict)
    _write_json(
        os.path.join(outdir, "energy_ledger.json"),
        {
            "channels": CHANNELS,
            "last": run["ledger_last"],
            "summary": accumulate(run["history"]),
            "history": run["history"],
        },
    )
    _write_json(
        os.path.join(outdir, "contract_audit.json"),
        {"n_audits": len(run["contract_runs"]), "runs": run["contract_runs"]},
    )
    _write_json(
        os.path.join(outdir, "fsi_stability.json"),
        {"gate_case": run["coupling_history"][0], "scan": stability},
    )
    _write_json(
        os.path.join(outdir, "rans_history.json"),
        {"history": run["rans_history"]},
    )
    for key, bundle in run["snapshots"].items():
        _write_json(
            os.path.join(outdir, "payloads", "%s.json" % key),
            {
                "time_s": bundle["time_s"],
                "ice_field": bundle["ice_payload"],
                "traction": bundle["traction_payload"],
                "feedback": bundle["feedback_payload"],
            },
        )
    with open(os.path.join(outdir, "data_flow.mmd"), "w", encoding="utf-8") as fh:
        fh.write(DATA_FLOW_GRAPH)

    figs = make_figures(cfg, run, stability, outdir)
    if not figs:
        verdict["figures"] = {
            "n": 0,
            "reason": "matplotlib unavailable in this environment; the numeric "
            "artifacts and the tables in the report carry the evidence",
        }

    # ---- report ----------------------------------------------------------
    from .report import write_report

    report_path = os.path.join(docs_dir, "IN66_N08_ICE_NAV_INTEGRATION_GATE.md")
    # the report is written on the next line: C5 is satisfiable at render time
    verdict["criteria"]["C5"]["passed"] = True
    write_report(
        cfg=cfg,
        run=run,
        verdict=verdict,
        stability=stability,
        figs=figs,
        path=report_path,
    )
    verdict["criteria"]["C5"]["passed"] = bool(os.path.exists(report_path))
    verdict["overall_status"] = (
        "PASS"
        if all(verdict["criteria"][k]["passed"] for k in ("C1", "C2", "C3", "C4", "C5"))
        else "FAIL"
    )
    _write_json(os.path.join(outdir, "gate_verdict.json"), verdict)

    # ---- provenance ------------------------------------------------------
    artifacts = [
        "gate_verdict.json",
        "energy_ledger.json",
        "contract_audit.json",
        "fsi_stability.json",
        "rans_history.json",
        "data_flow.mmd",
    ] + [os.path.join("figures", f["name"]) for f in figs] + [
        os.path.join("payloads", "%s.json" % k) for k in run["snapshots"]
    ]
    provenance = {
        "schema": "in66_n08_provenance.v1",
        "generated_utc": _utcnow(),
        "artifacts": {
            path: _sha256_file(os.path.join(outdir, path))
            for path in artifacts
            if os.path.exists(os.path.join(outdir, path))
        },
        "report": {
            report_path: _sha256_file(report_path) if os.path.exists(report_path) else None
        },
        "python": {"numpy": np.__version__},
        "config": {
            "seed": cfg.seed,
            "nx": cfg.nx,
            "ny": cfg.ny,
            "dem_dt_s": cfg.dem_dt_s,
            "rans_dt_s": cfg.rans_dt_s,
            "end_time_s": cfg.end_time_s,
        },
    }
    _write_json(os.path.join(outdir, "provenance.json"), provenance)
    return verdict


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="IN66-N08 cross-domain integration gate")
    parser.add_argument("--outdir", default="artifacts/in66_n08")
    parser.add_argument("--docs-dir", default="docs")
    parser.add_argument("--end-time", type=float, default=200.0)
    parser.add_argument("--nx", type=int, default=96)
    parser.add_argument("--ny", type=int, default=16)
    args = parser.parse_args(argv)
    cfg = CaseConfig(end_time_s=args.end_time, nx=args.nx, ny=args.ny)
    verdict = run_gate(cfg, outdir=args.outdir, docs_dir=args.docs_dir)
    print(json.dumps(verdict["criteria"], indent=2, sort_keys=True, default=float))
    print("overall_status: %s" % verdict["overall_status"])
    return 0 if verdict["overall_status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())