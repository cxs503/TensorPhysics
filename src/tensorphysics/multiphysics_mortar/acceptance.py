"""C1..C7 acceptance driver for campaign node ``P162W1-N06``.

Criteria (campaign table, verbatim thresholds):

C1  1D scalar transport mass residual ``R < 1e-12``
C2  2D FSI momentum ``R_p < 1e-8``, energy ``R_e < 1e-6``
C3  3D flow-ice mass ``R_m < 1e-6``, momentum ``R_p < 1e-5``
C4  matching meshes -> mortar degenerates to the identity (defect ``< 1e-12``)
C5  refinement -> conservation residual decreases monotonically
C6  vs nearest-neighbour interpolation the mortar residual is ``>= 3`` orders lower
C7  mortar construction time ``<= 10 %`` of the coupling computation time

Everything the driver reports is computed here; the reporting scripts only format
it, so no number in the report is hand-copied.
"""
from __future__ import annotations

import time
from typing import Sequence

import numpy as np

from .assembly import MortarTransfer
from .cases import (InterfaceCase, flow_ice_3d_case, fsi_2d_case,
                    matching_2d_case, matching_line_case, refined_flow_ice_case,
                    scalar_transport_1d_case)
from .conservation import (MaterialState, conservation_residuals,
                           local_balance_defect)
from .gate import GATE_TOLERANCES, ConservationGate

__all__ = [
    "C6_MIN_RATIO",
    "N_EXCHANGE_STEPS",
    "REFINEMENT_LEVELS",
    "run_acceptance",
    "state_from_conserved",
    "transport_conserved",
]

N_EXCHANGE_STEPS = 200
REFINEMENT_LEVELS: tuple[tuple[int, int], ...] = ((3, 6), (6, 12), (12, 24), (24, 48))
C6_MIN_RATIO = 1.0e3
N_COUPLED_STEPS = 200
COUPLED_GRID = 64
COUPLED_SWEEPS = 60
_FLOOR = 1e-300
# denominator floor for the C6 ratio: once the mortar residual is at round-off the
# ratio is only a lower bound, so it is capped here instead of exploding to 1e300
_RATIO_FLOOR = 1e-15


def state_from_conserved(mass: np.ndarray, momentum: np.ndarray, energy: np.ndarray,
                         specific_heat: float) -> MaterialState:
    """Recover ``(rho, u, T)`` from the transported conservative quantities."""
    rho = np.asarray(mass, dtype=float).ravel()
    mom = np.asarray(momentum, dtype=float)
    if mom.ndim == 1:
        mom = mom[:, None]
    en = np.asarray(energy, dtype=float).ravel()
    if np.any(rho <= 0.0):
        raise ValueError("non-positive transported density")
    vel = mom / rho[:, None]
    kinetic = np.sum(mom ** 2, axis=1) / (2.0 * rho)
    temp = (en - kinetic) / (rho * specific_heat)
    return MaterialState(density=rho, velocity=vel, temperature=temp,
                         specific_heat=specific_heat)


def transport_conserved(operator, state: MaterialState) -> MaterialState:
    """Send ``(rho, rho u, rho E)`` through ``operator`` and rebuild the state."""
    q = state.quantities()
    out = {name: np.asarray(operator.apply(vals), dtype=float)
           for name, vals in q.items()}
    return state_from_conserved(out["mass"][:, 0], out["momentum"],
                                out["energy"][:, 0], state.specific_heat)


def _residual_map(residuals) -> dict[str, float]:
    return {name: res.residual for name, res in residuals.items()}


def _operator_diagnostics(transfer: MortarTransfer) -> dict:
    asm = transfer.assembly
    return {
        "src_n": int(asm.src.n),
        "tgt_n": int(asm.tgt.n),
        "n_pieces": int(asm.n_pieces),
        "src_partition_of_unity_error": asm.src.partition_of_unity_error(),
        "tgt_partition_of_unity_error": asm.tgt.partition_of_unity_error(),
        "biorthogonality_defect": transfer.biorthogonality_defect(),
        "dual_partition_of_unity_defect": transfer.dual_partition_of_unity_defect(),
        "measure_row_sum_defect": asm.row_sum_defect(),
        "build_seconds": asm.seconds,
    }


def run_case(case: InterfaceCase, subdivisions: int = 1) -> dict:
    """Forward (master->slave) and reverse (slave->master) residuals + baseline."""
    forward = case.transfer(subdivisions)
    slave_state = transport_conserved(forward, case.master_state)
    fwd_res = conservation_residuals(case.master_space, case.master_state,
                                     case.slave_space, slave_state)

    reverse = MortarTransfer(case.slave_space, case.master_space,
                             subdivisions=subdivisions, label=case.key + "-rev")
    master_state = transport_conserved(reverse, case.slave_state)
    rev_res = conservation_residuals(case.slave_space, case.slave_state,
                                     case.master_space, master_state)

    nn = case.baseline()
    nn_state = transport_conserved(nn, case.master_state)
    nn_res = conservation_residuals(case.master_space, case.master_state,
                                    case.slave_space, nn_state)

    local = local_balance_defect(forward.assembly, case.master_state, slave_state)

    forward_map = _residual_map(fwd_res)
    nn_map = _residual_map(nn_res)
    ratios = {name: (nn_map[name] / max(forward_map[name], _RATIO_FLOOR))
              for name in forward_map}
    return {
        "key": case.key,
        "title": case.title,
        "parameters": case.parameters,
        "residuals_master_to_slave": forward_map,
        "residuals_slave_to_master": _residual_map(rev_res),
        "integrals_master_to_slave": {name: res.to_dict() for name, res in fwd_res.items()},
        "nearest_neighbor_residuals": nn_map,
        "mortar_over_nearest_neighbor": ratios,
        "operator": _operator_diagnostics(forward),
        "local_balance": {
            name: {"max": float(np.max(vals)), "sum": float(np.sum(vals)),
                   "per_dof": [float(v) for v in vals]}
            for name, vals in local.items()
        },
    }


def _identity_defect(case: InterfaceCase) -> dict:
    transfer = case.transfer()
    defects = {}
    for name, values in case.master_state.quantities().items():
        out = np.asarray(transfer.apply(values), dtype=float)
        den = max(float(np.max(np.abs(values))), _FLOOR)
        defects[name] = float(np.max(np.abs(out - values))) / den
    return {"max_over_quantities": max(defects.values()), "per_quantity": defects}


def _refinement_study() -> dict:
    rows = []
    for nu_master, nu_slave in REFINEMENT_LEVELS:
        case = refined_flow_ice_case(nu_master, nu_slave)
        payload = run_case(case)
        rows.append({
            "nu_master": nu_master, "nu_slave": nu_slave,
            "h": 0.9 / nu_slave,
            "residuals": payload["residuals_master_to_slave"],
            "n_pieces": payload["operator"]["n_pieces"],
        })
    mass = [row["residuals"]["mass"] for row in rows]
    momentum = [row["residuals"]["momentum"] for row in rows]
    deltas = [mass[i] - mass[i + 1] for i in range(len(mass) - 1)]
    return {
        "levels": rows,
        "monotone_mass": all(d > 0.0 for d in deltas),
        "monotone_momentum": all(momentum[i] > momentum[i + 1]
                                 for i in range(len(momentum) - 1)),
        "mass_ratios": [mass[i] / max(mass[i + 1], _FLOOR) for i in range(len(mass) - 1)],
        "observed_order_mass": _fit_order([row["h"] for row in rows], mass),
    }


def _fit_order(h: Sequence[float], residual: Sequence[float]) -> float | None:
    """Least-squares slope of ``log(residual)`` against ``log(h)``."""
    h = np.asarray(h, dtype=float)
    r = np.asarray(residual, dtype=float)
    mask = r > 0.0
    if mask.sum() < 2:
        return None
    slope = np.polyfit(np.log(h[mask]), np.log(r[mask]), 1)[0]
    return float(slope)


def _edge_nodes(space, axis: int = 1, value: float = 0.0, tol: float = 1e-9):
    """Interface-edge DOFs of a patch and their chart abscissa (sorted)."""
    coords = space.node_coords
    ids = np.nonzero(np.abs(coords[:, axis] - value) <= tol)[0]
    if ids.size < 2:
        raise ValueError("interface edge not found on the patch boundary")
    order = np.argsort(coords[ids, 0])
    ids = ids[order]
    return ids, coords[ids, 0]


def _timing_study(case: InterfaceCase, steps: int = N_COUPLED_STEPS,
                  grid: int = COUPLED_GRID, sweeps: int = COUPLED_SWEEPS) -> dict:
    """C7: mortar construction vs a *coupled run*, not vs a bare matvec loop.

    The exchange is only meaningful inside a coupled computation, so the benchmark
    runs a block-Gauss--Seidel partitioned loop in which the mortar output actually
    drives the solvers: each step transfers the conservative state to the ice side,
    sets the interface Dirichlet data of a 2D thermal solve from it, returns the
    ice-side response through the reverse mortar and solves the flow side the same
    way.  Both solves use the repository's own red-black Gauss--Seidel kernel
    (:mod:`tensorphysics.multigrid_poisson`), so the per-step cost is a measured
    implicit-solver cost, not a synthetic sleep.  The bare-matvec fraction is
    reported as well -- it is the *worst case* for amortisation.
    """
    from ..multigrid_poisson.core import red_black_gs_sweeps

    t0 = time.perf_counter()
    forward = MortarTransfer(case.master_space, case.slave_space, label="C7-fwd")
    reverse = MortarTransfer(case.slave_space, case.master_space, label="C7-rev")
    t_setup = time.perf_counter() - t0

    # bare-exchange reference (no solver work): worst case for amortisation
    q_master = case.master_state.quantities()
    q_slave = case.slave_state.quantities()
    forward.apply(q_master["mass"])
    reverse.apply(q_slave["mass"])
    t0 = time.perf_counter()
    for _ in range(steps):
        for values in q_master.values():
            forward.apply(values)
        for values in q_slave.values():
            reverse.apply(values)
    t_bare = time.perf_counter() - t0

    # coupled loop with solver work
    slave_edge_ids, slave_theta = _edge_nodes(case.slave_space)
    master_edge_ids, master_theta = _edge_nodes(case.master_space)
    theta_grid = np.linspace(master_theta[0], master_theta[-1], grid)
    h = 1.0 / (grid - 1)
    f = np.zeros((grid, grid))
    u_ice = np.zeros((grid, grid))
    u_flow = np.zeros((grid, grid))

    t0 = time.perf_counter()
    for _ in range(steps):
        ice_state = transport_conserved(forward, case.master_state)
        u_ice[0, :] = np.interp(theta_grid, slave_theta,
                                ice_state.temperature[slave_edge_ids])
        u_ice = red_black_gs_sweeps(u_ice, f, h, sweeps)

        back = ice_state.temperature.copy()
        back[slave_edge_ids] = np.interp(slave_theta, theta_grid, u_ice[1, :])
        ice_response = MaterialState(density=ice_state.density,
                                     velocity=ice_state.velocity,
                                     temperature=back,
                                     specific_heat=ice_state.specific_heat)
        flow_state = transport_conserved(reverse, ice_response)
        u_flow[0, :] = np.interp(theta_grid, master_theta,
                                 flow_state.temperature[master_edge_ids])
        u_flow = red_black_gs_sweeps(u_flow, f, h, sweeps)
    t_coupled = time.perf_counter() - t0

    total = t_setup + t_coupled
    per_step = t_coupled / steps
    return {
        "definition": ("coupled block-Gauss-Seidel loop: 1 thermal solve per side "
                       "(red-black GS, %d sweeps, %dx%d grid) + mortar exchange both "
                       "ways per step" % (sweeps, grid, grid)),
        "steps": steps,
        "setup_seconds": t_setup,
        "coupled_seconds": t_coupled,
        "total_seconds": total,
        "setup_fraction": t_setup / total,
        "per_step_seconds": per_step,
        "break_even_steps": 9.0 * t_setup / per_step,
        "bare_exchange": {
            "steps": steps,
            "exchange_seconds": t_bare,
            "setup_fraction": t_setup / (t_setup + t_bare),
            "note": "matvec-only exchange; upper bound on the setup share, not a coupled computation",
        },
    }


def _l2_error(space, values: np.ndarray, profile) -> float:
    """Relative L2 error of an interface field against an analytic profile."""
    pts = space.nodes if hasattr(space, "nodes") else space.node_coords
    exact = np.asarray(profile(pts)["density"], dtype=float).ravel()
    got = np.asarray(values, dtype=float).ravel()
    w = space.measures()
    den = float(np.sqrt(np.sum(w * exact ** 2)))
    return float(np.sqrt(np.sum(w * (got - exact) ** 2))) / max(den, _FLOOR)


def _accuracy_study(levels: Sequence[int] = (16, 32, 64, 128)) -> dict:
    """Consistency (not conservation): transfer error against the analytic field.

    Both sides are refined together, so the error of the transferred density is
    the *interface* interpolation error of the coupled pair.  The mortar projects
    onto the receiving P1 space (2nd order); the nearest-neighbour baseline is
    first order.  This is the evidence that conservation is not bought by
    smearing the field.
    """
    from .cases import _profile_1d  # noqa: PLC0415 - local analytic profile

    rows = []
    for n_master in levels:
        case = scalar_transport_1d_case(master_nodes=n_master,
                                        slave_nodes=n_master // 2)
        mortar = case.transfer()
        nn = case.baseline()
        rows.append({
            "master_nodes": n_master,
            "slave_nodes": n_master // 2,
            "h": 1.0 / (n_master // 2),
            "mortar_l2": _l2_error(case.slave_space, mortar.apply(case.master_state.density),
                                   _profile_1d),
            "nearest_neighbor_l2": _l2_error(case.slave_space, nn.apply(case.master_state.density),
                                             _profile_1d),
        })
    return {
        "levels": rows,
        "order_mortar": _fit_order([row["h"] for row in rows],
                                   [row["mortar_l2"] for row in rows]),
        "order_nearest_neighbor": _fit_order([row["h"] for row in rows],
                                             [row["nearest_neighbor_l2"] for row in rows]),
    }


def run_acceptance(subdivisions: int = 1) -> dict:
    """Run every criterion and return the JSON-ready payload."""
    case_c1 = scalar_transport_1d_case()
    case_c2 = fsi_2d_case()
    case_c3 = flow_ice_3d_case()

    payload_cases = {
        case.key: run_case(case, subdivisions) for case in (case_c1, case_c2, case_c3)
    }

    identity_1d = _identity_defect(matching_line_case())
    identity_2d = _identity_defect(matching_2d_case())
    refinement = _refinement_study()
    accuracy = _accuracy_study()
    timing = _timing_study(case_c3)

    # --- declared gate on the headline residuals ------------------------
    measured = {
        "C1_1d_mass": payload_cases["C1_1d_scalar_transport"]["residuals_master_to_slave"]["mass"],
        "C2_2d_momentum": payload_cases["C2_2d_fsi_panel"]["residuals_master_to_slave"]["momentum"],
        "C2_2d_energy": payload_cases["C2_2d_fsi_panel"]["residuals_master_to_slave"]["energy"],
        "C3_3d_mass": payload_cases["C3_3d_flow_ice_patch"]["residuals_master_to_slave"]["mass"],
        "C3_3d_momentum": payload_cases["C3_3d_flow_ice_patch"]["residuals_master_to_slave"]["momentum"],
        "C4_identity_1d": identity_1d["max_over_quantities"],
        "C4_identity_2d": identity_2d["max_over_quantities"],
    }
    gate_report = ConservationGate().evaluate(measured)

    ratios = {}
    for key, payload in payload_cases.items():
        for name, value in payload["mortar_over_nearest_neighbor"].items():
            ratios[f"{key}:{name}"] = value
    c6_ratio_min = min(ratios.values())

    criteria = {
        "C1": {"description": "1D scalar transport mass residual < 1e-12",
               "value": measured["C1_1d_mass"], "threshold": 1e-12,
               "passed": bool(measured["C1_1d_mass"] <= 1e-12)},
        "C2": {"description": "2D FSI momentum < 1e-8, energy < 1e-6",
               "value": {"momentum": measured["C2_2d_momentum"],
                         "energy": measured["C2_2d_energy"]},
               "threshold": {"momentum": 1e-8, "energy": 1e-6},
               "passed": bool(measured["C2_2d_momentum"] <= 1e-8
                              and measured["C2_2d_energy"] <= 1e-6)},
        "C3": {"description": "3D flow-ice mass < 1e-6, momentum < 1e-5",
               "value": {"mass": measured["C3_3d_mass"],
                         "momentum": measured["C3_3d_momentum"]},
               "threshold": {"mass": 1e-6, "momentum": 1e-5},
               "passed": bool(measured["C3_3d_mass"] <= 1e-6
                              and measured["C3_3d_momentum"] <= 1e-5)},
        "C4": {"description": "matching meshes -> identity mapping, defect < 1e-12",
               "value": {"1d": identity_1d["max_over_quantities"],
                         "2d": identity_2d["max_over_quantities"]},
               "threshold": 1e-12,
               "passed": bool(identity_1d["max_over_quantities"] <= 1e-12
                              and identity_2d["max_over_quantities"] <= 1e-12)},
        "C5": {"description": "mesh refinement -> monotonically decreasing residual",
               "value": {"monotone_mass": refinement["monotone_mass"],
                         "monotone_momentum": refinement["monotone_momentum"],
                         "observed_order_mass": refinement["observed_order_mass"]},
               "threshold": "strict decrease over 4 levels",
               "passed": bool(refinement["monotone_mass"])},
        "C6": {"description": "mortar residual >= 3 orders below nearest neighbour",
               "value": {"min_ratio": c6_ratio_min, "ratios": ratios},
               "threshold": C6_MIN_RATIO,
               "passed": bool(c6_ratio_min >= C6_MIN_RATIO)},
        "C7": {"description": "construction time <= 10% of coupling computation time",
               "value": timing["setup_fraction"],
               "threshold": 0.10,
               "passed": bool(timing["setup_fraction"] <= 0.10)},
    }

    return {
        "node": "P162W1-N06",
        "campaign": "P162 多物理场耦合",
        "repository": "TensorPhysics",
        "module": "tensorphysics.multiphysics_mortar",
        "operator_family": "P1 trace space, L2-projection mortar with dual (biorthogonal) basis",
        "gate_tolerances": dict(GATE_TOLERANCES),
        "gate": gate_report.to_dict(),
        "cases": payload_cases,
        "identity_controls": {"1d": identity_1d, "2d": identity_2d},
        "refinement": refinement,
        "accuracy": accuracy,
        "timing": timing,
        "criteria": criteria,
        "all_criteria_passed": all(item["passed"] for item in criteria.values()),
    }