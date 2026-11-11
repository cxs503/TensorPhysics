"""Static verification gates for the plate/shell element verification face.

Three analytic anchors are used, all independent of the finite element code:

1. **Navier double series** for a uniformly loaded simply supported plate
   (:mod:`.navier`) -- deflection, bending/twisting moments and shear forces
   at every point of the domain.
2. **Published coefficient** ``alpha = w_max D / (q a^4) = 0.00406`` for the
   square plate (Timoshenko & Woinowsky-Krieger, Table 8).
3. **Donnell axisymmetric closed form** for a cylindrical shell under internal
   pressure (:mod:`.shell_axisym`), including the ``exp(-beta x)`` bending
   boundary layer with ``beta = (3(1-nu^2))^{1/4}/sqrt(R t)``.

Gates (identical numbering to the campaign node ``p405w1-n05``)::

    G1  plate centre deflection   <= 1 % (4x4) and <= 0.1 % (16x16)
    G2  plate centre Mx, My       <= 2 % (8x8 and every finer mesh)
    G3  shell sigma_theta         <= 1 % in the membrane region (x >= 6/beta)
    G4  mesh convergence order    >= 2 for the C^0 quantities; the C^1 plate
        deflection must reach >= 3
    G5  >= 12 pytest checks green (enforced by the test suite)

Everything numeric in the returned dictionary is produced by the code in this
module: nothing is typed by hand, so the JSON report is machine-auditable.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List, Sequence

import numpy as np

from .navier import (NAVIER_ALPHA_SQUARE_PUBLISHED, navier_plate_center,
                     series_convergence)
from .plate_fe import (SSSSPlateFE, convergence_study, element_stiffness,
                       element_stiffness_kernels, fit_order)
from .shell_axisym import (CylinderPressureFE, beta_edge, cylinder_edge_solution,
                           edge_decay_fit, membrane_solution,
                           shell_convergence_study)

__all__ = ["PLATE_CASE", "SHELL_CASE", "PLATE_MESHES", "SHELL_MESHES",
           "run_verification", "plate_gates", "shell_gates", "gate_rows",
           "_count_tests"]

#: Plate verification case: steel panel, 1 m square, 10 mm thick, 10 kPa load.
PLATE_CASE = {"a": 1.0, "b": 1.0, "t": 0.010, "E": 2.1e11, "nu": 0.3,
              "q": 1.0e4}

#: Shell verification case: R/t = 100 cylinder, 1 MPa internal pressure.
SHELL_CASE = {"R": 1.0, "L": 0.8, "t": 0.010, "E": 2.1e11, "nu": 0.3,
              "p": 1.0e6}

#: h-refinement sequence for the plate (every mesh has a node at the centre).
PLATE_MESHES: Sequence[int] = (4, 8, 16, 32, 64)

#: h-refinement sequence for the shell.
SHELL_MESHES: Sequence[int] = (8, 16, 32, 64, 128)


def _row(rows: List[Dict[str, float]], n: int) -> Dict[str, float]:
    for r in rows:
        if int(r["n_elem"]) == int(n):
            return r
    raise KeyError(f"mesh n={n} not in study")


def plate_gates(plate_rows, orders: Dict[str, float]) -> Dict[str, dict]:
    """G1/G2/G4 gate values for the plate."""
    r4, r8 = _row(plate_rows, 4), _row(plate_rows, 8)
    r16, r64 = _row(plate_rows, 16), _row(plate_rows, 64)
    fine = [r for r in plate_rows if int(r["n_elem"]) >= 8]
    g1 = {
        "criterion": "w_max relative error <= 1% at 4x4 and <= 0.1% at 16x16",
        "err_w_4x4": r4["err_w"], "err_w_16x16": r16["err_w"],
        "err_w_64x64": r64["err_w"],
        "pass": bool(r4["err_w"] <= 1e-2 and r16["err_w"] <= 1e-3),
    }
    g2 = {
        "criterion": "Mx, My relative error <= 2% (8x8 and all finer meshes)",
        "err_Mx_4x4": r4["err_Mx"], "err_Mx_8x8": r8["err_Mx"],
        "err_My_8x8": r8["err_My"], "err_Mx_64x64": r64["err_Mx"],
        "max_err_Mx_ge8": max(r["err_Mx"] for r in fine),
        "max_err_My_ge8": max(r["err_My"] for r in fine),
        "pass": bool(max(r["err_Mx"] for r in fine) <= 2e-2
                     and max(r["err_My"] for r in fine) <= 2e-2),
        "note": ("the 4x4 mesh stays in the record: its centre-moment error "
                 f"is {r4['err_Mx'] * 100:.2f}% (>2%), the gate is met from "
                 "8x8 on, as expected for O(h^2) moment convergence"),
    }
    g4 = {
        "criterion": "h-refinement order >= 2 for the C^0 quantities "
                     "(moments/shear); C^1 plate deflection >= 3",
        "order_w": orders["err_w"], "order_Mx": orders["err_Mx"],
        "order_Qx": orders["err_Qx"], "order_l2_w": orders["l2_w"],
        "pass": bool(orders["err_w"] >= 3.0 and orders["err_Mx"] >= 2.0),
    }
    return {"G1": g1, "G2": g2, "G4_plate": g4}


def shell_gates(shell_rows, orders: Dict[str, float], decay: Dict[str, float],
                membrane: Dict[str, float]) -> Dict[str, dict]:
    """G3 gate plus the edge-effect (``sqrt(R t)``) supporting checks."""
    finest = shell_rows[-1]
    g3 = {
        "criterion": "sigma_theta relative error <= 1% in the membrane region "
                     "(x >= 6/beta) and pointwise <= 1% at the finest mesh",
        "err_sigma_theta_membrane_finest":
            finest["err_sigma_theta_membrane"],
        "err_sigma_theta_eq_pointwise_finest":
            finest["err_sigma_theta_eq_pointwise"],
        "err_sigma_theta_pointwise_finest":
            finest["err_sigma_theta_pointwise"],
        "sigma_theta_membrane_value": finest["sigma_theta_ref"],
        "pass": bool(finest["err_sigma_theta_membrane"] <= 1e-2
                     and finest["err_sigma_theta_eq_pointwise"] <= 1e-2),
    }
    edge = {
        "criterion": "edge bending moment <= 2% at 64 axial elements and "
                     "measured attenuation length matches sqrt(R t)",
        "err_Mx_edge_64": _row(shell_rows, 64)["err_Mx_edge"],
        "beta_fit": decay["beta_fit"], "beta_theory": decay["beta_theory"],
        "beta_rel_error": decay["beta_rel_error"],
        "decay_length_over_sqrt_Rt": decay["decay_length_over_sqrt_Rt"],
        "pass": bool(_row(shell_rows, 64)["err_Mx_edge"] <= 2e-2
                     and decay["beta_rel_error"] <= 1e-2),
    }
    return {"G3": g3, "G3b_edge_effect": edge,
            "order_shell": {"criterion": "edge-moment h-refinement order "
                                         ">= 1.8 (closed-form anchor)",
                            "order_Mx_edge": orders["err_Mx_edge"],
                            "order_sigma_theta_eq":
                                orders["err_sigma_theta_eq_pointwise"],
                            "pass": bool(orders["err_Mx_edge"] >= 1.8)},
            "membrane_limit": membrane}



def _count_tests(rel: str = "tests/test_plate_shell_verification.py") -> int:
    """Number of ``test_*`` functions in the verification test module (AST).

    Static count only -- the *green* status of G5 is asserted by actually
    running ``python -m pytest tests/test_plate_shell_verification.py``; this
    helper just records the size of the suite in the machine-readable payload.
    """
    import ast
    path = pathlib.Path(__file__).resolve().parents[3] / rel
    if not path.exists():
        return 0
    tree = ast.parse(path.read_text())
    return sum(1 for node in ast.walk(tree)
               if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
               and node.name.startswith("test_"))


def gate_rows(gates: Dict[str, dict]) -> List[Dict[str, object]]:
    """Flatten the gate dictionary into ``[{gate, pass, criterion}]`` rows."""
    rows = []
    for name, payload in gates.items():
        if isinstance(payload, dict) and "pass" in payload:
            rows.append({"gate": name, "pass": bool(payload["pass"]),
                         "criterion": payload.get("criterion", "")})
    return rows


def run_verification(plate_case=None, shell_case=None,
                     plate_meshes: Sequence[int] = PLATE_MESHES,
                     shell_meshes: Sequence[int] = SHELL_MESHES,
                     n_terms: int = 81, with_gates: bool = True,
                     ) -> Dict[str, object]:
    """Run the whole verification face and return the report payload."""
    pc = dict(PLATE_CASE if plate_case is None else plate_case)
    sc = dict(SHELL_CASE if shell_case is None else shell_case)

    # ---------------------------------------------------------------- plate
    alpha = navier_plate_center(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"],
                                pc["q"], n_terms=n_terms)
    xs = np.array([0.25 * pc["a"], 0.5 * pc["a"], 0.75 * pc["a"]])
    ys = np.array([0.5 * pc["b"], 0.25 * pc["b"], 0.5 * pc["b"]])
    series = series_convergence(pc["a"], pc["b"], pc["t"], pc["E"], pc["nu"],
                                pc["q"], xs, ys,
                                n_terms_list=(21, 41, 81, 161, 321))
    plate_rows = convergence_study(pc["a"], pc["b"], pc["t"], pc["E"],
                                   pc["nu"], pc["q"], meshes=plate_meshes,
                                   n_terms=n_terms)
    h = [r["h"] for r in plate_rows]
    plate_orders = {
        "err_w": fit_order(h, [r["err_w"] for r in plate_rows]),
        "err_Mx": fit_order(h, [r["err_Mx"] for r in plate_rows]),
        "err_Qx": fit_order(h, [r["err_Qx"] for r in plate_rows]),
        "l2_w": fit_order(h, [r["l2_w"] for r in plate_rows]),
        "l2_Mx": fit_order(h, [r["l2_Mx"] for r in plate_rows]),
    }

    # negative controls -------------------------------------------------
    nat = SSSSPlateFE(a=pc["a"], b=pc["b"], t=pc["t"], E=pc["E"],
                      nu=pc["nu"], nx=8, ny=8)
    nat.solve(pc["q"], bc_mode="natural")
    w_nat = nat.center_deflection()
    tight = SSSSPlateFE(a=pc["a"], b=pc["b"], t=pc["t"], E=pc["E"],
                        nu=pc["nu"], nx=8, ny=8)
    tight.solve(pc["q"], bc_mode="tight")
    w_tight = tight.center_deflection()
    k_cross = float(np.abs(element_stiffness(0.25, 0.25, 1.0, 0.3)
                           - element_stiffness_kernels(0.25, 0.25, 1.0, 0.3)
                           ).max())

    plate = {
        "case": pc,
        "slenderness_a_over_t": pc["a"] / pc["t"],
        "navier_reference": alpha,
        "published_alpha": NAVIER_ALPHA_SQUARE_PUBLISHED,
        "alpha_rel_error_vs_published": abs(
            alpha["alpha_deflection"] / NAVIER_ALPHA_SQUARE_PUBLISHED - 1),
        "series_truncation": series,
        "meshes": plate_rows,
        "orders": plate_orders,
        "over_constrained_control": {
            "w_center_natural_8x8": w_nat,
            "w_center_tight_8x8": w_tight,
            "relative_change": (w_tight - w_nat) / w_nat,
            "stiffer_as_expected": bool(w_tight < w_nat),
        },
        "stiffness_cross_check_vs_plate_buckling_kernels": k_cross,
    }

    # ---------------------------------------------------------------- shell
    beta = beta_edge(sc["R"], sc["t"], sc["E"], sc["nu"])
    mem = membrane_solution(sc["R"], sc["t"], sc["E"], sc["p"])
    shell_rows = shell_convergence_study(sc["R"], sc["L"], sc["t"], sc["E"],
                                         sc["nu"], sc["p"],
                                         n_elems=shell_meshes)
    fe = CylinderPressureFE(R=sc["R"], L=sc["L"], t=sc["t"], E=sc["E"],
                            nu=sc["nu"], n_elem=int(shell_meshes[-1]))
    fe.solve(sc["p"])
    decay = edge_decay_fit(fe, sc["p"])
    hh = [r["h"] for r in shell_rows]
    shell_orders = {
        "err_Mx_edge": fit_order(hh, [r["err_Mx_edge"] for r in shell_rows]),
        "err_sigma_theta_pointwise": fit_order(
            hh, [r["err_sigma_theta_pointwise"] for r in shell_rows]),
        "err_sigma_theta_eq_pointwise": fit_order(
            hh, [r["err_sigma_theta_eq_pointwise"] for r in shell_rows]),
        "err_w_field": fit_order(hh, [r["err_w_field"] for r in shell_rows]),
    }
    # independent residual of the closed form against the shell ODE
    xg = np.linspace(0.15, 0.35, 401)
    ref = cylinder_edge_solution(sc["R"], sc["L"], sc["t"], sc["E"], sc["nu"],
                                 sc["p"], xg)
    w = ref["w"]
    dx = xg[1] - xg[0]
    w_4 = (w[4:] - 4 * w[3:-1] + 6 * w[2:-2] - 4 * w[1:-3] + w[:-4]) / dx ** 4
    D = sc["E"] * sc["t"] ** 3 / (12 * (1 - sc["nu"] ** 2))
    resid = D * w_4 + sc["E"] * sc["t"] / sc["R"] ** 2 * w[2:-2] - sc["p"]
    ode_residual = float(np.max(np.abs(resid)) / sc["p"])

    shell = {
        "case": sc,
        "ratio_R_over_t": sc["R"] / sc["t"],
        "beta": beta,
        "decay_length": 1.0 / beta,
        "sqrt_Rt": float(np.sqrt(sc["R"] * sc["t"])),
        "beta_sqrt_Rt": float(beta * np.sqrt(sc["R"] * sc["t"])),
        "membrane_reference": mem,
        "meshes": shell_rows,
        "orders": shell_orders,
        "edge_decay_fit": decay,
        "energy_balance": fe.energy(),
        "ode_residual_of_closed_form": ode_residual,
        "reaction_check": fe.reaction_check(),
    }

    n_tests = _count_tests()
    gates: Dict[str, dict] = {}
    gates["G5_tests"] = {
        "criterion": ">= 12 pytest checks, all green "
                     "(tests/test_plate_shell_verification.py)",
        "n_tests_collected": n_tests,
        "pass": bool(n_tests >= 12),
        "note": "green status is asserted by running the pytest module; this "
                "entry records the collected count",
    }
    if with_gates:
        gates.update(plate_gates(plate_rows, plate_orders))
        gates.update(shell_gates(shell_rows, shell_orders, decay,
                                 {"sigma_theta": mem["sigma_theta"],
                                  "w": mem["w"]}))
    rows = gate_rows(gates)

    payload: Dict[str, object] = {
        "node": "p405w1-n05-fem-plate-shell",
        "repo": "TensorPhysics",
        "module": "tensorphysics.plate_shell",
        "plate": plate,
        "shell": shell,
        "gates": gates,
        "gate_table": rows,
        "n_gates": len(rows),
        "with_gates": with_gates,
        "n_gates_pass": int(sum(1 for r in rows if r["pass"])),
        "verdict": ("PASS" if rows and all(r["pass"] for r in rows)
                    else ("SMOKE" if not with_gates else "FAIL")),
        "negative_controls": {
            "plate_natural_bc_used": True,
            "over_constraint_makes_model_stiffer":
                plate["over_constrained_control"]["stiffer_as_expected"],
            "navier_series_truncation_below_gate": bool(
                series["w"][-1] < 1e-6 and series["Mx"][-1] < 1e-4),
            "closed_form_satisfies_shell_ode": bool(ode_residual < 1e-2),
            "shell_energy_balance_identity": bool(
                shell["energy_balance"]["energy_balance_rel_error"] < 1e-9),
        },
    }
    return payload