"""Acceptance gate for the conservative field-exchange operators.

``run_acceptance()`` executes the three verification cases and returns a JSON
payload in which every criterion carries its measured value, its threshold, the
comparison relation and its pass/fail verdict.  Nothing is hand-typed downstream:
the report and the figures are generated from this payload.
"""
from __future__ import annotations

import platform
import time

import numpy as np

from .verification import case_contact, case_couette, case_poiseuille

__all__ = ["CRITERIA", "run_acceptance"]

# (id, quantity path description, threshold, relation)
CRITERIA = [
    ("C1_FORCE_PANEL", "LBM wall-traction -> membrane: relative force residual",
     1.0e-6, "<="),
    ("C2_MOMENT_PANEL", "LBM wall-traction -> membrane: relative moment residual",
     1.0e-6, "<="),
    ("C3_FORCE_CONTACT", "DEM contact -> plate: relative force residual", 1.0e-6, "<="),
    ("C4_MOMENT_CONTACT", "DEM contact -> plate: relative moment residual", 1.0e-6, "<="),
    ("C5_NEGCTRL_MOMENT", "negative control (nearest-node lumping) moment residual "
     "must exceed the 1% band", 1.0e-2, ">="),
    ("C6_LBM_COUETTE_PROFILE", "LBM Couette velocity profile vs U*y/h (relative L2)",
     1.0e-2, "<="),
    ("C7_LBM_COUETTE_TAU", "LBM Couette wall shear vs mu*U/h", 3.0e-2, "<="),
    ("C8_MEMBRANE_TIP", "membrane tip displacement vs tau*L^2/(2*E*t)", 1.0e-2, "<="),
    ("C9_LBM_POISEUILLE_PROFILE", "LBM Poiseuille profile vs parabola (relative L2)",
     1.0e-2, "<="),
    ("C10_LBM_POISEUILLE_TAU", "LBM Poiseuille wall shear vs rho*g*h/2", 3.0e-2, "<="),
    ("C11_PLATE_NAVIER", "mapped pressure -> plate centre deflection vs Navier series",
     3.0e-2, "<="),
    ("C12_PLATE_LITERATURE", "uniform-pressure plate vs tabulated 0.00406 q a^4/D",
     1.0e-2, "<="),
    ("C13_CONTACT_ANALYTIC", "mapped contact forces -> strip deflection vs closed form",
     1.0e-2, "<="),
    ("C14_ENERGY_PLATE", "plate strain energy vs external work (energy audit)",
     1.0e-6, "<="),
    ("C15_ENERGY_STRIP", "strip strain energy vs external work (energy audit)",
     1.0e-6, "<="),
    ("C16_INTERFACE_WORK", "DEM interface work defect (sum f_i u_i vs sum P_j w_j)",
     1.0e-2, "<="),
    ("C17_PLATE_ENERGY_EXCHANGE", "plate exchange work defect (sum f_i u_i vs int t u dA)",
     1.0e-2, "<="),
    ("C18_COUPLE_COMPLETED", "contact couple carried after moment completion "
     "(relative to the applied couple)", 1.0e-6, "<="),
    ("C19_COUPLE_RAW_DETECTED", "contact couple lost by raw force-only lumping must be "
     "detected", 1.0e-2, ">="),
]


def _cmp(value: float, threshold: float, relation: str) -> bool:
    if relation == "<=":
        return bool(value <= threshold)
    if relation == ">=":
        return bool(value >= threshold)
    raise ValueError(relation)


def assemble_payload(cases: dict, elapsed_s: float = 0.0) -> dict:
    """Build the acceptance payload from already-computed case dictionaries."""
    a = cases["couette"]
    b = cases["poiseuille"]
    c = cases["contact"]
    values = {
        "C1_FORCE_PANEL": a["exchange"]["force_residual_rel"],
        "C2_MOMENT_PANEL": a["exchange"]["moment_residual_rel"],
        "C3_FORCE_CONTACT": c["exchange"]["conserved"]["force_residual_rel"],
        "C4_MOMENT_CONTACT": c["exchange"]["conserved"]["moment_residual_rel"],
        "C5_NEGCTRL_MOMENT": c["exchange"]["negative_control_nearest_node"]["moment_residual_rel"],
        "C6_LBM_COUETTE_PROFILE": a["lbm"]["profile_rel_L2_error"],
        "C7_LBM_COUETTE_TAU": a["lbm"]["tau_rel_error"],
        "C8_MEMBRANE_TIP": a["fem"]["u_tip_rel_error"],
        "C9_LBM_POISEUILLE_PROFILE": b["lbm"]["profile_rel_L2_error"],
        "C10_LBM_POISEUILLE_TAU": b["lbm"]["tau_w_rel_error"],
        "C11_PLATE_NAVIER": b["fem"]["rel_error_mapped_consistent"],
        "C12_PLATE_LITERATURE": b["fem"]["uniform_vs_literature_rel"],
        "C13_CONTACT_ANALYTIC": c["fem"]["rel_error_max"],
        "C14_ENERGY_PLATE": b["fem"]["energy_residual_rel"],
        "C15_ENERGY_STRIP": c["fem"]["energy_residual_rel"],
        "C16_INTERFACE_WORK": c["fem"]["interface_work_defect_rel"],
        "C17_PLATE_ENERGY_EXCHANGE": a["fem"]["exchange_work_defect_rel"],
        "C18_COUPLE_COMPLETED": c["exchange"]["moment_carrying"]["completed_couple_loss_rel"],
        "C19_COUPLE_RAW_DETECTED": c["exchange"]["moment_carrying"]["raw_couple_loss_rel"],
    }

    criteria = []
    for cid, desc, thr, rel in CRITERIA:
        v = float(values[cid])
        criteria.append({
            "id": cid,
            "description": desc,
            "value": v,
            "threshold": thr,
            "relation": rel,
            "passed": _cmp(v, thr, rel),
        })

    # extra information that is reported but not gated
    diagnostics = {
        "plate_w_center_mapped_nodal_rel_error": b["fem"]["rel_error_mapped_nodal"],
        "plate_navier_truncation_rel": b["fem"]["navier_truncation_rel"],
        "contact_raw_lumping_moment_residual": c["exchange"]["raw_lumping"]["moment_residual_rel"],
        "contact_raw_lumping_force_residual": c["exchange"]["raw_lumping"]["force_residual_rel"],
        "negative_control_force_residual": c["exchange"]["negative_control_nearest_node"]["force_residual_rel"],
        "contact_fem_rel_errors": c["fem"]["rel_error"],
        "contact_couple_applied_Nm": c["exchange"]["moment_carrying"]["applied_couple_Nm"],
    }

    return {
        "node": "P475W1-N05",
        "title": "Conservative cross-solver field exchange (LBM/DEM -> FEM) with audits",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "elapsed_s": elapsed_s,
        "criteria": criteria,
        "passed_all": all(cr["passed"] for cr in criteria),
        "n_passed": sum(1 for cr in criteria if cr["passed"]),
        "n_criteria": len(criteria),
        "cases": cases,
        "diagnostics": diagnostics,
    }


def run_acceptance(*, fast: bool = False) -> dict:
    t0 = time.time()
    cases = {
        "couette": case_couette(),
        "poiseuille": case_poiseuille(),
        "contact": case_contact(n_elem=100 if fast else 200),
    }
    return assemble_payload(cases, elapsed_s=time.time() - t0)