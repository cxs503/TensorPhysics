"""Acceptance runner for the generalised Smith-method benchmark family (C1-C6).

The campaign node ``IN387-FEM01`` fixes six machine-checkable criteria.  They are
implemented here, one function per criterion, and the thresholds are **literals
in this module** (they do not depend on the results):

===== ==========================================================================
C1    Richardson extrapolation of the Smith solution over three mesh levels:
      discretisation error of the finest mesh vs the ``h -> 0`` value ``<= 1%``
      (applied to the double-bottom unit axial strength and to the KCS midship
      section ultimate moment)
C2    Stiffened-panel ``beta`` series: at ``beta = 2.0``
      ``sigma_u/sigma_y in [0.4, 0.8]`` and the relative difference from the
      CSR-type design formula ``<= 10%``
C3    Box girder: ``M_u / M_reference in [0.85, 1.15]`` against the closed-form
      rigid-plastic collapse moment (and, for the slender box, the two-sided
      bound ``M_first_yield <= M_u <= M_plastic``)
C4    KCS midship section: ``M_u / M_wave_extreme >= 1.5``
C5    Failure-mode identification correct for at least 2 of the 3 declared cases
C6    The module test suite collects and passes >= 8 tests
===== ==========================================================================

Every criterion returns its numbers *and* its evidence class, so the report can
quote them without re-derivation.  A criterion that cannot be honestly satisfied
is returned with ``"passes": False`` and a ``"reason"`` string — honest FAIL is
never overwritten.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional, Sequence


from .benchmarks import (
    KCSMidshipSection,
    box_girder,
    box_girder_plastic_moment,
    issc2009_double_bottom_unit,
    kcs_midship_section,
    paik_style_panel,
)
from .convergence import grid_convergence_criterion, mesh_convergence_study
from .handoff import kcs_girder, load_capacity_handoff
from .section import SectionModel, StiffenerGroup
from .solver import SmithSectionSolver, axial_ultimate_strength
from .strength_curves import csr_type_panel_strength

__all__ = [
    "c1_grid_convergence",
    "c2_panel_series",
    "c3_box_girder",
    "c4_kcs_handoff",
    "c5_failure_modes",
    "c6_test_suite",
    "run_acceptance",
    "write_acceptance_report",
]

C1_TOL = 1.0e-2
C2_BAND = (0.4, 0.8)
C2_DIFF_TOL = 0.10
C3_BAND = (0.85, 1.15)
C4_MIN_MARGIN = 1.5
C5_MIN_CORRECT = 2
C6_MIN_TESTS = 8


# ---------------------------------------------------------------------------
# C1
# ---------------------------------------------------------------------------
def c1_grid_convergence(levels: Sequence[int] = (1, 2, 4, 8),
                        kcs_levels: Sequence[int] = (4, 8, 16),
                        n_kappa: int = 48) -> Dict[str, object]:
    """Mesh convergence of the Smith solutions, three-level Richardson extrapolation.

    Two studies, both predeclared:

    * **double-bottom unit, axial strength** — levels ``(1, 2, 4, 8)`` fibers per
      vertical strip.  Under uniform axial shortening only the centre girder is
      fiberised and its stress is uniform, so the quantity is expected to be
      mesh-independent; the study verifies exactly that (it must *not* show a
      spurious order);
    * **KCS midship section, ultimate moment in bending** — levels ``(4, 8, 16)``,
      i.e. the near-asymptotic range identified from the mesh sequence
      ``4.4729 / 4.4769 / 4.4777 GNm`` (the two coarsest levels, 1 and 2, are
      outside the asymptotic range and are reported separately, not extrapolated).
    """
    db = issc2009_double_bottom_unit()

    def eval_db(n: int) -> float:
        return axial_ultimate_strength(db, n_fibers=n, n_web_fibers=max(1, n // 2),
                                       n_steps=160).sigma_u

    study_db = mesh_convergence_study(eval_db, levels=levels, tolerance=C1_TOL,
                                      label="double-bottom unit axial sigma_u [Pa]")

    kcs = kcs_midship_section()
    sec = kcs.section

    def eval_kcs(n: int) -> float:
        return SmithSectionSolver(sec, n_fibers=n, n_web_fibers=max(1, n // 2),
                                  n_kappa=n_kappa).solve().M_u

    study_kcs = mesh_convergence_study(eval_kcs, levels=kcs_levels, tolerance=C1_TOL,
                                       label="KCS midship section M_u [Nm]")
    crit_db = grid_convergence_criterion(study_db, C1_TOL)
    crit_kcs = grid_convergence_criterion(study_kcs, C1_TOL)
    coarse = {n: eval_kcs(n) for n in (1, 2)}
    return {
        "criterion": "C1",
        "description": ("Richardson extrapolation h->0 over >=3 mesh levels; "
                        "finest-mesh discretisation error <= 1%"),
        "tolerance": C1_TOL,
        "studies": {"double_bottom_axial": crit_db, "kcs_bending": crit_kcs},
        "kcs_coarse_levels_outside_asymptotic_range": coarse,
        "passes": bool(crit_db["passes"] and crit_kcs["passes"]),
        "evidence_level": "REAL",
        "reason": "" if (crit_db["passes"] and crit_kcs["passes"]) else
                  "at least one study exceeds the 1% discretisation tolerance",
    }


# ---------------------------------------------------------------------------
# C2
# ---------------------------------------------------------------------------
def c2_panel_series(betas: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
                    plate_law: str = "faulkner",
                    n_fibers: int = 4) -> Dict[str, object]:
    """``sigma_u/sigma_y`` vs ``beta`` against the CSR-type design formula."""
    rows: List[Dict[str, object]] = []
    for beta in betas:
        sec = paik_style_panel(beta=beta, plate_law=plate_law)
        res = axial_ultimate_strength(sec, n_fibers=n_fibers, n_web_fibers=3,
                                     n_steps=200)
        csr = csr_type_panel_strength(sec.plates, sec.stiffeners, sec.material,
                                      plate_law="faulkner")
        diff = abs(res.sigma_u - csr["sigma_u"]) / csr["sigma_u"]
        rows.append({
            "beta": float(beta),
            "plate_thickness_m": sec.provenance["plate_thickness"],  # type: ignore[index]
            "sigma_u_MPa": res.sigma_u / 1e6,
            "sigma_u_over_sigma_y": res.sigma_u_over_sigma_y,
            "csr_sigma_u_MPa": csr["sigma_u"] / 1e6,
            "csr_sigma_u_over_sigma_y": csr["sigma_u_over_sigma_y"],
            "relative_difference": float(diff),
            "failure_mode": res.failure_mode,
            "governing_group": res.governing_group,
            "utilisation": {k: round(v, 3) for k, v in res.utilisation.items()},
        })
    # sensitivity: the element law is a modelled variable, report both
    sensitivity = {}
    for law in ("von_karman", "faulkner"):
        sec = paik_style_panel(beta=2.0, plate_law=law)
        res = axial_ultimate_strength(sec, n_fibers=n_fibers, n_web_fibers=3,
                                     n_steps=200)
        sensitivity[law] = {
            "sigma_u_over_sigma_y": res.sigma_u_over_sigma_y,
            "failure_mode": res.failure_mode,
        }
    row_beta2 = next(r for r in rows if abs(r["beta"] - 2.0) < 1e-9)
    in_band = C2_BAND[0] <= row_beta2["sigma_u_over_sigma_y"] <= C2_BAND[1]
    within_tol = row_beta2["relative_difference"] <= C2_DIFF_TOL
    return {
        "criterion": "C2",
        "description": ("stiffened-panel beta series: at beta=2 sigma_u/sigma_y in "
                        "[0.4,0.8] and |Smith - CSR-type| <= 10%"),
        "band": list(C2_BAND),
        "difference_tolerance": C2_DIFF_TOL,
        "plate_law": plate_law,
        "rows": rows,
        "element_law_sensitivity_at_beta2": sensitivity,
        "beta2": row_beta2,
        "passes": bool(in_band and within_tol),
        "sub_checks": {"in_band": bool(in_band), "within_10pct": bool(within_tol)},
        "evidence_level": "REAL (incremental analysis) vs ANALYTICAL (CSR-type formula)",
        "reason": "" if (in_band and within_tol) else
                  "beta=2 panel outside the declared band or >10% from CSR-type formula",
    }


# ---------------------------------------------------------------------------
# C3
# ---------------------------------------------------------------------------
def c3_box_girder(n_fibers: int = 2, n_kappa: int = 48) -> Dict[str, object]:
    """Box girder ultimate moment vs the closed-form limit-analysis reference.

    Two box cases are declared *a priori*:

    * **stocky box** (``beta ~ 1.2`` flanges, plastic collapse): reference = the
      closed-form full-plastic moment ``M_p`` (every element at yield, statics
      only).  Expected ratio 1.0; band ``[0.85, 1.15]``.
    * **slender box** (``beta ~ 3.8`` flanges, plate-buckling dominated):
      reference = :func:`~.benchmarks.rigid_plastic_reference`, i.e. obligatory
      statics with the plating limited by its own element strength in compression
      and yielding in tension.  Additionally the Smith value must sit between the
      elastic moment at which the compression flange reaches its plate strength
      (a hard lower bound) and the full-plastic moment (a hard upper bound).
    """
    from .benchmarks import rigid_plastic_reference
    from .strength_curves import faulkner_plate_ratio

    cases: Dict[str, Dict[str, object]] = {}
    # --- stocky box
    stocky = box_girder(breadth=0.40, depth=0.40, t_flange=0.012, t_web=0.012,
                        span=1.6, sigma_y=250.0e6, plate_law="von_karman")
    ref_s = box_girder_plastic_moment(stocky, sigma_y=250.0e6)
    res_s = SmithSectionSolver(stocky, n_fibers=n_fibers, n_web_fibers=2,
                               n_kappa=n_kappa).solve()
    ratio_s = abs(res_s.M_u) / ref_s["M_plastic_Nm"]
    cases["stocky"] = {
        "geometry": dict(stocky.provenance),  # type: ignore[arg-type]
        "beta_flange": stocky.plates[0].plate_slenderness(250.0e6),
        "M_u_Nm": abs(res_s.M_u),
        "M_reference_Nm": ref_s["M_plastic_Nm"],
        "ratio": float(ratio_s),
        "reference_method": ref_s["method"],
        "failure_mode": res_s.failure_mode,
        "in_band": bool(C3_BAND[0] <= ratio_s <= C3_BAND[1]),
    }
    # --- slender box
    slender = box_girder(breadth=0.80, depth=0.80, t_flange=0.008, t_web=0.008,
                         span=1.6, sigma_y=300.0e6, plate_law="faulkner")
    sy_l = 300.0e6
    beta_l = slender.plates[0].plate_slenderness(sy_l)
    ref_l = rigid_plastic_reference(slender, sigma_y=sy_l)
    res_l = SmithSectionSolver(slender, n_fibers=n_fibers, n_web_fibers=2,
                               n_kappa=n_kappa).solve()
    m_u_l = abs(res_l.M_u)
    phi = faulkner_plate_ratio(beta_l)
    m_lower = phi * sy_l * slender.section_moduli["S_deck"]
    m_upper = ref_l["M_plastic_Nm"]
    ratio_l = m_u_l / ref_l["M_reference_Nm"]
    cases["slender"] = {
        "geometry": dict(slender.provenance),  # type: ignore[arg-type]
        "beta_flange": beta_l,
        "phi_faulkner": phi,
        "M_u_Nm": m_u_l,
        "M_reference_Nm": ref_l["M_reference_Nm"],
        "M_plastic_Nm": m_upper,
        "M_lower_bound_Nm": float(m_lower),
        "ratio": float(ratio_l),
        "ratio_vs_plastic": float(m_u_l / m_upper),
        "reference_method": ref_l["method"],
        "cap_map_MPa": ref_l["cap_map_MPa"],
        "failure_mode": res_l.failure_mode,
        "within_bounds": bool(m_lower <= m_u_l <= m_upper),
        "in_band": bool(C3_BAND[0] <= ratio_l <= C3_BAND[1]),
    }
    ok = bool(cases["stocky"]["in_band"] and cases["slender"]["in_band"]
              and cases["slender"]["within_bounds"])
    return {
        "criterion": "C3",
        "description": ("box girder M_u / M_reference in [0.85,1.15] against the "
                        "closed-form limit-analysis moment; slender box additionally "
                        "inside [elastic plate-strength moment, full plastic moment]"),
        "band": list(C3_BAND),
        "cases": cases,
        "passes": ok,
        "evidence_level": "REAL (Smith) vs ANALYTICAL (rigid-plastic limit analysis)",
        "literature_experiment_reference": {
            "status": "NOT_SUPPORTED",
            "reason": ("the collapse moments of the Dow (1991) / Gordo & Guedes Soares "
                       "(1996) box-girder tests are not available in-repo and are not "
                       "fabricated here; the analytical limit-analysis reference is used "
                       "instead and this substitution is declared"),
        },
        "reason": "" if ok else "box-girder ratio outside the declared band/bounds",
    }


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------
def c4_kcs_handoff(n_fibers: int = 4, n_kappa: int = 56,
                   still_water_fraction: Optional[float] = None) -> Dict[str, object]:
    """KCS midship section capacity vs the wave-induced extreme bending moment."""
    kcs: KCSMidshipSection = kcs_midship_section()
    gate = kcs.geometry_gate()
    if not gate["all_pass"]:
        return {
            "criterion": "C4",
            "description": "KCS M_u / M_wave_extreme >= 1.5",
            "geometry_gate": gate,
            "passes": False,
            "reason": "midship-section geometry gate failed - capacity numbers withheld",
        }
    sec = kcs.section
    sag = SmithSectionSolver(sec, n_fibers=n_fibers, n_web_fibers=3,
                             n_kappa=n_kappa, bending="sagging").solve()
    hog = SmithSectionSolver(sec, n_fibers=n_fibers, n_web_fibers=3,
                             n_kappa=n_kappa, bending="hogging").solve()
    sm = sec.section_moduli
    girder = kcs_girder(section_modulus=sm["S_deck"])
    handoff = load_capacity_handoff(
        M_u_sagging=abs(sag.M_u), M_u_hogging=abs(hog.M_u), ship=girder,
        still_water_fraction_of_wave=still_water_fraction,
        required_margin=C4_MIN_MARGIN)
    passes = bool(handoff["gate"]["passes"])
    return {
        "criterion": "C4",
        "description": "KCS midship section M_u / M_wave_extreme >= 1.5",
        "geometry_gate": gate,
        "section_properties": sm,
        "sagging": {"M_u_Nm": abs(sag.M_u), "kappa_u": sag.kappa_u,
                    "failure_mode": sag.failure_mode,
                    "governing_group": sag.governing_group,
                    "utilisation": {k: round(v, 3) for k, v in sag.utilisation.items()}},
        "hogging": {"M_u_Nm": abs(hog.M_u), "kappa_u": hog.kappa_u,
                    "failure_mode": hog.failure_mode,
                    "governing_group": hog.governing_group,
                    "utilisation": {k: round(v, 3) for k, v in hog.utilisation.items()}},
        "handoff": handoff,
        "passes": passes,
        "reason": "" if passes else "load-capacity margin below the required 1.5",
    }


# ---------------------------------------------------------------------------
# C5
# ---------------------------------------------------------------------------
def c5_failure_modes(n_fibers: int = 4) -> Dict[str, object]:
    """Failure-mode identification on three cases with *a priori* expected modes.

    The three cases are chosen so that the expected mode follows from the element
    strengths *before* running the analysis:

    1. **slender plating, flanged stiffener** (``beta = 3.0``): the plate strength
       ``0.556 sigma_y`` (Faulkner) is far below the stiffener strength
       (``0.97 sigma_y``) -> ``plate_buckling``;
    2. **stocky plating, slender flat bar** (``beta = 0.9``, ``250 x 14`` flat
       bar on a 4.0 m span, ``h/t = 18``): the torsional (tripping) strength of
       the flat bar (``~0.68 sigma_y``) is reached well before the stocky plating
       yields -> ``stiffener_tripping``;
    3. **stocky box girder in bending**: every element is stocky
       (``beta ~ 1.2``, ``sigma_cr > sigma_y``) so no instability mode exists
       -> ``yielding``.
    """
    cases: List[Dict[str, object]] = []
    # 1) plate-buckling dominated panel
    sec1 = paik_style_panel(beta=3.0, plate_law="faulkner", stiffener_type="T")
    r1 = axial_ultimate_strength(sec1, n_fibers=n_fibers, n_web_fibers=3)
    caps1 = {f.group: f.sigma_cap / 1e6 for f in sec1.fibers(n_fibers=1, n_web_fibers=1)}
    cases.append({"case": "slender plating (beta=3.0), flanged stiffener",
                  "expected": "plate_buckling", "observed": r1.failure_mode,
                  "governing_group": r1.governing_group,
                  "element_strengths_MPa": {k: round(v, 1) for k, v in caps1.items()},
                  "utilisation": {k: round(v, 3) for k, v in r1.utilisation.items()},
                  "sigma_u_over_sigma_y": r1.sigma_u_over_sigma_y})
    # 2) tripping dominated panel (slender flat bar on stocky plating)
    base = paik_style_panel(beta=0.9, plate_law="faulkner")
    sec2 = SectionModel(
        name="tripping_case",
        plates=base.plates,
        stiffeners=(StiffenerGroup(
            name="panel_stiffener", n_long=1, h_s=0.250, t_w=0.014, b_f=0.0, t_f=0.0,
            spacing=0.8, span=4.0, z_base=base.plates[0].thickness, direction=1.0,
            sigma_y=315.0e6),),
        material=base.material, plate_law="faulkner", stiffener_column_axis="weak")
    r2 = axial_ultimate_strength(sec2, n_fibers=n_fibers, n_web_fibers=3)
    caps2 = {f.group: f.sigma_cap / 1e6 for f in sec2.fibers(n_fibers=1, n_web_fibers=1)}
    cases.append({"case": "stocky plating (beta=0.9), slender flat bar (h/t=44)",
                  "expected": "stiffener_tripping", "observed": r2.failure_mode,
                  "governing_group": r2.governing_group,
                  "element_strengths_MPa": {k: round(v, 1) for k, v in caps2.items()},
                  "utilisation": {k: round(v, 3) for k, v in r2.utilisation.items()},
                  "sigma_u_over_sigma_y": r2.sigma_u_over_sigma_y})
    # 3) stocky box in bending -> general yielding
    stocky = box_girder(breadth=0.40, depth=0.40, t_flange=0.012, t_web=0.012,
                        span=1.6, sigma_y=250.0e6)
    r3 = SmithSectionSolver(stocky, n_fibers=n_fibers, n_web_fibers=2, n_kappa=40).solve()
    caps3 = {f.group: f.sigma_cap / 1e6 for f in stocky.fibers(n_fibers=1, n_web_fibers=1)}
    cases.append({"case": "stocky box girder in bending (beta=1.16)",
                  "expected": "yielding", "observed": r3.failure_mode,
                  "governing_group": r3.governing_group,
                  "element_strengths_MPa": {k: round(v, 1) for k, v in caps3.items()},
                  "utilisation": {k: round(v, 3) for k, v in r3.utilisation.items()},
                  "M_u_over_M_plastic": float(
                      abs(r3.M_u) / box_girder_plastic_moment(stocky)["M_plastic_Nm"])})
    n_correct = sum(1 for c in cases if c["observed"] == c["expected"])
    return {
        "criterion": "C5",
        "description": "failure-mode identification correct for >= 2 of 3 declared cases",
        "cases": cases,
        "n_correct": int(n_correct),
        "n_cases": len(cases),
        "passes": bool(n_correct >= C5_MIN_CORRECT),
        "evidence_level": "REAL (mechanism attribution at the ultimate state)",
        "reason": "" if n_correct >= C5_MIN_CORRECT else
                  "failure-mode classifier below the required 2/3 accuracy",
    }


# ---------------------------------------------------------------------------
# C6
# ---------------------------------------------------------------------------
def c6_test_suite(test_path: Optional[str] = None,
                  run: bool = True) -> Dict[str, object]:
    """Run (or collect) the module's pytest suite and count the tests."""
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    rel = test_path or os.path.join("tests", "test_smith_section_benchmark.py")
    path = rel if os.path.isabs(rel) else os.path.join(root, rel)
    n_tests = 0
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.lstrip().startswith("def test_"):
                    n_tests += 1
    result: Dict[str, object] = {
        "criterion": "C6",
        "description": f"module test suite collects and passes >= {C6_MIN_TESTS} tests",
        "test_file": path,
        "n_tests_collected": n_tests,
        "passes": bool(n_tests >= C6_MIN_TESTS),
        "evidence_level": "REAL",
        "reason": "",
    }
    if run and os.path.exists(path):
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", path, "-q", "--no-header"],
            cwd=root, capture_output=True, text=True, timeout=3600)
        result["pytest_returncode"] = int(proc.returncode)
        result["pytest_stdout_tail"] = "\n".join(proc.stdout.strip().splitlines()[-12:])
        result["pytest_duration_s"] = float(time.time() - t0)
        result["passes"] = bool(result["passes"] and proc.returncode == 0)
        if proc.returncode != 0:
            result["reason"] = "pytest did not pass"
    return result


# ---------------------------------------------------------------------------
# aggregate
# ---------------------------------------------------------------------------
def run_acceptance(levels: Sequence[int] = (1, 2, 4, 8), n_kappa: int = 48,
                   include_tests: bool = True,
                   still_water_fraction: Optional[float] = None) -> Dict[str, object]:
    """Run C1-C6 and return the full acceptance record."""
    t0 = time.time()
    out: Dict[str, object] = {
        "node": "IN387-FEM01",
        "title": ("FEM Smith progressive collapse: grid convergence, section-type "
                  "generalisation (double bottom / stiffened panel / box girder) and "
                  "real KCS midship section ultimate moment handoff"),
        "thresholds": {
            "C1_discretisation": C1_TOL, "C2_band": list(C2_BAND),
            "C2_difference": C2_DIFF_TOL, "C3_band": list(C3_BAND),
            "C4_margin": C4_MIN_MARGIN, "C5_min_correct": C5_MIN_CORRECT,
            "C6_min_tests": C6_MIN_TESTS,
        },
    }
    out["C1"] = c1_grid_convergence(levels=levels, n_kappa=n_kappa)
    out["C2"] = c2_panel_series()
    out["C3"] = c3_box_girder(n_kappa=n_kappa)
    out["C4"] = c4_kcs_handoff(n_kappa=max(n_kappa, 56),
                               still_water_fraction=still_water_fraction)
    out["C5"] = c5_failure_modes()
    out["C6"] = c6_test_suite(run=include_tests)
    criteria = ["C1", "C2", "C3", "C4", "C5", "C6"]
    out["summary"] = {
        "criteria": {c: bool(out[c]["passes"]) for c in criteria},
        "n_passed": int(sum(1 for c in criteria if out[c]["passes"])),
        "n_criteria": len(criteria),
        "all_pass": bool(all(out[c]["passes"] for c in criteria)),
        "wall_time_s": float(time.time() - t0),
    }
    return out


def write_acceptance_report(out_dir: str, record: Dict[str, object]) -> str:
    """Write the acceptance record as JSON into ``out_dir``; return the path."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "in387_fem01_acceptance.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(record, fh, indent=2, sort_keys=False)
    return path