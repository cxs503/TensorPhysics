"""Independent closed-form reference for the damaged-section ultimate moment.

Purpose
-------
The RIF matrix is produced by the incremental (fiber) Smith solver.  To have a
*benchmark* for the residual strength of the damaged section that does not share
the numerical machinery of that solver, this module implements the published
simplified (rigid-plastic, effective-width) method:

    M_u = sum_i  A_i * sigma_u,i * |z_i - z_N|

with

- plating: ``sigma_u = phi(beta) * sigma_y`` using Faulkner's (1975) plate
  strength curve ``phi = 1`` for ``beta <= 1`` and ``phi = 2/beta - 1/beta^2``
  otherwise, ``beta = (b/t) sqrt(sigma_y/E)``;
- stiffener groups: ``sigma_u = sigma_y`` (stocky flat bars), optionally capped
  by the tripping stress when the damaged state is ``trip``;
- ``z_N`` the elastic (transformed-section) neutral axis of the *damaged*
  section;
- removed components contribute zero area; a debonded/removed stiffener also
  doubles the buckling width of the plating it supported, exactly as in the
  damage model.

This is the standard classification-society simplified residual-strength
approach (Paik & Thayamballi 2003, Ch. 5 "simplified method"; Faulkner 1975 for
the plate strength; also the basis of the IACS CSR "reduced section modulus"
check).  It is a *method-level* benchmark: no experimental damaged-hull-girder
dataset is bundled in this repository, so the comparison is against the
published closed-form method, and that limitation is carried in the report.

References
----------
- Faulkner, D. (1975). J. Ship Research 19(1), 1-17.
- Paik, J.K. & Thayamballi, A.K. (2003). Ultimate Limit State Design of
  Steel-Plated Structures. Wiley. (simplified method, Ch. 5)
- IACS (2021). Common Structural Rules for Bulk Carriers and Oil Tankers,
  Pt 1 Ch 5 (reduced section-modulus check for damaged conditions).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

from .damage import DamageSpec, DamagedSection, apply_damage
from .elements import tripping_stress
from .section import HullSection

__all__ = [
    "faulkner_plate_strength",
    "simplified_ultimate_moment",
    "benchmark_vs_smith",
]


def faulkner_plate_strength(beta: float) -> float:
    """Faulkner (1975) plate strength ratio ``phi(beta)``."""
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if beta <= 1.0:
        return 1.0
    return float(2.0 / beta - 1.0 / beta**2)


def simplified_ultimate_moment(
    section: HullSection, spec: Optional[DamageSpec] = None
) -> Dict[str, object]:
    """Closed-form rigid-plastic residual ultimate moment (published method)."""
    if spec is None:
        spec = DamageSpec(section.component_ids()[0], "none")
    dmg: DamagedSection = apply_damage(section, spec)
    mat = section.material
    sigma_y = mat.sigma_y0
    z_n = dmg.elastic_neutral_axis()

    contributions: List[Dict[str, object]] = []
    M = 0.0
    for p in dmg.plates:
        if p.removed:
            contributions.append({"id": p.id, "kind": "plate", "sigma_u_MPa": 0.0,
                                  "area_m2": 0.0, "lever_arm_m": 0.0,
                                  "moment_MNm": 0.0, "beta": None, "phi": 0.0})
            continue
        beta = (p.span_b / p.thickness) * math.sqrt(sigma_y / mat.E)
        phi = faulkner_plate_strength(beta)
        sigma_u = phi * sigma_y
        lever = abs(p.z - z_n)
        contrib = p.area * sigma_u * lever
        M += contrib
        contributions.append({"id": p.id, "kind": "plate",
                              "sigma_u_MPa": sigma_u / 1e6, "area_m2": p.area,
                              "lever_arm_m": lever, "moment_MNm": contrib / 1e6,
                              "beta": beta, "phi": phi})
    for s in dmg.stiffeners:
        if s.removed:
            contributions.append({"id": s.id, "kind": "stiffener",
                                  "sigma_u_MPa": 0.0, "area_m2": 0.0,
                                  "lever_arm_m": 0.0, "moment_MNm": 0.0,
                                  "beta": None, "phi": 1.0})
            continue
        sigma_u = sigma_y
        if s.cap is not None:
            sigma_u = min(sigma_u, s.cap)
        lever = abs(s.z - z_n)
        contrib = s.area * sigma_u * lever
        M += contrib
        contributions.append({"id": s.id, "kind": "stiffener",
                              "sigma_u_MPa": sigma_u / 1e6, "area_m2": s.area,
                              "lever_arm_m": lever, "moment_MNm": contrib / 1e6,
                              "beta": None, "phi": 1.0})
    return {
        "method": "simplified rigid-plastic effective-width method "
                  "(Faulkner 1975 plate strength; Paik & Thayamballi 2003 Ch.5)",
        "section": section.name,
        "damage": {"component_id": spec.component_id, "mode": spec.mode},
        "z_NA_elastic_m": z_n,
        "M_u_Nm": M,
        "M_u_MNm": M / 1e6,
        "contributions": contributions,
    }


def benchmark_vs_smith(
    section: HullSection,
    spec: Optional[DamageSpec] = None,
    smith_M_u: Optional[float] = None,
    solver_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    """Relative difference between the Smith solver and the closed-form method."""
    ref = simplified_ultimate_moment(section, spec)
    if smith_M_u is None:
        from .smith import SmithSectionSolver  # local import: avoid cycle

        dmg = apply_damage(section, spec) if spec is not None else apply_damage(
            section, DamageSpec(section.component_ids()[0], "none")
        )
        smith_M_u = SmithSectionSolver(dmg, **(solver_kwargs or {})).solve().M_u
    ref_M = float(ref["M_u_MNm"]) * 1e6
    rel_err = abs(smith_M_u - ref_M) / ref_M
    return {
        "reference": ref,
        "M_u_smith_MNm": smith_M_u / 1e6,
        "M_u_reference_MNm": ref_M / 1e6,
        "relative_error": rel_err,
        "within_15pct": bool(rel_err <= 0.15),
        "benchmark_type": "published closed-form simplified method (method-level)",
    }