"""Element strength curves and the CSR-type panel reference.

The Smith analysis of :mod:`tensorphysics.smith_section_benchmark.solver` needs
one stress per element (the *element strength*); the benchmark family also needs
an **independent** design-formula reference so that the incremental analysis is
compared with something that was not built from the same element curves.

Curves provided
---------------
``plate_ultimate_stress``
    Plate element strength:

    * ``plate_law="von_karman"`` — ``sqrt(sigma_cr sigma_y)`` (perfect plate,
      effective width; used by the repo's
      :mod:`~tensorphysics.elastoplastic_collapse.progressive_collapse`);
    * ``plate_law="faulkner"`` — ``(2/beta - 1/beta^2) sigma_y`` for
      ``beta > 1``, ``sigma_y`` otherwise (Faulkner 1975 strength curve, the
      same reduction as ``hullgirder.ultimate.plate_buckling_reduction``).

``faulkner_plate_ratio`` / ``von_karman_plate_ratio``
    The same two laws as pure ratios ``sigma_u/sigma_y`` (β series plotting).

``johnson_ostenfeld`` (re-export)
    Elasto-plastic column/plate strength used for stiffener elements.

``csr_type_panel_strength``
    The **independent reference** for the stiffened-panel benchmark: the
    area-weighted element-wise capacity check used by classification practice

        sigma_u,panel = (A_p phi_p sigma_y,p + sum_i A_i sigma_u,i) / A_total

    with the plating reduction ``phi_p`` (Faulkner law) and each stiffener at
    its own squash/column strength.  The stiffener term deliberately uses the
    **strong-axis panel column curve** (stiffener + associated plating, Johnson-
    Ostenfeld), i.e. a different mechanism set from the fiber analysis (which
    caps stiffeners by weak-axis column / tripping and resolves the plate
    through its own load-end shortening curve).  The two methods therefore are
    not circular: agreement means that the element-wise design formula and the
    incremental Smith analysis are consistent, and the residual difference is
    attributable to the mechanisms present in one and not the other.

References
----------
- Faulkner, D. (1975). "A review of effective plating for use in the analysis
  of stiffened plating in bending and compression." J. Ship Research 19(1), 1-17.
- von Karman, T., Sechler, E.E. & Donnell, L.H. (1932). Trans. ASME 54, 53-57.
- Johnson, J.B. & Ostenfeld, A. (1936). *Welded and Riveted Columns* (column
  curve; also IACS CSR Pt.1 Ch.8 buckling formulations).
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design of
  Steel-Plated Structures*. Wiley (element-wise panel capacity check).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Sequence

import numpy as np

from ..elastoplastic_collapse.von_mises_plasticity import VonMisesMaterial
from ..hull_redundancy.elements import johnson_ostenfeld

if TYPE_CHECKING:  # pragma: no cover - avoids a circular import at run time
    from .section import StiffenerGroup

__all__ = [
    "faulkner_plate_ratio",
    "von_karman_plate_ratio",
    "plate_ultimate_stress",
    "johnson_ostenfeld",
    "stiffener_strong_axis_column",
    "csr_type_panel_strength",
]


def faulkner_plate_ratio(beta: float) -> float:
    """Faulkner (1975) plating strength ratio ``sigma_u/sigma_y`` [-]."""
    if beta <= 0.0:
        raise ValueError("plate slenderness must be positive")
    if beta <= 1.0:
        return 1.0
    return float(min(1.0, 2.0 / beta - 1.0 / beta ** 2))


def von_karman_plate_ratio(beta: float) -> float:
    """Perfect-plate effective-width ratio ``1/beta`` (clipped to 1) [-]."""
    if beta <= 0.0:
        raise ValueError("plate slenderness must be positive")
    return float(min(1.0, 1.0 / beta))


def plate_ultimate_stress(beta: float, sigma_cr: float, sigma_y: float,
                          law: str = "von_karman") -> tuple[float, str]:
    """Plate element strength [Pa] and the identifier of the law used."""
    if law == "von_karman":
        return float(np.sqrt(sigma_cr * sigma_y)), "plate_von_karman"
    if law == "faulkner":
        return float(faulkner_plate_ratio(beta) * sigma_y), "plate_faulkner"
    raise ValueError("plate law must be 'von_karman' or 'faulkner'")


def stiffener_strong_axis_column(
    stiff: "StiffenerGroup",
    material: VonMisesMaterial,
    effective_plate_width: float | None = None,
) -> Dict[str, float]:
    """Strong-axis (panel) column strength of a stiffener with its plating.

    The standard panel-column check: the stiffener plus an associated plating
    strip of width ``min(spacing, 0.4 x span)`` forms a column of length
    ``span``; the Euler stress uses the radius of gyration about the axis
    parallel to the plating (bending in the vertical plane), and the
    Johnson-Ostenfeld elasto-plastic correction gives the ultimate stress of the
    *stiffener element*.

    Returns the Euler stress, the corrected strength and the geometric data
    (for reporting/audit).
    """
    sy = float(stiff.sigma_y if stiff.sigma_y is not None else material.sigma_y0)
    b_e = effective_plate_width if effective_plate_width is not None else min(
        stiff.spacing, 0.4 * stiff.span)
    plate_t = stiff.t_w  # attached plating thickness is not known here; use web
    a_st = stiff.area_one
    a_pl = b_e * plate_t
    # centroid of the combined T section measured from the plating line
    z_st = abs(stiff.z_centroid - stiff.z_base)
    z_bar = (a_st * z_st + a_pl * 0.5 * plate_t) / (a_st + a_pl)
    i_web = stiff.t_w * stiff.h_s ** 3 / 12.0 + a_st * (z_st - z_bar) ** 2
    i_pl = b_e * plate_t ** 3 / 12.0 + a_pl * (z_bar - 0.5 * plate_t) ** 2
    inertia = i_web + i_pl
    area = a_st + a_pl
    r = float(np.sqrt(inertia / area))
    sigma_e = float(np.pi ** 2 * material.E / (stiff.span / r) ** 2)
    return {
        "sigma_euler_strong_axis": sigma_e,
        "sigma_column_strong_axis": float(johnson_ostenfeld(sigma_e, sy)),
        "radius_of_gyration": r,
        "associated_plate_width": float(b_e),
        "combined_area": float(area),
    }


def csr_type_panel_strength(
    plates: Sequence,
    stiffeners: Sequence["StiffenerGroup"],
    material: VonMisesMaterial,
    plate_law: str = "faulkner",
) -> Dict[str, object]:
    """Area-weighted element-wise panel ultimate strength (design-formula view).

    Parameters
    ----------
    plates : sequence of PlateStrip
        Plating strips of the panel (only the *loaded* plating — for a panel
        benchmark this is the plate between the stiffeners).
    stiffeners : sequence of StiffenerGroup
        Longitudinal stiffener groups.
    material : VonMisesMaterial
    plate_law : str
        ``"faulkner"`` (default) or ``"von_karman"`` plating reduction.

    Returns
    -------
    dict
        ``sigma_u`` [Pa], ``sigma_u_over_sigma_y``, the participating
        reductions/strengths and the areas, plus ``sigma_y`` used (the minimum
        yield stress of the panel).
    """
    a_total = 0.0
    capacity = 0.0
    detail: Dict[str, Dict[str, float]] = {}
    for p in plates:
        sy = float(p.sigma_y if p.sigma_y is not None else material.sigma_y0)
        beta = p.plate_slenderness(sy, material.E)
        if plate_law == "faulkner":
            phi = faulkner_plate_ratio(beta)
        elif plate_law == "von_karman":
            phi = von_karman_plate_ratio(beta)
        else:
            raise ValueError("plate_law must be 'faulkner' or 'von_karman'")
        a = p.area
        a_total += a
        capacity += a * phi * sy
        detail[p.name] = {"area": a, "beta": beta, "phi": phi, "sigma_u": phi * sy,
                          "kind": "plate"}
    for s in stiffeners:
        sy = float(s.sigma_y if s.sigma_y is not None else material.sigma_y0)
        col = stiffener_strong_axis_column(s, material)
        sigma_u = float(min(sy, col["sigma_column_strong_axis"]))
        a = s.area
        a_total += a
        capacity += a * sigma_u
        detail[s.name] = {"area": a, "sigma_u": sigma_u,
                          "sigma_column": col["sigma_column_strong_axis"], "kind": "stiffener"}
    if a_total <= 0.0:
        raise ValueError("panel has no area")
    sigma_y_panel = float(min(
        [float(p.sigma_y if p.sigma_y is not None else material.sigma_y0) for p in plates]
        + [float(s.sigma_y if s.sigma_y is not None else material.sigma_y0)
           for s in stiffeners]))
    return {
        "sigma_u": float(capacity / a_total),
        "sigma_u_over_sigma_y": float(capacity / a_total / sigma_y_panel),
        "sigma_y": sigma_y_panel,
        "area_total": float(a_total),
        "plate_law": plate_law,
        "components": detail,
        "method": "area-weighted element capacity check (CSR-type panel strength)",
    }