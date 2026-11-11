"""Benchmark case family for the generalised Smith analysis.

Four families, each returning a :class:`~.section.SectionModel` plus a
machine-readable provenance block so that every number in the report can be
traced to a geometry and a method:

``issc2009_double_bottom_unit``
    A double-bottom structural unit (bottom shell + longitudinals + inner bottom
    + longitudinals + one centre girder), i.e. the structural assembly used as
    the *unit cell* in the ultimate-strength benchmark series of the ISSC
    Committee III.1 reports.  **Provenance note (honest):** the parameter table
    below is a *reconstruction* of the benchmark unit's structural arrangement
    (plate thicknesses 11-18 mm, longitudinal spacing 0.8 m, double-bottom height
    1.8 m — the ranges quoted in the benchmark descriptions), not a digitised
    copy of the committee drawing.  The benchmark's *reference value* therefore
    is not quoted from the committee report; the verification anchors used here
    are the grid-convergence study (C1) and the element-wise design formula
    (C2).  Evidence class of the literature comparison: WITHHELD (data not
    available in-repo, not fabricable offline).

``paik_style_panel`` / ``paik_style_beta_series``
    Stiffened-panel family varying the plate slenderness ``beta`` in
    ``[1.5, 3.0]`` with fixed panel aspect and stiffener, i.e. the
    ``sigma_u/sigma_y`` vs ``beta`` series that Paik's empirical formulations
    (Paik & Thayamballi 2003; Paik 2007) are fitted to and that CSR-type checks
    are calibrated on.  Same provenance caveat: reconstructed parameter set, the
    comparison is against the *formula*, not against a digitised test series.

``box_girder``
    A stiffened box girder in pure bending: the classical benchmark geometry
    (square box, plating flanges and webs) used by Dow (1991) and Gordo &
    Guedes Soares (1996).  Reference values used here are **closed-form limit
    analysis** (rigid-plastic collapse moment, and the same with the compression
    flange capped at the plate strength) — see
    :func:`box_girder_plastic_moment`.  The experimental collapse moments of the
    cited tests are **not** reproduced here (NOT_SUPPORTED offline); the
    acceptance check is therefore against the analytical reference plus a
    two-sided bond (first yield <= M_u <= full plastic).

``kcs_midship_section``
    A **real-ship** midship section: the KRISO Container Ship (KCS, 3600 TEU,
    ``Lpp = 230 m``, ``B = 32.2 m``, ``D = 19.0 m``, ``T = 10.8 m``,
    ``Cb = 0.651``, ``Cm = 0.977``), single-skin with double bottom, hopper
    tanks, deck strips + hatch coaming.  The section is *not* a box: the bilge
    radius, hopper slopes, deck openings and the mixed HT36 / mild-steel
    scantlings are all represented.  The published midship area coefficient
    ``Cm`` is used as the **geometry gate**: the bilge radius is the single
    derived parameter, solved so that the modelled underwater outline reproduces
    ``Cm = 0.977`` (see :meth:`KCSMidshipSection.geometry_gate`).

All scantling tables are "representative": they reproduce the structural
arrangement and the plate-thickness ranges of the class of ship, not a
particular yard drawing.  This is declared in every returned provenance block.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..elastoplastic_collapse.von_mises_plasticity import VonMisesMaterial
from .section import (
    STEEL_E,
    STEEL_NU,
    PlateStrip,
    SectionModel,
    StiffenerGroup,
)

__all__ = [
    "HULL_STRUCTURAL_STEEL",
    "kcs_particulars",
    "issc2009_double_bottom_unit",
    "paik_style_panel",
    "paik_style_beta_series",
    "box_girder",
    "box_girder_plastic_moment",
    "rigid_plastic_reference",
    "midship_area_from_outline",
    "kcs_midship_section",
    "KCSMidshipSection",
]

HULL_STRUCTURAL_STEEL = {
    "mild": 235.0e6,      # MS / Grade A
    "HT32": 315.0e6,
    "HT36": 355.0e6,
}


def _material(sigma_y: float = 235.0e6) -> VonMisesMaterial:
    return VonMisesMaterial(E=STEEL_E, nu=STEEL_NU, sigma_y0=float(sigma_y), H=0.0)


# ---------------------------------------------------------------------------
# ISSC-2009-style double-bottom unit
# ---------------------------------------------------------------------------
def issc2009_double_bottom_unit(
    plate_law: str = "von_karman",
    material_grade: str = "mild",
    n_long_bottom: int = 10,
    unit_width: float = 8.0,
) -> SectionModel:
    """Double-bottom structural unit: bottom + inner bottom + longitudinals + girder.

    Idealisation (unit cell of a double-bottom tank, 8.0 m of breadth, one centre
    girder, longitudinal stiffening both sides):

    =================================  ========================================
    component                          geometry
    =================================  ========================================
    bottom shell                       8.00 m x 18 mm, longitudinals @ 0.80 m
    bottom longitudinals               FB 300 x 12, 10 off, upward
    inner bottom                       8.00 m x 13 mm, longitudinals @ 0.80 m
    inner bottom longitudinals         FB 250 x 11, 10 off, downward
    centre girder (vertical plating)   1.78 m x 12 mm, plate span 1.78 x 2.40 m
    =================================  ========================================

    Frame spacing 2.40 m (used for all plate/element spans).
    """
    sy = HULL_STRUCTURAL_STEEL[material_grade]
    mat = _material(sy)
    a = 2.40
    t_bottom = 0.018
    z_ib = 1.80
    t_ib = 0.013
    plates = [
        PlateStrip(name="bottom_shell", thickness=t_bottom, span_b=0.80, span_a=a,
                   z_low=0.0, z_high=t_bottom, width=unit_width, sigma_y=sy),
        PlateStrip(name="inner_bottom", thickness=t_ib, span_b=0.80, span_a=a,
                   z_low=z_ib, z_high=z_ib + t_ib, width=unit_width, sigma_y=sy),
        # centre girder: vertical plating between bottom and inner bottom
        PlateStrip(name="centre_girder", thickness=0.012, span_b=z_ib, span_a=a,
                   z_low=t_bottom, z_high=z_ib, sigma_y=sy, orientation="vertical"),
    ]
    stiffeners = [
        StiffenerGroup(name="bottom_long", n_long=n_long_bottom, h_s=0.300, t_w=0.012,
                       b_f=0.0, t_f=0.0, spacing=unit_width / n_long_bottom, span=a,
                       z_base=t_bottom, direction=1.0, sigma_y=sy),
        StiffenerGroup(name="inner_bottom_long", n_long=n_long_bottom, h_s=0.250,
                       t_w=0.011, b_f=0.0, t_f=0.0, spacing=unit_width / n_long_bottom,
                       span=a, z_base=z_ib, direction=-1.0, sigma_y=sy),
    ]
    sec = SectionModel(name="issc2009_double_bottom_unit", plates=tuple(plates),
                       stiffeners=tuple(stiffeners), material=mat, depth=z_ib + t_ib,
                       plate_law=plate_law)
    sec.provenance = {  # type: ignore[attr-defined]
        "family": "ISSC-2009-style double-bottom unit",
        "evidence_level": "RECONSTRUCTED_PARAMETER_SET",
        "reference": None,
        "note": ("structural arrangement and scantling ranges reconstructed from the "
                 "ISSC Committee III.1 ultimate-strength benchmark description; the "
                 "committee reference value is not quoted (data not available in-repo)"),
        "material": material_grade,
        "plate_law": plate_law,
    }
    return sec


# ---------------------------------------------------------------------------
# Paik-style stiffened panel family
# ---------------------------------------------------------------------------
def paik_style_panel(
    beta: float = 2.0,
    plate_law: str = "faulkner",
    stiffener_type: str = "T",
    sigma_y: float = 315.0e6,
    a: float = 3.2,
    b: float = 0.8,
    column_axis: str = "strong",
) -> SectionModel:
    """Stiffened panel with a prescribed plate slenderness ``beta``.

    Parameters
    ----------
    beta : float
        Plate slenderness ``(b/t) sqrt(sigma_y/E)``; the plate thickness is
        derived from it (``b``, ``sigma_y`` and ``E`` are held fixed), so the
        series varies **only** the plate thickness.
    plate_law : str
        ``"faulkner"`` (default, imperfection-inclusive) or ``"von_karman"``.
    stiffener_type : str
        ``"T"`` — T-bar 180 x 11 + 80 x 12 (flanged, tripping restrained);
        ``"FB"`` — flat bar 180 x 12 (tripping active).
    sigma_y, a, b : float
        Yield stress [Pa], frame spacing [m], stiffener spacing [m].

    Notes
    -----
    ``column_axis="strong"`` (default for the panel family) puts the stiffener
    cap on the *panel-column* curve (stiffener + associated plating about the
    axis parallel to the plating), because that is the mechanism the CSR-type
    design formula also models.  The hull-girder cases use the section-level
    default ``"weak"`` (lateral / tripping), which is the relevant mode there.
    """
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    mat = _material(sigma_y)
    t = float(b * np.sqrt(sigma_y / STEEL_E) / beta)
    plates = [PlateStrip(name="panel_plate", thickness=t, span_b=b, span_a=a,
                         z_low=0.0, z_high=t, width=b, sigma_y=sigma_y,
                         orientation="horizontal")]
    if stiffener_type.upper() == "T":
        st = StiffenerGroup(name="panel_stiffener", n_long=1, h_s=0.180, t_w=0.011,
                            b_f=0.080, t_f=0.012, spacing=b, span=a, z_base=t,
                            direction=1.0, sigma_y=sigma_y)
    elif stiffener_type.upper() == "FB":
        st = StiffenerGroup(name="panel_stiffener", n_long=1, h_s=0.180, t_w=0.012,
                            b_f=0.0, t_f=0.0, spacing=b, span=a, z_base=t,
                            direction=1.0, sigma_y=sigma_y)
    else:
        raise ValueError("stiffener_type must be 'T' or 'FB'")
    sec = SectionModel(name=f"paik_panel_beta{beta:.2f}", plates=tuple(plates),
                       stiffeners=(st,), material=mat,
                       depth=float(t + st.h_s + st.t_f), plate_law=plate_law,
                       stiffener_column_axis=column_axis)
    sec.provenance = {  # type: ignore[attr-defined]
        "family": "Paik-style stiffened panel (beta series)",
        "evidence_level": "RECONSTRUCTED_PARAMETER_SET",
        "note": ("beta varied through the plate thickness with fixed a/b, stiffener "
                 "and yield stress; comparison is against the CSR-type design "
                 "formula, not against a digitised test series"),
        "beta": float(beta),
        "plate_thickness": t,
        "plate_law": plate_law,
        "stiffener_type": stiffener_type,
        "column_axis": column_axis,
        "sigma_y": sigma_y,
    }
    return sec


def paik_style_beta_series(betas: Sequence[float] = (1.5, 2.0, 2.5, 3.0),
                           plate_law: str = "faulkner",
                           stiffener_type: str = "T") -> List[SectionModel]:
    """The ``beta`` sweep of :func:`paik_style_panel`."""
    return [paik_style_panel(beta=b, plate_law=plate_law,
                             stiffener_type=stiffener_type) for b in betas]


# ---------------------------------------------------------------------------
# box girder
# ---------------------------------------------------------------------------
def box_girder(
    breadth: float = 0.80,
    depth: float = 0.80,
    t_flange: float = 0.008,
    t_web: float = 0.005,
    span: float = 3.20,
    sigma_y: float = 250.0e6,
    plate_law: str = "von_karman",
    web_stiffener_spacing: Optional[float] = None,
) -> SectionModel:
    """Stiffened box girder in pure bending (four plating strips, no stiffeners).

    The classic box-girder benchmark arrangement (Dow 1991; Gordo & Guedes Soares
    1996): a square box built from four plates, loaded in pure bending.  The
    compression flange buckles between the webs (``span_b = breadth``), the webs
    are supported by the flanges and a stiffener spacing ``web_stiffener_spacing``
    (default ``span/4``, representing transverse stiffeners).

    Parameters are *lengths in metres*; the default set is a slender box
    (``beta ~ 3.5`` for the flange), which is the case dominated by plate
    buckling.  Use thick plates / small breadth for the stocky (plastic-collapse
    dominated) variant.
    """
    mat = _material(sigma_y)
    b = float(breadth)
    d = float(depth)
    sb_web = float(web_stiffener_spacing if web_stiffener_spacing is not None
                   else span / 4.0)
    plates = [
        PlateStrip(name="bottom_flange", thickness=t_flange, span_b=b, span_a=span,
                   z_low=0.0, z_high=t_flange, width=b, sigma_y=sigma_y,
                   orientation="horizontal"),
        PlateStrip(name="top_flange", thickness=t_flange, span_b=b, span_a=span,
                   z_low=d - t_flange, z_high=d, width=b, sigma_y=sigma_y,
                   orientation="horizontal"),
        PlateStrip(name="web_port", thickness=t_web, span_b=sb_web, span_a=span,
                   z_low=t_flange, z_high=d - t_flange, sigma_y=sigma_y,
                   orientation="vertical"),
        PlateStrip(name="web_starboard", thickness=t_web, span_b=sb_web,
                   span_a=span, z_low=t_flange, z_high=d - t_flange,
                   sigma_y=sigma_y, orientation="vertical"),
    ]
    sec = SectionModel(name="box_girder", plates=tuple(plates), stiffeners=(), material=mat,
                       depth=d, plate_law=plate_law)
    sec.provenance = {  # type: ignore[attr-defined]
        "family": "box girder in pure bending",
        "evidence_level": "RECONSTRUCTED_PARAMETER_SET",
        "note": ("four-plate square box; reference = closed-form limit analysis "
                 "(box_girder_plastic_moment), the cited tests' collapse moments are "
                 "not reproduced offline"),
        "breadth": b, "depth": d, "t_flange": t_flange, "t_web": t_web,
        "span": span, "sigma_y": sigma_y, "plate_law": plate_law,
    }
    return sec


def box_girder_plastic_moment(
    j_section: SectionModel,
    sigma_y: Optional[float] = None,
    compression_flange_cap: Optional[float] = None,
    compression_flange_ids: Sequence[str] = ("top_flange",),
) -> Dict[str, object]:
    """Closed-form rigid-plastic collapse moment of a box section.

    Two limit-analysis values are returned:

    ``M_plastic``
        full-plastic moment: every element at ``sigma_y``, neutral axis placed so
        that the areas in tension and compression balance, tension and
        compression both at yield;
    ``M_capped``
        the same analysis with the compression flange(s) stress limited to
        ``compression_flange_cap`` — the classical "effective width" lower
        bound, i.e. the plastic collapse moment of the section whose compression
        plating only carries its ultimate average stress.

    Both are computed with **statics only** (no fiber analysis, no incremental
    sweep, no element curves), so they are an independent reference for the Smith
    solution.

    Returns
    -------
    dict with ``M_plastic``, ``M_capped``, the neutral axis positions, the areas
    and the yield stress used.
    """
    sy = float(sigma_y if sigma_y is not None else j_section.material.sigma_y0)
    comp_ids = tuple(compression_flange_ids)
    comp_area = sum(p.area for p in j_section.plates if p.name in comp_ids)
    ten_area = j_section.area - comp_area
    if comp_area <= 0.0:
        raise ValueError("compression flange area must be positive")
    # 1) full plastic moment: NA splits the area in half
    z_pl = _plastic_neutral_axis(j_section, cap_map={}, sigma_y=sy)
    m_pl = _plastic_moment(j_section, z_pl, cap_map={}, sigma_y=sy)
    # 2) capped compression flange
    cap = float(compression_flange_cap if compression_flange_cap is not None
                else sy)
    cap_map = {name: cap for name in comp_ids}
    z_cap = _plastic_neutral_axis(j_section, cap_map=cap_map, sigma_y=sy)
    m_cap = _plastic_moment(j_section, z_cap, cap_map=cap_map, sigma_y=sy)
    return {
        "M_plastic_Nm": float(m_pl),
        "M_capped_Nm": float(m_cap),
        "z_plastic_axis": float(z_pl),
        "z_plastic_axis_capped": float(z_cap),
        "sigma_y": sy,
        "compression_flange_area": float(comp_area),
        "tension_area": float(ten_area),
        "compression_flange_cap": cap,
        "method": "rigid-plastic limit analysis (statics only)",
    }


def _strip_capacities(p: PlateStrip, cap_map: Dict[str, float],
                      sigma_y: float) -> Tuple[float, float]:
    """``(compression cap, tension cap)`` of one strip [Pa].

    In tension a strip yields at its own yield stress (no buckling), in
    compression it is limited by its *element strength* (plate strength law or
    yield), which the caller passes in ``cap_map`` keyed by component name.
    """
    sy = float(p.sigma_y if p.sigma_y is not None else sigma_y)
    cap_c = float(cap_map[p.name]) if p.name in cap_map else sy
    return cap_c, sy


def _strip_statics(p: PlateStrip, z_p: float, cap_c: float,
                   cap_t: float) -> Tuple[float, float, float]:
    """``(force_above, force_below, |moment| about z_p)`` of one strip.

    Horizontal plating is a concentrated fiber at its centroid; vertical or
    inclined plating is integrated over its extent (``area / (z_high - z_low)``
    per metre of height), which matters: lumping a web at its centroid throws
    away its whole contribution to the plastic moment.  Material above the
    neutral axis is at its compression capacity, material below at its tension
    capacity.
    """
    if not p.is_vertical:
        f_c = p.area * cap_c
        f_t = p.area * cap_t
        if p.z_centroid >= z_p:
            return f_c, 0.0, f_c * (p.z_centroid - z_p)
        return 0.0, f_t, f_t * (z_p - p.z_centroid)
    z1, z2 = p.z_low, p.z_high
    rho = p.area / (z2 - z1)
    above = rho * cap_c * max(0.0, z2 - max(z1, z_p))
    below = rho * cap_t * max(0.0, min(z2, z_p) - z1)
    moment = 0.0
    if z2 > z_p:
        moment += rho * cap_c * (z2 - z_p) ** 2 / 2.0
    if z1 < z_p:
        moment += rho * cap_t * (z_p - z1) ** 2 / 2.0
    return above, below, moment


def _plastic_neutral_axis(j_section: SectionModel, cap_map: Dict[str, float],
                          sigma_y: float) -> float:
    """Depth ``z_p`` balancing the compression and tension capacities (bisection)."""
    z_lo = min(p.z_low for p in j_section.plates)
    z_hi = max(p.z_high for p in j_section.plates)

    def imbalance(z_p: float) -> float:
        above = below = 0.0
        for p in j_section.plates:
            cap_c, cap_t = _strip_capacities(p, cap_map, sigma_y)
            a, b, _ = _strip_statics(p, z_p, cap_c, cap_t)
            above += a
            below += b
        return above - below

    lo, hi = z_lo, z_hi
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if imbalance(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def _plastic_moment(j_section: SectionModel, z_p: float,
                    cap_map: Dict[str, float], sigma_y: float) -> float:
    """Plastic moment about the strength neutral axis (compression above)."""
    m = 0.0
    for p in j_section.plates:
        cap_c, cap_t = _strip_capacities(p, cap_map, sigma_y)
        _a, _b, mom = _strip_statics(p, z_p, cap_c, cap_t)
        m += mom
    return float(m)


def rigid_plastic_reference(j_section: SectionModel,
                            cap_map: Optional[Dict[str, float]] = None,
                            sigma_y: Optional[float] = None) -> Dict[str, object]:
    """Rigid-plastic (limit analysis) collapse moment with element strength caps.

    This is the independent reference used by criterion C3: the section is
    assumed rigid-perfectly plastic, each plating strip carrying at most its
    *element strength* in compression (plate strength law of the section's
    ``plate_law``) and its yield stress in tension, with the neutral axis placed
    so that the compression and tension resultants balance.  It uses **statics
    only** — no incremental sweep, no element load-end shortening curve, no fiber
    discretisation — so agreement with the Smith solution checks the incremental
    solver, not the element curves.

    Parameters
    ----------
    j_section : SectionModel
    cap_map : dict or None
        ``{component name: compression cap}``; ``None`` takes the caps that the
        section itself assigns (``SectionModel.fibers``), i.e. the same element
        strengths the Smith analysis uses.
    sigma_y : float or None
        Yield stress used for the tension side; ``None`` uses each strip's own.

    Returns
    -------
    dict
        ``M_reference_Nm``, ``M_plastic_Nm`` (uncapped, full plasticity),
        ``z_axis``, the cap map and the method tag.
    """
    if cap_map is None:
        fibers = j_section.fibers(n_fibers=1, n_web_fibers=1)
        cap_map = {}
        for f in fibers:
            cap_map.setdefault(f.group, f.sigma_cap)
    sy = float(sigma_y if sigma_y is not None else j_section.material.sigma_y0)
    z_p = _plastic_neutral_axis(j_section, cap_map, sy)
    m_cap = _plastic_moment(j_section, z_p, cap_map, sy)
    z_pl = _plastic_neutral_axis(j_section, {}, sy)
    m_pl = _plastic_moment(j_section, z_pl, {}, sy)
    return {
        "M_reference_Nm": float(m_cap),
        "M_plastic_Nm": float(m_pl),
        "z_axis": float(z_p),
        "z_plastic_axis": float(z_pl),
        "cap_map_MPa": {k: v / 1e6 for k, v in cap_map.items()},
        "sigma_y": sy,
        "method": ("rigid-plastic limit analysis: element caps in compression, "
                   "yield in tension, strength-balanced neutral axis (statics only)"),
    }


# ---------------------------------------------------------------------------
# KCS real midship section
# ---------------------------------------------------------------------------
def midship_area_from_outline(outline: Sequence[Tuple[float, float]],
                              draft: float,
                              n_levels: int = 4000) -> float:
    """Underwater sectional area of a closed section outline [m^2].

    ``outline`` is the *half-breadth* curve of the section, i.e. a list of
    ``(x >= 0, z)`` points ordered along the surface from the keel to the deck
    edge (straight segments are interpolated linearly, so a bilge arc must be
    given as a sufficiently fine polyline).  The area below ``draft`` is
    ``2 * integral(x dz)`` evaluated with the trapezoid rule on ``n_levels + 1``
    evenly spaced waterlines (end points included).

    This is the adapter used to bring a section outline produced elsewhere
    (a CAD/section-extraction tool such as TensorCAD, a lines plan, a
    digitised drawing) into the Smith analysis: any outline can be gated against
    a published midship area coefficient before it is idealised into plating
    strips.  It is also used as an independent cross-check of the closed-form
    ``B*T - 2(1 - pi/4) R^2`` value of the KCS idealisation.
    """
    if draft <= 0.0:
        raise ValueError("draft must be positive")
    pts = [(float(x), float(z)) for x, z in outline]
    if len(pts) < 2:
        raise ValueError("outline needs at least two points")
    if any(x < 0.0 for x, _ in pts):
        raise ValueError("outline must be the half-breadth curve (x >= 0)")
    # endpoint-inclusive grid: the breadth is discontinuous at the deck edge, so a
    # midpoint grid loses an O(dz) sliver of the integration domain
    z = np.linspace(0.0, float(draft), int(n_levels) + 1)
    x = np.zeros_like(z)
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        if z1 == z0:
            continue
        lo, hi = min(z0, z1), max(z0, z1)
        m = (z >= lo) & (z <= hi)
        if not np.any(m):
            continue
        xi = x0 + (x1 - x0) * (z[m] - z0) / (z1 - z0)
        x[m] = np.maximum(x[m], xi)
    return float(2.0 * np.trapezoid(x, z))


def kcs_particulars() -> Dict[str, float]:
    """Published principal particulars of the KRISO Container Ship (KCS).

    ``Lpp``, ``B``, ``D``, ``T``, ``Cb``, ``Cm``, ``Cwp``, ``S`` from the
    publicly distributed KCS geometry description (MOERI/KRISO; the hull form
    used in the Tokyo-2015 workshop series).  The values are *inputs* here: they
    are used to build and gate the midship section, not fitted from it.
    """
    return {
        "Lpp": 230.0,
        "B": 32.2,
        "D": 19.0,
        "T": 10.8,
        "Cb": 0.651,
        "Cm": 0.977,
        "Cwp": 0.821,
        "displacement_m3": 52030.0,
    }


@dataclass
class KCSMidshipSection:
    """Real (non-box) KCS midship section with its geometry gate.

    Attributes
    ----------
    section : SectionModel
        The idealised midship section.
    particulars : dict
        Published KCS particulars (see :func:`kcs_particulars`).
    bilge_radius : float
        Derived bilge radius [m] (the only derived parameter).
    plate_law : str
    """

    section: SectionModel
    particulars: Dict[str, float]
    bilge_radius: float
    plate_law: str = "von_karman"
    provenance: Dict[str, object] = field(default_factory=dict)

    # -- geometry gate -----------------------------------------------------
    def midship_area(self, draft: Optional[float] = None) -> float:
        """Underwater midship area of the modelled *outline* [m^2].

        The outline is the flat of bottom + the two bilge quarter circles + the
        vertical sides up to the draft, so
        ``A = B*T - 2*(1 - pi/4) R^2`` for ``T >= R``.
        """
        p = self.particulars
        t = float(p["T"] if draft is None else draft)
        r = self.bilge_radius
        if t < r:
            raise ValueError("draft below the bilge radius: outline formula not valid")
        return float(p["B"] * t - 2.0 * (1.0 - np.pi / 4.0) * r ** 2)

    def midship_area_coefficient(self, draft: Optional[float] = None) -> float:
        p = self.particulars
        t = float(p["T"] if draft is None else draft)
        return float(self.midship_area(t) / (p["B"] * t))

    def outline(self, n_arc: int = 180) -> List[Tuple[float, float]]:
        """Half-breadth outline of the modelled section, keel to deck edge [m].

        Flat of bottom -> bilge quarter circle -> side shell.  This is the array a
        CAD/section-extraction tool would supply; the geometry gate integrates it
        independently of the closed-form area.
        """
        p = self.particulars
        r = self.bilge_radius
        xc = p["B"] / 2.0 - r
        pts: List[Tuple[float, float]] = [(0.0, 0.0), (xc, 0.0)]
        theta = np.linspace(-np.pi / 2.0, 0.0, n_arc)
        pts.extend((float(xc + r * np.cos(t)), float(r + r * np.sin(t))) for t in theta)
        pts.append((float(p["B"] / 2.0), float(p["D"])))
        return pts

    def geometry_gate(self, tolerance: float = 2.0e-3) -> Dict[str, object]:
        """Machine-checkable geometry gate (``Cm`` and the principal dimensions).

        Two independent area evaluations are compared: the closed form
        ``B*T - 2(1 - pi/4) R^2`` and the waterline integration of the section
        :meth:`outline` (the value a CAD section-extraction would supply).  The
        published ``Cm`` is matched by construction (the bilge radius is solved
        from it), so the gate is a check of the *implementation* and of the
        outline/idealisation consistency, not a fit.
        """
        p = self.particulars
        cm_model = self.midship_area_coefficient()
        area_outline = midship_area_from_outline(self.outline(), p["T"])
        cm_outline = float(area_outline / (p["B"] * p["T"]))
        checks = {
            "Cm_matches_published": bool(abs(cm_model - p["Cm"]) <= tolerance),
            "Lpp_positive": bool(p["Lpp"] > 0.0),
            "B_over_T_reasonable": bool(2.5 <= p["B"] / p["T"] <= 4.0),
            "Lpp_over_B_reasonable": bool(5.0 <= p["Lpp"] / p["B"] <= 8.5),
            "D_over_T_reasonable": bool(1.3 <= p["D"] / p["T"] <= 2.0),
            "bilge_radius_physical": bool(0.5 <= self.bilge_radius <= 0.25 * p["B"]),
            "section_gate_passes": bool(self.section.geometry_validation()["all_pass"]),
            "outline_area_matches_closed_form": bool(
                abs(area_outline - self.midship_area()) <= 1e-3 * self.midship_area()),
        }
        return {
            "case": "KCS midship section",
            "particulars": dict(p),
            "Cm_model": cm_model,
            "Cm_outline": cm_outline,
            "Cm_published": p["Cm"],
            "Cm_error": float(cm_model - p["Cm"]),
            "midship_area_m2": float(self.midship_area()),
            "midship_area_outline_m2": area_outline,
            "bilge_radius": float(self.bilge_radius),
            "checks": checks,
            "all_pass": bool(all(checks.values())),
        }


def kcs_midship_section(
    plate_law: str = "von_karman",
    double_bottom_height: float = 2.0,
    hatch_half_width: float = 13.1,
    cm_tolerance: float = 2.0e-3,
) -> KCSMidshipSection:
    """Build the KCS midship section (single skin, double bottom, hopper tanks).

    Structural arrangement (``x`` from centreline, ``z`` from baseline):

    ==========================  ===============================================
    component                   geometry
    ==========================  ===============================================
    bottom shell                23.56 m wide x 20 mm (HT36), long. @ 0.85 m
    bilge plate (P/S)           quarter circle R = 4.32 m, 16 mm (HT36)
    side shell lower            z 4.32 -> 10.80 m, 16 mm (HT36)
    side shell upper            z 10.80 -> 19.00 m, 13 mm (HT36)
    hopper plate (P/S)          from inner-bottom edge down to the bottom shell,
                                14 mm (mild), 54.7 deg
    inner bottom                19.56 m wide x 13 mm (mild), long. @ 0.85 m
    deck strip (P/S)            3.00 m wide x 25 mm (HT36), long. @ 0.85 m
    hatch coaming (P/S)         1.00 m high x 25 mm (HT36)
    ==========================  ===============================================

    The bilge radius is **derived** from the published midship area coefficient
    ``Cm = 0.977``:
    ``R = sqrt((1 - Cm) B T / (2 (1 - pi/4)))``, i.e. the idealisation
    reproduces the published geometry rather than being tuned to the answer.
    """
    p = kcs_particulars()
    B, T, D = p["B"], p["T"], p["D"]
    s_ms, s_ht = HULL_STRUCTURAL_STEEL["mild"], HULL_STRUCTURAL_STEEL["HT36"]
    mat = _material(s_ht)
    a = 2.5          # transverse frame spacing [m]
    s_long = 0.85    # longitudinal spacing [m]

    r_bilge = float(np.sqrt((1.0 - p["Cm"]) * B * T / (2.0 * (1.0 - np.pi / 4.0))))
    flat_bottom = B - 2.0 * r_bilge
    h_db = float(double_bottom_height)

    plates = [
        PlateStrip(name="bottom_shell", thickness=0.022, span_b=s_long, span_a=a,
                   z_low=0.0, z_high=0.022, width=flat_bottom, sigma_y=s_ht),
        PlateStrip(name="bilge_port", thickness=0.020, span_b=s_long, span_a=a,
                   z_low=0.0, z_high=r_bilge, arc_length=np.pi * r_bilge / 2.0,
                   sigma_y=s_ht, orientation="inclined"),
        PlateStrip(name="bilge_starboard", thickness=0.020, span_b=s_long, span_a=a,
                   z_low=0.0, z_high=r_bilge, arc_length=np.pi * r_bilge / 2.0,
                   sigma_y=s_ht, orientation="inclined"),
        PlateStrip(name="side_lower_port", thickness=0.018, span_b=s_long, span_a=a,
                   z_low=r_bilge, z_high=12.0, sigma_y=s_ht, orientation="vertical"),
        PlateStrip(name="side_lower_starboard", thickness=0.018, span_b=s_long,
                   span_a=a, z_low=r_bilge, z_high=12.0, sigma_y=s_ht,
                   orientation="vertical"),
        PlateStrip(name="side_upper_port", thickness=0.016, span_b=s_long, span_a=a,
                   z_low=12.0, z_high=D, sigma_y=s_ht, orientation="vertical"),
        PlateStrip(name="side_upper_starboard", thickness=0.016, span_b=s_long,
                   span_a=a, z_low=12.0, z_high=D, sigma_y=s_ht,
                   orientation="vertical"),
        # deck: the hatch opening takes most of the beam, so the deck strips at the
        # hatch sides are the heavily loaded compression flange (thick HT36 plate)
        PlateStrip(name="deck_port", thickness=0.035, span_b=s_long, span_a=a,
                   z_low=D, z_high=D + 0.035, width=3.0, sigma_y=s_ht),
        PlateStrip(name="deck_starboard", thickness=0.035, span_b=s_long, span_a=a,
                   z_low=D, z_high=D + 0.035, width=3.0, sigma_y=s_ht),
        # hatch side girder: deep longitudinal girder under the deck strip edge
        PlateStrip(name="hatch_side_girder_port", thickness=0.015, span_b=1.20,
                   span_a=a, z_low=D - 1.20, z_high=D, sigma_y=s_ht,
                   orientation="vertical"),
        PlateStrip(name="hatch_side_girder_starboard", thickness=0.015,
                   span_b=1.20, span_a=a, z_low=D - 1.20, z_high=D, sigma_y=s_ht,
                   orientation="vertical"),
        # hatch coaming: 1.40 m high box above the deck, a major longitudinal member
        PlateStrip(name="hatch_coaming_port", thickness=0.035, span_b=1.40, span_a=a,
                   z_low=D, z_high=D + 1.40, sigma_y=s_ht, orientation="vertical"),
        PlateStrip(name="hatch_coaming_starboard", thickness=0.035, span_b=1.40,
                   span_a=a, z_low=D, z_high=D + 1.40, sigma_y=s_ht,
                   orientation="vertical"),
        PlateStrip(name="inner_bottom", thickness=0.014, span_b=s_long, span_a=a,
                   z_low=h_db, z_high=h_db + 0.014, width=2.0 * 9.78, sigma_y=s_ms),
    ]

    # hopper: from the inner-bottom edge (x = +-9.78, z = h_db) down to the
    # bottom shell, i.e. the slope runs inboard 1.40 m over the full DB height
    # less one plate thickness -> 54.7 deg
    hop_dz = h_db - 0.022
    hop_dx = 1.40
    hop_arc = float(np.hypot(hop_dz, hop_dx))
    # topside tank: sloped plate from the hatch-side deck girder down to the side
    # shell, the standard container-ship upper-wing structure (it carries the
    # upper part of the compression flange in sagging)
    ts_dz = D - 13.5
    ts_dx = (B / 2.0) - hatch_half_width
    ts_arc = float(np.hypot(ts_dz, ts_dx))
    plates.extend([
        PlateStrip(name="topside_port", thickness=0.012, span_b=0.85, span_a=a,
                   z_low=13.5, z_high=D, arc_length=ts_arc, sigma_y=s_ht,
                   orientation="inclined"),
        PlateStrip(name="topside_starboard", thickness=0.012, span_b=0.85, span_a=a,
                   z_low=13.5, z_high=D, arc_length=ts_arc, sigma_y=s_ht,
                   orientation="inclined"),
    ])
    plates.extend([
        PlateStrip(name="hopper_port", thickness=0.014, span_b=0.90, span_a=a,
                   z_low=0.022, z_high=h_db, arc_length=hop_arc, sigma_y=s_ms,
                   orientation="inclined"),
        PlateStrip(name="hopper_starboard", thickness=0.014, span_b=0.90, span_a=a,
                   z_low=0.022, z_high=h_db, arc_length=hop_arc, sigma_y=s_ms,
                   orientation="inclined"),
    ])

    n_bot = int(round(flat_bottom / s_long))
    n_ib = int(round(2.0 * 9.78 / s_long))
    stiffeners = [
        # bottom longitudinals: bulb/flat bars on 0.85 m spacing
        StiffenerGroup(name="bottom_long", n_long=n_bot, h_s=0.350, t_w=0.012,
                       b_f=0.0, t_f=0.0, spacing=s_long, span=a, z_base=0.022,
                       direction=1.0, sigma_y=s_ht),
        StiffenerGroup(name="inner_bottom_long", n_long=n_ib, h_s=0.220, t_w=0.010,
                       b_f=0.0, t_f=0.0, spacing=s_long, span=a, z_base=h_db,
                       direction=-1.0, sigma_y=s_ms),
        StiffenerGroup(name="hopper_long", n_long=4, h_s=0.200, t_w=0.009,
                       b_f=0.0, t_f=0.0, spacing=0.65, span=a, z_base=0.9,
                       direction=1.0, sigma_y=s_ms, z_offset=0.0),
        # horizontal side-shell longitudinals: webs point inboard, so the group is
        # lumped at its own level (z_offset = 0) instead of being fiberised
        StiffenerGroup(name="side_long_lower", n_long=2, h_s=0.250, t_w=0.010,
                       b_f=0.0, t_f=0.0, spacing=0.85, span=a, z_base=7.0,
                       direction=1.0, sigma_y=s_ht, z_offset=0.0),
        StiffenerGroup(name="side_long_upper", n_long=2, h_s=0.250, t_w=0.010,
                       b_f=0.0, t_f=0.0, spacing=0.85, span=a, z_base=14.0,
                       direction=1.0, sigma_y=s_ht, z_offset=0.0),
        # deck longitudinals under the 3 m deck strips (4 per side)
        StiffenerGroup(name="deck_long", n_long=8, h_s=0.250, t_w=0.012,
                       b_f=0.0, t_f=0.0, spacing=s_long, span=a, z_base=D,
                       direction=-1.0, sigma_y=s_ht),
        # topside tank stiffeners (transverse frames on the slope, lumped at mid-height)
        StiffenerGroup(name="topside_long", n_long=4, h_s=0.200, t_w=0.010,
                       b_f=0.0, t_f=0.0, spacing=0.80, span=a, z_base=16.2,
                       direction=1.0, sigma_y=s_ht, z_offset=0.0),
        # coaming longitudinals / hatch cover rail stiffening (lumped)
        StiffenerGroup(name="coaming_long", n_long=4, h_s=0.200, t_w=0.012,
                       b_f=0.0, t_f=0.0, spacing=0.70, span=a, z_base=D + 0.2,
                       direction=1.0, sigma_y=s_ht, z_offset=0.0),
    ]
    sec = SectionModel(name="kcs_midship_section", plates=tuple(plates),
                       stiffeners=tuple(stiffeners), material=mat,
                       depth=D + 1.4, plate_law=plate_law)
    provenance = {
        "family": "KCS (KRISO Container Ship) real midship section",
        "evidence_level": "RECONSTRUCTED_PARAMETER_SET",
        "particulars_source": "published KCS particulars (MOERI/KRISO hull form)",
        "scantlings": ("representative 3600 TEU single-skin container-ship scantlings "
                       "reproducing the structural arrangement and plate-thickness "
                       "ranges; not a yard drawing"),
        "bilge_radius_derivation": "solved from the published midship area coefficient Cm",
        "plate_law": plate_law,
    }
    return KCSMidshipSection(section=sec, particulars=p, bilge_radius=r_bilge,
                             plate_law=plate_law, provenance=provenance)