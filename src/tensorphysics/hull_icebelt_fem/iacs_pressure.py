"""IACS Polar Class ice-belt design pressure: the pressure-area law.

Physical background
-------------------
The ice belt of a Polar Class hull is loaded by a *local* ice pressure patch.
IACS UR I (and the IMO Polar Code that mirrors it) idealises that patch as a
rectangle whose height is ``0.6 h_I`` with a 4:1 width/height aspect ratio (see
:mod:`tensorphysics.plate_buckling.loads`), while the *magnitude* of the
pressure is governed by the area effect: a large contact patch cannot sustain
the crushing pressure of a small one because the ice fails progressively and
the highest-pressure zones are statistically rare.  That is the classical
pressure-area relation

.. math::

    p(A) = p_1 \\left(\\frac{A}{A_1}\\right)^{-e}, \\qquad e \\approx 0.5

with ``e`` inside the IACS/ISO 19906 band 0.45-0.55 for local patch pressures.

Anchor provenance (declared, not machine-verified)
--------------------------------------------------
This repository does **not** carry the IACS UR I rule text, and
``benchmarks/ice_nav_refs`` (the IN10-L2 anchor product) is not present in
``main``.  The anchor below is therefore *self-built and declared*:

* class table ``h_I``: upper bound of the published Polar Class ice-thickness
  range (IACS UR I Part 1 Ch.3 class descriptions), evidence grade
  ``EXTERNAL_CITATION_DECLARED``;
* reference pressure ``p_1 = 1.2 MPa`` at ``A_1 = 1 m²``: the same
  ISO 19906 / IACS Polar Class pressure level already used inside this
  repository by :mod:`tensorphysics.hullgirder.ice_load`;
* exponent ``e = 0.5``: mid-band of the IACS/ISO 0.45-0.55 interval.

Every consumer (see :mod:`tensorphysics.hull_icebelt_fem.xcheck`) must report
:data:`ANCHOR_PROVENANCE` verbatim so that the strength of the claim is never
stronger than the evidence.  The *numerical* verification in that module (FEM
vs closed-form Navier solution, < 5 %) is independent of this anchor; only the
engineering numbers inherit the citation-only evidence grade.

Units: SI throughout (m, m², Pa).  The JSON layer converts to MPa/mm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

__all__ = [
    "IACSPressureError",
    "POLAR_CLASS_ICE_THICKNESS",
    "ANCHOR_PROVENANCE",
    "PATCH_HEIGHT_FACTOR",
    "PATCH_ASPECT_RATIO",
    "REFERENCE_PRESSURE",
    "REFERENCE_AREA",
    "PRESSURE_AREA_EXPONENT",
    "design_ice_thickness",
    "pressure_area_law",
    "patch_area",
    "IcePatchDesignPressure",
    "design_patch_pressure",
]


class IACSPressureError(ValueError):
    """Raised for an invalid ice class, patch area or pressure input."""


#: IACS UR I load-patch height factor (patch height = 0.6 h_I).
PATCH_HEIGHT_FACTOR = 0.6
#: IACS UR I load-patch aspect ratio (patch width / patch height).
PATCH_ASPECT_RATIO = 4.0
#: Reference pressure of the pressure-area law at the reference area [Pa].
REFERENCE_PRESSURE = 1.2e6
#: Reference area of the pressure-area law [m²].
REFERENCE_AREA = 1.0
#: Pressure-area exponent (IACS/ISO local patch band 0.45-0.55).
PRESSURE_AREA_EXPONENT = 0.5

#: Design (upper-bound) ice thickness per Polar Class [m], IACS UR I class
#: descriptions.  ``declared_citation`` -- see :data:`ANCHOR_PROVENANCE`.
POLAR_CLASS_ICE_THICKNESS: Dict[str, float] = {
    "PC1": 3.5,
    "PC2": 3.0,
    "PC3": 2.5,
    "PC4": 2.0,
    "PC5": 1.75,
    "PC6": 1.5,
    "PC7": 1.0,
}

ANCHOR_PROVENANCE: Dict[str, str] = {
    "ice_class_table": (
        "IACS UR I (Rev. 2021) Part 1 Ch.3 Polar Class descriptions, upper "
        "bound of the design ice thickness range "
        "(PC1 3.5 / PC2 3.0 / PC3 2.5 / PC4 2.0 / PC5 1.75 / PC6 1.5 / "
        "PC7 1.0 m). Declared citation: the rule text is not part of this "
        "repository and was not machine-checked against the PDF."
    ),
    "pressure_area_law": (
        "p(A) = p_1 (A/A_1)^(-e) with p_1 = 1.2 MPa at A_1 = 1 m^2 (same "
        "ISO 19906 / IACS Polar Class pressure level already used by "
        "tensorphysics.hullgirder.ice_load) and e = 0.5 (mid-band of the "
        "IACS/ISO 0.45-0.55 local-patch exponent)."
    ),
    "patch_geometry": (
        "Rectangular patch, height 0.6 h_I, aspect 4:1 (IACS UR I), reused "
        "from tensorphysics.plate_buckling.loads."
    ),
    "benchmarks_ice_nav_refs_in_main": "absent -> anchor self-built (declared)",
    "evidence_grade": "EXTERNAL_CITATION_DECLARED",
    "not_claimed": (
        "Verbatim IACS UR I rule compliance or class-society approval of the "
        "resulting scantlings is NOT claimed; only the physics chain and its "
        "numerical verification are."
    ),
}


def _positive(value: float, name: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise IACSPressureError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(v):
        raise IACSPressureError(f"{name} must be finite, got {v!r}")
    if v <= 0.0:
        raise IACSPressureError(f"{name} must be positive, got {v!r}")
    return v


def design_ice_thickness(ice_class: str,
                         table: Optional[Mapping[str, float]] = None) -> float:
    """Design ice thickness ``h_I`` [m] of a Polar Class.

    Raises
    ------
    IACSPressureError
        If ``ice_class`` is not a Polar Class of the IACS table (e.g. ``"PC9"``,
        ``"1A"``, ``"pc6-super"``) -- an unknown class must never silently fall
        back to a default thickness.
    """
    tbl = POLAR_CLASS_ICE_THICKNESS if table is None else table
    if not isinstance(ice_class, str):
        raise IACSPressureError(f"ice class must be a string, got {ice_class!r}")
    key = ice_class.strip().upper()
    if key not in tbl:
        raise IACSPressureError(
            f"unknown Polar Class {ice_class!r}; expected one of "
            f"{sorted(tbl)}")
    return float(tbl[key])


def pressure_area_law(area: float, p_ref: float = REFERENCE_PRESSURE,
                      a_ref: float = REFERENCE_AREA,
                      exponent: float = PRESSURE_AREA_EXPONENT) -> float:
    """Local ice pressure [Pa] on a contact patch of area ``area`` [m²].

    ``p(A) = p_ref (A/a_ref)^(-exponent)``.  The law is strictly decreasing in
    ``A`` (larger patch -> lower sustainable pressure), which is the physical
    content of the IACS/ISO pressure-area family.

    Raises
    ------
    IACSPressureError
        For a non-positive or non-finite area, a non-positive reference
        pressure/area, or an exponent outside ``(0, 1]``.
    """
    A = _positive(area, "patch area")
    p_ref = _positive(p_ref, "reference pressure")
    a_ref = _positive(a_ref, "reference area")
    e = float(exponent)
    if not (0.0 < e <= 1.0):
        raise IACSPressureError(
            f"pressure-area exponent must lie in (0, 1], got {exponent!r}")
    return float(p_ref * (A / a_ref) ** (-e))


def patch_area(h_ice: float, stiffener_spacing: float,
               panel_length: Optional[float] = None) -> float:
    """Design patch area of the plating check [m²].

    The rule patch is ``0.6 h_I`` high and ``4 x 0.6 h_I = 2.4 h_I`` wide.  Two
    conventions are used in the literature and both are supported here:

    * ``panel_length=None`` -> *unclipped rule patch*,
      ``A = 0.6 h_I * 2.4 h_I`` (the contact area the rule is written for);
    * ``panel_length=a`` -> *loaded area on the panel*,
      ``A = min(0.6 h_I, s) * min(2.4 h_I, a)``, i.e. the patch intersected with
      the panel that the finite element model actually sees.  This is the
      convention of :func:`design_patch_pressure` when ``panel_length`` is
      given, because the FEM then applies the pressure uniformly over exactly
      that area -- the pressure and the loaded area stay consistent.

    Raises
    ------
    IACSPressureError
        If ``h_ice``, ``stiffener_spacing`` or ``panel_length`` is non-positive.
    """
    h = _positive(h_ice, "design ice thickness")
    s = _positive(stiffener_spacing, "stiffener spacing")
    height = PATCH_HEIGHT_FACTOR * h
    width = PATCH_ASPECT_RATIO * height
    if panel_length is not None:
        a = _positive(panel_length, "panel length")
        height = min(height, s)
        width = min(width, a)
    return float(height * width)


@dataclass(frozen=True)
class IcePatchDesignPressure:
    """IACS-style design pressure on one ice-belt plate panel.

    Attributes
    ----------
    ice_class, h_ice, stiffener_spacing:
        Class label, design ice thickness [m] and transverse stiffener spacing
        ``s`` [m].
    patch_height, patch_width:
        Rule patch dimensions [m] (height ``0.6 h_I``, aspect 4:1).
    area:
        Design patch area [m²] used in the pressure-area law.
    pressure:
        Design patch pressure [Pa].
    """

    ice_class: str
    h_ice: float
    stiffener_spacing: float
    patch_height: float
    patch_width: float
    area: float
    pressure: float
    p_ref: float = REFERENCE_PRESSURE
    a_ref: float = REFERENCE_AREA
    exponent: float = PRESSURE_AREA_EXPONENT

    def as_dict(self) -> Dict[str, float]:
        """JSON-ready view in SI plus MPa / mm for readability."""
        return {
            "ice_class": self.ice_class,
            "h_ice_m": float(self.h_ice),
            "h_ice_mm": float(self.h_ice) * 1e3,
            "stiffener_spacing_m": float(self.stiffener_spacing),
            "patch_height_m": float(self.patch_height),
            "patch_width_m": float(self.patch_width),
            "patch_area_m2": float(self.area),
            "p_ref_MPa": float(self.p_ref) / 1e6,
            "a_ref_m2": float(self.a_ref),
            "exponent": float(self.exponent),
            "pressure_Pa": float(self.pressure),
            "pressure_MPa": float(self.pressure) / 1e6,
        }


def design_patch_pressure(ice_class: str, stiffener_spacing: float,
                          p_ref: float = REFERENCE_PRESSURE,
                          a_ref: float = REFERENCE_AREA,
                          exponent: float = PRESSURE_AREA_EXPONENT,
                          panel_length: Optional[float] = None) -> IcePatchDesignPressure:
    """Assemble the IACS-style design pressure for one panel.

    Raises
    ------
    IACSPressureError
        For an unknown ice class, a non-positive stiffener spacing or a
        non-positive pressure reference.
    """
    h = design_ice_thickness(ice_class)
    s = _positive(stiffener_spacing, "stiffener spacing")
    area = patch_area(h, s, panel_length=panel_length)
    p = pressure_area_law(area, p_ref=p_ref, a_ref=a_ref, exponent=exponent)
    if p <= 0.0:  # pragma: no cover - guarded by pressure_area_law
        raise IACSPressureError("design pressure must be positive")
    return IcePatchDesignPressure(
        ice_class=ice_class.strip().upper(), h_ice=h, stiffener_spacing=s,
        patch_height=PATCH_HEIGHT_FACTOR * h,
        patch_width=PATCH_ASPECT_RATIO * PATCH_HEIGHT_FACTOR * h,
        area=area, pressure=p, p_ref=float(p_ref), a_ref=float(a_ref),
        exponent=float(exponent))