"""Generic plated-section idealisation for Smith-method progressive collapse.

The pre-existing Smith machinery in this repository is specialised in two
directions:

``elastoplastic_collapse.progressive_collapse``
    a *single-span single-stiffener* panel (axial compression, column
    equilibrium about the panel mid-span);
``hull_redundancy.smith``
    a *damaged* hull-girder section expressed as ``PlateStrip`` /
    ``StiffenerGroup`` components with a fixed 3-fiber web discretisation.

Neither generalises: the first cannot represent a box girder or a full midship
section, the second has no mesh-refinement parameter (so no grid-convergence
study is possible) and no element-strength caps for stiffeners.

This module supplies the missing *geometric* layer: a small, explicit component
model

* :class:`PlateStrip` — a plating strip, horizontal (bottom, deck, inner
  bottom) or vertical (side shell, inner side, hopper, web);
* :class:`StiffenerGroup` — ``n_long`` identical longitudinals lumped into one
  super-element (the standard Smith "element group" idealisation), attached to
  a named plating strip;
* :class:`SectionModel` — the assembly, with geometric section properties
  (area, elastic neutral axis, second moment, section moduli) and a rational
  fiberisation :meth:`SectionModel.fibers`.

The fiberisation is the *mesh* of the Smith analysis: every vertical strip is
divided into ``n_fibers`` equal fibers in the ``z`` direction and every
stiffener web into ``n_web_fibers`` fibers (plus one fiber through the flange
thickness).  ``n_fibers`` / ``n_web_fibers`` are therefore the refinement
parameters used by the grid-convergence study
(:mod:`tensorphysics.smith_section_benchmark.convergence`).

Sign conventions
----------------
``z`` is measured **upward from the baseline** [m]; ``sigma`` is **positive in
compression**; the strain passed to the element curves is the *shortening*
``s = -eps_axial`` (positive in compression).  A sagging bending moment is
positive and produces shortening at the deck.

Notes on the idealisation (declared limitations)
-----------------------------------------------
* Each plating strip carries the *average* stress of a panel of buckling width
  ``span_b`` (longitudinal spacing) and span ``span_a`` (transverse frame
  spacing); the strip may be transversely much wider than ``span_b``, in which
  case the plate is assumed to buckle into ``span_b``-wide cells.
* Stiffener elements are lumped (area and centroid exact, own inertia from the
  real cross-section); local web/flange buckling and tripping are represented
  only through the element strength *caps* computed in
  :mod:`tensorphysics.smith_section_benchmark.strength_curves`.
* Residual stresses and initial distortions are not modelled explicitly (they
  enter the element curves only through the softening branch, as in
  :mod:`tensorphysics.elastoplastic_collapse.progressive_collapse`).
* Shear lag and cross-section distortion are ignored (rigid-section assumption).

References
----------
- Smith, C.S. (1977). Trans. SNAME 85, 1-18.
- Paik, J.K. & Thayamballi, A.K. (2003). *Ultimate Limit State Design of
  Steel-Plated Structures*. Wiley.
- Gordo, J.M. & Guedes Soares, C. (1996). "Approximate load shortening curves
  for stiffened plates under uniaxial compression." In *Integrity of Offshore
  Structures V*, 189-211.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..elastoplastic_collapse.von_mises_plasticity import VonMisesMaterial
from ..hull_redundancy.elements import plate_buckling_coefficient
from .strength_curves import plate_ultimate_stress

__all__ = [
    "PlateStrip",
    "StiffenerGroup",
    "SectionModel",
    "SectionFiber",
    "STEEL_E",
    "STEEL_NU",
    "STEEL_SHEAR_MODULUS",
]

STEEL_E = 206.0e9          # Pa
STEEL_NU = 0.3
STEEL_SHEAR_MODULUS = STEEL_E / (2.0 * (1.0 + STEEL_NU))


def _default_material(sigma_y: float = 235.0e6) -> VonMisesMaterial:
    return VonMisesMaterial(E=STEEL_E, nu=STEEL_NU, sigma_y0=float(sigma_y), H=0.0)


# ---------------------------------------------------------------------------
# components
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlateStrip:
    """A plating strip of the section.

    Parameters
    ----------
    name : str
        Component identifier (used for failure-mode attribution).
    thickness : float
        Plate thickness ``t`` [m].
    span_b : float
        Buckling width of the plating [m] (longitudinal stiffener spacing; for
        an unstiffened plate the panel width).
    span_a : float
        Panel span [m] (transverse frame spacing).
    z_low, z_high : float
        Vertical extent of the strip [m].  For horizontal plating
        ``z_high - z_low == thickness`` (the strip is one fiber); for vertical
        plating the extent is the physical height and the strip is fiberised
        over it.
    width : float or None
        Transverse extent [m] for horizontal plating.  ``None`` marks a vertical
        (or inclined) plate, whose area is ``thickness * (z_high - z_low)``.
    sigma_y : float or None
        Yield stress of the strip [Pa]; ``None`` inherits the section material.
    orientation : str
        "horizontal" | "vertical" | "inclined" — informational, drives only the
        reporting of the idealisation.
    arc_length : float or None
        Developed (arc) length of an inclined strip — bilge plate, hopper slope,
        hatch coaming web.  When given, the strip area is
        ``arc_length * thickness`` and the strip is fiberised over its vertical
        extent ``[z_low, z_high]`` exactly like vertical plating.
    """

    name: str
    thickness: float
    span_b: float
    span_a: float
    z_low: float
    z_high: float
    width: Optional[float] = None
    sigma_y: Optional[float] = None
    orientation: str = "horizontal"
    arc_length: Optional[float] = None

    def __post_init__(self) -> None:
        if self.thickness <= 0.0:
            raise ValueError(f"{self.name}: thickness must be positive")
        if self.span_b <= 0.0 or self.span_a <= 0.0:
            raise ValueError(f"{self.name}: span_b and span_a must be positive")
        if self.z_high <= self.z_low:
            raise ValueError(f"{self.name}: z_high must exceed z_low")
        if self.width is not None and self.width <= 0.0:
            raise ValueError(f"{self.name}: width must be positive when given")
        if self.arc_length is not None and self.arc_length <= 0.0:
            raise ValueError(f"{self.name}: arc_length must be positive when given")
        if self.width is not None and self.arc_length is not None:
            raise ValueError(f"{self.name}: give either width or arc_length, not both")
        if (self.width is not None
                and abs((self.z_high - self.z_low) - self.thickness) > 1e-9):
            raise ValueError(
                f"{self.name}: horizontal strip must have z_high - z_low == thickness")
        if (self.arc_length is not None
                and self.arc_length < (self.z_high - self.z_low) - 1e-9):
            raise ValueError(
                f"{self.name}: arc_length cannot be shorter than the vertical extent")

    @property
    def area(self) -> float:
        """Cross-sectional area of the strip [m^2]."""
        if self.arc_length is not None:
            return float(self.arc_length * self.thickness)
        if self.width is None:
            return float(self.thickness * (self.z_high - self.z_low))
        return float(self.width * self.thickness)

    @property
    def z_centroid(self) -> float:
        return 0.5 * (self.z_low + self.z_high)

    @property
    def is_vertical(self) -> bool:
        """True when the strip extends over ``z`` and must be fiberised."""
        return self.width is None or self.arc_length is not None

    def plate_slenderness(self, sigma_y: float, E: float = STEEL_E) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y / E)`` [-]."""
        return float((self.span_b / self.thickness) * np.sqrt(sigma_y / E))

    def buckling_stress(self, sigma_y: float, E: float = STEEL_E,
                        nu: float = STEEL_NU) -> float:
        """Elastic plate buckling stress of the strip [Pa]."""
        k = plate_buckling_coefficient(self.span_a / self.span_b)
        return float(k * np.pi ** 2 * E / (12.0 * (1.0 - nu ** 2))
                     * (self.thickness / self.span_b) ** 2)


@dataclass(frozen=True)
class StiffenerGroup:
    """``n_long`` identical longitudinals lumped into one Smith super-element.

    Parameters
    ----------
    name : str
        Component identifier.
    n_long : int
        Number of identical stiffeners in the group.
    h_s, t_w : float
        Web height and thickness [m].
    b_f, t_f : float
        Flange width / thickness [m]; use ``0.0`` for a flat bar.
    spacing : float
        Spacing between the stiffeners of the group [m].
    span : float
        Stiffener column length [m] (transverse frame spacing).
    z_base : float
        ``z`` of the plating to which the stiffener is welded [m].
    direction : float
        ``+1`` if the stiffener extends upward from ``z_base``, ``-1`` downward.
    sigma_y : float or None
        Yield stress [Pa]; ``None`` inherits the section material.
    tripped : bool
        Include the torsional (tripping) strength cap.
    z_offset : float or None
        When given, the group is *not* fiberised over a web: the whole group
        area is lumped into a single fiber at ``z_base + z_offset``.  This is how
        transverse-direction material (e.g. horizontal side-shell longitudinals,
        whose webs point inboard rather than vertically) is represented without
        introducing a spurious vertical centroid shift.
    """

    name: str
    n_long: int
    h_s: float
    t_w: float
    b_f: float
    t_f: float
    spacing: float
    span: float
    z_base: float
    direction: float = 1.0
    sigma_y: Optional[float] = None
    tripped: bool = True
    z_offset: Optional[float] = None

    def __post_init__(self) -> None:
        if self.n_long < 1:
            raise ValueError(f"{self.name}: n_long must be >= 1")
        if self.h_s <= 0.0 or self.t_w <= 0.0:
            raise ValueError(f"{self.name}: web dimensions must be positive")
        if self.b_f < 0.0 or self.t_f < 0.0:
            raise ValueError(f"{self.name}: flange dimensions must be non-negative")
        if (self.b_f == 0.0) != (self.t_f == 0.0):
            raise ValueError(f"{self.name}: flange width and thickness given together")
        if self.spacing <= 0.0 or self.span <= 0.0:
            raise ValueError(f"{self.name}: spacing and span must be positive")
        if self.direction not in (-1.0, 1.0):
            raise ValueError(f"{self.name}: direction must be +1 or -1")

    # -- single-stiffener section properties -------------------------------
    @property
    def area_one(self) -> float:
        """Area of one stiffener [m^2]."""
        return float(self.h_s * self.t_w + self.b_f * self.t_f)

    @property
    def area(self) -> float:
        """Area of the whole group [m^2]."""
        return float(self.n_long * self.area_one)

    @property
    def z_centroid(self) -> float:
        """``z`` of the group centroid [m] (all stiffeners share one ``z``)."""
        a_w = self.h_s * self.t_w
        a_f = self.b_f * self.t_f
        z_loc = (a_w * 0.5 * self.h_s + a_f * (self.h_s + 0.5 * self.t_f)) / self.area_one
        return float(self.z_base + self.direction * z_loc)

    @property
    def z_flange(self) -> float:
        """``z`` of the flange mid-thickness [m] (``z_base`` for a flat bar)."""
        if self.b_f <= 0.0:
            return float(self.z_base + self.direction * self.h_s)
        return float(self.z_base + self.direction * (self.h_s + 0.5 * self.t_f))

    def weak_axis_inertia(self) -> float:
        """Stiffener own inertia about the weld line [m^4] (``t_w h^3/3`` form)."""
        return float(self.t_w * self.h_s ** 3 / 3.0 + self.b_f * self.t_f ** 3 / 12.0)


# ---------------------------------------------------------------------------
# fibers
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SectionFiber:
    """One axial fiber of the section with its element parameters."""

    area: float
    z: float
    group: str
    kind: str                 # "plate" | "stiffener"
    sigma_y: float
    sigma_cr: float           # elastic buckling stress (inf for stocky/stiffener)
    sigma_cap: float          # element compression ultimate stress
    cap_source: str           # "yield" | "plate_von_karman" | "tripping" | ...
    beta: float               # plate slenderness (-inf for stiffener fibers)


# ---------------------------------------------------------------------------
# section model
# ---------------------------------------------------------------------------
@dataclass
class SectionModel:
    """Assembly of plating strips and stiffener groups forming a section beam.

    Parameters
    ----------
    name : str
        Case identifier.
    plates : sequence of PlateStrip
    stiffeners : sequence of StiffenerGroup
    material : VonMisesMaterial
        Parent material (per-component ``sigma_y`` overrides the yield stress).
    depth : float or None
        Section depth [m] used for reporting/validation; ``None`` derives it
        from the component extents.
    plate_law : str
        Post-buckling plateau law of the plating element:

        ``"von_karman"``
            perfect-plate effective width, ``sigma_u,p = sqrt(sigma_cr sigma_y)``
            (the law used by
            :mod:`tensorphysics.elastoplastic_collapse.progressive_collapse`);
        ``"faulkner"``
            imperfection-inclusive strength curve
            ``sigma_u,p = (2/beta - 1/beta^2) sigma_y`` for ``beta > 1``
            (Faulkner 1975; identical to ``hullgirder.ultimate``'s reduction).

        The distinction matters: at ``beta = 2`` the two laws give
        ``0.50 sigma_y`` and ``0.75 sigma_y`` respectively, so the element law is
        a *modelled variable* and is reported with every result.
    """

    name: str
    plates: Tuple[PlateStrip, ...]
    stiffeners: Tuple[StiffenerGroup, ...] = ()
    material: VonMisesMaterial = field(default_factory=lambda: _default_material())
    depth: Optional[float] = None
    plate_law: str = "von_karman"
    stiffener_column_axis: str = "weak"

    def __post_init__(self) -> None:
        self.plates = tuple(self.plates)
        self.stiffeners = tuple(self.stiffeners)
        if not self.plates:
            raise ValueError("a section needs at least one plate strip")
        if self.plate_law not in ("von_karman", "faulkner"):
            raise ValueError("plate_law must be 'von_karman' or 'faulkner'")
        if self.stiffener_column_axis not in ("weak", "strong"):
            raise ValueError("stiffener_column_axis must be 'weak' or 'strong'")
        names = [p.name for p in self.plates] + [s.name for s in self.stiffeners]
        if len(set(names)) != len(names):
            raise ValueError("component names must be unique")
        if self.depth is None:
            self.depth = float(max(p.z_high for p in self.plates)
                               - min(p.z_low for p in self.plates))

    # -- geometry ----------------------------------------------------------
    @property
    def components(self) -> Tuple[object, ...]:
        return self.plates + self.stiffeners

    def sigma_y_of(self, comp) -> float:
        return float(comp.sigma_y if comp.sigma_y is not None else self.material.sigma_y0)

    @property
    def area(self) -> float:
        return float(sum(c.area for c in self.components))

    @property
    def first_moment(self) -> float:
        return float(sum(c.area * c.z_centroid for c in self.components))

    @property
    def neutral_axis(self) -> float:
        """Elastic neutral axis height above baseline [m]."""
        return float(self.first_moment / self.area)

    @property
    def second_moment(self) -> float:
        """Second moment about the elastic neutral axis [m^4].

        Component own inertia is included where it is resolvable (vertical
        plating strips, ``t (h)^3/12``); horizontal plating contributes only the
        parallel-axis term (its own inertia is negligible for ``t << B``).
        """
        z_na = self.neutral_axis
        total = 0.0
        for c in self.components:
            total += c.area * (c.z_centroid - z_na) ** 2
            if isinstance(c, PlateStrip) and c.is_vertical:
                h = c.z_high - c.z_low
                total += c.thickness * h ** 3 / 12.0
        return float(total)

    @property
    def section_moduli(self) -> Dict[str, float]:
        """Elastic section moduli at deck and keel [m^3]."""
        z_na = self.neutral_axis
        z_top = max(p.z_high for p in self.plates)
        z_bot = min(p.z_low for p in self.plates)
        i = self.second_moment
        return {
            "z_neutral_axis": z_na,
            "z_deck": float(z_top),
            "z_keel": float(z_bot),
            "S_deck": float(i / max(z_top - z_na, 1e-9)),
            "S_keel": float(i / max(z_na - z_bot, 1e-9)),
            "I": float(i),
        }

    def geometry_validation(self) -> Dict[str, object]:
        """Machine-checkable sanity gate on the idealisation.

        Returns the section area, neutral axis, section moduli and per-strip
        plate slenderness, plus hard checks that the idealisation is physical
        (positive area/moduli, neutral axis inside the depth, no component
        outside the depth envelope).
        """
        z_top = max(p.z_high for p in self.plates)
        z_bot = min(p.z_low for p in self.plates)
        sm = self.section_moduli
        checks = {
            "positive_area": bool(self.area > 0.0),
            "na_inside_depth": bool(z_bot < sm["z_neutral_axis"] < z_top),
            "positive_section_modulus": bool(sm["S_deck"] > 0.0 and sm["S_keel"] > 0.0),
            "stiffeners_within_depth": all(
                min(s.z_base, s.z_flange) >= z_bot - 1e-9
                and max(s.z_base, s.z_flange) <= z_top + 1e-9
                for s in self.stiffeners),
        }
        return {
            "name": self.name,
            "depth": float(z_top - z_bot),
            "area": self.area,
            "section_moduli": sm,
            "checks": checks,
            "all_pass": bool(all(checks.values())),
        }

    # -- fiberisation ------------------------------------------------------
    def fibers(self, n_fibers: int = 2, n_web_fibers: int = 2) -> List[SectionFiber]:
        """Discretise the section into axial fibers.

        Parameters
        ----------
        n_fibers : int
            Number of fibers through the vertical extent of each *vertical*
            plating strip (horizontal plating is always a single fiber).  This
            is the primary mesh-refinement parameter.
        n_web_fibers : int
            Number of fibers through each stiffener web height; the flange is
            one fiber.
        """
        if n_fibers < 1 or n_web_fibers < 1:
            raise ValueError("fiber counts must be >= 1")
        out: List[SectionFiber] = []
        for p in self.plates:
            sy = self.sigma_y_of(p)
            beta = p.plate_slenderness(sy, self.material.E)
            sigma_cr = p.buckling_stress(sy, self.material.E, self.material.nu)
            if sigma_cr >= sy:
                cap, source = sy, "yield"
            else:
                cap, source = plate_ultimate_stress(
                    beta=beta, sigma_cr=sigma_cr, sigma_y=sy, law=self.plate_law)
            n = 1 if not p.is_vertical else int(n_fibers)
            dz = (p.z_high - p.z_low) / n
            area_f = p.area / n
            for i in range(n):
                out.append(SectionFiber(
                    area=area_f,
                    z=p.z_low + (i + 0.5) * dz,
                    group=p.name,
                    kind="plate",
                    sigma_y=sy,
                    sigma_cr=sigma_cr,
                    sigma_cap=cap,
                    cap_source=source,
                    beta=beta,
                ))
        for s in self.stiffeners:
            caps = stiffener_strength_caps(
                s, self.material, tripped=s.tripped,
                column_axis=self.stiffener_column_axis)
            sy = caps["sigma_y"]
            if s.z_offset is not None:
                out.append(SectionFiber(
                    area=s.area,
                    z=float(s.z_base + s.z_offset),
                    group=s.name,
                    kind="stiffener",
                    sigma_y=sy,
                    sigma_cr=float("inf"),
                    sigma_cap=caps["sigma_cap"],
                    cap_source=caps["cap_source"],
                    beta=float("-inf"),
                ))
                continue
            # web fibers
            z_lo = s.z_base
            direction = s.direction
            dz = s.h_s / n_web_fibers
            a_w = s.t_w * dz
            for i in range(n_web_fibers):
                out.append(SectionFiber(
                    area=a_w * s.n_long,
                    z=z_lo + direction * (i + 0.5) * dz,
                    group=s.name,
                    kind="stiffener",
                    sigma_y=sy,
                    sigma_cr=float("inf"),
                    sigma_cap=caps["sigma_cap"],
                    cap_source=caps["cap_source"],
                    beta=float("-inf"),
                ))
            if s.b_f > 0.0 and s.t_f > 0.0:
                out.append(SectionFiber(
                    area=s.b_f * s.t_f * s.n_long,
                    z=s.z_flange,
                    group=s.name,
                    kind="stiffener",
                    sigma_y=sy,
                    sigma_cr=float("inf"),
                    sigma_cap=caps["sigma_cap"],
                    cap_source=caps["cap_source"],
                    beta=float("-inf"),
                ))
        if not out:
            raise ValueError("fiberisation produced no fibers")
        if abs(sum(f.area for f in out) - self.area) > 1e-9 * max(self.area, 1.0):
            raise RuntimeError("fiber areas do not sum to the section area")
        return out

    def component_areas(self) -> Dict[str, float]:
        return {c.name: float(c.area) for c in self.components}


# ---------------------------------------------------------------------------
# stiffener element strength caps
# ---------------------------------------------------------------------------
def stiffener_strength_caps(
    stiff: StiffenerGroup,
    material: VonMisesMaterial,
    tripped: bool = True,
    attached_plate_t: Optional[float] = None,
    column_axis: str = "weak",
) -> Dict[str, object]:
    """Compression strength caps of a stiffener super-element.

    Three mechanisms are evaluated and the *lowest* stress is used as the
    element cap (zero-tangent plateau outside the elastic range):

    =========================  ==============================================
    mechanism                  expression
    =========================  ==============================================
    squash (yield)             ``sigma_y`` (+ linear hardening H, if any)
    column (weak axis)         ``Johnson-Ostenfeld(pi^2 E / (span/r)^2)`` with
                               ``r^2 = I_p / A``, ``I_p = t_w h^3/3 + b_f t_f^3/12``;
                               with ``column_axis="strong"`` the Euler stress of
                               the stiffener *with its associated plating* about
                               the axis parallel to the plating is used instead
                               (panel-column mode, see
                               :func:`strength_curves.stiffener_strong_axis_column`)
    tripping (torsional)       ``Johnson-Ostenfeld(G J / I_p)`` with
                               ``J = h t_w^3/3`` (flat-bar torsion, Guedes
                               Soares / Smith form, see
                               :func:`hull_redundancy.elements.tripping_stress`)
    =========================  ==============================================

    The returned dict contains the participating stresses and the identifier of
    the governing mechanism (used by the failure-mode classifier).
    """
    from ..hull_redundancy.elements import johnson_ostenfeld, tripping_stress

    sy = float(stiff.sigma_y if stiff.sigma_y is not None else material.sigma_y0)
    # squash including isotropic hardening
    sigma_squash = sy
    # weak-axis column
    inertia = stiff.weak_axis_inertia()
    area = stiff.area_one
    r = float(np.sqrt(inertia / area))
    sigma_e = float(np.pi ** 2 * material.E / (stiff.span / r) ** 2)
    sigma_col = johnson_ostenfeld(sigma_e, sy)
    caps: Dict[str, object] = {
        "sigma_y": sy,
        "sigma_squash": sigma_squash,
        "sigma_col_weak_axis": sigma_col,
        "sigma_euler_weak_axis": sigma_e,
    }
    candidates: Dict[str, float] = {"yield": sigma_squash, "stiffener_column": sigma_col}
    if column_axis == "strong":
        from .strength_curves import stiffener_strong_axis_column

        strong = stiffener_strong_axis_column(stiff, material,
                                              effective_plate_width=stiff.spacing)
        caps["sigma_column_strong_axis"] = strong["sigma_column_strong_axis"]
        caps["sigma_euler_strong_axis"] = strong["sigma_euler_strong_axis"]
        candidates["stiffener_column"] = min(sigma_col,
                                             strong["sigma_column_strong_axis"])
    if tripped and stiff.b_f <= 0.0:
        # the closed-form tripping stress is derived for a flat bar
        plate_t = attached_plate_t if attached_plate_t is not None else stiff.t_w
        sigma_trip = tripping_stress(material, stiff.h_s, stiff.t_w, stiff.span, plate_t,
                                     n_long=stiff.n_long)
        caps["sigma_tripping"] = float(sigma_trip)
        candidates["stiffener_tripping"] = float(sigma_trip)
    cap_source = min(candidates, key=candidates.get)
    # an instability mechanism whose strength is within 0.5% of the squash strength
    # is numerically indistinguishable from yielding; label it as yield so the
    # failure-mode classifier does not report a spurious buckling mode
    if candidates[cap_source] >= 0.995 * sy:
        cap_source = "yield"
    caps["sigma_cap"] = float(candidates[cap_source])
    caps["cap_source"] = cap_source
    return caps