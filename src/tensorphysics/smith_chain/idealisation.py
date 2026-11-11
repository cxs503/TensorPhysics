"""Cross-scale sectional idealisation for the incremental Smith method.

The module describes *any* thin-walled steel cross-section that carries a
longitudinal bending moment with a single, dimension-independent vocabulary:

``PlatePanel``
    A strip of plating.  Carry-over area ``width * thickness``; the buckling
    width is the *panel* width ``spacing_b`` (stiffener spacing for a flange,
    panel depth for a web) and the panel length is the frame spacing ``span_a``.

``StiffenerGroup``
    ``n`` identical longitudinal profiles (flat bar or T-bar) welded to a named
    plate panel.  The profile is split into web fibres (distributed over the web
    height, so the lever arm is integrated) plus one flange fibre for T-bars.

``GirderSection``
    A depth ``D`` (z measured from the bottom fibre upwards) populated with the
    two element families above, plus the material.  Elastic properties (neutral
    axis, second moment, section moduli, first-yield moment) and the
    rigid-plastic modulus are derived from the same idealisation used by the
    Smith solver, so anchors and solver agree at the model level; the
    *independent* closed-form anchors live in :mod:`.benchmarks`.

Why a new idealisation instead of :mod:`tensorphysics.hull_redundancy`?
------------------------------------------------------------------------
That package idealises *one* 350 m-class bulker midship section for damage
redundancy work.  The Smith cross-scale chain needs the same mechanics on
three structurally different objects (a stiffened panel, a box girder and a
double-bottom hull girder), therefore the element set is generalised here and
the section *type* is inferred from the geometry (:meth:`GirderSection.kind`).
The idealisation intentionally references hull_redundancy's verified element
laws and elastic routines where they apply, instead of duplicating them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from ..elastoplastic_collapse import VonMisesMaterial

__all__ = [
    "STEEL",
    "PlatePanel",
    "StiffenerGroup",
    "GirderSection",
    "classify_section",
]

#: Default structural steel used by the benchmark suite.  ``H = 0``
#: (elastic-perfectly plastic) is deliberate: the cross-scale verification chain
#: compares the solver against *rigid-plastic limit loads* (``M_p = sigma_y Z_p``)
#: as analytic references, and a hardening modulus would put the numerical
#: collapse moment above those references for purely numerical reasons.  Users
#: who need hardening pass their own ``VonMisesMaterial``.
STEEL = VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=315.0e6, H=0.0)


@dataclass(frozen=True)
class PlatePanel:
    """A plating strip contributing axial area at one height.

    Parameters
    ----------
    id : str
        Unique element name (used by the failure-mode attribution).
    z : float
        Height of the plate mid-surface above the section bottom [m].
    width : float
        Transverse extent of plating lumped into this element [m] (area =
        ``width * thickness``).
    thickness : float
        Plate thickness [m].
    spacing_b : float
        Buckling width of the panel [m]: longitudinal spacing for a flange,
        unsupported panel depth for a web.
    span_a : float
        Panel length in the load direction [m] (frame spacing).
    k_factor : float, optional
        Elastic plate buckling coefficient; ``None`` selects the simply
        supported ``k(a/b)`` law of :func:`tensorphysics.hull_redundancy.elements.plate_buckling_coefficient`.
    label : str
        Free-form description for reports.
    """

    id: str
    z: float
    width: float
    thickness: float
    spacing_b: float
    span_a: float
    k_factor: Optional[float] = None
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("width", "thickness", "spacing_b", "span_a"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{self.id}: {name} must be positive")
        if self.thickness <= 0.0:
            raise ValueError(f"{self.id}: thickness must be positive")

    @property
    def area(self) -> float:
        return float(self.width * self.thickness)

    @property
    def beta(self) -> float:
        """Plate slenderness ``beta = (b/t) sqrt(sigma_y/E)`` (material free)."""
        return float(self.spacing_b / self.thickness)


@dataclass(frozen=True)
class StiffenerGroup:
    """``n`` identical longitudinal profiles carried by a plating panel.

    ``h`` is the total profile depth measured from the plate mid-surface to the
    outer fibre of the stiffener (web height for a flat bar, web + flange
    thickness for a T-bar).  ``b_f = 0`` selects a flat bar.
    """

    id: str
    z_attach: float
    n: int
    h: float
    t_w: float
    spacing_b: float
    span_a: float
    plate_id: str
    b_f: float = 0.0
    t_f: float = 0.0
    plate_t: float = 0.0
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("h", "t_w", "spacing_b", "span_a"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{self.id}: {name} must be positive")
        if self.n < 1:
            raise ValueError(f"{self.id}: n must be >= 1")
        if self.b_f < 0.0 or self.t_f < 0.0:
            raise ValueError(f"{self.id}: flange dimensions must be >= 0")
        if (self.b_f > 0.0) != (self.t_f > 0.0):
            raise ValueError(f"{self.id}: b_f and t_f must be given together")

    @property
    def is_tee(self) -> bool:
        return self.b_f > 0.0

    @property
    def web_height(self) -> float:
        return self.h - self.t_f if self.is_tee else self.h

    @property
    def area_one(self) -> float:
        return float(self.web_height * self.t_w + self.b_f * self.t_f)

    @property
    def area(self) -> float:
        return float(self.n * self.area_one)

    @property
    def z_web_bottom(self) -> float:
        """Bottom fibre of the profile measured from the plate mid-surface."""
        return float(self.z_attach - 0.5 * self.plate_t)

    def profile_points(self) -> List[Tuple[float, float]]:
        """``(z, area)`` breakpoints of the profile relative to the plate."""
        z0 = self.z_web_bottom
        pts = [(z0, self.web_height * self.t_w), (z0 + self.web_height, self.b_f * self.t_f)]
        return [(z, a) for z, a in pts if a > 0.0]


@dataclass
class GirderSection:
    """A thin-walled cross-section analysed by the incremental Smith method."""

    name: str
    depth: float
    plates: Tuple[PlatePanel, ...]
    stiffeners: Tuple[StiffenerGroup, ...] = ()
    material: VonMisesMaterial = field(default=STEEL)
    label: str = ""

    def __post_init__(self) -> None:
        self.plates = tuple(self.plates)
        self.stiffeners = tuple(self.stiffeners)
        if self.depth <= 0.0:
            raise ValueError("section depth must be positive")
        if not self.plates:
            raise ValueError("section needs at least one plate panel")
        ids = [p.id for p in self.plates] + [s.id for s in self.stiffeners]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate element ids in section {self.name}: {ids}")
        for s in self.stiffeners:
            if s.plate_id not in {p.id for p in self.plates}:
                raise ValueError(f"{s.id}: unknown plate_id {s.plate_id!r}")
            if s.z_attach < -0.5 or s.z_attach > self.depth + 0.5:
                raise ValueError(f"{s.id}: z_attach outside the section depth")

    # -- element access ----------------------------------------------------
    def element_ids(self) -> Tuple[str, ...]:
        return tuple([p.id for p in self.plates] + [s.id for s in self.stiffeners])

    # -- elastic properties ------------------------------------------------
    @property
    def area_total(self) -> float:
        return float(sum(p.area for p in self.plates) + sum(s.area for s in self.stiffeners))

    @property
    def z_neutral_elastic(self) -> float:
        num = sum(p.area * p.z for p in self.plates)
        den = sum(p.area for p in self.plates)
        for s in self.stiffeners:
            for z, a in s.profile_points():
                num += s.n * a * z
                den += s.n * a
        return float(num / den)

    def second_moment(self) -> float:
        """Second moment about the elastic neutral axis [m^4] (thin-wall)."""
        z0 = self.z_neutral_elastic
        I = 0.0
        for p in self.plates:
            I += p.area * (p.z - z0) ** 2 + p.width * p.thickness**3 / 12.0
        for s in self.stiffeners:
            for z, a in s.profile_points():
                I += s.n * a * (z - z0) ** 2
        return float(I)

    def section_modulus(self, extreme: str = "bottom") -> float:
        z0 = self.z_neutral_elastic
        c = z0 if extreme == "bottom" else self.depth - z0
        return float(self.second_moment() / c)

    def first_yield_moment(self) -> float:
        """Elastic first-yield moment ``sigma_y * min(S_bottom, S_deck)`` [N m]."""
        return float(self.material.sigma_y0 * min(self.section_modulus("bottom"),
                                                  self.section_modulus("deck")))

    # -- rigid-plastic properties (dense fibre integration) ----------------
    def plastic_modulus(self, n_dense: int = 200) -> float:
        """Rigid-plastic modulus ``Z_p`` of the idealisation [m^3]."""
        z, a = fibre_arrays(self, n_sub=n_dense)
        z_na = _plastic_neutral_axis(z, a)
        return float(np.sum(np.abs(z - z_na) * a))

    def plastic_moment(self, n_dense: int = 200) -> float:
        return float(self.material.sigma_y0 * self.plastic_modulus(n_dense))

    # -- geometry classification ------------------------------------------
    @property
    def kind(self) -> str:
        return classify_section(self)

    def breadth(self) -> float:
        """Approximate section breadth: widest plating level extent [m]."""
        levels = _plate_levels(self.plates)
        return float(max(sum(p.width for p in lvl) for lvl in levels))

    def describe(self) -> Dict[str, object]:
        z0 = self.z_neutral_elastic
        return {
            "name": self.name,
            "kind": self.kind,
            "depth_m": self.depth,
            "breadth_m": self.breadth(),
            "aspect_D_over_B": self.depth / max(self.breadth(), 1e-12),
            "area_m2": self.area_total,
            "n_plates": len(self.plates),
            "n_stiffener_groups": len(self.stiffeners),
            "z_neutral_elastic_m": z0,
            "I_m4": self.second_moment(),
            "S_bottom_m3": self.section_modulus("bottom"),
            "S_deck_m3": self.section_modulus("deck"),
            "M_first_yield_MNm": self.first_yield_moment() / 1e6,
            "Z_plastic_m3": self.plastic_modulus(),
            "M_plastic_MNm": self.plastic_moment() / 1e6,
            "sigma_y_MPa": self.material.sigma_y0 / 1e6,
            "element_ids": list(self.element_ids()),
        }


# ---------------------------------------------------------------------------
# fibre generation (shared by the solver and by the plastic-modulus anchors)
# ---------------------------------------------------------------------------
def _plate_levels(plates: Sequence[PlatePanel], tol: float = 0.02) -> List[List[PlatePanel]]:
    """Cluster plate panels into distinct height levels (tolerance ``tol`` m)."""
    levels: List[List[PlatePanel]] = []
    for p in sorted(plates, key=lambda q: q.z):
        if levels and abs(p.z - levels[-1][0].z) <= tol:
            levels[-1].append(p)
        else:
            levels.append([p])
    return levels


def _plastic_neutral_axis(z: np.ndarray, a: np.ndarray, n_iter: int = 200) -> float:
    """Equal-area axis of the fibre set (bisection on the area moment)."""
    lo, hi = float(np.min(z)), float(np.max(z))
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        if np.sum(a[z > mid]) - np.sum(a[z < mid]) > 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))


def fibre_arrays(section: GirderSection, n_sub: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """Discretise a section into ``(z, area)`` fibres.

    ``n_sub`` fibres are used across the thickness of every plate panel and
    across the web height of every stiffener profile (plus one fibre per
    flange), so ``n_sub`` is the single refinement parameter of the Smith grid
    study.
    """
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1")
    zs: List[float] = []
    areas: List[float] = []
    for p in section.plates:
        dz = p.thickness / n_sub
        for i in range(n_sub):
            zs.append(p.z - 0.5 * p.thickness + (i + 0.5) * dz)
            areas.append(p.width * dz)
    for s in section.stiffeners:
        h_w = s.web_height
        if h_w > 0.0 and s.t_w > 0.0:
            dz = h_w / n_sub
            for i in range(n_sub):
                zs.append(s.z_web_bottom + (i + 0.5) * dz)
                areas.append(s.n * s.t_w * dz)
        if s.is_tee:
            zs.append(s.z_web_bottom + h_w + 0.5 * s.t_f)
            areas.append(s.n * s.b_f * s.t_f)
    return np.asarray(zs, dtype=float), np.asarray(areas, dtype=float)


def classify_section(section: GirderSection) -> str:
    """Infer the structural archetype of a section from its geometry.

    The classification only routes *reporting and anchor selection* - the Smith
    mechanics are identical for every archetype.  Rules (breadth ``B`` = widest
    plating level extent, ``D/B`` = aspect ratio, ``n_skins`` = number of plate
    levels whose cumulative width exceeds 50 % of ``B``):

    ==========================  ==============================================
    condition                   class
    ==========================  ==============================================
    one plating level, no stiff. ``"plate"``
    one plating level, stiff.    ``"stiffened_panel"``
    >= 3 skins                   ``"hull_girder"``   (double bottom / box hull)
    2 skins and D/B >= 0.25      ``"box_girder"``
    2 skins and D/B <  0.25      ``"double_bottom"``
    ==========================  ==============================================
    """
    levels = _plate_levels(section.plates)
    B = max(sum(p.width for p in lvl) for lvl in levels)
    n_skins = sum(1 for lvl in levels if sum(p.width for p in lvl) >= 0.5 * B)
    has_stiff = bool(section.stiffeners)
    if len(levels) == 1:
        return "stiffened_panel" if has_stiff else "plate"
    if n_skins >= 3:
        return "hull_girder"
    if section.depth / max(B, 1e-12) >= 0.25:
        return "box_girder"
    return "double_bottom"