"""Component-level idealisation of the P350W1-N03 hull-girder section.

Two target sections are provided, both described as a list of *components*:

``PlateStrip``
    A plating strip idealised as one fiber at its centroid.  ``span_b`` is the
    buckling width between longitudinals (NOT the transverse extent of the
    strip): a strip of transverse width 8 m made of ten longitudinals spaced
    0.8 m apart buckles as a 0.8 m panel, so the plate element carries the
    *grouped* area but the *local* buckling width.

``StiffenerGroup``
    One equivalent longitudinal group (``n_long`` real flat bars lumped into one
    element) attached to a named plate strip (``plate_id``).  The grouping is
    the standard Smith-method "super-element" idealisation; it keeps the
    enumeration tractable (<= 12 plate strips x <= 8 stiffener groups) while
    conserving area, centroid and buckling slenderness.

Target section ``p350w1_n03_combined`` (primary)
-----------------------------------------------
Idealised midship box of the P350W1-N03 plate-beam combination closed by a
double-hull side, all steel MS (E = 206 GPa, nu = 0.3, sigma_y = 235 MPa):

============================  =========  ==========  =========  =============
component class               extent     thickness   spacing    z centroid
============================  =========  ==========  =========  =============
bottom plating (3 strips)     24.0 m     18 mm       0.8 m      0.009 m
bottom longitudinals (3 grp)  --         350x12 FB  0.8 m      0.184 m
deck plating (3 strips)       24.0 m     16 mm       0.8 m      19.984 m
deck longitudinals (3 grp)    --         300x11 FB  0.8 m      19.841 m
outer shell (3 strips)        20.0 m     14 mm       0.8 m      3.333/10.0/16.667 m
inner shell (3 strips)        18.0 m     12 mm       0.8 m      5.0/11.0/17.0 m
============================  =========  ==========  =========  =============

Frame spacing a = 2.4 m (longitudinal span between transverse frames), i.e. the
standard panel aspect a/b = 3 used for the plate buckling coefficient.

Target section ``double_hull_side`` (secondary)
----------------------------------------------
The double-hull side assembly alone (outer shell + inner shell + closing
plating), 8 plate strips and 6 stiffener groups, analysed as a stand-alone
girder so that the redundancy ranking can be checked to be section-independent.

Declared limitation: the idealisation is a *section-level* Smith idealisation -
local flange/web buckling and residual stresses are not modelled, and the two
sections are geometrically representative of a 350 m-class double-hull bulker,
not a project-specific scantling drawing.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..elastoplastic_collapse import VonMisesMaterial

__all__ = [
    "PlateStrip",
    "StiffenerGroup",
    "HullSection",
    "DEFAULT_MATERIAL",
    "p350w1_n03_combined_section",
    "double_hull_side_section",
]

DEFAULT_MATERIAL = VonMisesMaterial(E=2.06e11, nu=0.3, sigma_y0=235.0e6, H=1.0e9)


@dataclass(frozen=True)
class PlateStrip:
    """A plating strip carrying hull-girder axial load at one height."""

    id: str
    z: float
    width: float
    thickness: float
    span_b: float
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("width", "thickness", "span_b"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{self.id}: {name} must be positive")

    @property
    def area(self) -> float:
        return float(self.width * self.thickness)

    def with_span(self, span_b: float) -> "PlateStrip":
        return replace(self, span_b=float(span_b))


@dataclass(frozen=True)
class StiffenerGroup:
    """An equivalent longitudinal group (``n_long`` flat bars) on one strip."""

    id: str
    z: float
    area: float
    h: float
    t_w: float
    span_b: float
    plate_t: float
    n_long: int
    plate_id: str
    label: str = ""

    def __post_init__(self) -> None:
        for name in ("area", "h", "t_w", "span_b", "plate_t"):
            if getattr(self, name) <= 0.0:
                raise ValueError(f"{self.id}: {name} must be positive")
        if self.n_long < 1:
            raise ValueError(f"{self.id}: n_long must be >= 1")


@dataclass
class HullSection:
    """A hull-girder cross-section made of plating strips and stiffener groups."""

    name: str
    depth: float
    frame_spacing: float
    plates: Sequence[PlateStrip]
    stiffeners: Sequence[StiffenerGroup]
    material: VonMisesMaterial = DEFAULT_MATERIAL
    meta: Dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.plates:
            raise ValueError("section needs at least one plate strip")
        if self.depth <= 0.0 or self.frame_spacing <= 0.0:
            raise ValueError("depth and frame_spacing must be positive")
        ids = [p.id for p in self.plates] + [s.id for s in self.stiffeners]
        if len(ids) != len(set(ids)):
            raise ValueError("component ids must be unique")
        plate_ids = {p.id for p in self.plates}
        for s in self.stiffeners:
            if s.plate_id not in plate_ids:
                raise ValueError(f"{s.id}: unknown plate_id {s.plate_id}")

    # -- integral section properties ---------------------------------------
    def component_ids(self) -> List[str]:
        return [p.id for p in self.plates] + [s.id for s in self.stiffeners]

    def plate(self, cid: str) -> PlateStrip:
        for p in self.plates:
            if p.id == cid:
                return p
        raise KeyError(cid)

    def stiffener(self, cid: str) -> StiffenerGroup:
        for s in self.stiffeners:
            if s.id == cid:
                return s
        raise KeyError(cid)

    def component_kind(self, cid: str) -> str:
        if cid in {p.id for p in self.plates}:
            return "plate"
        if cid in {s.id for s in self.stiffeners}:
            return "stiffener"
        raise KeyError(cid)

    def area(self) -> float:
        return float(
            sum(p.area for p in self.plates) + sum(s.area for s in self.stiffeners)
        )

    def first_moment(self) -> float:
        return float(
            sum(p.area * p.z for p in self.plates)
            + sum(s.area * s.z for s in self.stiffeners)
        )

    def neutral_axis(self) -> float:
        return self.first_moment() / self.area()

    def second_moment(self) -> float:
        z_n = self.neutral_axis()
        return float(
            sum(p.area * (p.z - z_n) ** 2 for p in self.plates)
            + sum(s.area * (s.z - z_n) ** 2 for s in self.stiffeners)
        )

    def section_moduli(self) -> Dict[str, float]:
        z_n = self.neutral_axis()
        I = self.second_moment()
        return {
            "I_m4": I,
            "z_NA_m": z_n,
            "S_deck_m3": I / (self.depth - z_n),
            "S_bottom_m3": I / z_n,
        }

    def geometry_validation(self) -> Dict[str, object]:
        """Independent re-computation of the integral properties (negative control).

        The same quantities are accumulated with plain python floats in a
        different order, so a transposition error in the dataclass bookkeeping
        shows up as a mismatch instead of silently passing.
        """
        A = 0.0
        S = 0.0
        entries: List[Tuple[float, float]] = []
        for p in self.plates:
            entries.append((p.width * p.thickness, p.z))
        for s in self.stiffeners:
            entries.append((s.area, s.z))
        for a_i, z_i in entries:
            A += a_i
            S += a_i * z_i
        z_n = S / A
        I = sum(a_i * (z_i - z_n) ** 2 for a_i, z_i in entries)
        checks = {
            "area_m2": A,
            "z_NA_m": z_n,
            "I_m4": I,
            "n_components": len(entries),
            "NA_inside_depth": bool(0.0 < z_n < self.depth),
            "positive_section": bool(I > 0.0 and A > 0.0),
        }
        checks["area_rel_err_vs_model"] = abs(A - self.area()) / self.area()
        checks["I_rel_err_vs_model"] = abs(I - self.second_moment()) / self.second_moment()
        checks["valid"] = bool(
            checks["NA_inside_depth"]
            and checks["positive_section"]
            and checks["area_rel_err_vs_model"] < 1e-12
            and checks["I_rel_err_vs_model"] < 1e-12
        )
        return checks


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def p350w1_n03_combined_section(
    material: Optional[VonMisesMaterial] = None,
    frame_spacing: float = 2.4,
    stiffener_spacing: float = 0.8,
) -> HullSection:
    """P350W1-N03 plate-beam combination closed by a double-hull side (primary)."""
    mat = DEFAULT_MATERIAL if material is None else material
    B, D = 24.0, 20.0
    t_bot, t_deck = 0.018, 0.016
    t_out, t_in = 0.014, 0.012
    h_bot, w_bot = 0.350, 0.012
    h_deck, w_deck = 0.300, 0.011

    n_groups = 3
    group_width = B / n_groups                 # 8.0 m per group
    n_long_per_group = int(round(group_width / stiffener_spacing))  # 10 bars
    b_local = group_width / n_long_per_group   # 0.8 m longitudinal spacing

    plates: List[PlateStrip] = []
    stiffeners: List[StiffenerGroup] = []

    for i in range(n_groups):
        pid = f"PB{i + 1}"                     # bottom plating strip
        plates.append(
            PlateStrip(id=pid, z=t_bot / 2.0, width=group_width,
                       thickness=t_bot, span_b=b_local,
                       label=f"bottom plating bay {i + 1}")
        )
        stiffeners.append(
            StiffenerGroup(
                id=f"SB{i + 1}",
                z=t_bot + h_bot / 2.0,
                area=n_long_per_group * h_bot * w_bot,
                h=h_bot, t_w=w_bot, span_b=b_local, plate_t=t_bot,
                n_long=n_long_per_group, plate_id=pid,
                label=f"bottom longitudinals FB {int(h_bot*1000)}x{int(w_bot*1000)} group {i + 1}",
            )
        )

    for i in range(n_groups):
        pid = f"PD{i + 1}"                     # deck plating strip
        plates.append(
            PlateStrip(id=pid, z=D - t_deck / 2.0, width=group_width,
                       thickness=t_deck, span_b=b_local,
                       label=f"deck plating bay {i + 1}")
        )
        stiffeners.append(
            StiffenerGroup(
                id=f"SD{i + 1}",
                z=D - t_deck - h_deck / 2.0,
                area=n_long_per_group * h_deck * w_deck,
                h=h_deck, t_w=w_deck, span_b=b_local, plate_t=t_deck,
                n_long=n_long_per_group, plate_id=pid,
                label=f"deck longitudinals FB {int(h_deck*1000)}x{int(w_deck*1000)} group {i + 1}",
            )
        )

    # double-hull side: outer shell z in [0, D], inner shell z in [t_bot, D]
    z_edges = [0.0, D / 3.0, 2.0 * D / 3.0, D]
    for i in range(3):
        z0, z1 = z_edges[i], z_edges[i + 1]
        plates.append(
            PlateStrip(id=f"PO{i + 1}", z=0.5 * (z0 + z1), width=(z1 - z0),
                       thickness=t_out, span_b=b_local,
                       label=f"outer shell strip {i + 1} (z={z0:.2f}-{z1:.2f} m)")
        )
    z_in_edges = [t_bot, t_bot + (D - t_bot) / 3.0,
                  t_bot + 2.0 * (D - t_bot) / 3.0, D]
    for i in range(3):
        z0, z1 = z_in_edges[i], z_in_edges[i + 1]
        plates.append(
            PlateStrip(id=f"PI{i + 1}", z=0.5 * (z0 + z1), width=(z1 - z0),
                       thickness=t_in, span_b=b_local,
                       label=f"inner shell strip {i + 1} (z={z0:.2f}-{z1:.2f} m)")
        )

    return HullSection(
        name="p350w1_n03_combined",
        depth=D,
        frame_spacing=frame_spacing,
        plates=tuple(plates),
        stiffeners=tuple(stiffeners),
        material=mat,
        meta={
            "breadth_m": B,
            "depth_m": D,
            "stiffener_spacing_m": b_local,
            "frame_spacing_m": frame_spacing,
            "n_plate_strips": len(plates),
            "n_stiffener_groups": len(stiffeners),
            "idealisation": "grouped Smith super-elements (10 longitudinals/group)",
            "source": "P350W1-N03 plate-beam combination + double-hull side",
        },
    )


def double_hull_side_section(
    material: Optional[VonMisesMaterial] = None,
    frame_spacing: float = 2.4,
    stiffener_spacing: float = 0.8,
) -> HullSection:
    """Stand-alone double-hull side assembly (secondary target section)."""
    mat = DEFAULT_MATERIAL if material is None else material
    D = 20.0
    t_out, t_in = 0.014, 0.012
    h_s, w_s = 0.300, 0.011
    b_local = stiffener_spacing
    n_long = int(round(D / 3.0 / b_local))

    plates: List[PlateStrip] = []
    stiffeners: List[StiffenerGroup] = []
    edges = [0.0, D / 3.0, 2.0 * D / 3.0, D]
    for i in range(3):
        z0, z1 = edges[i], edges[i + 1]
        plates.append(PlateStrip(id=f"PO{i + 1}", z=0.5 * (z0 + z1),
                                 width=(z1 - z0), thickness=t_out,
                                 span_b=b_local,
                                 label=f"outer shell strip {i + 1}"))
        stiffeners.append(StiffenerGroup(id=f"SO{i + 1}",
                                         z=0.5 * (z0 + z1),
                                         area=n_long * h_s * w_s,
                                         h=h_s, t_w=w_s, span_b=b_local,
                                         plate_t=t_out, n_long=n_long,
                                         plate_id=f"PO{i + 1}",
                                         label=f"outer shell longitudinals group {i + 1}"))
    for i in range(3):
        z0, z1 = edges[i], edges[i + 1]
        plates.append(PlateStrip(id=f"PI{i + 1}", z=0.5 * (z0 + z1),
                                 width=(z1 - z0), thickness=t_in,
                                 span_b=b_local,
                                 label=f"inner shell strip {i + 1}"))
        stiffeners.append(StiffenerGroup(id=f"SI{i + 1}",
                                         z=0.5 * (z0 + z1),
                                         area=n_long * h_s * w_s,
                                         h=h_s, t_w=w_s, span_b=b_local,
                                         plate_t=t_in, n_long=n_long,
                                         plate_id=f"PI{i + 1}",
                                         label=f"inner shell longitudinals group {i + 1}"))
    # closing plating at top and bottom of the side assembly
    plates.append(PlateStrip(id="PC1", z=0.0, width=1.2, thickness=t_out,
                             span_b=b_local, label="bottom closing plate of side assembly"))
    plates.append(PlateStrip(id="PC2", z=D, width=1.2, thickness=t_out,
                             span_b=b_local, label="top closing plate of side assembly"))

    return HullSection(
        name="double_hull_side",
        depth=D,
        frame_spacing=frame_spacing,
        plates=tuple(plates),
        stiffeners=tuple(stiffeners),
        material=mat,
        meta={
            "depth_m": D,
            "stiffener_spacing_m": b_local,
            "frame_spacing_m": frame_spacing,
            "n_plate_strips": len(plates),
            "n_stiffener_groups": len(stiffeners),
            "idealisation": "double-hull side assembly analysed as a stand-alone girder",
            "source": "P350W1-N03 double-hull side",
        },
    )