"""Component damage states of the RIF campaign node p351w1_n08.

Three component failure modes are defined on top of the intact section:

``remove``
    The component is physically absent.  For a plate strip the strip area is
    deleted (plate-band tearing).  For a stiffener group both its area and its
    support to the associated plating are lost, so the plating buckles at twice
    the longitudinal spacing (the panel spans between the remaining connected
    longitudinals).

``trip``
    Stiffener lateral-torsional buckling (tripping).  The longitudinal remains
    welded to the plating (shear connection retained) but its lateral-torsional
    stiffness is exhausted: the element stress is capped at
    ``sigma_cr,trip`` with zero tangent stiffness beyond (ideal-plastic
    plateau).  The plating keeps its support, so only the stiffener degrades.

``debond``
    Loss of stiffener-plate composite action.  The weld line no longer
    transfers the lateral/transverse action, so (i) the stiffener becomes a
    bare strut with Euler (Johnson-Ostenfeld) capacity about its own weak axis
    and (ii) the associated plating loses its longitudinal support line and
    buckles at twice the spacing.  Severity ranking by construction:
    ``remove > debond > trip > none``.

``none``
    Negative control: the unmodified section, which must return RIF = 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Tuple

from ..elastoplastic_collapse import VonMisesMaterial
from .elements import (
    bare_strut_stress,
    effective_plate_buckling_stress,
    tripping_stress,
)
from .section import HullSection, PlateStrip, StiffenerGroup

__all__ = [
    "MODES",
    "MODE_DESCRIPTION",
    "DamageSpec",
    "DamagedPlate",
    "DamagedStiffener",
    "DamagedSection",
    "apply_damage",
]

MODES: Tuple[str, ...] = ("none", "remove", "trip", "debond")

MODE_DESCRIPTION: Dict[str, str] = {
    "none": "intact section (negative control, RIF must equal 1)",
    "remove": "component physically removed (plate-band tearing / stiffener torn away)",
    "trip": "stiffener tripping: stress capped at sigma_cr,trip, shear connection retained",
    "debond": "loss of composite action: bare-strut cap + plating support loss (2 x spacing)",
}

#: severity rank used for the ordering assertions (higher = more severe)
SEVERITY_RANK: Dict[str, int] = {"none": 0, "trip": 1, "debond": 2, "remove": 3}


@dataclass(frozen=True)
class DamageSpec:
    """A single-component damage state."""

    component_id: str
    mode: str

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"unknown damage mode {self.mode!r}; expected {MODES}")
        if not self.component_id:
            raise ValueError("component_id must be non-empty")

    @property
    def severity(self) -> int:
        return SEVERITY_RANK[self.mode]


@dataclass(frozen=True)
class DamagedPlate:
    id: str
    z: float
    area: float
    thickness: float
    span_b: float
    sigma_cr: float
    removed: bool = False
    support_lost: bool = False


@dataclass(frozen=True)
class DamagedStiffener:
    id: str
    z: float
    area: float
    h: float
    t_w: float
    cap: Optional[float]
    removed: bool = False
    plate_id: str = ""


@dataclass
class DamagedSection:
    """The section with one component (or none) degraded."""

    section: HullSection
    spec: DamageSpec
    plates: Tuple[DamagedPlate, ...]
    stiffeners: Tuple[DamagedStiffener, ...]
    notes: List[str] = field(default_factory=list)

    @property
    def material(self) -> VonMisesMaterial:
        return self.section.material

    def area(self) -> float:
        return float(
            sum(p.area for p in self.plates if not p.removed)
            + sum(s.area for s in self.stiffeners if not s.removed)
        )

    def first_moment(self) -> float:
        return float(
            sum(p.area * p.z for p in self.plates if not p.removed)
            + sum(s.area * s.z for s in self.stiffeners if not s.removed)
        )

    def elastic_neutral_axis(self) -> float:
        return self.first_moment() / self.area()

    def summary(self) -> Dict[str, object]:
        return {
            "component_id": self.spec.component_id,
            "mode": self.spec.mode,
            "severity": self.spec.severity,
            "area_m2": self.area(),
            "z_NA_elastic_m": self.elastic_neutral_axis(),
            "removed_components": [
                c.id
                for c in list(self.plates) + list(self.stiffeners)
                if getattr(c, "removed", False)
            ],
            "support_lost_plates": [p.id for p in self.plates if p.support_lost],
            "stress_caps_MPa": {
                s.id: (None if s.cap is None else s.cap / 1e6) for s in self.stiffeners
            },
            "notes": list(self.notes),
        }


def apply_damage(section: HullSection, spec: DamageSpec) -> DamagedSection:
    """Build the damaged section for ``spec``."""
    mat = section.material
    a_frame = section.frame_spacing
    if spec.component_id not in set(section.component_ids()):
        raise KeyError(f"unknown component {spec.component_id!r}")

    kind = section.component_kind(spec.component_id)
    notes: List[str] = []
    support_lost: set = set()
    removed_plates: set = set()
    removed_stiffeners: set = set()
    caps: Dict[str, float] = {}

    if spec.mode == "none":
        notes.append("intact section: negative control")
    elif kind == "plate":
        if spec.mode in ("remove", "debond", "trip"):
            # plate strips have a single tearing mode; the other modes are
            # mapped onto removal because a plate strip carries no lateral
            # stiffness of its own to lose.
            removed_plates.add(spec.component_id)
            notes.append(
                f"plate strip {spec.component_id}: plate-band tearing -> element removed"
            )
    else:  # stiffener group
        sg: StiffenerGroup = section.stiffener(spec.component_id)
        if spec.mode == "remove":
            removed_stiffeners.add(sg.id)
            support_lost.add(sg.plate_id)
            notes.append(
                f"stiffener {sg.id} removed; plating {sg.plate_id} loses its support line"
            )
        elif spec.mode == "trip":
            caps[sg.id] = tripping_stress(
                mat, h=sg.h, t_w=sg.t_w, span=a_frame, plate_t=sg.plate_t,
                n_long=sg.n_long,
            )
            notes.append(
                f"stiffener {sg.id} tripping: stress capped at "
                f"{caps[sg.id] / 1e6:.1f} MPa (shear connection retained)"
            )
        elif spec.mode == "debond":
            caps[sg.id] = bare_strut_stress(
                mat, h=sg.h, t_w=sg.t_w, span=a_frame
            )
            support_lost.add(sg.plate_id)
            notes.append(
                f"stiffener {sg.id} debonded: bare-strut cap "
                f"{caps[sg.id] / 1e6:.1f} MPa; plating {sg.plate_id} panel doubles"
            )

    plates: List[DamagedPlate] = []
    for p in section.plates:
        span_b = 2.0 * p.span_b if p.id in support_lost else p.span_b
        removed = p.id in removed_plates
        sigma_cr = effective_plate_buckling_stress(
            mat, b=span_b, t=p.thickness, a=a_frame
        )
        plates.append(
            DamagedPlate(
                id=p.id,
                z=p.z,
                area=0.0 if removed else p.area,
                thickness=p.thickness,
                span_b=span_b,
                sigma_cr=sigma_cr,
                removed=removed,
                support_lost=p.id in support_lost,
            )
        )

    stiffeners: List[DamagedStiffener] = []
    for s in section.stiffeners:
        removed = s.id in removed_stiffeners
        stiffeners.append(
            DamagedStiffener(
                id=s.id,
                z=s.z,
                area=0.0 if removed else s.area,
                h=s.h,
                t_w=s.t_w,
                cap=caps.get(s.id),
                removed=removed,
                plate_id=s.plate_id,
            )
        )

    return DamagedSection(
        section=section,
        spec=spec,
        plates=tuple(plates),
        stiffeners=tuple(stiffeners),
        notes=notes,
    )