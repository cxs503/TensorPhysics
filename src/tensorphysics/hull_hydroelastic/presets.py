"""Reference hull forms used by the springing / whipping node (KCS, KVLCC2).

Both hulls are the standard public benchmark geometries (KRISO container ship
and KRISO very large crude carrier) reduced to the principal particulars needed
by the hull-girder idealisation of :class:`~tensorphysics.hullgirder.model.ShipGirder`.

Section modulus
---------------
The midship elastic section modulus is **not** published with the benchmark
particulars (it is a structural-design output), so it is taken from the IACS UR
S7 rule minimum

    S_M = C L^2 B (C_b + 0.7) kappa x 1e-6   [m^3]
    C   = 10.75 ((300 - L)/100)^1.5 for 90 < L < 300, else 10.75

with ``kappa = 1`` (mild steel).  Real scantlings are usually 5-25 % above the
rule minimum; the utilisation gate of this node is *proportional* and therefore
insensitive to that offset, but any absolute utilisation number inherits the
rule-minimum assumption.  Evidence grade for ``S_M``: RULE ESTIMATE.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..hullgirder.model import ShipGirder

__all__ = ["HullCase", "KCS", "KVLCC2", "rule_section_modulus", "hull_cases"]

YOUNG_STEEL = 2.06e11


def rule_section_modulus(length: float, breadth: float, block: float, kappa: float = 1.0) -> float:
    """IACS UR S7 minimum midship section modulus (m^3)."""
    if 90.0 < length < 300.0:
        c = 10.75 * ((300.0 - length) / 100.0) ** 1.5
    else:
        c = 10.75
    return float(c * length ** 2 * breadth * (block + 0.7) * kappa * 1.0e-6)


@dataclass(frozen=True)
class HullCase:
    """Benchmark hull plus the parameters this node needs beyond the hull form."""

    name: str
    length: float
    breadth: float
    draught: float
    depth: float
    block: float
    design_speed_kn: float
    section_exponent: float = 4.0
    yield_stress: float = 315.0e6
    plate_slenderness: float = 1.6
    n_elements: int = 120

    @property
    def section_modulus(self) -> float:
        return rule_section_modulus(self.length, self.breadth, self.block)

    @property
    def froude_number(self) -> float:
        v = self.design_speed_kn * 0.514444
        return float(v / (9.81 * self.length) ** 0.5)

    def girder(self) -> ShipGirder:
        return ShipGirder(
            length=self.length,
            breadth=self.breadth,
            draught=self.draught,
            block_coefficient=self.block,
            depth=self.depth,
            section_modulus=self.section_modulus,
            yield_stress=self.yield_stress,
            plate_slenderness=self.plate_slenderness,
            n_nodes=801,
        )

    def as_dict(self) -> dict:
        g = self.girder()
        return {
            "name": self.name,
            "length_m": self.length,
            "breadth_m": self.breadth,
            "draught_m": self.draught,
            "depth_m": self.depth,
            "block_coefficient": self.block,
            "displacement_t": g.displacement_mass / 1000.0,
            "waterplane_coefficient": g.waterplane_coefficient,
            "section_modulus_m3_rule": self.section_modulus,
            "design_speed_kn": self.design_speed_kn,
            "froude_number_design": self.froude_number,
            "yield_stress_MPa": self.yield_stress / 1e6,
            "section_exponent": self.section_exponent,
        }


KCS = HullCase(
    name="KCS",
    length=230.0,
    breadth=32.2,
    draught=10.8,
    depth=19.0,
    block=0.651,
    design_speed_kn=24.0,
    section_exponent=4.0,
    yield_stress=315.0e6,
)

KVLCC2 = HullCase(
    name="KVLCC2",
    length=320.0,
    breadth=58.0,
    draught=20.8,
    depth=30.0,
    block=0.8098,
    design_speed_kn=15.5,
    section_exponent=6.0,
    yield_stress=315.0e6,
)


def hull_cases() -> tuple[HullCase, ...]:
    return (KCS, KVLCC2)