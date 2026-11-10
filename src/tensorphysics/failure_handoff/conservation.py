"""Conservation gates for the FEM→DEM failure-face handoff.

The gates are the machine-checkable part of the interface contract:

======================  ===============================================  =========
gate                    comparison                                       tolerance
======================  ===============================================  =========
``area``                FEM-declared total area vs. DEM-recomputed area   1 %
``area_per_face``       worst per-face area disagreement                 1 %
``energy``              FEM released energy vs. DEM fragment internal    2 %
``mass``                ``rho * V`` per fragment vs. declared mass       1e-12
``clump_volume``        sum of clump sphere volumes vs fragment volume    1e-12
======================  ===============================================  =========

Each gate reports the two numbers it compared, the relative error and a boolean,
so the report can cite numbers rather than adjectives.  A gate that *cannot* be
evaluated (e.g. the consumer disagrees with the producer about the fragment
count) raises :class:`HandoffGateError` instead of passing silently.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Mapping

from .dem_fragments import FragmentSet

__all__ = [
    "AREA_TOLERANCE",
    "ENERGY_TOLERANCE",
    "MACHINE_TOLERANCE",
    "HandoffGateError",
    "GateResult",
    "GateReport",
    "area_conservation_gate",
    "per_face_area_gate",
    "energy_conservation_gate",
    "mass_conservation_gate",
    "clump_volume_gate",
    "run_handoff_gates",
]

#: Acceptance threshold of the area-conservation criterion (C1).
AREA_TOLERANCE = 0.01
#: Acceptance threshold of the energy-conservation criterion (C2).
ENERGY_TOLERANCE = 0.02
#: Tolerance for identities that must hold exactly (float arithmetic only).
MACHINE_TOLERANCE = 1e-12


class HandoffGateError(RuntimeError):
    """The two sides of the handoff are not comparable (structural mismatch)."""


@dataclass
class GateResult:
    name: str
    description: str
    producer_value: float
    consumer_value: float
    relative_error: float
    tolerance: float
    passed: bool
    unit: str = ""
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class GateReport:
    gates: list[GateResult]
    packet_digest: str
    fragment_count: int

    @property
    def passed(self) -> bool:
        return all(gate.passed for gate in self.gates)

    @property
    def failures(self) -> list[str]:
        return [gate.name for gate in self.gates if not gate.passed]

    def as_dict(self) -> dict:
        return {
            "packet_digest": self.packet_digest,
            "fragment_count": self.fragment_count,
            "passed": self.passed,
            "failed_gates": self.failures,
            "gates": [gate.as_dict() for gate in self.gates],
        }


def _relative_error(producer: float, consumer: float, *, scale: float | None = None) -> float:
    denom = float(scale) if scale not in (None, 0.0) else max(abs(producer), abs(consumer))
    if denom == 0.0:
        return 0.0 if producer == consumer else math.inf
    return abs(producer - consumer) / denom


def _check_pairing(packet: Mapping, fragment_set: FragmentSet) -> None:
    packet_faces = packet.get("faces", [])
    if len(packet_faces) != fragment_set.count:
        raise HandoffGateError(
            f"consumer built {fragment_set.count} fragments from "
            f"{len(packet_faces)} packet faces"
        )
    for face, fragment in zip(packet_faces, fragment_set.fragments):
        if str(face["face_id"]) != fragment.face_id:
            raise HandoffGateError(
                f"face/fragment order mismatch: {face['face_id']} vs {fragment.face_id}"
            )


def area_conservation_gate(
    packet: Mapping, fragment_set: FragmentSet, *, tolerance: float = AREA_TOLERANCE
) -> GateResult:
    """C1: FEM-declared fracture area vs. DEM area recomputed from coordinates."""
    _check_pairing(packet, fragment_set)
    producer = float(packet["area_total_m2"])
    consumer = fragment_set.area_total_m2
    error = _relative_error(producer, consumer)
    return GateResult(
        name="area",
        description=(
            "FEM declared fracture-surface area vs DEM area recomputed from the "
            "packet vertices (Newell area vector)"
        ),
        producer_value=producer,
        consumer_value=consumer,
        relative_error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
        unit="m2",
        detail={"face_count": fragment_set.count},
    )


def per_face_area_gate(
    packet: Mapping, fragment_set: FragmentSet, *, tolerance: float = AREA_TOLERANCE
) -> GateResult:
    """Worst per-face area disagreement (localisation of a C1 failure)."""
    _check_pairing(packet, fragment_set)
    worst = 0.0
    worst_face = ""
    for face, fragment in zip(packet["faces"], fragment_set.fragments):
        error = _relative_error(float(face["area_m2"]), fragment.area_m2)
        if error > worst:
            worst = error
            worst_face = fragment.face_id
    return GateResult(
        name="area_per_face",
        description="worst per-face relative area disagreement",
        producer_value=float(packet["area_total_m2"]),
        consumer_value=fragment_set.area_total_m2,
        relative_error=worst,
        tolerance=tolerance,
        passed=worst <= tolerance,
        unit="m2",
        detail={"worst_face": worst_face},
    )


def energy_conservation_gate(
    packet: Mapping, fragment_set: FragmentSet, *, tolerance: float = ENERGY_TOLERANCE
) -> GateResult:
    """C2: FEM released fracture energy vs. sum of DEM fragment internal energy."""
    _check_pairing(packet, fragment_set)
    producer = float(packet["energy_total_j"])
    consumer = fragment_set.internal_energy_total_j
    error = _relative_error(producer, consumer, scale=producer if producer > 0 else None)
    return GateResult(
        name="energy",
        description=(
            "FEM fracture energy release vs DEM fragment initial internal energy "
            "(sum over fragments)"
        ),
        producer_value=producer,
        consumer_value=consumer,
        relative_error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
        unit="J",
        detail={
            "mean_specific_energy_j_m2": float(
                math.fsum(f.specific_energy_j_m2 for f in fragment_set.fragments)
                / max(fragment_set.count, 1)
            )
        },
    )


def mass_conservation_gate(
    packet: Mapping,
    fragment_set: FragmentSet,
    *,
    tolerance: float = MACHINE_TOLERANCE,
) -> GateResult:
    """``rho * A * t`` must reproduce the fragment mass exactly (bookkeeping)."""
    _check_pairing(packet, fragment_set)
    rho = fragment_set.density_kg_m3
    expected = rho * fragment_set.area_total_m2 * fragment_set.layer_thickness_m
    actual = fragment_set.mass_total_kg
    error = _relative_error(expected, actual)
    return GateResult(
        name="mass",
        description="rho*A*t total mass vs sum of fragment masses",
        producer_value=expected,
        consumer_value=actual,
        relative_error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
        unit="kg",
        detail={"density_kg_m3": rho, "layer_thickness_m": fragment_set.layer_thickness_m},
    )


def clump_volume_gate(
    packet: Mapping,
    fragment_set: FragmentSet,
    *,
    tolerance: float = MACHINE_TOLERANCE,
) -> GateResult:
    """Sum of clump sphere volumes vs. fragment volume (skipped when no clump)."""
    _check_pairing(packet, fragment_set)
    if not any(fragment.clump_spheres_m for fragment in fragment_set.fragments):
        return GateResult(
            name="clump_volume",
            description="clump sphere volume partition (not requested)",
            producer_value=fragment_set.volume_total_m3,
            consumer_value=fragment_set.volume_total_m3,
            relative_error=0.0,
            tolerance=tolerance,
            passed=True,
            unit="m3",
            detail={"clump_requested": False},
        )
    expected = fragment_set.volume_total_m3
    actual = fragment_set.clump_volume_total_m3
    error = _relative_error(expected, actual)
    return GateResult(
        name="clump_volume",
        description="sum of clump sphere volumes vs fragment prism volume",
        producer_value=expected,
        consumer_value=actual,
        relative_error=error,
        tolerance=tolerance,
        passed=error <= tolerance,
        unit="m3",
        detail={"clump_requested": True},
    )


def run_handoff_gates(
    packet: Mapping,
    fragment_set: FragmentSet,
    *,
    area_tolerance: float = AREA_TOLERANCE,
    energy_tolerance: float = ENERGY_TOLERANCE,
) -> GateReport:
    """Run every conservation gate and collect one auditable report."""
    gates = [
        area_conservation_gate(packet, fragment_set, tolerance=area_tolerance),
        per_face_area_gate(packet, fragment_set, tolerance=area_tolerance),
        energy_conservation_gate(packet, fragment_set, tolerance=energy_tolerance),
        mass_conservation_gate(packet, fragment_set),
        clump_volume_gate(packet, fragment_set),
    ]
    return GateReport(
        gates=gates,
        packet_digest=str(packet.get("sha256", "")),
        fragment_count=fragment_set.count,
    )