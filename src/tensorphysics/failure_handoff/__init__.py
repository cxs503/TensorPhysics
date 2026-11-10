"""Implementation-level FEM→DEM fracture-face handoff (``failure_faces`` v3).

The package turns the *document-level* ``failure_faces`` interface (a frozen JSON
packet that both solvers agreed on paper) into an executable pipeline:

``fem_export``        FEM fracture-face extraction → sealed ``failure_faces`` v3 packet
``dem_fragments``     DEM consumer: packet → polyhedral / clump fragments + internal energy
``conservation``      area (≤1 %), energy (≤2 %), mass and clump-volume gates
``dem_rigid``         rigid-fragment integrator closing the energy chain (free fall + impact)
``three_point_bend``  notched three-point-bend FEM driver used for the end-to-end case
``pipeline``          the end-to-end run that produces the audited artifacts
``schema``            frozen JSON Schema, canonical digest, seal and version upgrades
``geometry``          polygon primitives shared by both sides (independent recomputation)

Entry points: :func:`run_handoff`, :func:`packet_from_result`,
:func:`fragments_from_packet`, :func:`run_handoff_gates`, :func:`RigidDEM`,
:func:`validate_packet`.
"""

from __future__ import annotations

from .conservation import (
    AREA_TOLERANCE,
    ENERGY_TOLERANCE,
    GateReport,
    GateResult,
    HandoffGateError,
    area_conservation_gate,
    clump_volume_gate,
    energy_conservation_gate,
    mass_conservation_gate,
    per_face_area_gate,
    run_handoff_gates,
)
from .dem_fragments import Fragment, FragmentSet, fragment_from_face, fragments_from_packet
from .dem_rigid import EnergyHistory, RigidDEM, RigidFragmentState
from .fem_export import (
    FaceRecord,
    FEMFractureResult,
    FractureMaterial,
    element_faces,
    extract_failure_faces,
    faces_on_plane,
    packet_from_result,
    packet_to_json_text,
    write_packet,
)
from .geometry import polygon_area, polygon_centroid
from .pipeline import run_handoff
from .schema import (
    FAILURE_FACES_SCHEMA,
    SCHEMA_FINGERPRINT,
    SCHEMA_VERSION,
    SUPPORTED_VERSIONS,
    VERSION,
    FailureFaceSchemaError,
    packet_digest,
    seal,
    upgrade_legacy_packet,
    validate_packet,
    verify_seal,
)
from .three_point_bend import (
    BeamSpec,
    BendModel,
    build_bend_model,
    solve_bend,
    verify_against_beam_theory,
)

__all__ = [
    "AREA_TOLERANCE",
    "ENERGY_TOLERANCE",
    "BeamSpec",
    "BendModel",
    "EnergyHistory",
    "FAILURE_FACES_SCHEMA",
    "FEMFractureResult",
    "FaceRecord",
    "FailureFaceSchemaError",
    "Fragment",
    "FragmentSet",
    "FractureMaterial",
    "GateReport",
    "GateResult",
    "HandoffGateError",
    "RigidDEM",
    "RigidFragmentState",
    "SCHEMA_FINGERPRINT",
    "SCHEMA_VERSION",
    "SUPPORTED_VERSIONS",
    "VERSION",
    "area_conservation_gate",
    "build_bend_model",
    "clump_volume_gate",
    "element_faces",
    "energy_conservation_gate",
    "extract_failure_faces",
    "faces_on_plane",
    "fragment_from_face",
    "fragments_from_packet",
    "mass_conservation_gate",
    "packet_digest",
    "packet_from_result",
    "packet_to_json_text",
    "per_face_area_gate",
    "polygon_area",
    "polygon_centroid",
    "run_handoff",
    "run_handoff_gates",
    "seal",
    "solve_bend",
    "upgrade_legacy_packet",
    "validate_packet",
    "verify_against_beam_theory",
    "verify_seal",
    "write_packet",
]