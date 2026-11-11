"""Implementation-level FEM→DEM fracture-face handoff (``failure_faces`` v3).

The package turns the *document-level* ``failure_faces`` interface (a frozen JSON
packet that both solvers agreed on paper) into an executable pipeline:

``fem_export``        FEM fracture-face extraction → sealed ``failure_faces`` v3 packet
``dem_fragments``     DEM consumer: packet → polyhedral / clump fragments + internal energy
``conservation``      area (≤1 %), energy (≤2 %), mass and clump-volume gates
``dem_rigid``         rigid-fragment integrator closing the energy chain (free fall + impact)
``three_point_bend``  notched three-point-bend FEM driver used for the end-to-end case
``mixed_mode``        crack-tip K by displacement correlation, MTS kink angle, Williams
                      stress-field fit and element-deletion crack advance
``transfer``          mass / inertia / velocity inheritance and the volume, momentum and
                      energy budgets of the reconstructed fragments
``protocol``          the continuous-crack -> discrete-fragment protocol (C1-C4 + capability
                      record) on a mixed-mode ice three-point-bend specimen
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
from .mixed_mode import (
    EFFECTIVE_KAPPA_CASES,
    asymptotic_kink_angle_deg,
    asymptotic_sigma_theta,
    mts_kink_angle_deg,
    notch_face_node_pairs,
    propagate_crack,
    ring_kink_angle,
    senb_k1_reference,
    solve_step,
    stress_intensity_factors,
)
from .protocol import (
    ICE_DENSITY_KG_M3,
    ICE_MATERIAL,
    ICE_SPEC,
    ProtocolRun,
    gate_sensitivity,
    run_protocol,
    write_protocol_artifacts,
)
from .transfer import (
    TransferFragment,
    build_transfer_fragment,
    element_volume,
    element_volumes,
    energy_budget,
    fragment_inertia,
    momentum_budget,
    polyhedron_volume,
    polygon_second_moments,
    rigid_body_fit,
    volume_budget,
)
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
from .beam_fem import (
    BeamFractureSpec,
    BeamMesh,
    FractureEvent,
    ReleaseState,
    SIX_CASES,
    packet_from_fracture,
    release_analysis,
    release_sidecar,
    resolve_fracture,
)
from .beam_cases import (
    CaseRun,
    FragmentCloud,
    build_fragment_cloud,
    energy_gate,
    mass_gate,
    momentum_gate,
    mesh_convergence,
    negative_controls,
    run_case,
    run_sweep,
    schema_gate,
    size_consistency_gate,
)
from .three_point_bend import (
    BeamSpec,
    BendModel,
    build_bend_model,
    solve_bend,
    verify_against_beam_theory,
)

__all__ = [
    "BeamFractureSpec",
    "BeamMesh",
    "CaseRun",
    "FragmentCloud",
    "FractureEvent",
    "ReleaseState",
    "SIX_CASES",
    "build_fragment_cloud",
    "energy_gate",
    "mass_gate",
    "mesh_convergence",
    "momentum_gate",
    "negative_controls",
    "packet_from_fracture",
    "release_analysis",
    "release_sidecar",
    "resolve_fracture",
    "run_case",
    "run_sweep",
    "schema_gate",
    "size_consistency_gate",
    "EFFECTIVE_KAPPA_CASES",
    "ICE_DENSITY_KG_M3",
    "ICE_MATERIAL",
    "ICE_SPEC",
    "ProtocolRun",
    "gate_sensitivity",
    "TransferFragment",
    "asymptotic_kink_angle_deg",
    "asymptotic_sigma_theta",
    "build_transfer_fragment",
    "element_volume",
    "element_volumes",
    "energy_budget",
    "fragment_inertia",
    "momentum_budget",
    "mts_kink_angle_deg",
    "notch_face_node_pairs",
    "polyhedron_volume",
    "polygon_second_moments",
    "propagate_crack",
    "rigid_body_fit",
    "ring_kink_angle",
    "run_protocol",
    "senb_k1_reference",
    "solve_step",
    "stress_intensity_factors",
    "volume_budget",
    "write_protocol_artifacts",
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