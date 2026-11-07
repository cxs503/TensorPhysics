"""Campaign orchestration contracts for coupled ice-region campaigns.

This package is deliberately thin on numerics and thick on *contracts*: the
solver kernels live in the domain libraries (TensorLBM / TensorFVM / TensorFEM),
while this layer only fixes the semantics that a multi-library campaign needs to
be reproducible:

* :mod:`tensorphysics.campaign.clock` -- M12 master/sub clock ratio and exchange
  cadence on an exact integer time lattice.
* :mod:`tensorphysics.campaign.ledger` -- dual mass / energy / impulse books and
  their machine-precision reconciliation.
* :mod:`tensorphysics.campaign.geometry` -- ``geom_pc6`` consumer registration
  and the backward-compatible ``failure_faces`` interface extension.
* :mod:`tensorphysics.campaign.schema` / ``schema_defs`` -- closed artifact
  schemas with fingerprints that lock producer and consumer together.
"""
from .clock import (
    DISPOSITION,
    RATIO_RTOL,
    TIME_BASE,
    ClockContractError,
    CouplingClock,
    CouplingClockSpec,
    ExchangeEvent,
    clock_drift,
    expected_exchange_count,
    verify_alignment,
)
from .geometry import (
    CLOSURE_RTOL,
    FAILURE_MODES,
    SUPPORTED_FACE_VERSIONS,
    FailureFace,
    FailureFacePacket,
    FailureFaceV1,
    GeometryContractError,
    GeometryContractRegistry,
    GeometryHandle,
    PrincipalDimensions,
    tetrahedron_face_packet,
)
from .ledger import (
    DEFAULT_ATOL_ULPS,
    DIMENSION,
    EPS,
    UNITS,
    ComponentResidual,
    LedgerEntry,
    LedgerError,
    QuantityKind,
    QuantityReconciliation,
    ReferenceFluxes,
    SideLedger,
    build_reference_dual_ledgers,
    ledger_totals,
    reconcile,
    reconcile_books,
    verify_same_batch,
)
from .schema import (
    SchemaError,
    canonical_json,
    payload_digest,
    validate,
)
from .schema_defs import (
    ARTIFACT_SCHEMAS,
    artifact_bytes,
    schema_fingerprint,
    schema_fingerprints,
    schema_for,
    validate_artifact,
    validate_quantity_book,
)

__all__ = [
    "ARTIFACT_SCHEMAS",
    "CLOSURE_RTOL",
    "DEFAULT_ATOL_ULPS",
    "DIMENSION",
    "DISPOSITION",
    "EPS",
    "FAILURE_MODES",
    "RATIO_RTOL",
    "SUPPORTED_FACE_VERSIONS",
    "TIME_BASE",
    "UNITS",
    "ClockContractError",
    "ComponentResidual",
    "CouplingClock",
    "CouplingClockSpec",
    "ExchangeEvent",
    "FailureFace",
    "FailureFacePacket",
    "FailureFaceV1",
    "GeometryContractError",
    "GeometryContractRegistry",
    "GeometryHandle",
    "LedgerEntry",
    "LedgerError",
    "PrincipalDimensions",
    "QuantityKind",
    "QuantityReconciliation",
    "ReferenceFluxes",
    "SchemaError",
    "SideLedger",
    "artifact_bytes",
    "build_reference_dual_ledgers",
    "canonical_json",
    "clock_drift",
    "expected_exchange_count",
    "ledger_totals",
    "payload_digest",
    "reconcile",
    "reconcile_books",
    "schema_fingerprint",
    "schema_fingerprints",
    "schema_for",
    "validate",
    "validate_artifact",
    "tetrahedron_face_packet",
    "validate_quantity_book",
    "verify_alignment",
    "verify_same_batch",
]