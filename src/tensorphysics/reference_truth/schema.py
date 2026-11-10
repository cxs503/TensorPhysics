"""Reference-Truth Certificate schema definition.

Defines the data structure for convergence certificates that vouch for
reference values used as acceptance criteria in campaign nodes.

Schema version: 1.0.0
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


CERTIFICATE_SCHEMA_VERSION = "1.0.0"


class CertificateType(str, Enum):
    """Four evidence classes for reference truth."""
    ANALYTICAL = "analytical"
    SELF_CONVERGENCE = "self_convergence"
    EXPERIMENTAL = "experimental"
    STATISTICAL = "statistical"


class CertificateStatus(str, Enum):
    """Lifecycle states of a certificate."""
    DRAFT = "draft"
    VALID = "valid"
    EXPIRED = "expired"
    REVOKED = "revoked"
    INSUFFICIENT = "insufficient"  # evidence does not meet minimum requirements


@dataclass(frozen=True)
class RefinementLevel:
    """One level in a mesh/time refinement sequence."""
    level_index: int          # 0=coarsest
    resolution_parameter: float  # h, dt, N, etc.
    quantity_of_interest: float  # computed value at this level
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RichardsonExtrapolation:
    """Richardson extrapolation results from a refinement sequence."""
    observed_order_p: float   # observed convergence order
    extrapolated_value: float  # h→0 (or dt→0) estimate
    gci_fine: float           # Grid Convergence Index on finest pair
    gci_coarse: float         # GCI on coarse pair (for safety factor check)
    safety_factor: float = 1.25  # Roache safety factor (1.25 for 3+ grids)
    asymptotic_range: bool = True  # is p close to theoretical?


@dataclass(frozen=True)
class StatisticalEstimate:
    """Monte-Carlo / sampling estimate metadata."""
    n_samples: int
    mean: float
    standard_error: float     # SEM = sigma / sqrt(N)
    confidence_level: float   # e.g. 0.95
    ci_lower: float
    ci_upper: float
    effective_sample_size: Optional[int] = None  # for correlated samples
    power_analysis: Optional[dict[str, float]] = None  # effect size, power, alpha


@dataclass(frozen=True)
class ExperimentalEvidence:
    """Experimental measurement metadata."""
    source_reference: str     # paper/report citation
    measurement_method: str
    stated_uncertainty: float  # e.g. ±2% at 95% confidence
    confidence_level: float = 0.95
    reynolds_number_range: Optional[tuple[float, float]] = None
    facility: Optional[str] = None


@dataclass
class ReferenceTruthCertificate:
    """A convergence certificate for a reference value.

    This is the central artifact: it vouches that a reference value is
    trustworthy to a stated precision, and declares the boundaries within
    which it may be cited as a verification baseline.
    """
    # Identity
    reference_id: str                    # unique identifier (e.g. "IR-L01")
    reference_name: str                  # human-readable name
    schema_version: str = CERTIFICATE_SCHEMA_VERSION

    # Classification
    certificate_type: CertificateType = CertificateType.ANALYTICAL
    status: CertificateStatus = CertificateStatus.DRAFT

    # Core evidence
    reference_value: float = 0.0         # the reference number being vouched
    unit: str = ""                       # physical unit
    quantity_description: str = ""       # what physical quantity

    # Type-specific evidence (only one should be populated)
    analytical_evidence: Optional[dict[str, Any]] = None
    refinement_sequence: Optional[list[dict[str, Any]]] = None
    richardson: Optional[dict[str, Any]] = None
    statistical: Optional[dict[str, Any]] = None
    experimental: Optional[dict[str, Any]] = None

    # Caliber & boundaries
    caliber_declaration: str = ""        # what this reference is valid for
    disclaimer_boundaries: list[str] = field(default_factory=list)  # where NOT to use

    # Provenance
    source_citation: str = ""            # literature source
    version_sha256: str = ""             # content hash of the evidence payload
    certified_by: str = ""               # mission/agent that issued this
    certification_timestamp: str = ""    # ISO-8601

    # Validation outcome
    validation_notes: list[str] = field(default_factory=list)
    minimum_requirements_met: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        d = asdict(self)
        d["certificate_type"] = self.certificate_type.value
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ReferenceTruthCertificate":
        """Deserialize from a dictionary."""
        d = dict(d)
        d["certificate_type"] = CertificateType(d.get("certificate_type", "analytical"))
        d["status"] = CertificateStatus(d.get("status", "draft"))
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def certificate_digest(certificate: ReferenceTruthCertificate) -> str:
    """Compute SHA-256 digest of the certificate's evidence payload.

    This hashes the type-specific evidence fields to create a content-addressable
    fingerprint. If the underlying data changes, the digest changes, invalidating
    any downstream citations.
    """
    payload = {
        "reference_id": certificate.reference_id,
        "certificate_type": certificate.certificate_type.value,
        "reference_value": certificate.reference_value,
        "analytical_evidence": certificate.analytical_evidence,
        "refinement_sequence": certificate.refinement_sequence,
        "richardson": certificate.richardson,
        "statistical": certificate.statistical,
        "experimental": certificate.experimental,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def build_certificate(
    reference_id: str,
    reference_name: str,
    certificate_type: CertificateType,
    reference_value: float,
    unit: str = "",
    quantity_description: str = "",
    source_citation: str = "",
    caliber_declaration: str = "",
    disclaimer_boundaries: Optional[list[str]] = None,
    # Type-specific
    analytical_evidence: Optional[dict[str, Any]] = None,
    refinement_sequence: Optional[list[dict[str, Any]]] = None,
    richardson: Optional[dict[str, Any]] = None,
    statistical: Optional[dict[str, Any]] = None,
    experimental: Optional[dict[str, Any]] = None,
    # Metadata
    certified_by: str = "",
    certification_timestamp: str = "",
) -> ReferenceTruthCertificate:
    """Factory function to build a certificate with automatic digest computation."""
    cert = ReferenceTruthCertificate(
        reference_id=reference_id,
        reference_name=reference_name,
        certificate_type=certificate_type,
        reference_value=reference_value,
        unit=unit,
        quantity_description=quantity_description,
        analytical_evidence=analytical_evidence,
        refinement_sequence=refinement_sequence,
        richardson=richardson,
        statistical=statistical,
        experimental=experimental,
        source_citation=source_citation,
        caliber_declaration=caliber_declaration,
        disclaimer_boundaries=disclaimer_boundaries or [],
        certified_by=certified_by,
        certification_timestamp=certification_timestamp,
    )
    cert.version_sha256 = certificate_digest(cert)
    return cert
