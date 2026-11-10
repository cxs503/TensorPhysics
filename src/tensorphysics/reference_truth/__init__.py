"""Reference-Truth Certificate system for TensorPhysics.

This module provides a convergence-certification framework for reference values
used as acceptance criteria in campaign nodes.  Un-certified references are
barred from serving as downstream verification baselines.

Certificate types:
    - analytical:     exact closed-form solution (symbolic equivalence check)
    - self_convergence: mesh/time refinement study (Richardson extrapolation, GCI)
    - experimental:   measured data with stated uncertainty
    - statistical:    Monte-Carlo / sampling estimate (SEM + CI + power)

Each certificate carries:
    - type, evidence class, refinement sequence
    - Richardson observed order p, GCI (if applicable)
    - MC standard error & confidence interval (if applicable)
    - caliber declaration (what the reference is valid for)
    - version sha256 (content-addressable)
    - disclaimer boundaries (where it must NOT be used)

The reference gate checks certificate status before any criterion that cites
a reference_id can be used as an acceptance baseline.
"""
from .schema import (
    CertificateType,
    CertificateStatus,
    ReferenceTruthCertificate,
    build_certificate,
    certificate_digest,
    CERTIFICATE_SCHEMA_VERSION,
)
from .validator import validate_certificate, ValidationResult
from .gate import ReferenceGate, GateDecision
from .store import CertificateStore

__all__ = [
    "CertificateType",
    "CertificateStatus",
    "ReferenceTruthCertificate",
    "build_certificate",
    "certificate_digest",
    "CERTIFICATE_SCHEMA_VERSION",
    "validate_certificate",
    "ValidationResult",
    "ReferenceGate",
    "GateDecision",
    "CertificateStore",
]
