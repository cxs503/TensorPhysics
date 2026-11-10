"""Reference gate: check certificate status before citing references.

The gate enforces the rule: un-certified references cannot be used as
downstream acceptance criteria.  When a task criterion cites a reference_id,
the gate checks:
    1. Does a certificate exist for that reference_id?
    2. Is the certificate status VALID?
    3. Has the certificate been revoked or expired?

If any check fails, the gate returns a blocking decision with a clear reason.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

from .schema import CertificateStatus, ReferenceTruthCertificate
from .store import CertificateStore


class GateDecision(str, Enum):
    """Outcome of a reference gate check."""
    ALLOWED = "allowed"           # certificate is valid, reference may be cited
    BLOCKED_NO_CERT = "blocked_no_certificate"  # no certificate exists
    BLOCKED_INVALID = "blocked_invalid_status"  # certificate exists but not VALID
    BLOCKED_REVOKED = "blocked_revoked"         # certificate was revoked
    BLOCKED_EXPIRED = "blocked_expired"         # certificate has expired


@dataclass
class GateResult:
    """Detailed result of a gate check."""
    decision: GateDecision
    reference_id: str
    reason: str
    certificate: Optional[ReferenceTruthCertificate] = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision.value,
            "reference_id": self.reference_id,
            "reason": self.reason,
            "certificate_status": self.certificate.status.value if self.certificate else None,
            "details": self.details,
        }


class ReferenceGate:
    """Gate that enforces certificate requirements for reference citations.

    Usage:
        store = CertificateStore()
        store.load_from_file("certificates.json")
        gate = ReferenceGate(store)

        # Before using a reference as acceptance criterion:
        result = gate.check("IR-L01")
        if result.decision != GateDecision.ALLOWED:
            raise RuntimeError(f"Cannot cite {result.reference_id}: {result.reason}")
    """

    def __init__(self, store: CertificateStore):
        self.store = store

    def check(self, reference_id: str) -> GateResult:
        """Check if a reference may be cited as an acceptance criterion.

        Returns a GateResult with the decision and reason.
        """
        cert = self.store.get(reference_id)

        if cert is None:
            return GateResult(
                decision=GateDecision.BLOCKED_NO_CERT,
                reference_id=reference_id,
                reason=f"No certificate found for reference_id '{reference_id}'. "
                       f"Un-certified references cannot be used as acceptance criteria.",
            )

        if cert.status == CertificateStatus.REVOKED:
            return GateResult(
                decision=GateDecision.BLOCKED_REVOKED,
                reference_id=reference_id,
                reason=f"Certificate for '{reference_id}' has been revoked. "
                       f"Reason: {cert.validation_notes}",
                certificate=cert,
            )

        if cert.status == CertificateStatus.EXPIRED:
            return GateResult(
                decision=GateDecision.BLOCKED_EXPIRED,
                reference_id=reference_id,
                reason=f"Certificate for '{reference_id}' has expired. "
                       f"Certified at: {cert.certification_timestamp}",
                certificate=cert,
            )

        if cert.status != CertificateStatus.VALID:
            return GateResult(
                decision=GateDecision.BLOCKED_INVALID,
                reference_id=reference_id,
                reason=f"Certificate for '{reference_id}' has status '{cert.status.value}', "
                       f"not VALID. Validation notes: {cert.validation_notes}",
                certificate=cert,
            )

        # All checks passed
        return GateResult(
            decision=GateDecision.ALLOWED,
            reference_id=reference_id,
            reason=f"Certificate for '{reference_id}' is VALID. "
                   f"Type: {cert.certificate_type.value}, "
                   f"sha256: {cert.version_sha256[:16]}...",
            certificate=cert,
            details={
                "certificate_type": cert.certificate_type.value,
                "reference_value": cert.reference_value,
                "unit": cert.unit,
                "caliber_declaration": cert.caliber_declaration,
                "disclaimer_boundaries": cert.disclaimer_boundaries,
                "version_sha256": cert.version_sha256,
            },
        )

    def check_multiple(self, reference_ids: list[str]) -> list[GateResult]:
        """Check multiple references at once."""
        return [self.check(rid) for rid in reference_ids]

    def all_allowed(self, reference_ids: list[str]) -> tuple[bool, list[GateResult]]:
        """Check if all references are allowed. Returns (all_ok, results)."""
        results = self.check_multiple(reference_ids)
        all_ok = all(r.decision == GateDecision.ALLOWED for r in results)
        return all_ok, results
