"""Certificate store: JSON-backed persistence for reference-truth certificates.

The store manages a collection of certificates, supporting:
    - Add / update / remove certificates
    - Query by reference_id, type, status
    - Load / save to JSON file
    - Aggregate statistics
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator, Optional

from .schema import (
    CertificateStatus,
    CertificateType,
    ReferenceTruthCertificate,
    certificate_digest,
)


class CertificateStore:
    """In-memory store with JSON persistence for reference-truth certificates.

    Usage:
        store = CertificateStore()
        store.add(certificate)
        store.save_to_file("certificates.json")

        # Later:
        store2 = CertificateStore()
        store2.load_from_file("certificates.json")
        cert = store2.get("IR-L01")
    """

    def __init__(self):
        self._certificates: dict[str, ReferenceTruthCertificate] = {}

    def add(self, certificate: ReferenceTruthCertificate) -> None:
        """Add or update a certificate."""
        # Recompute digest to ensure consistency
        certificate.version_sha256 = certificate_digest(certificate)
        self._certificates[certificate.reference_id] = certificate

    def get(self, reference_id: str) -> Optional[ReferenceTruthCertificate]:
        """Get a certificate by reference_id."""
        return self._certificates.get(reference_id)

    def remove(self, reference_id: str) -> bool:
        """Remove a certificate. Returns True if it existed."""
        if reference_id in self._certificates:
            del self._certificates[reference_id]
            return True
        return False

    def list_all(self) -> list[ReferenceTruthCertificate]:
        """List all certificates."""
        return list(self._certificates.values())

    def list_by_type(self, cert_type: CertificateType) -> list[ReferenceTruthCertificate]:
        """List certificates of a specific type."""
        return [c for c in self._certificates.values() if c.certificate_type == cert_type]

    def list_by_status(self, status: CertificateStatus) -> list[ReferenceTruthCertificate]:
        """List certificates with a specific status."""
        return [c for c in self._certificates.values() if c.status == status]

    def __len__(self) -> int:
        return len(self._certificates)

    def __iter__(self) -> Iterator[ReferenceTruthCertificate]:
        return iter(self._certificates.values())

    def __contains__(self, reference_id: str) -> bool:
        return reference_id in self._certificates

    def save_to_file(self, path: str | Path) -> None:
        """Save all certificates to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "schema_version": "1.0.0",
            "certificate_count": len(self._certificates),
            "certificates": [c.to_dict() for c in self._certificates.values()],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_from_file(self, path: str | Path) -> int:
        """Load certificates from a JSON file. Returns count loaded."""
        path = Path(path)
        if not path.exists():
            return 0

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self._certificates.clear()
        for cert_dict in data.get("certificates", []):
            cert = ReferenceTruthCertificate.from_dict(cert_dict)
            self._certificates[cert.reference_id] = cert

        return len(self._certificates)

    def statistics(self) -> dict[str, Any]:
        """Compute aggregate statistics."""
        by_type: dict[str, int] = {}
        by_status: dict[str, int] = {}

        for cert in self._certificates.values():
            t = cert.certificate_type.value
            s = cert.status.value
            by_type[t] = by_type.get(t, 0) + 1
            by_status[s] = by_status.get(s, 0) + 1

        return {
            "total": len(self._certificates),
            "by_type": by_type,
            "by_status": by_status,
            "valid_rate": by_status.get("valid", 0) / max(len(self._certificates), 1),
        }

    def revoke(self, reference_id: str, reason: str) -> bool:
        """Revoke a certificate. Returns True if it existed."""
        cert = self._certificates.get(reference_id)
        if cert is None:
            return False
        cert.status = CertificateStatus.REVOKED
        cert.validation_notes.append(f"REVOKED: {reason}")
        return True

    def expire(self, reference_id: str, reason: str = "") -> bool:
        """Mark a certificate as expired. Returns True if it existed."""
        cert = self._certificates.get(reference_id)
        if cert is None:
            return False
        cert.status = CertificateStatus.EXPIRED
        if reason:
            cert.validation_notes.append(f"EXPIRED: {reason}")
        return True
