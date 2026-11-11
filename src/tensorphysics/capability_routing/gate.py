"""Evidence gate: the fail-closed check placed in front of every route.

The gate answers exactly one question -- "may this implementation be executed
for this capability, given the grade on file and the admissibility policy?" --
and it answers it from data only.  Two structural properties matter:

1. **No default-allow path.**  An unknown capability, an unknown solver, or an
   unparseable grade is a *rejection*, not a fallback to "carry on".
2. **No silent placeholder.**  ``MOCK``/``STUB`` output is refused unless the
   caller opted in through a ``development``-tier policy with ``allow_mock``;
   ``NOT_SUPPORTED`` is refused in every tier, so a missing capability raises
   :class:`RoutingRejected` instead of returning fabricated numbers.

Every admission carries the reason code and the grade, so the routing layer can
record *why* a solver was chosen (audit chain requirement).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .evidence import AdmissibilityPolicy, EvidenceLevel, UnknownEvidenceLevel
from .ledger import CapabilityLedger, SolverRecord

__all__ = [
    "REASON_CODES",
    "Admission",
    "RoutingRejected",
    "EvidenceGate",
]

#: Closed vocabulary of admission / rejection reasons.
REASON_CODES: dict[str, str] = {
    "ADMITTED": "grade is inside the policy's admissible set",
    "CAPABILITY_UNKNOWN": "capability has no row in the evidence ledger",
    "SOLVER_UNKNOWN": "solver has no grade on file for this capability",
    "EVIDENCE_NOT_SUPPORTED": "capability is graded NOT_SUPPORTED: refusing instead of fabricating",
    "EVIDENCE_FABRICATED_NO_CONSENT": "grade is MOCK/STUB and the policy did not opt in",
    "EVIDENCE_BELOW_POLICY": "grade is outside the tier/rank floor of the policy",
    "GRADE_UNPARSEABLE": "grade string outside the closed vocabulary: treated as not admissible",
    "SOLVER_UNAVAILABLE": "solver is declared unavailable (ledger flag or runtime)",
    "NO_ADMISSIBLE_SOLVER": "every candidate of the capability was rejected by the gate",
    "CAPABILITY_NOT_IN_LEDGER_CANDIDATES": "capability exists but has no candidate left after filtering",
}


class RoutingRejected(RuntimeError):
    """Raised instead of silently degrading to a placeholder implementation."""

    def __init__(self, code: str, message: str, *, capability_id: str,
                 solver_id: "str | None" = None,
                 level: "EvidenceLevel | str | None" = None,
                 examined: Iterable = ()) -> None:
        if code not in REASON_CODES:
            raise ValueError(f"unknown rejection code {code!r}")
        super().__init__(message)
        self.code = code
        self.capability_id = capability_id
        self.solver_id = solver_id
        self.level = level.value if isinstance(level, EvidenceLevel) else level
        self.examined = tuple(examined)

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "capability_id": self.capability_id,
            "solver_id": self.solver_id,
            "level": self.level,
            "message": str(self),
            "examined": list(self.examined),
        }


@dataclass(frozen=True)
class Admission:
    """Outcome of one gate check (never raises; ``require`` raises)."""

    capability_id: str
    solver_id: str
    admitted: bool
    code: str
    level: "EvidenceLevel | None" = None
    provisional: bool = False
    mock_consumed: bool = False
    record: "SolverRecord | None" = field(default=None, repr=False)

    def as_dict(self) -> dict:
        return {
            "capability_id": self.capability_id,
            "solver_id": self.solver_id,
            "admitted": self.admitted,
            "code": self.code,
            "level": self.level.value if self.level else None,
            "provisional": self.provisional,
            "mock_consumed": self.mock_consumed,
        }


class EvidenceGate:
    """Fail-closed admissibility check backed by a capability ledger."""

    def __init__(self, ledger: CapabilityLedger,
                 policy: "AdmissibilityPolicy | None" = None) -> None:
        if not isinstance(ledger, CapabilityLedger):
            raise TypeError("ledger must be a CapabilityLedger")
        self.ledger = ledger
        self.policy = policy or AdmissibilityPolicy()

    # ------------------------------------------------------------------ core
    def admit(self, capability_id: str, solver_id: str,
              unavailable: Iterable[str] = ()) -> Admission:
        """Grade check without raising.  Absence of evidence is a rejection."""
        excluded = {unavailable} if isinstance(unavailable, str) else set(unavailable)
        record = self.ledger.record(capability_id, solver_id)
        if record is None:
            code = ("CAPABILITY_UNKNOWN" if not self.ledger.candidates(capability_id)
                    else "SOLVER_UNKNOWN")
            return Admission(capability_id, solver_id, False, code)
        level = record.level
        try:
            level = EvidenceLevel.parse(level)
        except UnknownEvidenceLevel:
            return Admission(capability_id, solver_id, False, "GRADE_UNPARSEABLE", record=record)
        if level is EvidenceLevel.NOT_SUPPORTED:
            return Admission(capability_id, solver_id, False, "EVIDENCE_NOT_SUPPORTED",
                             level=level, record=record)
        if not self.policy.admits(level):
            code = ("EVIDENCE_FABRICATED_NO_CONSENT" if level.fabricated
                    else "EVIDENCE_BELOW_POLICY")
            return Admission(capability_id, solver_id, False, code, level=level, record=record)
        if not record.available or solver_id in excluded:
            return Admission(capability_id, solver_id, False, "SOLVER_UNAVAILABLE",
                             level=level, record=record)
        return Admission(
            capability_id, solver_id, True, "ADMITTED",
            level=level,
            provisional=not level.verified,
            mock_consumed=level.fabricated,
            record=record,
        )

    def require(self, capability_id: str, solver_id: str,
                unavailable: Iterable[str] = ()) -> Admission:
        """Grade check that raises :class:`RoutingRejected` when not admitted."""
        admission = self.admit(capability_id, solver_id, unavailable)
        if admission.admitted:
            return admission
        raise RoutingRejected(
            admission.code,
            self._message(admission),
            capability_id=capability_id,
            solver_id=solver_id,
            level=admission.level,
        )

    def _message(self, admission: Admission) -> str:
        reason = REASON_CODES[admission.code]
        shown = admission.level.value if admission.level else "no grade"
        detail = ""
        candidates = self.ledger.candidates(admission.capability_id)
        if candidates:
            detail = "; known implementations: " + ", ".join(
                f"{r.solver_id}={r.level.value}" for r in candidates
            )
        return (
            f"evidence gate refused {admission.capability_id}/{admission.solver_id} "
            f"[{shown}] tier={self.policy.tier}: {reason}{detail}"
        )

    def explain(self, capability_id: str) -> dict:
        """Full admissibility table of a capability (audit / report artifact)."""
        return {
            "capability_id": capability_id,
            "policy": self.policy.as_dict(),
            "candidates": [
                self.admit(capability_id, record.solver_id).as_dict()
                for record in self.ledger.candidates(capability_id)
            ],
        }