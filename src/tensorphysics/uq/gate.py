"""Pre-evidence gate: no engineering error claim without a UQ budget.

The engineering rule of the campaign is blunt -- *a comparison whose error
exceeds 15 % is not acceptable*.  On its own that rule is unenforceable, because
"the error is 4 %" is a statement about two numbers, neither of which is exact:
the simulation has numerical uncertainty and the inputs have a spread.  This
module makes the UQ budget a **precondition** of the error adjudication:

    claim (relative error)  +  UQ study  ->  PASS / FAIL / WITHHELD
                                              ^
                                              +-- WITHHELD if the UQ study is
                                                  itself withheld, or if no
                                                  study is presented at all

Two consequences are deliberate:

* A claim without a UQ record is *withheld*, not passed.  The gate cannot be
  satisfied by the absence of evidence.
* The UQ half-width is exported as the ASME V&V 20 ``U_input`` term, so the same
  record can be chained into :func:`tensorphysics.vv.validation.validate` next
  to the discretisation uncertainty ``U_num`` (GCI).

Every decision carries the sha256 digest of its :class:`PreEvidence` payload, so
a number printed in a report can be traced back to the exact UQ evidence that
admitted it.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

from .verdict import FAIL, PASS, WITHHELD, UQVerdict

__all__ = [
    "PreEvidence",
    "GateDecision",
    "GateReason",
    "SCHEMA_VERSION",
    "UQEvidenceError",
    "canonical_digest",
    "chain_asme_validation",
    "engineering_error_gate",
    "pre_evidence",
    "require_pass",
]

#: Version tag of the evidence payload; bump when a field changes meaning.
SCHEMA_VERSION = "tensorphysics.uq.pre_evidence/v1"

#: The engineering acceptance threshold: a band or an error above it is FAIL.
DEFAULT_THRESHOLD = 0.15


class UQEvidenceError(RuntimeError):
    """Raised when a claim is presented without admissible UQ evidence."""


def canonical_digest(payload: Any) -> str:
    """sha256 of the canonical JSON encoding (sorted keys, 12-digit floats)."""
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _json_default(value: Any) -> Any:
    try:  # numpy scalars / arrays
        return value.item()
    except AttributeError:
        pass
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serialisable: {type(value)!r}")


@dataclass(frozen=True)
class PreEvidence:
    """Portable record of one UQ study, digest-addressed."""

    status: str
    threshold: float
    reason_codes: tuple[str, ...]
    blocking_codes: tuple[str, ...]
    sample_hash: str
    spec_fingerprint: str
    n_samples: int
    degree: int
    eps_loo: float
    n_valid: int
    indices: dict[str, Any] = field(default_factory=dict)
    cross_check: dict[str, Any] | None = None
    budget: dict[str, Any] | None = None
    solver_evaluations: int = 0
    label: str = ""
    schema: str = SCHEMA_VERSION
    created_at: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "label": self.label,
            "status": self.status,
            "threshold": self.threshold,
            "reason_codes": list(self.reason_codes),
            "blocking_codes": list(self.blocking_codes),
            "sample_hash": self.sample_hash,
            "spec_fingerprint": self.spec_fingerprint,
            "n_samples": int(self.n_samples),
            "degree": int(self.degree),
            "eps_loo": self.eps_loo,
            "n_valid": int(self.n_valid),
            "indices": self.indices,
            "cross_check": self.cross_check,
            "budget": self.budget,
            "solver_evaluations": int(self.solver_evaluations),
            "extra": self.extra,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.payload())

    @property
    def admissible(self) -> bool:
        """Only a PASS study may support an engineering claim."""
        return self.status == PASS

    def as_dict(self) -> dict[str, Any]:
        out = self.payload()
        out["created_at"] = self.created_at
        out["digest"] = self.digest
        return out


def pre_evidence(study: Any, *, label: str = "", extra: Mapping[str, Any] | None = None) -> PreEvidence:
    """Build the digest-addressed evidence record from a ``UQStudy``."""
    verdict: UQVerdict = study.verdict
    indices: dict[str, Any] = {}
    if study.pce_sobol is not None:
        indices["pce"] = study.pce_sobol.as_dict(study.spec.names)
    if study.mc is not None:
        indices["mc"] = study.mc.as_dict(study.spec.names)
    budget = None if study.budget is None else study.budget.as_dict()
    cross = None
    if study.cross_check is not None:
        cross = study.cross_check.as_dict()
    elif verdict.details.get("cross_check") is not None:
        cross = verdict.details["cross_check"]
    solver_evaluations = int(study.results.n)
    if study.mc is not None:
        solver_evaluations += int(study.mc.n_model_evaluations)
    return PreEvidence(
        status=str(verdict.status),
        threshold=float(verdict.threshold),
        reason_codes=tuple(verdict.reason_codes),
        blocking_codes=tuple(verdict.blocking_codes),
        sample_hash=str(study.samples.hash),
        spec_fingerprint=str(study.spec.fingerprint()),
        n_samples=int(study.samples.n),
        degree=int(study.surrogate.degree) if study.surrogate is not None else 0,
        eps_loo=float(study.surrogate.eps_loo) if study.surrogate is not None else math.inf,
        n_valid=int(verdict.details.get("n_valid") or 0),
        indices=indices,
        cross_check=cross,
        budget=budget,
        solver_evaluations=solver_evaluations,
        label=label,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        extra=dict(extra or {}),
    )


@dataclass(frozen=True)
class GateReason:
    """Why the gate decided what it decided."""

    code: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message}


@dataclass(frozen=True)
class GateDecision:
    """Outcome of the pre-evidence gate for one engineering claim."""

    status: str
    threshold: float
    reasons: tuple[GateReason, ...] = ()
    evidence_digest: str = ""
    u_input: float = 0.0
    relative_error: float | None = None
    band_relative_width: float | None = None
    label: str = ""

    @property
    def passed(self) -> bool:
        return self.status == PASS

    @property
    def codes(self) -> tuple[str, ...]:
        return tuple(r.code for r in self.reasons)

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "status": self.status,
            "threshold": float(self.threshold),
            "relative_error": self.relative_error,
            "band_relative_width": self.band_relative_width,
            "u_input": float(self.u_input),
            "evidence_digest": self.evidence_digest,
            "codes": list(self.codes),
            "reasons": [r.as_dict() for r in self.reasons],
        }


def engineering_error_gate(
    *,
    study: Any | None,
    relative_error: float | None = None,
    threshold: float = DEFAULT_THRESHOLD,
    label: str = "",
    evidence: PreEvidence | None = None,
) -> GateDecision:
    """Adjudicate an engineering claim against the UQ budget and the 15 % rule.

    ``relative_error`` is the claim under test (``|f_sim - f_ref| / f_ref``).
    When it is ``None`` only the width of the UQ band is adjudicated, which is
    the "no reference value yet" case: the band itself may not exceed the
    threshold.
    """
    if threshold <= 0.0 or not math.isfinite(threshold):
        raise ValueError(f"threshold must be positive and finite, got {threshold!r}")
    if relative_error is not None:
        if not isinstance(relative_error, (int, float)) or not math.isfinite(relative_error):
            raise ValueError(f"relative_error must be finite, got {relative_error!r}")
        relative_error = abs(float(relative_error))

    if study is None and evidence is None:
        return GateDecision(
            status=WITHHELD,
            threshold=threshold,
            reasons=(
                GateReason(
                    "UQ-G01-NO-EVIDENCE",
                    "no uncertainty budget was presented: an error figure without a UQ "
                    "record cannot be adjudicated",
                ),
            ),
            u_input=0.0,
            relative_error=relative_error,
            label=label,
        )
    record = evidence if evidence is not None else pre_evidence(study, label=label)
    digest = record.digest
    band = None
    u_input = 0.0
    if record.budget is not None:
        band = float(record.budget.get("relative_width", math.nan))
        u_input = 0.5 * float(record.budget.get("width", 0.0))

    reasons: list[GateReason] = []
    if record.status == WITHHELD:
        reasons.append(
            GateReason(
                "UQ-G02-UQ-WITHHELD",
                "the UQ study itself is withheld ("
                + ", ".join(record.blocking_codes or record.reason_codes)
                + "): no claim can be certified",
            )
        )
    if record.status == FAIL:
        reasons.append(
            GateReason(
                "UQ-G04-BAND",
                f"the 95 % UQ band is {band:.4g} of the mean, above the "
                f"{threshold:.3g} engineering threshold",
            )
        )
    if relative_error is not None and relative_error > threshold:
        reasons.append(
            GateReason(
                "UQ-G03-ERROR",
                f"the comparison error {relative_error:.4g} exceeds the "
                f"{threshold:.3g} engineering threshold",
            )
        )
    if relative_error is None and record.status == PASS and (
        band is None or not math.isfinite(band)
    ):
        reasons.append(
            GateReason("UQ-G05-NO-BAND", "the UQ record carries no output band to adjudicate")
        )

    status = WITHHELD if any(r.code.startswith(("UQ-G01", "UQ-G02", "UQ-G05")) for r in reasons) else (
        FAIL if reasons else PASS
    )
    return GateDecision(
        status=status,
        threshold=threshold,
        reasons=tuple(reasons),
        evidence_digest=digest,
        u_input=u_input,
        relative_error=relative_error,
        band_relative_width=None if band is None or not math.isfinite(band) else band,
        label=label,
    )


def require_pass(decision: GateDecision) -> GateDecision:
    """Raise :class:`UQEvidenceError` unless the gate passed."""
    if not decision.passed:
        raise UQEvidenceError(
            f"gate did not pass ({decision.status}): "
            + "; ".join(f"{r.code}: {r.message}" for r in decision.reasons)
        )
    return decision


def chain_asme_validation(
    *,
    study: Any,
    f_sim: float,
    f_exp: float,
    u_num: float,
    threshold: float = DEFAULT_THRESHOLD,
    label: str = "",
) -> dict[str, Any]:
    """Chain the UQ record into the ASME V&V 20 validation comparison.

    ``u_input`` is the UQ half-width (``U = 0.5 * band width``), ``u_num`` is the
    caller's numerical uncertainty (typically the GCI of the finest grid).  The
    returned dict keeps both verdicts: the ASME comparison and the pre-evidence
    gate, so a report cannot quote one without the other.
    """
    from ..vv.validation import validate

    record = pre_evidence(study, label=label)
    u_input = 0.0 if record.budget is None else 0.5 * float(record.budget.get("width", 0.0))
    asme = validate(f_sim, f_exp, u_num=u_num, u_input=u_input, label=label)
    relative_error = 0.0 if f_exp == 0.0 else abs(f_sim - f_exp) / abs(f_exp)
    decision = engineering_error_gate(
        study=study, relative_error=relative_error, threshold=threshold, label=label
    )
    return {
        "label": label,
        "asme": {
            "f_sim": asme.f_sim,
            "f_exp": asme.f_exp,
            "u_num": asme.u_num,
            "u_input": asme.u_input,
            "u_val": asme.u_val,
            "error": asme.error,
            "ratio": asme.ratio,
            "passed": bool(asme.passed),
        },
        "gate": decision.as_dict(),
        "evidence": record.as_dict(),
        "combined_status": (
            PASS if (asme.passed and decision.passed) else (WITHHELD if decision.status == WITHHELD else FAIL)
        ),
    }