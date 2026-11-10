"""Zero-rewrite gate for revival requests.

A revival may only re-measure the *pre-registered* criterion of a registered
FAIL.  The request body therefore carries exactly two digests -- the criterion
digest and the input digest -- and nothing else.  Anything that could move the
goalposts (a threshold, a tolerance, a metric name, a verdict, a code-version
override) is rejected with a classified reason, and the rejection is a
``GateDecision`` (never an exception the caller could swallow).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .model import FailRecord, RevivalRequest, is_hex64

GATE_SCHEMA = "revival_gate.v1"

#: reason codes -- one per rejection class, used by the acceptance audit
EDITABLE_FIELD_FORBIDDEN = "EDITABLE_FIELD_FORBIDDEN"
HASH_FORMAT_INVALID = "HASH_FORMAT_INVALID"
CRITERION_HASH_MISMATCH = "CRITERION_HASH_MISMATCH"
UNREGISTERED_FAIL = "UNREGISTERED_FAIL"
INPUT_HASH_MISSING = "INPUT_HASH_MISSING"
INPUT_HASH_INVALID = "INPUT_HASH_INVALID"
VERDICT_INJECTION_FORBIDDEN = "VERDICT_INJECTION_FORBIDDEN"
ACCEPTED = "ACCEPTED"

#: fields whose presence anywhere in the payload is a rewrite attempt
FORBIDDEN_FIELD_HINTS = (
    "threshold", "tolerance", "allowed", "limit", "target", "criterion_text",
    "metric", "verdict", "band", "margin", "code_version", "caliber",
)


@dataclass(frozen=True)
class GateDecision:
    accepted: bool
    reason: str
    detail: str
    fail_key: str | None = None
    criterion_sha256: str | None = None
    input_sha256: str | None = None
    schema: str = GATE_SCHEMA

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "accepted": self.accepted,
            "reason": self.reason,
            "detail": self.detail,
            "fail_key": self.fail_key,
            "criterion_sha256": self.criterion_sha256,
            "input_sha256": self.input_sha256,
        }


def request_from_payload(payload: Mapping[str, Any]) -> RevivalRequest:
    """Split a raw request payload into the allowed core and the extras."""
    known = set(RevivalRequest.ALLOWED)
    core = {k: payload.get(k) for k in known if k in payload}
    extra = {k: v for k, v in payload.items() if k not in known}
    return RevivalRequest(
        fail_key=core.get("fail_key") or "",
        criterion_sha256=core.get("criterion_sha256") or "",
        input_sha256=core.get("input_sha256") or "",
        registry_criterion_sha256=core.get("registry_criterion_sha256"),
        extra=extra,
    )


def zero_rewrite_gate(request: RevivalRequest, registry: Mapping[str, FailRecord]) -> GateDecision:
    """Decide whether a revival request may be admitted (pure, deterministic)."""
    # 1. any extra field at all is a rewrite attempt; verdict/threshold get their
    #    own, more specific code so the acceptance audit can classify the reason.
    forbidden = request.forbidden_fields()
    if forbidden:
        lowered = [f.lower() for f in forbidden]
        if any("verdict" in f for f in lowered):
            return GateDecision(False, VERDICT_INJECTION_FORBIDDEN,
                                "payload carries a verdict field: %s" % sorted(forbidden))
        if any(any(h in f for h in FORBIDDEN_FIELD_HINTS) for f in lowered):
            return GateDecision(False, EDITABLE_FIELD_FORBIDDEN,
                                "payload carries editable field(s): %s" % sorted(forbidden))
        return GateDecision(False, EDITABLE_FIELD_FORBIDDEN,
                            "payload carries field(s) outside the allowed core: %s" % sorted(forbidden))

    # 2. digest well-formedness
    if not is_hex64(request.criterion_sha256):
        return GateDecision(False, HASH_FORMAT_INVALID,
                            "criterion_sha256 is not a 64-hex digest")
    if not request.input_sha256:
        return GateDecision(False, INPUT_HASH_MISSING, "input_sha256 is required")
    if not is_hex64(request.input_sha256):
        return GateDecision(False, INPUT_HASH_INVALID,
                            "input_sha256 is not a 64-hex digest")

    # 3. the request must name a registered FAIL
    record = registry.get(request.fail_key)
    if record is None:
        return GateDecision(False, UNREGISTERED_FAIL,
                            "fail_key %r is not in the registry" % (request.fail_key,))

    # 4. the criterion digest must be the registered one -- byte for byte
    if request.criterion_sha256 != record.criterion_sha256:
        return GateDecision(False, CRITERION_HASH_MISMATCH,
                            "request digest %s != registered %s"
                            % (request.criterion_sha256[:12], record.criterion_sha256[:12]))
    if request.registry_criterion_sha256 and request.registry_criterion_sha256 != record.criterion_sha256:
        return GateDecision(False, CRITERION_HASH_MISMATCH,
                            "caller claims a registry digest that does not match")

    return GateDecision(True, ACCEPTED, "request matches the registered criterion",
                        fail_key=record.key(), criterion_sha256=record.criterion_sha256,
                        input_sha256=request.input_sha256)